"""Competition-only views. No external Pokemon tables or move corrections."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


def load(path):
    return [json.loads(line) for line in open(path)]


def describe(b):
    team = tuple(sorted(p['name'] for p in b['p1_team_details']))
    items = tuple(sorted((p['name'], p.get('item', ''), p['level']) for p in b['p1_team_details']))
    states = {}; moves = {}; first = {}; last_at = {}
    for side in ('p1', 'p2'):
        states[side] = {}; moves[side] = defaultdict(set); first[side] = {}; last_at[side] = {}
        for i, t in enumerate(b['battle_timeline']):
            st = t[side+'_pokemon_state']; m = t[side+'_move_details']; n = st['name']
            states[side][n] = st; first[side].setdefault(n, i); last_at[side][n] = i
            if m: moves[side][n].add(m['name'])
    seen = frozenset(states['p2']) | {b['p2_lead_details']['name']}
    # Unknown moves must stay absent. Reading a feature must never insert an
    # empty species into the learned move-prior population.
    moves = {side: dict(by_species) for side, by_species in moves.items()}
    return dict(team=team, items=items, lead=b['p1_team_details'][0]['name'],
                p2lead=b['p2_lead_details']['name'], seen=seen, states=states,
                moves=moves, first=first, last_at=last_at,
                levels={p['name']:p['level'] for p in b['p1_team_details']},
                group=hashlib.sha256(json.dumps(b['battle_timeline'],sort_keys=True).encode()).hexdigest())


def observation_features(b, v):
    """Add observation channels without altering the documented move fields."""
    f = {}; tl = b['battle_timeline']
    for side, opp in [('p1','p2'),('p2','p1')]:
        prefix = side+'_obs_'; lastmove = {}; same_dmg=[]; change_dmg=[]; changed_moves=0
        counters=Counter(); moves_by_species=v['moves'][side]
        for i,t in enumerate(tl):
            s=t[side+'_pokemon_state']; m=t[side+'_move_details']; n=s['name']
            if i:
                prev=tl[i-1]; a=prev[side+'_pokemon_state']; d=prev[opp+'_pokemon_state']; e=t[opp+'_pokemon_state']
                changed=a['name']!=n
                counters['switches']+=changed
                counters['return_move_repeat']+=bool(changed and m and lastmove.get(n)==m)
                counters['sleep_repeat']+=bool(not changed and a['status']=='slp' and s['status']=='slp' and m and lastmove.get(n)==m)
                changed_moves+=bool(m and m!=lastmove.get(n))
                counters['after_faint_change']+=bool(changed and a['status']=='fnt')
                if d['name']==e['name']:
                    change=d['hp_pct']-e['hp_pct']
                    (change_dmg if changed else same_dmg).append(change)
                if i>=20:
                    counters['last10_switches']+=changed
                    counters['last10_sleep']+=s['status']=='slp'
                    counters['last10_new_moves']+=bool(m and m!=lastmove.get(n))
            lastmove[n]=m
        f.update({prefix+k:float(val) for k,val in counters.items()})
        f[prefix+'new_move_observations']=changed_moves
        f[prefix+'distinct_species_moves']=sum(len(m) for m in moves_by_species.values())
        for name,values in [('continuous_net_hp',same_dmg),('switch_net_hp',change_dmg)]:
            f[prefix+name+'_sum']=sum(values);f[prefix+name+'_mean']=float(np.mean(values)) if values else 0
        active=tl[-1][side+'_pokemon_state']
        for stat,val in active['boosts'].items():f[prefix+'active_'+stat]=val
        for effect in ['substitute','trapped','perish','leechseed','confusion','encore','destinybond']:
            f[prefix+'active_'+effect]=int(effect in active['effects'])
        states=v['states'][side]
        alive={n:s for n,s in states.items() if s['status']!='fnt'}
        for age in [3,10,20]:
            old=[s for n,s in alive.items() if 29-v['last_at'][side][n]>=age]
            f[prefix+f'old{age}_count']=len(old);f[prefix+f'old{age}_hp']=sum(s['hp_pct'] for s in old)
            f[prefix+f'old{age}_low']=sum(s['hp_pct']<.3 for s in old)
        for role,names in [('rest',{'rest'}),('talk',{'sleeptalk'}),('spin',{'rapidspin'}),('spikes',{'spikes'}),('boom',{'explosion','selfdestruct'}),('phaze',{'roar','whirlwind'}),('bell',{'healbell'})]:
            users=[n for n,ms in moves_by_species.items() if ms&names]
            living=[n for n in users if n in alive]
            f[prefix+role+'_known']=len(users);f[prefix+role+'_alive']=len(living)
            f[prefix+role+'_hp']=sum(alive[n]['hp_pct'] for n in living)
        f[prefix+'asleep_resttalk']=sum(s['status']=='slp' and {'rest','sleeptalk'}<=moves_by_species.get(n,set()) for n,s in alive.items())
        f[prefix+'asleep_no_observed_talk']=sum(s['status']=='slp' and 'sleeptalk' not in moves_by_species.get(n,set()) for n,s in alive.items())
        f[prefix+'active_stint']=next((j for j in range(1,30) if tl[-1-j][side+'_pokemon_state']['name']!=active['name']),30)
        f[prefix+'hp_sq']=sum(s['hp_pct']**2 for s in alive.values())
        f[prefix+'low_hp_count']=sum(s['hp_pct']<.25 for s in alive.values())
    for suffix in sorted({k[len('p1_obs_'):] for k in f if k.startswith('p1_obs_')}):
        f['diff_obs_'+suffix]=f.get('p1_obs_'+suffix,0)-f.get('p2_obs_'+suffix,0)
    return f


def identity_features(b,v):
    out={}
    for side in ['p1','p2']:
        s=b['battle_timeline'][-1][side+'_pokemon_state']
        out[side+'_active_species']=s['name'];out[side+'_active_status']=s['status']
        out[side+'_active_moves']='|'.join(sorted(v['moves'][side].get(s['name'],set())))
    out['p1_roster']='|'.join(v['team']);out['p1_items']=str(v['items'])
    out['p1_lead']=v['lead'];out['p2_lead']=v['p2lead']
    return out


def sparse_tokens(v):
    out={}
    for side in ['p1','p2']:
        roster=set(v['team']) if side=='p1' else set(v['seen'])
        for n in roster:
            s=v['states'][side].get(n);hp=s['hp_pct'] if s else (1. if side=='p1' else 0.)
            out[f'{side}:species:{n}']=1.;out[f'{side}:hp:{n}']=hp
            for move in v['moves'][side].get(n,[]):out[f'{side}:set:{n}:{move}']=1.
        for a,b in __import__('itertools').combinations(sorted(roster),2):out[f'{side}:pair:{a}:{b}']=1.
    return out
