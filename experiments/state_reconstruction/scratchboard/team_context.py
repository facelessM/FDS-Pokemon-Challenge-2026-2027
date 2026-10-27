"""Cross-fitted team records, roster uncertainty, and opponent-adjusted strength."""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def top_candidate_indices(weights, count=3):
    """Break numerically tied posterior weights by canonical roster order.

    Candidate rosters are sorted in candidates(). Rounding only the ranking
    key prevents floating-point accumulation noise from selecting another
    roster at the cutoff; the original probabilities retain their values.
    """
    weights=np.asarray(weights,float)
    return np.lexsort((np.arange(len(weights)),-np.round(weights,12)))[:count]


class TeamContext:
    def __init__(self, enriched=False, moves=False, graph=False):
        self.enriched=enriched;self.use_moves=moves;self.graph=graph

    def candidates(self,v):
        if len(v['seen'])>6:return []
        sets=[self.by.get(n,set()) for n in v['seen']]
        return sorted(set.intersection(*sets)) if sets else []

    def fit(self,views,y):
        self.count=Counter(v['team'] for v in views)
        if self.enriched:self.count.update(tuple(sorted(v['seen'])) for v in views if len(v['seen'])==6)
        self.by=defaultdict(set)
        for k in self.count:
            for n in k:self.by[n].add(k)
        self.games=Counter();self.wins=defaultdict(float);self.item_games=Counter();self.item_wins=defaultdict(float)
        self.lead_games=Counter();self.lead_wins=defaultdict(float)
        self.ms=Counter();self.mt=Counter();self.gs=Counter();self.gt=Counter()
        graphrows=[]
        for v,z in zip(views,y):
            k=v['team'];self.games[k]+=1;self.wins[k]+=float(z)
            ik=v['items'];self.item_games[ik]+=1;self.item_wins[ik]+=float(z)
            lk=(k,v['lead']);self.lead_games[lk]+=1;self.lead_wins[lk]+=float(z)
            ca=self.candidates(v);op=ca[0] if len(ca)==1 else None
            if op is not None:self.games[op]+=1;self.wins[op]+=1-float(z)
            graphrows.append((k,op,int(z)))
            for side,tk in [('p1',k),('p2',op)]:
                for n,ms in sorted(v['moves'][side].items()):
                    if not ms:continue
                    self.gs[n]+=1
                    if tk is not None:self.ms[tk,n]+=1
                    for m in sorted(ms):
                        self.gt[n,m]+=1
                        if tk is not None:self.mt[tk,n,m]+=1
        self.skills={}
        if self.graph:
            ids={k:i for i,k in enumerate(sorted(self.count))};rr=[];cc=[];dd=[];target=[]
            for j,(a,b,z) in enumerate(graphrows):
                rr.append(j);cc.append(ids[a]);dd.append(1.)
                if b is not None:rr.append(j);cc.append(ids[b]);dd.append(-1.)
                target.append(z)
            M=csr_matrix((dd,(rr,cc)),shape=(len(target),len(ids)))
            model=LogisticRegression(C=.5,fit_intercept=True,max_iter=300,solver='lbfgs').fit(M,target)
            self.skills={k:float(model.coef_[0,i]) for k,i in ids.items()}
        return self

    def weights(self,v,ca):
        if not ca:return np.array([])
        log=np.log(np.array([self.count[k] for k in ca],float))
        if self.use_moves:
            for n,ms in sorted(v['moves']['p2'].items()):
                for m in sorted(ms):
                    prior=(self.gt[n,m]+1)/(self.gs[n]+4)
                    log+=np.log([(self.mt[k,n,m]+8*prior)/(self.ms[k,n]+8) for k in ca])
        log-=log.max();w=np.exp(log);return w/w.sum()

    def rate(self,k):return (self.wins.get(k,0)+5)/(self.games.get(k,0)+10)

    def transform(self,views,mode='base'):
        result=[]
        for v in views:
            k=v['team'];r1=self.rate(k);g1=self.games.get(k,0);ca=self.candidates(v)
            w=self.weights(v,ca);rates=np.array([self.rate(q) for q in ca]);g=np.array([self.games.get(q,0) for q in ca])
            best=int(top_candidate_indices(w,1)[0]) if len(w) else -1
            r2=float(rates[best]) if len(w) else .5;g2=float(g[best]) if len(w) else 0.
            out={'p1_team_record':r1,'p1_team_battles':g1,'p2_team_record':r2,'p2_team_battles':g2,'diff_team_record':r1-r2,'p2_team_candidates':len(ca)}
            if mode!='base':
                mu=float(w@rates) if len(w) else .5
                out.update(p2_record_mean=mu,p2_record_sd=float(np.sqrt(w@(rates-mu)**2)) if len(w) else 0.,
                    p2_record_min=float(rates.min()) if len(w) else .5,p2_record_max=float(rates.max()) if len(w) else .5,
                    p2_match_mass=float(w.max()) if len(w) else 0.,p2_match_entropy=float(-(w*np.log(w+1e-15)).sum()) if len(w) else 0.,
                    p2_expected_support=float(w@g) if len(w) else 0.,diff_expected_record=r1-mu,p2_six_known=int(len(v['seen'])==6))
            if mode=='hierarchy':
                ik=v['items'];lk=(k,v['lead']);gi=self.item_games.get(ik,0);gl=self.lead_games.get(lk,0)
                out.update(p1_item_record=(self.item_wins.get(ik,0)+10*r1)/(gi+10),p1_item_support=gi,
                    p1_lead_record=(self.lead_wins.get(lk,0)+10*r1)/(gl+10),p1_lead_support=gl)
                s1=self.skills.get(k,0.);s2=float(w@np.array([self.skills.get(q,0.) for q in ca])) if len(w) else 0.
                out.update(p1_skill=s1,p2_expected_skill=s2,diff_skill=s1-s2,skill_probability=float(expit(s1-s2)))
            result.append(out)
        return pd.DataFrame(result).astype(float)
