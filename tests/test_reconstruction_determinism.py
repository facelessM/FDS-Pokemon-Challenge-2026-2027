"""Regression checks for near-tied roster selection and unordered move evidence."""
import copy,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd

STUDY=Path(__file__).resolve().parents[1]/'experiments/state_reconstruction'
sys.path.insert(0,str(STUDY/'scratchboard'));sys.path.insert(0,str(STUDY/'deep_features'))
from team_context import top_candidate_indices
from move_records import MoveRecords
from data_views import describe,observation_features,identity_features
from features import resources

class ReconstructionDeterminism(unittest.TestCase):
    def test_numerical_ties_use_canonical_roster_order(self):
        self.assertEqual(top_candidate_indices([.1,.1+1e-15,.8],2).tolist(),[2,0])
        self.assertEqual(top_candidate_indices([.1,.1001,.7999],2).tolist(),[2,1])

    def test_equivalent_move_order_does_not_change_records(self):
        views=[];labels=[]
        for i in range(30):
            team=tuple(sorted(['a','b','c','d','e'+str(i%5),'f'+str(i%5)]))
            moves={'a':['move3','move1','move2'],'b':['move4','move2']}
            states={n:{'status':'nostatus','hp_pct':.7} for n in ['a','b']}
            views.append(dict(team=team,items=tuple((n,'leftovers',100) for n in team),lead='a',seen=frozenset(['a','b']),moves={'p1':copy.deepcopy(moves),'p2':copy.deepcopy(moves)},states={'p1':copy.deepcopy(states),'p2':copy.deepcopy(states)}))
            labels.append(int((i%7)<3))
        other=copy.deepcopy(views)
        for view in other:
            for side in ['p1','p2']:
                view['moves'][side]={name:list(reversed(ms)) for name,ms in reversed(list(view['moves'][side].items()))}
        model=MoveRecords().fit(views,np.array(labels));alternative=MoveRecords().fit(other,np.array(labels))
        pd.testing.assert_frame_equal(model.transform(views),model.transform(other))
        pd.testing.assert_frame_equal(model.transform(views),alternative.transform(other))
        pd.testing.assert_frame_equal(model.transform(views),model.transform(list(reversed(views))).iloc[::-1].reset_index(drop=True))

    def test_feature_reads_do_not_create_unknown_move_observations(self):
        def state(name,status):return dict(name=name,status=status,hp_pct=.8,effects=[],boosts={'atk':0,'def':0,'spa':0,'spd':0,'spe':0})
        battle={'battle_id':1,'p1_team_details':[dict(name=n,item='leftovers',level=100) for n in ['a','b','c','d','e','f']],'p2_lead_details':{'name':'g'},'battle_timeline':[]}
        for turn in range(1,31):battle['battle_timeline'].append(dict(turn=turn,p1_pokemon_state=state('a','slp'),p2_pokemon_state=state('g','nostatus'),p1_move_details=None,p2_move_details=None))
        view=describe(battle);before=copy.deepcopy(view)
        for function in [observation_features,identity_features,resources]:
            function(battle,view)
            self.assertEqual(view,before)
        self.assertEqual(view['moves'],{'p1':{},'p2':{}})

if __name__=='__main__':unittest.main()
