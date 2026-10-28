"""Entry point for isolated state-reconstruction experiments and predictions."""
import argparse,os,shutil,subprocess,sys
from pathlib import Path
from cache_guard import validate_cache

STUDY=Path(__file__).resolve().parent
REPO=STUDY.parents[1]
SEEDS=[20261026,20261105,193]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['cv','review','predict'])
    ap.add_argument('--data',type=Path,default=REPO/'data')
    ap.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    ap.add_argument('--output',type=Path,default=REPO/'submissions/submission_state_reconstruction_deterministic.csv')
    ap.add_argument('--refresh-cache',action='store_true',help='Archive generated caches and predictions before rebuilding them')
    args=ap.parse_args();data=args.data.resolve();env=dict(os.environ)
    env['POKEMON_DATA_DIR']=str(data);env.setdefault('OPENBLAS_NUM_THREADS','1');env.setdefault('OMP_NUM_THREADS','2');env.setdefault('PYTHONHASHSEED','0')
    def run(script,*params):
        subprocess.run([sys.executable,str(STUDY/script),*map(str,params)],cwd=REPO,env=env,check=True)
    if args.action in ['cv','predict']:
        for name in ['train.jsonl','type_chart_gen2.csv']:
            if not (data/name).is_file():ap.error(f'Missing competition file: {data/name}')
        run('scratchboard/bootstrap_reference.py','--repo',REPO)
        if args.action=='predict' and not (data/'test.jsonl').is_file():ap.error(f'Missing competition file: {data/"test.jsonl"}')
        archived=validate_cache(STUDY,data,args.action,refresh=args.refresh_cache)
        if archived:print(f'Previous cached files: {archived}',flush=True)
    if args.action=='cv':
        for seed in args.seeds:
            run('scratchboard/prepare.py','--data',data,'--seed',seed)
            run('scratchboard/experiment.py','--seed',seed,'--models','base_lgb','base_xgb','combined_lgb','combined_xgb','cat_combined')
            run('deep_features/prepare.py','--seed',seed)
            run('deep_features/auxiliary.py','--seed',seed,'--entry')
            run('deep_features/move_records.py','--seed',seed)
            run('deep_features/experiment.py','--seed',seed,'--models','prior_state_lgb','prior_state_xgb','prior_entryko_lgb','prior_entryko_xgb','deep_entry_lgb','deep_entry_xgb','deep_entry_sets_lgb','deep_entry_sets_xgb')
            run('merge_check/experiment.py','--seed',seed)
        run('deep_features/corrected_baseline.py','--seeds',*args.seeds)
        run('deep_features/summarize.py',*args.seeds)
    if args.action=='review' or (args.action=='cv' and set(SEEDS)<=set(args.seeds)):
        run('deep_features/validate.py');run('merge_check/review.py')
    if args.action=='predict':
        if not (data/'test.jsonl').is_file():ap.error(f'Missing competition file: {data/"test.jsonl"}')
        run('deep_features/predict.py','--selection',STUDY/'merge_check/selection.json')
        import json
        selection=json.loads((STUDY/'merge_check/selection.json').read_text())
        source=STUDY/'deep_features'/selection['filename'];output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
        if output!=source:shutil.copy2(source,output)
        print(f'Submission: {output}')

if __name__=='__main__':main()
