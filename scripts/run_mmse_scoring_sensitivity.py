"""Sensitivity analysis: stored versus item-reconstructed Wave 1 MMSE.

Requires train_baseline_adjusted.py and dataset_preflight.py alongside the script,
or provide --stage2-script path. Never edits the source dataset.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--audit',type=Path,required=True,help='w1_participant_scoring_audit.csv')
p.add_argument('--out',type=Path,default=Path('results_mmse_scoring_sensitivity'))
p.add_argument('--stage2-script',type=Path,default=Path('train_baseline_adjusted.py'))
p.add_argument('--bootstrap',type=int,default=5000)
args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(args.stage2_script.resolve().parent))
from train_baseline_adjusted import SETS,preprocessing

raw=pd.read_csv(args.data); audit=pd.read_csv(args.audit)
assert raw.prim_key.notna().all() and raw.prim_key.is_unique
assert audit.prim_key.notna().all() and audit.prim_key.is_unique
assert set(raw.prim_key.astype(str))==set(audit.prim_key.astype(str)), 'Participant ID sets differ'
d=raw.merge(audit[['prim_key','mmse_total_stored','mmse_total_reconstructed','reconstruction_minus_stored','mmse_status']],on='prim_key',validate='one_to_one',sort=False)
assert d.prim_key.tolist()==raw.prim_key.tolist(),'Row order changed'
assert np.allclose(d.mmse_w1,d.mmse_total_stored),'Stored baseline MMSE differs from audit'
assert np.allclose(d.mmse_total_reconstructed-d.mmse_total_stored,d.reconstruction_minus_stored)
assert set(d.reconstruction_minus_stored.unique()).issubset({0.,5.}),'Unexpected reconstruction differences'
assert d.mmse_w2.notna().all() and d.mmse_w1.notna().all()
changed=d.reconstruction_minus_stored.ne(0)
print(f'Validated {len(d)} IDs; reconstructed baseline differs for {changed.sum()} participants')

# Outcome stays the same across scenarios; baseline is the only changed model input.
diagnostics=[]
for label,baseline in [('original',d.mmse_w1),('alternative_reconstructed',d.mmse_total_reconstructed)]:
    delta=d.mmse_w2-baseline
    for group,mask in [('all',np.ones(len(d),dtype=bool)),('affected_62',changed.to_numpy()),('unaffected',(~changed).to_numpy())]:
        diagnostics.append(dict(scenario=label,group=group,n=int(mask.sum()),baseline_mean=float(baseline[mask].mean()),followup_mean=float(d.mmse_w2[mask].mean()),mean_change=float(delta[mask].mean()),median_change=float(delta[mask].median()),baseline_followup_pearson_r=float(np.corrcoef(baseline[mask],d.mmse_w2[mask])[0,1])))
pd.DataFrame(diagnostics).to_csv(args.out/'cognitive_change_summary.csv',index=False)

outer=list(KFold(n_splits=5,shuffle=True,random_state=42).split(d))
inner=KFold(n_splits=3,shuffle=True,random_state=19)
pred=[];fold_rows=[]
for scenario in ['original','alternative_reconstructed']:
    frame=d.copy()
    if scenario=='alternative_reconstructed': frame['mmse_w1']=frame['mmse_total_reconstructed']
    for set_name,features in SETS.items():
        X=frame[features]; y=frame.mmse_w2.to_numpy(dtype=float)
        for fold,(tr,te) in enumerate(outer,1):
            pipe=Pipeline([('prep',preprocessing(features)),('model',Ridge())])
            search=GridSearchCV(pipe,{'model__alpha':[0.1,1,10,100]},cv=inner,scoring='neg_mean_absolute_error',n_jobs=1,refit=True)
            search.fit(X.iloc[tr],y[tr]); yh=search.predict(X.iloc[te]); actual=y[te]
            fold_rows.append(dict(scenario=scenario,feature_set=set_name,fold=fold,n_test=len(te),mae=mean_absolute_error(actual,yh),rmse=np.sqrt(mean_squared_error(actual,yh)),r2=r2_score(actual,yh),best_alpha=search.best_params_['model__alpha']))
            for idx,yy,pp in zip(te,actual,yh):
                pred.append(dict(prim_key=d.iloc[idx].prim_key,scenario=scenario,feature_set=set_name,fold=fold,actual_mmse_w2=float(yy),predicted_mmse_w2=float(pp),affected_w1_scoring=bool(changed.iloc[idx])))
        print('Finished',scenario,set_name,flush=True)
preds=pd.DataFrame(pred);preds.to_csv(args.out/'out_of_fold_predictions.csv',index=False)
pd.DataFrame(fold_rows).to_csv(args.out/'fold_metrics.csv',index=False)
summary=[]
for (sc,fs),g in preds.groupby(['scenario','feature_set'],sort=False):
    summary.append(dict(scenario=sc,feature_set=fs,n=len(g),mae=mean_absolute_error(g.actual_mmse_w2,g.predicted_mmse_w2),rmse=np.sqrt(mean_squared_error(g.actual_mmse_w2,g.predicted_mmse_w2)),r2=r2_score(g.actual_mmse_w2,g.predicted_mmse_w2)))
summary=pd.DataFrame(summary);summary.to_csv(args.out/'model_summary.csv',index=False)

# Bootstrap paired per-person absolute-error differences (no independent fold assumption).
rng=np.random.default_rng(42)
wide=preds.pivot(index='prim_key',columns=['scenario','feature_set'],values='predicted_mmse_w2')
truth=d.set_index('prim_key').loc[wide.index,'mmse_w2'].to_numpy(dtype=float)
comparisons=[]
def compare(label, a, b):
    diff=np.abs(truth-wide[b].to_numpy())-np.abs(truth-wide[a].to_numpy())
    n=len(diff); inds=rng.integers(0,n,size=(args.bootstrap,n)); boot=diff[inds].mean(axis=1)
    comparisons.append(dict(comparison=label,reference=f'{a[0]}:{a[1]}',alternative=f'{b[0]}:{b[1]}',n=n,delta_mae=float(diff.mean()),ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975))))
for fs in SETS:
    compare(f'scoring_change_{fs}',('original',fs),('alternative_reconstructed',fs))
for sc in ['original','alternative_reconstructed']:
    compare(f'{sc}_S2_minus_S1',(sc,'S1_baseline_core'),(sc,'S2_baseline_core_engineered_sleep'))
    compare(f'{sc}_S3_minus_S1',(sc,'S1_baseline_core'),(sc,'S3_baseline_core_historical_sleep'))
pd.DataFrame(comparisons).to_csv(args.out/'paired_bootstrap_comparisons.csv',index=False)

pd.DataFrame({'prim_key':d.prim_key,'original_mmse_w1':d.mmse_w1,'reconstructed_mmse_w1':d.mmse_total_reconstructed,'difference':d.reconstruction_minus_stored,'mmse_w2':d.mmse_w2,'mmse_status_w1':d.mmse_status,'original_delta':d.mmse_w2-d.mmse_w1,'alternative_delta':d.mmse_w2-d.mmse_total_reconstructed}).to_csv(args.out/'participant_score_changes.csv',index=False)
manifest={'analysis':'Alternative Wave 1 MMSE scoring sensitivity; NOT official corrected data','n':len(d),'n_changed':int(changed.sum()),'outcome':'mmse_w2 unchanged','cv':'same 5 outer folds seed 42; 3 inner folds seed 19; Ridge alpha grid 0.1,1,10,100','bootstrap_resamples':args.bootstrap,'caution':'Candidate item-reconstruction anomaly unverified; missingness and timing validity unresolved; do not replace official scores.'}
(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('\nCognitive change:\n',pd.DataFrame(diagnostics).to_string(index=False))
print('\nModel performance:\n',summary.to_string(index=False))
print('\nPaired comparisons:\n',pd.DataFrame(comparisons).to_string(index=False))
print('Saved to:',args.out.resolve())
