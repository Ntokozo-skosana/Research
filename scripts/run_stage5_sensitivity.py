"""Stage 5 sensitivity: separate missingness SHAP and test LIME seed stability.
Run beside Stage 5 training modules and dataset_preflight.py.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from lime.lime_tabular import LimeTabularExplainer

EXPECTED_SHA='77E2B9D4A6B72D846A9D5767FF86F54C77413536DA30BC1B8D82827F5F46C36E'
SLEEP_TOKENS=('duration','quality','insufficiency','bedtime')

def group_feature(name):
    if 'missingindicator_' in name:return 'sleep_missingness' if any(t in name for t in SLEEP_TOKENS) else 'other_missingness'
    if any(t in name for t in SLEEP_TOKENS):return 'measured_sleep'
    return 'core_predictors'

def main(a):
    import train_baseline_adjusted as s2
    import train_stage3_xgboost as s3
    import run_stage5_explainability as s5
    from dataset_preflight import check
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.shap_global)
    need={'experiment','feature','mean_abs_shap'}
    if not need.issubset(raw.columns):raise ValueError('Invalid SHAP CSV columns')
    raw['feature_group']=raw.feature.map(group_feature)
    grouped=raw.groupby(['experiment','feature_group'],as_index=False).mean_abs_shap.sum()
    grouped.to_csv(out/'shap_feature_groups.csv',index=False)
    totals=grouped.pivot(index='experiment',columns='feature_group',values='mean_abs_shap').fillna(0)
    for col in ['core_predictors','measured_sleep','sleep_missingness','other_missingness']:
        if col not in totals:totals[col]=0.
    totals['total']=totals[['core_predictors','measured_sleep','sleep_missingness','other_missingness']].sum(axis=1)
    totals['sleep_share_pct']=100*(totals.measured_sleep+totals.sleep_missingness)/totals.total
    totals['missingness_share_pct']=100*totals.sleep_missingness/totals.total
    totals.reset_index().to_csv(out/'shap_sleep_missingness_summary.csv',index=False)
    print('\nSHAP feature group summary:')
    print(totals[['measured_sleep','sleep_missingness','sleep_share_pct','missingness_share_pct']].round(4).to_string())

    data=Path(a.data)
    sha=hashlib.sha256(data.read_bytes()).hexdigest().upper()
    if sha!=EXPECTED_SHA:raise ValueError('Cleaned dataset checksum mismatch')
    ref=Path(a.reference)
    df=check(data,s5.SETS,ref)
    folds=list(KFold(n_splits=5,shuffle=True,random_state=42).split(df))
    tr,te=folds[a.fold-1]
    inner=KFold(n_splits=3,shuffle=True,random_state=19)
    results=[];detail=[]
    for key in a.models:
        features=s5.SETS[key]
        if key.startswith('ridge'):
            pipe=Pipeline([('prep',s2.preprocessing(features)),('model',s2.MODELS['ridge'][0])]);grid=s2.MODELS['ridge'][1]
        else:
            pipe=s3.make_pipeline(features);grid=s3.GRID
        fit=GridSearchCV(pipe,grid,cv=inner,scoring='neg_mean_absolute_error',n_jobs=1,refit=True)
        fit.fit(df.iloc[tr][features],df.iloc[tr].mmse_w2.to_numpy(float))
        best=fit.best_estimator_;prep=best.named_steps['prep'];model=best.named_steps['model']
        Xtr=prep.transform(df.iloc[tr][features]);Xte=prep.transform(df.iloc[te][features]);
        if hasattr(Xtr,'toarray'):Xtr=Xtr.toarray()
        if hasattr(Xte,'toarray'):Xte=Xte.toarray()
        Xtr=np.asarray(Xtr,dtype=float);Xte=np.asarray(Xte,dtype=float)
        names=list(prep.get_feature_names_out())
        j=a.example_index
        if j>=len(te):raise ValueError('Example index exceeds held-out fold size')
        baseline=float(model.predict(Xte[j:j+1])[0]); pk=df.iloc[te[j]].prim_key
        for seed in a.seeds:
            explainer=LimeTabularExplainer(Xtr,feature_names=names,mode='regression',discretize_continuous=True,random_state=seed)
            explanation=explainer.explain_instance(Xte[j],model.predict,num_features=min(a.top_k,len(names)),num_samples=a.lime_samples)
            weights=dict(explanation.as_map()[1])
            missing_abs=sum(abs(w) for idx,w in weights.items() if 'missingindicator_' in names[idx])
            total_abs=sum(abs(w) for w in weights.values())
            results.append({'experiment':key,'fold':a.fold,'prim_key':pk,'seed':seed,'lime_score_r2':float(explanation.score),'prediction':baseline,'lime_local_prediction':float(np.ravel(explanation.local_pred)[0]),'abs_local_prediction_error':abs(float(np.ravel(explanation.local_pred)[0])-baseline),'missingness_abs_weight':missing_abs,'total_abs_weight':total_abs,'missingness_weight_share_pct':100*missing_abs/total_abs if total_abs else 0,'top_features':'; '.join(names[i] for i,_ in sorted(weights.items(),key=lambda t:-abs(t[1]))[:5])})
            for idx,w in weights.items():detail.append({'experiment':key,'seed':seed,'feature':names[idx],'lime_weight':float(w),'abs_lime_weight':abs(float(w)),'feature_group':group_feature(names[idx])})
            print(f'{key} seed={seed} LIME R2={explanation.score:.3f} missingness share={100*missing_abs/total_abs if total_abs else 0:.1f}%',flush=True)
    r=pd.DataFrame(results);d=pd.DataFrame(detail)
    r.to_csv(out/'lime_seed_stability_runs.csv',index=False);d.to_csv(out/'lime_seed_feature_weights.csv',index=False)
    summary=r.groupby('experiment',as_index=False).agg(n_seeds=('seed','count'),mean_local_r2=('lime_score_r2','mean'),min_local_r2=('lime_score_r2','min'),max_local_r2=('lime_score_r2','max'),mean_abs_local_prediction_error=('abs_local_prediction_error','mean'),mean_missingness_weight_share_pct=('missingness_weight_share_pct','mean'),min_missingness_weight_share_pct=('missingness_weight_share_pct','min'),max_missingness_weight_share_pct=('missingness_weight_share_pct','max'))
    summary.to_csv(out/'lime_stability_summary.csv',index=False)
    (out/'sensitivity_manifest.json').write_text(json.dumps({'dataset_sha256':sha,'fold':a.fold,'example_index':a.example_index,'seeds':a.seeds,'lime_samples':a.lime_samples,'note':'LIME uses perturbed transformed features; missingness and one-hot perturbations may be unrealistic. R2 is local surrogate fit, not predictive model test R2. SHAP group sums are descriptive, not independent effects.'},indent=2))
    print('\nLIME stability summary:');print(summary.round(4).to_string(index=False));print('\nSaved to:',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--reference',default='haalsi_original_reference.csv');p.add_argument('--shap-global',required=True)
    p.add_argument('--out',default='results_stage5_sensitivity');p.add_argument('--fold',type=int,default=1,choices=[1,2,3,4,5]);p.add_argument('--example-index',type=int,default=0)
    p.add_argument('--models',nargs='+',choices=['ridge_S2','ridge_S3','xgboost_S2','xgboost_S3'],default=['ridge_S3','xgboost_S3'])
    p.add_argument('--seeds',nargs='+',type=int,default=[11,21,42,73,101,202,303,404,505,606]);p.add_argument('--lime-samples',type=int,default=2000);p.add_argument('--top-k',type=int,default=12)
    a=p.parse_args();main(a)
