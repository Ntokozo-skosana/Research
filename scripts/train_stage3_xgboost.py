"""Stage 3: nested-CV XGBoost and controlled sleep-domain ablations.
Run: py train_stage3_xgboost.py --data ../haalsi_final_model_dataset_404.csv
Use the same participant-level outer folds as Stage 2 (seed 42).
This analysis predicts W2 MMSE, not a diagnosis of MCI.
"""
from dataset_preflight import check as check_cleaned_dataset
import argparse, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

CORE=['age_w2','sex','education_w1','cesd_w2','wealth_quintile_w2']
ENG=['duration_mean','duration_change','duration_variability','quality_mean','quality_change','insufficiency_mean','insufficiency_change','bedtime_mean_shifted','bedtime_change']
HIST=[f'{s}_w{w}' for s in ('duration','quality','insufficiency','bedtime') for w in (1,2)]
BASE=['mmse_w1']+CORE
SETS={
 'S0_baseline_mmse':['mmse_w1'],
 'S1_baseline_core':BASE,
 'S2_baseline_core_engineered_sleep':BASE+ENG,
 'S3_baseline_core_historical_sleep':BASE+HIST,
 'A1_without_duration':BASE+[f for f in ENG if not f.startswith('duration_')],
 'A2_without_quality':BASE+[f for f in ENG if not f.startswith('quality_')],
 'A3_without_insufficiency':BASE+[f for f in ENG if not f.startswith('insufficiency_')],
 'A4_without_bedtime':BASE+[f for f in ENG if not f.startswith('bedtime_')],
}
CAT={'sex','education_w1','wealth_quintile_w2'}
GRID={'model__n_estimators':[100,250], 'model__max_depth':[2,3], 'model__learning_rate':[0.03,0.1], 'model__min_child_weight':[3], 'model__subsample':[0.85], 'model__colsample_bytree':[0.85]}

def make_pipeline(features):
    numeric=[c for c in features if c not in CAT]
    categorical=[c for c in features if c in CAT]
    pre=ColumnTransformer([
        ('numeric',SimpleImputer(strategy='median',add_indicator=True),numeric),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)
    ],sparse_threshold=0)
    return Pipeline([('prep',pre),('model',XGBRegressor(objective='reg:squarederror',random_state=42,n_jobs=1,tree_method='hist',verbosity=0))])

def main(args):
    data=Path(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=check_cleaned_dataset(data,SETS,Path(__file__).resolve().parent/"haalsi_original_reference.csv")
    if df['prim_key'].isna().any() or df['prim_key'].duplicated().any(): raise ValueError('Participant IDs missing or duplicated')
    required={'prim_key','mmse_w1','mmse_w2','delta_mmse'}|{c for fs in SETS.values() for c in fs}
    if required-set(df.columns):raise ValueError(f'Missing columns: {sorted(required-set(df.columns))}')
    if df[['mmse_w1','mmse_w2']].isna().any().any():raise ValueError('Missing MMSE scores')
    if not np.allclose(df.mmse_w2-df.mmse_w1,df.delta_mmse):raise ValueError('Inconsistent MMSE change')
    y=df.mmse_w2.to_numpy(float)
    outer=list(KFold(n_splits=5,shuffle=True,random_state=42).split(df))
    inner=KFold(n_splits=3,shuffle=True,random_state=19)
    rows=[];pred=[]
    for fs,features in SETS.items():
        X=df[features]
        for fold,(train,test) in enumerate(outer,1):
            search=GridSearchCV(make_pipeline(features),GRID,scoring='neg_mean_absolute_error',cv=inner,n_jobs=1,refit=True)
            search.fit(X.iloc[train],y[train]);yhat=search.predict(X.iloc[test])
            rows.append({'feature_set':fs,'model':'xgboost','fold':fold,'n_test':len(test),
                'mae':mean_absolute_error(y[test],yhat),'rmse':np.sqrt(mean_squared_error(y[test],yhat)),
                'r2':r2_score(y[test],yhat),'params':json.dumps(search.best_params_,sort_keys=True)})
            for ix,v in zip(test,yhat):pred.append({'prim_key':df.iloc[ix].prim_key,'feature_set':fs,'model':'xgboost',
                'fold':fold,'actual_mmse_w2':y[ix],'predicted_mmse_w2':v,'mmse_w1':df.iloc[ix].mmse_w1})
        print('Finished',fs,flush=True)
    pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
    predictions=pd.DataFrame(pred);predictions.to_csv(out/'out_of_fold_predictions.csv',index=False)
    summary=[]
    for fs,g in predictions.groupby('feature_set',sort=False):
        summary.append({'feature_set':fs,'model':'xgboost','n':len(g),
            'oof_mae':mean_absolute_error(g.actual_mmse_w2,g.predicted_mmse_w2),
            'oof_rmse':np.sqrt(mean_squared_error(g.actual_mmse_w2,g.predicted_mmse_w2)),
            'oof_r2':r2_score(g.actual_mmse_w2,g.predicted_mmse_w2)})
    summary=pd.DataFrame(summary);summary.to_csv(out/'summary.csv',index=False)
    wide=predictions.pivot(index='prim_key',columns='feature_set',values='predicted_mmse_w2')
    actual=predictions[predictions.feature_set=='S1_baseline_core'].set_index('prim_key')['actual_mmse_w2'].reindex(wide.index)
    pairs=[('S0_baseline_mmse','S1_baseline_core'),('S1_baseline_core','S2_baseline_core_engineered_sleep'),('S1_baseline_core','S3_baseline_core_historical_sleep')]
    pairs += [('S2_baseline_core_engineered_sleep',name) for name in SETS if name.startswith('A')]
    # Bootstrap the PAIRED participant-level loss differences; uncertainty is descriptive,
    # not a correction for model selection or repeated CV dependence.
    rng=np.random.default_rng(42);comparisons=[]
    for base,extra in pairs:
        differences=(actual-wide[extra]).abs().to_numpy()-(actual-wide[base]).abs().to_numpy()
        draws=rng.integers(0,len(differences),size=(2000,len(differences)))
        boot=differences[draws].mean(axis=1)
        comparisons.append({'baseline':base,'augmented':extra,'n':len(differences),
            'baseline_mae':float((actual-wide[base]).abs().mean()),'augmented_mae':float((actual-wide[extra]).abs().mean()),
            'delta_mae_augmented_minus_baseline':float(differences.mean()),
            'bootstrap_ci_low':float(np.percentile(boot,2.5)),'bootstrap_ci_high':float(np.percentile(boot,97.5))})
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    # Compare with Stage 2 existing out-of-fold predictions if provided.
    if args.stage2_predictions:
        old=pd.read_csv(args.stage2_predictions)
        baseline=old[old.feature_set.isin(['S0_baseline_mmse','S1_baseline_core','S2_baseline_core_engineered_sleep','S3_baseline_core_historical_sleep'])]
        common=summary[summary.feature_set.isin(baseline.feature_set.unique())]
        previous=baseline.groupby(['feature_set','model']).apply(lambda g:pd.Series({'oof_mae':mean_absolute_error(g.actual_mmse_w2,g.predicted_mmse_w2),'oof_rmse':np.sqrt(mean_squared_error(g.actual_mmse_w2,g.predicted_mmse_w2)),'oof_r2':r2_score(g.actual_mmse_w2,g.predicted_mmse_w2)}),include_groups=False).reset_index()
        pd.concat([previous,common],ignore_index=True).sort_values(['feature_set','oof_mae']).to_csv(out/'stage2_vs_xgboost.csv',index=False)
    manifest={'n_participants':len(df),'target':'mmse_w2','outer_cv':'5-fold KFold seed 42','inner_cv':'3-fold KFold seed 19','selection_metric':'MAE',
        'grid':GRID,'feature_sets':SETS,'python':platform.python_version(),'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,
        'warning':'S2 and ablations use W3 sleep; core covariate timing needs review. These models are not necessarily prospective forecasts.',
        'note':'Bootstrap CIs quantify participant-level paired loss uncertainty but do not correct for nested-CV/model-selection dependence.'}
    (out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\n',summary.to_string(index=False))
    print('\nPaired comparisons:\n',pd.DataFrame(comparisons).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=str(Path(__file__).resolve().parent.parent/'haalsi_final_model_dataset_404.csv'))
    parser.add_argument('--out',default=str(Path(__file__).resolve().parent/'results_cleaned_stage3'))
    parser.add_argument('--stage2-predictions',default=None)
    main(parser.parse_args())
