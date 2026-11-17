"""Reproducible nested-CV baseline-adjusted HAALSI follow-up MMSE experiment.

Run: python train_baseline_adjusted.py --data ../haalsi_final_model_dataset_404.csv
S2 uses W3 sleep, which may be observed during the cognitive follow-up interval:
it is a follow-up-informed analysis, NOT a baseline-only prospective forecast.
"""
from __future__ import annotations
from dataset_preflight import check as check_cleaned_dataset
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CORE=['age_w2','sex','education_w1','cesd_w2','wealth_quintile_w2']
ENGINEERED=['duration_mean','duration_change','duration_variability','quality_mean','quality_change','insufficiency_mean','insufficiency_change','bedtime_mean_shifted','bedtime_change']
HISTORICAL=[f'{s}_w{w}' for s in ('duration','quality','insufficiency','bedtime') for w in (1,2)]
CATEGORICAL={'sex','education_w1','wealth_quintile_w2'}
SETS={
 'S0_baseline_mmse':['mmse_w1'],
 'S1_baseline_core':['mmse_w1']+CORE,
 'S2_baseline_core_engineered_sleep':['mmse_w1']+CORE+ENGINEERED,
 'S3_baseline_core_historical_sleep':['mmse_w1']+CORE+HISTORICAL,
}
MODELS={
 'dummy':(DummyRegressor(strategy='mean'),{}),
 'ridge':(Ridge(),{'model__alpha':[0.1,1,10,100]}),
 'random_forest':(RandomForestRegressor(random_state=42,n_jobs=1),{'model__n_estimators':[100],'model__max_depth':[3,6],'model__min_samples_leaf':[5,15]}),
}

def preprocessing(features):
    cat=[c for c in features if c in CATEGORICAL]
    num=[c for c in features if c not in CATEGORICAL]
    return ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat),
    ])

def run(data:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    df=check_cleaned_dataset(data,SETS,Path(__file__).resolve().parent/"haalsi_original_reference.csv")
    if df['prim_key'].isna().any() or df['prim_key'].duplicated().any():
        raise ValueError('Participant IDs must be present and unique')
    required=set(['prim_key','mmse_w1','mmse_w2','delta_mmse'])|{v for vals in SETS.values() for v in vals}
    absent=sorted(required-set(df.columns))
    if absent: raise ValueError(f'Missing columns: {absent}')
    if df[['mmse_w1','mmse_w2']].isna().any().any():raise ValueError('Missing MMSE outcomes')
    if not np.allclose(df.mmse_w2-df.mmse_w1,df.delta_mmse):raise ValueError('Inconsistent delta MMSE')
    y=df.mmse_w2.to_numpy(dtype=float)
    outer=list(KFold(n_splits=5,shuffle=True,random_state=42).split(df))
    inner=KFold(n_splits=3,shuffle=True,random_state=19)
    metrics=[];predictions=[]
    for set_name,features in SETS.items():
        X=df[features].copy()
        for model_name,(estimator,grid) in MODELS.items():
            for fold,(tr,te) in enumerate(outer,1):
                pipeline=Pipeline([('prep',preprocessing(features)),('model',estimator)])
                if grid:
                    search=GridSearchCV(pipeline,grid,cv=inner,scoring='neg_mean_absolute_error',n_jobs=1,refit=True)
                    search.fit(X.iloc[tr],y[tr]);fitted=search.best_estimator_;params=search.best_params_
                else:
                    fitted=pipeline.fit(X.iloc[tr],y[tr]);params={}
                p=fitted.predict(X.iloc[te]); actual=y[te]
                metrics.append({'feature_set':set_name,'model':model_name,'fold':fold,'n_test':len(te),
                                'mae':mean_absolute_error(actual,p),'rmse':np.sqrt(mean_squared_error(actual,p)),
                                'r2':r2_score(actual,p),'params':json.dumps(params,sort_keys=True)})
                for index,a,pred in zip(te,actual,p):
                    predictions.append({'prim_key':df.iloc[index]['prim_key'],'feature_set':set_name,'model':model_name,
                                        'fold':fold,'actual_mmse_w2':a,'predicted_mmse_w2':pred,
                                        'mmse_w1':df.iloc[index]['mmse_w1']})
        print('Finished',set_name,flush=True)
    m=pd.DataFrame(metrics);p=pd.DataFrame(predictions)
    m.to_csv(out/'fold_metrics.csv',index=False)
    p.to_csv(out/'out_of_fold_predictions.csv',index=False)
    summary=[]
    for (fs,model),g in p.groupby(['feature_set','model']):
        summary.append({'feature_set':fs,'model':model,'n':len(g),
                        'oof_mae':mean_absolute_error(g.actual_mmse_w2,g.predicted_mmse_w2),
                        'oof_rmse':np.sqrt(mean_squared_error(g.actual_mmse_w2,g.predicted_mmse_w2)),
                        'oof_r2':r2_score(g.actual_mmse_w2,g.predicted_mmse_w2)})
    summary=pd.DataFrame(summary).sort_values(['feature_set','oof_mae'])
    summary.to_csv(out/'summary.csv',index=False)
    # Paired out-of-fold comparisons for each algorithm: negative delta_mae means adding features helped.
    comparisons=[]
    for model in MODELS:
        pp=p[p.model==model].pivot(index='prim_key',columns='feature_set',values='predicted_mmse_w2')
        truth=p[(p.model==model)&(p.feature_set=='S0_baseline_mmse')].set_index('prim_key')['actual_mmse_w2'].reindex(pp.index)
        for base,extra in [('S0_baseline_mmse','S1_baseline_core'),('S1_baseline_core','S2_baseline_core_engineered_sleep'),('S1_baseline_core','S3_baseline_core_historical_sleep')]:
            base_error=np.abs(truth-pp[base]);extra_error=np.abs(truth-pp[extra])
            comparisons.append({'model':model,'baseline':base,'augmented':extra,'n':len(truth),
                                'baseline_mae':base_error.mean(),'augmented_mae':extra_error.mean(),
                                'delta_mae_augmented_minus_baseline':(extra_error-base_error).mean()})
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    # Diagnostic scatterplots: illustrative held-out predictions, not training-fit charts.
    for fs in SETS:
        g=p[(p.feature_set==fs)&(p.model=='ridge')]
        fig,ax=plt.subplots(figsize=(5.5,5))
        ax.scatter(g.actual_mmse_w2,g.predicted_mmse_w2,alpha=.45,s=18)
        ax.plot([0,30],[0,30],linestyle='--',linewidth=1)
        ax.set(xlabel='Observed W2 MMSE',ylabel='Out-of-fold predicted W2 MMSE',title=f'{fs} — Ridge')
        fig.tight_layout();fig.savefig(out/f'{fs}_ridge_actual_vs_predicted.png',dpi=150);plt.close(fig)
    manifest={'dataset':str(data.resolve()),'n_participants':len(df),'target':'mmse_w2','outer_cv':'5-fold KFold shuffle=True seed=42',
              'inner_cv':'3-fold KFold shuffle=True seed=19','selection_metric':'MAE',
              'feature_sets':SETS,'caution':'W3 sleep may occur after baseline cognitive assessment; S2 is not a baseline-only prospective model.',
              'additional_caution':'Check timing of W2 core covariates relative to intended prediction date.'}
    (out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary.to_string(index=False));print('\nPaired comparisons:\n',pd.DataFrame(comparisons).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'haalsi_final_model_dataset_404.csv')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results_cleaned_stage2')
    args=parser.parse_args();run(args.data,args.out)
