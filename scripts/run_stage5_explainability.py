"""Stage 5: fold-held-out SHAP and LIME explanations for HAALSI MMSE regression."""
import argparse, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import sklearn
import shap
from lime.lime_tabular import LimeTabularExplainer

SHA='77E2B9D4A6B72D846A9D5767FF86F54C77413536DA30BC1B8D82827F5F46C36E'
CORE=['age_w2','sex','education_w1','cesd_w2','wealth_quintile_w2']
ENG=['duration_mean','duration_change','duration_variability','quality_mean','quality_change','insufficiency_mean','insufficiency_change','bedtime_mean_shifted','bedtime_change']
HIST=[f'{s}_w{w}' for s in ('duration','quality','insufficiency','bedtime') for w in (1,2)]
SETS={'ridge_S1':['mmse_w1']+CORE,'ridge_S2':['mmse_w1']+CORE+ENG,'ridge_S3':['mmse_w1']+CORE+HIST,'xgboost_S1':['mmse_w1']+CORE,'xgboost_S2':['mmse_w1']+CORE+ENG,'xgboost_S3':['mmse_w1']+CORE+HIST}
REF={'ridge_S1':('S1_baseline_core','ridge'),'ridge_S2':('S2_baseline_core_engineered_sleep','ridge'),'ridge_S3':('S3_baseline_core_historical_sleep','ridge'),'xgboost_S1':('S1_baseline_core','xgboost'),'xgboost_S2':('S2_baseline_core_engineered_sleep','xgboost'),'xgboost_S3':('S3_baseline_core_historical_sleep','xgboost')}

def plot_bar(frame, path, title, n=20):
    shown=frame.head(n).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,max(4,0.36*len(shown)+1)))
    ax.barh(shown.feature,shown.mean_abs_shap)
    ax.set_xlabel('Mean absolute SHAP contribution (MMSE points)')
    ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def main(args):
    import train_baseline_adjusted as stage2
    import train_stage3_xgboost as stage3
    path=Path(args.data)
    digest=hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if digest!=SHA:raise ValueError(f'Dataset SHA mismatch: {digest}')
    from dataset_preflight import check as check_cleaned_dataset
    df=check_cleaned_dataset(path,SETS,Path(__file__).resolve().parent/'haalsi_original_reference.csv')
    if len(df)!=404 or df.prim_key.isna().any() or df.prim_key.duplicated().any():raise ValueError('Expected 404 unique IDs')
    if df[['mmse_w1','mmse_w2']].isna().any().any():raise ValueError('Missing outcome values')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    y=df.mmse_w2.to_numpy(float)
    folds=list(KFold(n_splits=5,shuffle=True,random_state=42).split(df))
    inner=KFold(n_splits=3,shuffle=True,random_state=19)
    saved={}
    for k,source in [('ridge',args.stage2_predictions),('xgboost',args.stage3_predictions)]:
        if source:
            pred=pd.read_csv(source)
            saved[k]=pred
    all_global=[];all_local=[];validation=[]
    for key,features in SETS.items():
        if key not in args.models:continue
        alg='ridge' if key.startswith('ridge') else 'xgboost'
        fs,model_name=REF[key]
        for fold,(train_idx,test_idx) in enumerate(folds,1):
            if args.fold and fold!=args.fold:continue
            Xtr=df.iloc[train_idx][features]
            Xte=df.iloc[test_idx][features]
            if alg=='ridge':
                estimator=stage2.MODELS['ridge'][0]
                grid=stage2.MODELS['ridge'][1]
                pipe=Pipeline([('prep',stage2.preprocessing(features)),('model',estimator)])
            else:
                pipe=stage3.make_pipeline(features)
                grid=stage3.GRID
            search=GridSearchCV(pipe,grid,cv=inner,scoring='neg_mean_absolute_error',n_jobs=1,refit=True)
            search.fit(Xtr,y[train_idx]);fitted=search.best_estimator_
            pred=fitted.predict(Xte)
            v={'experiment':key,'fold':fold,'n_test':len(test_idx),'heldout_mae':float(mean_absolute_error(y[test_idx],pred)),'best_params':json.dumps(search.best_params_)}
            if alg in saved:
                s=saved[alg]
                s=s[(s.feature_set==fs)&(s.model==model_name)&(s.fold==fold)]
                expected=pd.DataFrame({'prim_key':df.iloc[test_idx].prim_key.to_numpy(),'prediction':pred}).merge(s[['prim_key','predicted_mmse_w2']],on='prim_key',validate='one_to_one')
                if len(expected)!=len(test_idx):raise ValueError(f'{key} fold {fold}: participant mismatch')
                v['max_prediction_difference']=float(np.max(np.abs(expected.prediction-expected.predicted_mmse_w2)))
                if v['max_prediction_difference']>args.tolerance:
                    raise ValueError(f'{key} fold {fold}: refitted predictions differ from saved Stage 2/3 (max {v["max_prediction_difference"]:.6f}); refusing to explain inconsistent model')
            else:
                raise ValueError(f'Missing saved {alg} predictions; pass both --stage2-predictions and --stage3-predictions')
            validation.append(v)
            prep=fitted.named_steps['prep'];model=fitted.named_steps['model']
            train_t=prep.transform(Xtr)
            test_t=prep.transform(Xte)
            if hasattr(train_t,'toarray'):train_t=train_t.toarray()
            if hasattr(test_t,'toarray'):test_t=test_t.toarray()
            train_t=np.asarray(train_t,dtype=float);test_t=np.asarray(test_t,dtype=float)
            names=np.asarray(prep.get_feature_names_out(),dtype=str)
            if len(names)!=train_t.shape[1]:raise ValueError('Transformed feature-name mismatch')
            if alg=='ridge':
                explainer=shap.LinearExplainer(model,train_t)
            else:
                explainer=shap.TreeExplainer(model)
            explanation=explainer(test_t)
            values=np.asarray(explanation.values)
            if values.ndim!=2 or values.shape!=test_t.shape:raise ValueError(f'Unexpected SHAP shape {values.shape}')
            global_importance=np.mean(np.abs(values),axis=0)
            for name,score in zip(names,global_importance):
                all_global.append({'experiment':key,'fold':fold,'feature':name,'mean_abs_shap':float(score),'n_test':len(test_idx)})
            # Local example: first held-out participant in each fold, chosen independently of outcome/error.
            j=0
            for name,feature_value,contribution in zip(names,test_t[j],values[j]):
                all_local.append({'experiment':key,'fold':fold,'prim_key':df.iloc[test_idx[j]].prim_key,'feature':name,'transformed_value':float(feature_value),'shap_value':float(contribution),'predicted_mmse_w2':float(pred[j]),'actual_mmse_w2':float(y[test_idx[j]])})
            if args.lime and fold==(args.fold or 1):
                lime=LimeTabularExplainer(train_t,feature_names=list(names),mode='regression',discretize_continuous=True,random_state=42)
                local=lime.explain_instance(test_t[j],model.predict,num_features=min(12,len(names)),num_samples=args.lime_samples)
                pd.DataFrame(local.as_list(),columns=['rule','lime_weight']).to_csv(out/f'{key}_fold{fold}_lime_example.csv',index=False)
                fig=local.as_pyplot_figure();fig.tight_layout();fig.savefig(out/f'{key}_fold{fold}_lime_example.png',dpi=160);plt.close(fig)
            print(f'Explained {key} fold {fold}/5 | held-out MAE {v["heldout_mae"]:.3f}',flush=True)
    if not all_global:raise ValueError('No experiments selected')
    raw=pd.DataFrame(all_global)
    raw.to_csv(out/'shap_fold_importance.csv',index=False)
    pd.DataFrame(all_local).to_csv(out/'shap_local_examples.csv',index=False)
    pd.DataFrame(validation).to_csv(out/'prediction_reproduction_check.csv',index=False)
    global_rows=[]
    for exp,g in raw.groupby('experiment'):
        # Weighted across held-out participants; each appears exactly once for all-fold run.
        ag=g.assign(weighted=g.mean_abs_shap*g.n_test).groupby('feature',as_index=False).agg(weighted=('weighted','sum'),n=('n_test','sum'))
        ag['mean_abs_shap']=ag.weighted/ag.n
        ag=ag[['feature','mean_abs_shap']].sort_values('mean_abs_shap',ascending=False)
        ag.insert(0,'experiment',exp)
        global_rows.append(ag)
        plot_bar(ag,out/f'{exp}_shap_global.png',f'{exp}: held-out SHAP importance')
    pd.concat(global_rows,ignore_index=True).to_csv(out/'shap_global_importance.csv',index=False)
    (out/'stage5_manifest.json').write_text(json.dumps({'dataset_sha256':digest,'target':'mmse_w2','participants':len(df),'outer_cv':'5-fold shuffled seed 42','inner_cv':'3-fold shuffled seed 19','experiments':args.models,'fold':args.fold or 'all','shap':shap.__version__,'sklearn':sklearn.__version__,'python':platform.python_version(),'note':'SHAP calculated only on outer-held-out participants. Explanations describe fitted model predictions, not causal effects. One-hot and missingness indicator columns may appear separately. Some W2/W3 variables may be measured after baseline.'},indent=2))
    print('\nTop features:')
    for frame in global_rows:
        print('\n',frame.experiment.iloc[0], '\n',frame.head(8).to_string(index=False))
    print('\nSaved to:',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--stage2-predictions')
    p.add_argument('--stage3-predictions')
    p.add_argument('--out',default='results_stage5')
    p.add_argument('--fold',type=int,choices=[1,2,3,4,5],default=None,help='Use one fold for quick test; default all five')
    p.add_argument('--models',nargs='+',choices=list(SETS),default=list(SETS))
    p.add_argument('--lime',action='store_true',help='Run local LIME examples on first fold')
    p.add_argument('--lime-samples',type=int,default=2000)
    p.add_argument('--tolerance',type=float,default=1e-5)
    a=p.parse_args()
    if not a.stage2_predictions or not a.stage3_predictions:
        p.error('Both --stage2-predictions and --stage3-predictions are required for reproduction checks')
    main(a)
