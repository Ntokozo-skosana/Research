"""Stage 4 cleaned-data robustness: retrain no-sleep LSTM, paired comparisons, fold verification.
Place beside train_stage4_sequences.py and dataset_preflight.py.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import train_stage4_sequences as stage4

ORIGINAL_PREPROCESS=stage4.preprocess

def no_sleep_preprocess(train,test,waves):
    (xtr,ctr),(xte,cte)=ORIGINAL_PREPROCESS(train,test,waves)
    # Preserve the same network dimensions, parameter count, optimization, and
    # train-only core preprocessing. Disable ALL sequence channels including
    # missingness indicators and positional information.
    return (np.zeros_like(xtr),ctr),(np.zeros_like(xte),cte)

def load_preds(path, name):
    d=pd.read_csv(path)
    needed={'prim_key','fold','actual_mmse_w2','predicted_mmse_w2'}
    if not needed.issubset(d):raise ValueError(f'{name} missing {needed-set(d)}')
    if d[list(needed)].isna().any().any():raise ValueError(f'{name} has missing values')
    return d

def pick(df,feature,model,name):
    d=df[(df.feature_set==feature)&(df.model==model)].copy()
    if len(d)!=404:raise ValueError(f'{name}: expected 404 rows, got {len(d)}')
    if d.prim_key.duplicated().any():raise ValueError(f'{name}: duplicate participants')
    return d[['prim_key','fold','actual_mmse_w2','predicted_mmse_w2']].rename(columns={'predicted_mmse_w2':name})

def compare(a,b, an,bn,boot,seed):
    m=a.merge(b,on=['prim_key','fold'],how='outer',validate='one_to_one',indicator=True,suffixes=('_a','_b'))
    if len(m)!=404 or not (m._merge=='both').all():raise ValueError(f'Fold or ID mismatch: {an}, {bn}')
    if not np.allclose(m.actual_mmse_w2_a,m.actual_mmse_w2_b,atol=1e-7):raise ValueError('Actual outcomes differ')
    y=m.actual_mmse_w2_a.to_numpy()
    diff=np.abs(y-m[an].to_numpy())-np.abs(y-m[bn].to_numpy())
    rng=np.random.default_rng(seed)
    # Participant bootstrap, chunked to keep RAM small
    means=[]
    for _ in range(0,boot,250):
        size=min(250,boot-len(means))
        inds=rng.integers(0,len(diff),size=(size,len(diff)))
        means.extend(diff[inds].mean(axis=1).tolist())
    lo,hi=np.percentile(means,[2.5,97.5])
    return {'model_a':an,'model_b':bn,'n':len(m),'mae_a':float(np.abs(y-m[an]).mean()),'mae_b':float(np.abs(y-m[bn]).mean()),'delta_mae_a_minus_b':float(diff.mean()),'bootstrap_ci_low':float(lo),'bootstrap_ci_high':float(hi),'ci_includes_zero':bool(lo<=0<=hi)}

def main(args):
    stage4.torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data=Path(args.data)
    df=stage4.check_cleaned_dataset(data,{'core':stage4.CORE,'sleep':[f'{s}_w{w}' for s in stage4.SLEEP+['bedtime'] for w in (1,2,3)]},Path(stage4.__file__).resolve().parent/'haalsi_original_reference.csv')
    sha=hashlib.sha256(data.read_bytes()).hexdigest().upper()
    if sha!='77E2B9D4A6B72D846A9D5767FF86F54C77413536DA30BC1B8D82827F5F46C36E':raise ValueError('Unexpected cleaned dataset checksum')
    seq=load_preds(args.stage4_predictions,'stage4');ridge=load_preds(args.stage2_predictions,'stage2')
    names={}
    names['ridge_s1']=pick(ridge,'S1_baseline_core','ridge','ridge_s1')
    for variant,suffix in [('T1_three_waves','3w'),('T2_historical_waves','2w')]:
        for kind in ['lstm','gru']:
            key=f'{kind}_{suffix}';names[key]=pick(seq,variant,kind,key)
    # Validate outer CV folds against actual dataframe row ordering.
    foldmap={}
    for fold,(_,te) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42).split(df),1):
        for ix in te:foldmap[str(df.iloc[ix].prim_key)]=fold
    for name,d in names.items():
        actual=d.prim_key.astype(str).map(foldmap)
        if actual.isna().any() or not np.array_equal(actual.to_numpy(),d.fold.to_numpy()):raise ValueError(f'{name}: fold assignment mismatch')
    print('PASS: all supplied predictions have identical 404 participants and expected outer folds',flush=True)

    # No-sleep control: same T1 (three-timestep) architecture and hyperparameter
    # selection, with every sleep channel set to zero before model sees it.
    stage4.preprocess=no_sleep_preprocess
    preds=[];foldrows=[]
    try:
        for fold,(tr,te) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42).split(df),1):
            train=df.iloc[tr].reset_index(drop=True);test=df.iloc[te].reset_index(drop=True)
            inner=list(KFold(n_splits=3,shuffle=True,random_state=19).split(train))
            scores=[]
            for config in stage4.CONFIGS:
                errors=[]
                for j,(itr,ival) in enumerate(inner):
                    pred,_=stage4.fit_predict(train.iloc[itr].reset_index(drop=True),train.iloc[ival].reset_index(drop=True),'lstm',[1,2,3],config,seed=1000+fold*100+j,epochs=args.epochs,patience=args.patience)
                    errors.append(mean_absolute_error(train.iloc[ival].mmse_w2,pred))
                scores.append(float(np.mean(errors)))
            idx=int(np.argmin(scores));best=stage4.CONFIGS[idx]
            prediction,selected_epochs=stage4.fit_predict(train,test,'lstm',[1,2,3],best,seed=2000+fold,epochs=args.epochs,patience=args.patience)
            for i,value in enumerate(prediction):
                preds.append({'prim_key':test.iloc[i].prim_key,'feature_set':'T0_no_sleep','model':'lstm','fold':fold,'actual_mmse_w2':float(test.iloc[i].mmse_w2),'predicted_mmse_w2':float(value)})
            foldrows.append({'fold':fold,'mae':float(mean_absolute_error(test.mmse_w2,prediction)),'inner_mae':scores[idx],'best_config':json.dumps(best,sort_keys=True),'refit_epochs':selected_epochs})
            print(f'Finished no-sleep LSTM fold {fold}/5 | MAE={foldrows[-1]["mae"]:.4f}',flush=True)
    finally:
        stage4.preprocess=ORIGINAL_PREPROCESS
    p=pd.DataFrame(preds);p.to_csv(out/'no_sleep_oof_predictions.csv',index=False)
    pd.DataFrame(foldrows).to_csv(out/'no_sleep_fold_metrics.csv',index=False)
    names['lstm_no_sleep']=pick(p,'T0_no_sleep','lstm','lstm_no_sleep')
    y=p.actual_mmse_w2.to_numpy();yh=p.predicted_mmse_w2.to_numpy()
    summary={'n':len(p),'oof_mae':float(mean_absolute_error(y,yh)),'oof_rmse':float(np.sqrt(mean_squared_error(y,yh))),'oof_r2':float(r2_score(y,yh))}
    (out/'no_sleep_summary.json').write_text(json.dumps(summary,indent=2))
    pairs=[('lstm_3w','ridge_s1'),('lstm_2w','ridge_s1'),('gru_3w','ridge_s1'),('gru_2w','ridge_s1'),('lstm_3w','lstm_no_sleep'),('lstm_2w','lstm_no_sleep'),('lstm_no_sleep','ridge_s1'),('lstm_3w','lstm_2w'),('gru_3w','gru_2w'),('gru_3w','lstm_3w'),('gru_2w','lstm_2w')]
    comparisons=pd.DataFrame([compare(names[a],names[b],a,b,args.bootstrap,42+i) for i,(a,b) in enumerate(pairs)])
    comparisons.to_csv(out/'paired_robustness_comparisons.csv',index=False)
    manifest={'dataset_sha256':sha,'n':404,'epochs':args.epochs,'patience':args.patience,'bootstrap_samples':args.bootstrap,'seed_outer':42,'seed_inner':19,'control':'Three-wave LSTM identical to Stage 4 architecture; all 6 sequence channels (including missingness) zeroed in both training and prediction; core covariates retained; inner tuning and early stopping repeated.','bootstrap_note':'Participant-level bootstrap of paired held-out absolute errors; confidence intervals descriptive and do not incorporate uncertainty from retraining or model selection.','caution':'Sleep signal not isolated in GRU because no GRU no-sleep control trained. Historical and three-wave variants differ in sequence length. Temporal alignment and partially completed MMSE assessments remain unresolved.'}
    (out/'robustness_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nNo-sleep control:',summary,flush=True)
    print('\nPaired comparisons (negative difference favors model A):\n'+comparisons.to_string(index=False),flush=True)
    print('\nSaved results to:',out.resolve(),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--stage2-predictions',required=True)
    ap.add_argument('--stage4-predictions',required=True)
    ap.add_argument('--out',default='results_cleaned_stage4_robustness')
    ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--patience',type=int,default=12)
    ap.add_argument('--bootstrap',type=int,default=5000)
    main(ap.parse_args())
