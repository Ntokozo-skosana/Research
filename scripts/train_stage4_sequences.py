"""Stage 4: reproducible small LSTM/GRU models for HAALSI MMSE W2.
Run: py train_stage4_sequences.py --data ..\\haalsi_final_model_dataset_404.csv
Outputs held-out out-of-fold predictions using same 5-fold KFold seed 42 as Stages 2/3.
No test-fold leakage: preprocessing and early stopping use training participants only.
"""
from dataset_preflight import check as check_cleaned_dataset
import argparse, hashlib, json, platform, random
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

CORE=['mmse_w1','age_w2','sex','education_w1','cesd_w2','wealth_quintile_w2']
CAT=['sex','education_w1','wealth_quintile_w2']
SLEEP=['duration','quality','insufficiency']
CONFIGS=[{'hidden':8,'dropout':0.15,'weight_decay':0.01,'lr':0.001},
         {'hidden':16,'dropout':0.30,'weight_decay':0.05,'lr':0.001}]

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    try: torch.use_deterministic_algorithms(True)
    except Exception: pass

def preprocess(train, test, waves):
    """Fit transformations on training rows only; bedtime is circular, not linear."""
    nums=[c for c in CORE if c not in CAT]
    pre=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),nums),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)
    ],sparse_threshold=0)
    c_train=pre.fit_transform(train[CORE]).astype('float32')
    c_test=pre.transform(test[CORE]).astype('float32')
    # Sleep sequence: duration, quality, insufficiency, sin(bedtime), cos(bedtime)
    def build(frame):
        arrays=[]
        for w in waves:
            bed=frame[f'bedtime_w{w}'].to_numpy(dtype=float)
            angle=2*np.pi*(np.mod(bed,24)/24)
            arrays.append(np.column_stack([*[frame[f'{s}_w{w}'].to_numpy(dtype=float) for s in SLEEP],np.sin(angle),np.cos(angle)]))
        return np.stack(arrays,axis=1)
    x_train=build(train);x_test=build(test)
    # Explicit duration-missing indicator, derived from raw missingness before imputation.
    miss_train=(~np.isfinite(x_train[:,:,0])).astype("float32")[:,:,None]
    miss_test=(~np.isfinite(x_test[:,:,0])).astype("float32")[:,:,None]
    # Impute and standardize each sleep feature across training participants AND training timesteps.
    flat=x_train.reshape(-1,x_train.shape[-1]);med=np.nanmedian(flat,axis=0)
    med=np.where(np.isfinite(med),med,0.)
    def impute(a):return np.where(np.isfinite(a),a,med)
    flat=impute(flat);mu=flat.mean(axis=0);sd=flat.std(axis=0);sd=np.where(sd<1e-8,1.,sd)
    x_train=(impute(x_train)-mu)/sd;x_test=(impute(x_test)-mu)/sd
    x_train=np.concatenate([x_train,miss_train],axis=2)
    x_test=np.concatenate([x_test,miss_test],axis=2)
    return (x_train.astype('float32'),c_train),(x_test.astype('float32'),c_test)

class SleepNet(nn.Module):
    def __init__(self,kind,seq_dim,core_dim,hidden,dropout):
        super().__init__()
        rnn=nn.LSTM if kind=='lstm' else nn.GRU
        self.encoder=rnn(seq_dim,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden+core_dim,16),nn.ReLU(),nn.Dropout(dropout),nn.Linear(16,1))
    def forward(self,seq,core):
        _,hidden=self.encoder(seq)
        h=hidden[0][-1] if isinstance(hidden,tuple) else hidden[-1]
        return self.head(torch.cat([h,core],dim=1)).squeeze(1)

def fit_predict(train_df, predict_df, kind, waves, config, seed, epochs, patience):
    seed_everything(seed)
    # Internal early stopping is performed ONLY on a held-out part of this training fold.
    a,b=train_test_split(np.arange(len(train_df)),test_size=0.18,random_state=seed,shuffle=True)
    train_inner=train_df.iloc[a];valid_inner=train_df.iloc[b]
    (xt,ct),(xv,cv)=preprocess(train_inner,valid_inner,waves)
    y=train_inner.mmse_w2.to_numpy(dtype='float32');yv=valid_inner.mmse_w2.to_numpy(dtype='float32')
    # Standardize target using training subset only.
    ymu=float(y.mean());ysd=float(y.std());ysd=max(ysd,1e-6)
    model=SleepNet(kind,xt.shape[-1],ct.shape[-1],config['hidden'],config['dropout'])
    opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    loss_fn=nn.MSELoss();best=float('inf');best_epoch=1;best_weights=None;wait=0
    tensors=[torch.tensor(t) for t in (xt,ct,(y-ymu)/ysd,xv,cv,(yv-ymu)/ysd)]
    X,C,Y,V,VC,VY=tensors
    for epoch in range(1,epochs+1):
        model.train()
        order=torch.randperm(len(X))
        for batch in order.split(32):
            opt.zero_grad();loss=loss_fn(model(X[batch],C[batch]),Y[batch]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        model.eval()
        with torch.no_grad(): vloss=loss_fn(model(V,VC),VY).item()
        if vloss<best-1e-5:
            best=vloss;best_epoch=epoch;wait=0
            best_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=patience:break
    # Refit selected number of epochs on ALL participants in this training fold.
    seed_everything(seed)
    (xt,ct),(xp,cp)=preprocess(train_df,predict_df,waves)
    y=train_df.mmse_w2.to_numpy(dtype='float32');ymu=float(y.mean());ysd=max(float(y.std()),1e-6)
    X=torch.tensor(xt);C=torch.tensor(ct);Y=torch.tensor((y-ymu)/ysd)
    model=SleepNet(kind,xt.shape[-1],ct.shape[-1],config['hidden'],config['dropout'])
    opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    for epoch in range(best_epoch):
        model.train()
        for batch in torch.randperm(len(X)).split(32):
            opt.zero_grad();loss=loss_fn(model(X[batch],C[batch]),Y[batch]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    model.eval()
    with torch.no_grad():pred=model(torch.tensor(xp),torch.tensor(cp)).numpy()*ysd+ymu
    return pred,best_epoch

def main(args):
    torch.set_num_threads(1)
    data=Path(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=check_cleaned_dataset(data,{"stage4_static":CORE,"stage4_sequence":[f"{s}_w{w}" for s in SLEEP+["bedtime"] for w in (1,2,3)]},Path(__file__).resolve().parent/"haalsi_original_reference.csv")
    required={'prim_key','mmse_w1','mmse_w2','delta_mmse'}|set(CORE)|{f'{s}_w{w}' for s in SLEEP+['bedtime'] for w in (1,2,3)}
    if missing:=required-set(df.columns):raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.prim_key.isna().any() or df.prim_key.duplicated().any():raise ValueError('Duplicate/missing IDs')
    if df[['mmse_w1','mmse_w2']].isna().any().any():raise ValueError('Missing target/baseline MMSE')
    if not np.allclose(df.mmse_w2-df.mmse_w1,df.delta_mmse):raise ValueError('Inconsistent MMSE change')
    digest=hashlib.sha256(data.read_bytes()).hexdigest().upper()
    expected='77E2B9D4A6B72D846A9D5767FF86F54C77413536DA30BC1B8D82827F5F46C36E'
    print('Dataset SHA256:',digest,flush=True)
    if digest!=expected:print('WARNING: Dataset checksum differs from verified Stage 2/3 dataset!',flush=True)
    outer=list(KFold(n_splits=5,shuffle=True,random_state=42).split(df))
    rows=[];preds=[]
    variants={'T1_three_waves':[1,2,3],'T2_historical_waves':[1,2]}
    for variant,waves in variants.items():
      for kind in ('lstm','gru'):
        for fold,(tr,te) in enumerate(outer,1):
            train=df.iloc[tr].reset_index(drop=True);test=df.iloc[te].reset_index(drop=True)
            inner=list(KFold(n_splits=3,shuffle=True,random_state=19).split(train))
            scores=[]
            for ci,config in enumerate(CONFIGS):
                losses=[]
                for j,(itr,ival) in enumerate(inner):
                    pred,_=fit_predict(train.iloc[itr].reset_index(drop=True),train.iloc[ival].reset_index(drop=True),kind,waves,config,seed=1000+fold*100+j,epochs=args.epochs,patience=args.patience)
                    losses.append(mean_absolute_error(train.iloc[ival].mmse_w2,pred))
                scores.append(float(np.mean(losses)))
            best_idx=int(np.argmin(scores));best=CONFIGS[best_idx]
            yh,selected_epochs=fit_predict(train,test,kind,waves,best,seed=2000+fold,epochs=args.epochs,patience=args.patience)
            actual=test.mmse_w2.to_numpy(float)
            rows.append({'feature_set':variant,'model':kind,'fold':fold,'n_test':len(test),'mae':mean_absolute_error(actual,yh),'rmse':np.sqrt(mean_squared_error(actual,yh)),'r2':r2_score(actual,yh),'inner_mae':scores[best_idx],'best_config':json.dumps(best,sort_keys=True),'refit_epochs':selected_epochs})
            for ix,pr in enumerate(yh):
                preds.append({'prim_key':test.iloc[ix].prim_key,'feature_set':variant,'model':kind,'fold':fold,'actual_mmse_w2':actual[ix],'predicted_mmse_w2':float(pr),'mmse_w1':test.iloc[ix].mmse_w1})
            print(f'Finished {variant} {kind} fold {fold}/5 | MAE={rows[-1]["mae"]:.3f}',flush=True)
    pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
    predictions=pd.DataFrame(preds);predictions.to_csv(out/'out_of_fold_predictions.csv',index=False)
    summaries=[]
    for (variant,kind),g in predictions.groupby(['feature_set','model'],sort=False):
        summaries.append({'feature_set':variant,'model':kind,'n':len(g),'oof_mae':mean_absolute_error(g.actual_mmse_w2,g.predicted_mmse_w2),'oof_rmse':np.sqrt(mean_squared_error(g.actual_mmse_w2,g.predicted_mmse_w2)),'oof_r2':r2_score(g.actual_mmse_w2,g.predicted_mmse_w2)})
    summary=pd.DataFrame(summaries);summary.to_csv(out/'summary.csv',index=False)
    manifest={'dataset_sha256':digest,'n':len(df),'python':platform.python_version(),'torch':torch.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__,'seed_outer':42,'seed_inner':19,'seed_training':'1000+fold*100+inner_fold; 2000+outer_fold','outer_cv':'5-fold shuffled KFold','inner_cv':'3-fold shuffled KFold','selection_metric':'inner mean MAE','early_stopping':'18% internal training-only validation, MSE; then refit full training fold','epochs_max':args.epochs,'patience':args.patience,'configs':CONFIGS,'variants':variants,'warning':'W3 sleep may postdate baseline dementia assessment; W2 core covariates may also postdate baseline. T2 is not automatically a strictly prospective design.'}
    (out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    print('\n',summary.to_string(index=False),flush=True)
    if args.stage2_predictions:
        prior=pd.read_csv(args.stage2_predictions)
        ridge=prior[(prior.feature_set=='S1_baseline_core')&(prior.model=='ridge')][['prim_key','fold','actual_mmse_w2','predicted_mmse_w2']].rename(columns={'predicted_mmse_w2':'ridge_prediction'})
        comparison=[]
        for (variant,kind),g in predictions.groupby(['feature_set','model']):
            merged=g.merge(ridge,on=['prim_key','fold','actual_mmse_w2'],validate='one_to_one')
            if len(merged)!=len(df):raise ValueError('Fold/participant mismatch with Stage 2!')
            difference=np.abs(merged.actual_mmse_w2-merged.predicted_mmse_w2)-np.abs(merged.actual_mmse_w2-merged.ridge_prediction)
            rng=np.random.default_rng(42);draws=rng.integers(0,len(difference),size=(2000,len(difference)))
            ci=np.percentile(difference.to_numpy()[draws].mean(axis=1),[2.5,97.5])
            comparison.append({'feature_set':variant,'model':kind,'n':len(merged),'ridge_mae':np.abs(merged.actual_mmse_w2-merged.ridge_prediction).mean(),'sequence_mae':np.abs(merged.actual_mmse_w2-merged.predicted_mmse_w2).mean(),'delta_mae_sequence_minus_ridge':difference.mean(),'bootstrap_ci_low':ci[0],'bootstrap_ci_high':ci[1]})
        pd.DataFrame(comparison).to_csv(out/'vs_stage2_ridge.csv',index=False)
        print('\nCompared with Stage 2 Ridge:\n',pd.DataFrame(comparison).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='../haalsi_final_model_dataset_404.csv')
    p.add_argument('--out',default='results_cleaned_stage4')
    p.add_argument('--stage2-predictions',default=None)
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--patience',type=int,default=12)
    args=p.parse_args();main(args)
