"""Refined research UI: cached experiment replay, honest seed summaries, validated inference."""
from pathlib import Path
from functools import lru_cache
import csv,io,json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score,confusion_matrix
from src.inference import PortableXGBoost
from src.refinement_audit import sources
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results';EVAL=RESULTS/'deliverable3/evaluation'
MODELS=['XGBoost','MLP','GraphSAGE'];POLICIES=['frozen','expanding','rolling','recency']
COLUMNS=[f'local_{i:02}' for i in range(93)]

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,np.generic):v=v.item()
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def validate_selection(model,policy,seed,step,budget):
    if model not in MODELS or policy not in POLICIES:raise ValueError('Select an available model and adaptation policy.')
    if type(seed) is not int or seed not in [0,1,2]:raise ValueError('Seed must be 0, 1 or 2.')
    if type(step) is not int or not 35<=step<=49:raise ValueError('Time step must be an integer from 35 to 49.')
    if type(budget) is not int or not 1<=budget<=100:raise ValueError('Review budget must be a whole percentage from 1 to 100.')

@lru_cache(maxsize=18)
def _frame(model,seed,adaptive):
    f,a,_,_,_=sources(model,seed)
    return pd.read_csv(a if adaptive else f)

@lru_cache(maxsize=1)
def _metrics():return pd.read_csv(EVAL/'independently_audited_metrics.csv')

@lru_cache(maxsize=9)
def _settings(model,seed):return json.loads(sources(model,seed)[4].read_text())

def selected_rows(model,policy,seed,step,budget):
    validate_selection(model,policy,seed,step,budget)
    data=_frame(model,seed,policy!='frozen')
    if policy!='frozen':data=data[data.policy==policy]
    g=data[data.time_step==step].copy()
    if g.empty:raise ValueError('No saved results for this selection.')
    return g

def replay(model='XGBoost',policy='recency',step=49,budget=10,seed=0):
    g=selected_rows(model,policy,seed,step,budget);y=g.y.to_numpy();p=g.score.to_numpy();d=g.prediction.to_numpy();k=max(1,int(np.ceil(len(g)*budget/100)))
    ranked=g.sort_values('score',ascending=False,kind='stable').head(k)
    tn,fp,fn,tp=confusion_matrix(y,d,labels=[0,1]).ravel()
    m=dict(mode='saved prediction replay; no retraining',model=model,policy=policy,seed=seed,time_step=step,labeled=len(g),illicit=int(y.sum()),average_precision=average_precision_score(y,p) if len(np.unique(y))==2 else None,precision=precision_score(y,d,zero_division=0),recall=recall_score(y,d,zero_division=0),f1=f1_score(y,d,zero_division=0),review_k=k,precision_at_k=ranked.y.mean(),recall_at_k=ranked.y.sum()/y.sum() if y.sum() else None,threshold=float(_settings(model,seed)['threshold']),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),scores_are_calibrated_probabilities=False)
    if y.sum()<10:m['low_support_warning']='Fewer than 10 known illicit transactions. Snapshot metrics are unstable.'
    history=_metrics();history=history[(history.model==model)&(history.policy==policy)&(history.seed==seed)]
    return clean(dict(metrics=m,queue=ranked[['txId','time_step','y','score','prediction']].head(100).to_dict(orient='records'),history=history[['time_step','average_precision','f1','n_illicit']].to_dict(orient='records')))

def export_queue(model,policy,seed,step,budget):
    g=selected_rows(model,policy,seed,step,budget);k=max(1,int(np.ceil(len(g)*budget/100)));ranked=g.sort_values('score',ascending=False,kind='stable').head(k).copy()
    ranked['model']=model;ranked['policy']=policy;ranked['seed']=seed
    ranked['score_type']='uncalibrated weighted-model score';ranked['usage']='research; human review only'
    return ranked.to_csv(index=False)

@lru_cache(maxsize=1)
def comparison():
    summary=pd.read_csv(EVAL/'three_seed_summary.csv');baseline=pd.read_csv(EVAL/'deliverable2_vs_deliverable3.csv')[['model','policy','D2_seed0_AP']]
    rows=summary.merge(baseline,on=['model','policy'],validate='one_to_one')
    gains=pd.read_csv(EVAL/'paired_adaptation_gains.csv').groupby('model',as_index=False).agg(mean_gain=('recency_minus_expanding','mean'),sd_gain=('recency_minus_expanding','std'))
    return clean(dict(rows=rows.to_dict(orient='records'),gains=gains.to_dict(orient='records'),note='Post-shift steps 44–49: average snapshots within each seed, then summarize seeds 0/1/2. Sample SD measures run variation, not future-data uncertainty. Seed 0 appears in both Deliverable 2 and Deliverable 3. XGBoost runs are identical under this configuration.'))

@lru_cache(maxsize=1)
def _predictor():
    path=RESULTS/'xgb_frozen_seed0.json';portable=PortableXGBoost(path)
    try:import xgboost as xgb
    except ImportError:return portable,'portable numeric-tree evaluator; verified on 50 real test rows',None
    booster=xgb.Booster();booster.load_model(str(path));booster.set_param({'device':'cpu','nthread':2})
    return booster,'native XGBoost CPU inference',xgb

def checked_array(values):
    if not isinstance(values,list) or not values:raise ValueError('Provide a nonempty JSON list of numeric feature values.')
    flat=values if not isinstance(values[0],list) else [x for row in values for x in row]
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in flat):raise ValueError('Every feature must be a number, without nulls or booleans.')
    try:
        with np.errstate(over='ignore',invalid='ignore'):a=np.asarray(values,dtype=np.float32)
    except (ValueError,TypeError):raise ValueError('Each row must contain exactly 93 numeric values.') from None
    if a.ndim==1:a=a[None,:]
    if a.ndim!=2 or a.shape[1]!=93 or not 1<=len(a)<=200:raise ValueError('Provide 1–200 rows, each containing exactly 93 values.')
    if not np.isfinite(a).all():raise ValueError('All features must be finite and representable as float32. NaN and infinity are rejected.')
    return a

def infer(values):
    a=checked_array(values);predictor,engine,xgb=_predictor()
    scores=predictor.predict(xgb.DMatrix(a,feature_names=COLUMNS)) if xgb else predictor.predict(a)
    return [dict(row=i+1,score=float(p),threshold=.7,predicted_label=int(p>=.7),display_label='flag for human review' if p>=.7 else 'below review threshold') for i,p in enumerate(scores)],engine

def live(values):
    if isinstance(values,str):values=json.loads(values)
    a=checked_array(values)
    if len(a)!=1:raise ValueError('The JSON tab accepts exactly one row. Use CSV for batch inference.')
    rows,engine=infer(a.tolist());return dict(mode='live frozen XGBoost, seed 0',**rows[0],input_features=93,engine=engine,note='Uncalibrated score; no legal determination or automatic sanctions.')

def example():
    v=json.loads((ROOT/'docs/interface_verification.json').read_text())
    return dict(txId=54751137,time_step=35,observed_label='licit',features=v['real_feature_array'],expected_score=v['live_output']['score'],note='One labeled demonstration case, not representative performance evidence.')

def template_csv():
    out=io.StringIO();w=csv.writer(out);w.writerow(COLUMNS);w.writerow(example()['features']);return out.getvalue()

def batch_csv(text):
    if not isinstance(text,str) or len(text.encode('utf-8'))>2_000_000:raise ValueError('CSV must be text smaller than 2 MB.')
    reader=csv.reader(io.StringIO(text.lstrip('\ufeff')))
    try:header=next(reader)
    except StopIteration:raise ValueError('CSV is empty.') from None
    if len(header)!=93 or len(set(header))!=93 or set(header)!=set(COLUMNS):raise ValueError('CSV must have exactly local_00 through local_92, with no missing, extra or duplicate columns. Download the template.')
    rows=[]
    for number,row in enumerate(reader,start=2):
        if not row:continue
        if len(rows)>=200:raise ValueError('CSV limit is 200 data rows.')
        if len(row)!=93:raise ValueError(f'CSV line {number}: expected 93 values.')
        try:rows.append([float(row[header.index(c)]) for c in COLUMNS])
        except ValueError:raise ValueError(f'CSV line {number}: every feature must be numeric.') from None
    result,engine=infer(rows)
    return dict(rows=result,engine=engine,count=len(result),note='Columns aligned by name. Scores are uncalibrated. Unknown labels are not inferred as established truth.')
