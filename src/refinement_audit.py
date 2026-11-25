"""Recompute the three-seed study from saved predictions; no checkpoint loading."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];D3=ROOT/'results/deliverable3'
PREFIX={'XGBoost':'xgb','MLP':'mlp','GraphSAGE':'graphsage'}
POLICIES=['frozen','expanding','rolling','recency']
def sources(model,seed):
 p=PREFIX[model]
 if seed==0:
  r=D3/'baseline_seed0'
  return tuple(r/f'{p}_{s}' for s in ['frozen_test_predictions.csv','adaptation_seed0_predictions.csv','frozen_metrics_by_step.csv','adaptation_seed0_metrics.csv','frozen_run_info.json'])
 r=D3/('xgboost' if model=='XGBoost' else p)/f'seed_{seed}'
 return tuple(r/s for s in ['frozen_test_predictions.csv','adaptation/test_predictions.csv','frozen_metrics_by_step.csv','adaptation/metrics_by_step.csv','run_info.json'])
def per_step(frame,model,policy,seed):
 records=[]
 for step,g in frame.groupby('time_step',sort=True):
  y=g.y.to_numpy();p=g.score.to_numpy();d=g.prediction.to_numpy();tn,fp,fn,tp=confusion_matrix(y,d,labels=[0,1]).ravel()
  row=dict(model=model,policy=policy,seed=seed,time_step=int(step),n_labeled=len(g),n_illicit=int(y.sum()),illicit_prevalence=float(y.mean()),average_precision=average_precision_score(y,p) if len(np.unique(y))==2 else np.nan,precision=precision_score(y,d,zero_division=0),recall=recall_score(y,d,zero_division=0),f1=f1_score(y,d,zero_division=0),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
  order=np.argsort(-p,kind='stable')
  for b in [1,5,10,20]:
   k=max(1,int(np.ceil(len(g)*b/100)));top=y[order[:k]];row[f'review_k_{b}pct']=k;row[f'precision_at_{b}pct']=float(top.mean());row[f'recall_at_{b}pct']=float(top.sum()/y.sum()) if y.sum() else np.nan
  row['review_k']=row['review_k_10pct'];records.append(row)
 return pd.DataFrame(records)
def main():
 out=D3/'evaluation';out.mkdir(exist_ok=True);metrics=[];log=[];reference=None
 cols=['average_precision','precision','recall','f1','n_labeled','n_illicit','review_k','precision_at_10pct','recall_at_10pct']
 for model in PREFIX:
  for seed in range(3):
   frozen,adaptive,fm,am,settings=sources(model,seed);f=pd.read_csv(frozen).assign(policy='frozen');a=pd.read_csv(adaptive);assert set(a.policy)==set(POLICIES[1:])
   pred=pd.concat([f,a],ignore_index=True);assert len(pred)==16670*4 and not pred.duplicated(['policy','txId']).any()
   assert pred.y.isin([0,1]).all() and pred.prediction.isin([0,1]).all() and np.isfinite(pred.score).all() and pred.score.between(0,1).all()
   cfg=json.loads(settings.read_text());threshold=float(cfg['threshold']);np.testing.assert_array_equal(pred.prediction,(pred.score>=threshold).astype(int))
   stored=pd.concat([pd.read_csv(fm).assign(policy='frozen'),pd.read_csv(am)],ignore_index=True);assert len(stored)==60
   for policy,g in pred.groupby('policy'):
    observed=g.set_index('txId')[['time_step','y']].sort_index()
    if reference is None:reference=observed
    pd.testing.assert_frame_equal(reference,observed)
    r=per_step(g,model,policy,seed);s=stored[stored.policy==policy].sort_values('time_step');np.testing.assert_array_equal(r.time_step,s.time_step)
    for c in cols:np.testing.assert_allclose(r[c],s[c],atol=1e-10,rtol=1e-9,equal_nan=True)
    metrics.append(r)
   log.append(dict(model=model,seed=seed,prediction_rows=len(pred),metric_rows=60,threshold=threshold,selected_epoch=cfg.get('selected_epoch'),frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),adaptive_sha256=hashlib.sha256(adaptive.read_bytes()).hexdigest()))
 m=pd.concat(metrics,ignore_index=True);assert len(m)==540 and not m.duplicated(['model','policy','seed','time_step']).any()
 uploaded=pd.read_csv(out/'audited_metrics_all_seeds.csv').sort_values(['model','policy','seed','time_step']);ordered=m.sort_values(['model','policy','seed','time_step'])
 for c in cols:np.testing.assert_allclose(ordered[c],uploaded[c],atol=1e-10,rtol=1e-9,equal_nan=True)
 m.to_csv(out/'independently_audited_metrics.csv',index=False)
 m['phase']=np.select([m.time_step.between(35,42),m.time_step.eq(43)],['before','transition'],default='after')
 phase=m.groupby(['model','policy','seed','phase'],as_index=False).agg(mean_AP=('average_precision','mean'),mean_F1=('f1','mean'),mean_recall=('recall','mean'),mean_precision_at_10pct=('precision_at_10pct','mean'),mean_recall_at_10pct=('recall_at_10pct','mean'));phase.to_csv(out/'phase_results_all_seeds.csv',index=False)
 post=phase[phase.phase=='after'];summary=post.groupby(['model','policy'],as_index=False).agg(AP_mean=('mean_AP','mean'),AP_sd=('mean_AP','std'),F1_mean=('mean_F1','mean'),F1_sd=('mean_F1','std'),seeds=('seed','nunique'));summary.to_csv(out/'three_seed_summary.csv',index=False)
 gains=post.pivot(index=['model','seed'],columns='policy',values='mean_AP').reset_index();gains['recency_minus_expanding']=gains['recency']-gains['expanding'];gains['rolling_minus_expanding']=gains['rolling']-gains['expanding'];gains.to_csv(out/'paired_adaptation_gains.csv',index=False)
 recovery=[]
 for (model,policy,seed),g in m.groupby(['model','policy','seed']):
  baseline=float(g[g.time_step.between(35,42)].f1.mean());target=.9*baseline;g=g.set_index('time_step');eligible=[];qualifying=[]
  for t in range(44,49):
   pair=g.loc[[t,t+1]]
   if ((pair.n_illicit>=10)&((pair.n_labeled-pair.n_illicit)>=10)).all():
    eligible.append([t,t+1])
    if (pair.f1>=target).all():qualifying.append([t,t+1])
  recovery.append(dict(model=model,policy=policy,seed=int(seed),pre_shift_mean_F1=baseline,target_F1=target,eligible_pairs=len(eligible),recovery_status='observed' if qualifying else 'not observed by 49',confirmation_step=qualifying[0][1] if qualifying else None))
 recovery=pd.DataFrame(recovery);recovery.to_csv(out/'recovery_all_seeds.csv',index=False)
 m.groupby(['model','policy','seed','phase'],as_index=False)[['tn','fp','fn','tp']].sum().to_csv(out/'confusion_counts_by_phase.csv',index=False)
 nb=json.loads((ROOT/'notebooks/AML_GNN_Deliverable_3.ipynb').read_text());errors=[i for i,c in enumerate(nb['cells']) for o in c.get('outputs',[]) if o.get('output_type')=='error'];assert not errors
 report=dict(status='passed',metric_rows=540,prediction_rows=600120,seed_count=3,notebook_cells=len(nb['cells']),saved_notebook_errors=errors,recovery_observed=int((recovery.recovery_status=='observed').sum()),recovery_total=len(recovery),full_training_repeated_here=False,records=log);(out/'independent_audit.json').write_text(json.dumps(report,indent=2))
 print(summary.round(5).to_string(index=False));print('RECOVERY:',report['recovery_observed'],'/',report['recovery_total']);print('AUDIT PASSED: 600,120 prediction rows; 540 step metrics.')
if __name__=='__main__':main()
