from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
r=Path(__file__).resolve().parents[1];res=r/'results';records=[];proof=[]
for name,prefix in [('XGBoost','xgb'),('MLP','mlp'),('GraphSAGE','graphsage')]:
 frozen=pd.read_csv(res/f'{prefix}_frozen_test_predictions.csv').assign(model=name,policy='frozen',seed=0)
 adaptive=pd.read_csv(res/f'{prefix}_adaptation_seed0_predictions.csv')
 pred=pd.concat([frozen,adaptive],ignore_index=True)
 assert len(pred)==16670*4
 assert not pred.duplicated(['policy','txId']).any()
 known_reference=frozen.set_index('txId')[['time_step','y']].sort_index()
 tables=pd.concat([pd.read_csv(res/f'{prefix}_frozen_metrics_by_step.csv').assign(policy='frozen'),pd.read_csv(res/f'{prefix}_adaptation_seed0_metrics.csv')])
 for policy,g in pred.groupby('policy'):
  pd.testing.assert_frame_equal(known_reference,g.set_index('txId')[['time_step','y']].sort_index())
  for t,s in g.groupby('time_step'):
   y=s.y.to_numpy();p=s.score.to_numpy();d=s.prediction.to_numpy();k=int(np.ceil(.1*len(s)));top=y[np.argsort(-p,kind='stable')[:k]]
   row={'model':name,'policy':policy,'seed':0,'time_step':int(t),'n_labeled':len(s),'n_illicit':int(y.sum()),'illicit_prevalence':float(y.mean()),'average_precision':average_precision_score(y,p),'precision':precision_score(y,d,zero_division=0),'recall':recall_score(y,d,zero_division=0),'f1':f1_score(y,d,zero_division=0),'review_k':k,'precision_at_10pct':top.mean(),'recall_at_10pct':top.sum()/y.sum()}
   saved=tables[(tables.policy==policy)&(tables.time_step==t)].iloc[0]
   for key in ['average_precision','precision','recall','f1','precision_at_10pct','recall_at_10pct']:
    assert np.isclose(row[key],saved[key],atol=1e-10),(name,policy,t,key)
   records.append(row)
 proof.append({'model':name,'prediction_rows':len(pred),'metric_rows':60,'metrics_recomputed':'passed'})
metrics=pd.DataFrame(records);metrics.to_csv(res/'audited_metrics_seed0.csv',index=False)
metrics['phase']=np.select([metrics.time_step.between(35,42),metrics.time_step.eq(43)],['before','transition'],default='after')
summary=metrics.groupby(['model','policy','phase']).agg(mean_AP=('average_precision','mean'),mean_F1=('f1','mean'),mean_recall=('recall','mean'),mean_AP_excess=('illicit_prevalence','mean')).reset_index()
summary['mean_AP_excess']=summary.mean_AP-summary.mean_AP_excess
summary.to_csv(res/'audited_phase_summary.csv',index=False)
print(summary[summary.phase=='after'].to_string(index=False))
nb=json.loads((r/'notebooks/AML_GNN_Real_Data_Setup.ipynb').read_text());errors=[i for i,c in enumerate(nb['cells']) for o in c.get('outputs',[]) if o.get('output_type')=='error'];assert not errors
out={'metric_rows':180,'saved_prediction_rows_checked':200040,'known_transactions_per_combination':16670,'recomputation':proof,'notebook_saved_errors':errors,'full_training_rerun_here':False,'seed_count':1}
(r/'docs/audit_reproduced.json').write_text(json.dumps(out,indent=2))
print('AUDIT PASSED',out['saved_prediction_rows_checked'])
