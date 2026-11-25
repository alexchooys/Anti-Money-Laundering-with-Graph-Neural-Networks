from pathlib import Path
import pandas as pd,numpy as np,json,sys
r=Path(__file__).resolve().parents[1]
from src.refinement_audit import sources,PREFIX
from sklearn.metrics import average_precision_score,f1_score,recall_score,precision_score
e=r/'results/deliverable3/evaluation';degree=pd.read_csv(e/'transaction_incoming_degree.csv');saved=pd.read_csv(e/'degree_error_analysis.csv')
assert len(degree)==203769 and not degree.txId.duplicated().any();assert degree.incoming_degree.ge(0).all();assert degree.incoming_degree.sum()==234355
expected=np.select([degree.incoming_degree.eq(0),degree.incoming_degree.eq(1)],['0 incoming','1 incoming'],default='2+ incoming');np.testing.assert_array_equal(degree.degree_group,expected)
count=0
for model in PREFIX:
 for seed in range(3):
  f,a,*_=sources(model,seed);p=pd.concat([pd.read_csv(f).assign(policy='frozen'),pd.read_csv(a)],ignore_index=True).merge(degree,on='txId',validate='many_to_one');assert len(p)==66680
  p['phase']=np.select([p.time_step.between(35,42),p.time_step.eq(43)],['before','transition'],default='after')
  for (policy,phase,group),g in p[p.phase!='transition'].groupby(['policy','phase','degree_group']):
   y=g.y.to_numpy();s=saved[(saved.model==model)&(saved.seed==seed)&(saved.policy==policy)&(saved.phase==phase)&(saved.degree_group==group)].iloc[0]
   assert len(g)==s.n_labeled and y.sum()==s.n_illicit
   values=dict(pooled_AP=average_precision_score(y,g.score) if y.sum() and y.sum()!=len(y) else np.nan,f1=f1_score(y,g.prediction,zero_division=0),recall=recall_score(y,g.prediction,zero_division=0),precision=precision_score(y,g.prediction,zero_division=0))
   for key,v in values.items():np.testing.assert_allclose(v,s[key],atol=1e-10,rtol=1e-9,equal_nan=True)
   count+=1
assert count==216
(r/'docs/degree_audit.json').write_text(json.dumps({'status':'passed','diagnostic_rows_recomputed':count,'degree_nodes':len(degree),'incoming_degree_sum':int(degree.incoming_degree.sum()),'raw_edge_to_degree_computation':'executed in Colab; degree-map joins and metrics recomputed here'},indent=2))
print('Degree audit passed:',count,'rows')
