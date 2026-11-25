"""Optional retrospective error diagnostic using graph structure, never neighbor labels.
Requires original CSVs. AP here pools rows within a phase and degree group;
it differs from the snapshot-macro AP used for the primary comparison.
"""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,f1_score,recall_score,precision_score
from src.data import load_elliptic
from src.refinement_audit import ROOT,D3,PREFIX,POLICIES,sources

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data-dir',required=True);args=parser.parse_args()
 nodes,edges=load_elliptic(args.data_dir)
 degree=edges.txId2.value_counts().reindex(nodes.txId,fill_value=0)
 info=pd.DataFrame({'txId':nodes.txId.to_numpy(),'incoming_degree':degree.to_numpy()})
 info['degree_group']=np.select([info.incoming_degree.eq(0),info.incoming_degree.eq(1)],['0 incoming','1 incoming'],default='2+ incoming')
 rows=[]
 for model in PREFIX:
  for seed in range(3):
   f,a,*_=sources(model,seed);pred=pd.concat([pd.read_csv(f).assign(policy='frozen'),pd.read_csv(a)],ignore_index=True)
   pred=pred.merge(info,on='txId',validate='many_to_one');assert len(pred)==16670*4
   pred['phase']=np.select([pred.time_step.between(35,42),pred.time_step.eq(43)],['before','transition'],default='after')
   for (policy,phase,group),g in pred[pred.phase!='transition'].groupby(['policy','phase','degree_group']):
    y=g.y.to_numpy();positive=int(y.sum());negative=len(y)-positive
    rows.append(dict(model=model,seed=seed,policy=policy,phase=phase,degree_group=group,n_labeled=len(g),n_illicit=positive,n_licit=negative,support_at_least_10_each=positive>=10 and negative>=10,pooled_AP=average_precision_score(y,g.score) if positive and negative else np.nan,f1=f1_score(y,g.prediction,zero_division=0),recall=recall_score(y,g.prediction,zero_division=0),precision=precision_score(y,g.prediction,zero_division=0)))
 out=D3/'evaluation';pd.DataFrame(rows).to_csv(out/'degree_error_analysis.csv',index=False)
 info.to_csv(out/'transaction_incoming_degree.csv',index=False)
 table=pd.DataFrame(rows);print(table[(table.phase=='after')&(table.policy=='recency')][['model','seed','degree_group','n_labeled','n_illicit','support_at_least_10_each','pooled_AP','recall']].round(4).to_string(index=False))
 print('Diagnostic saved. AP pools phase/degree rows; do not compare it directly with macro snapshot AP. No features or thresholds changed.')
if __name__=='__main__':main()
