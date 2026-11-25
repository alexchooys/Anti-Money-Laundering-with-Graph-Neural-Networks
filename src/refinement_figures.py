"""Regenerate publication figures from audited data, not screenshots."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];EV=ROOT/'results/deliverable3/evaluation';OUT=ROOT/'docs/figures_d3'
def main():
 OUT.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
 models=['XGBoost','MLP','GraphSAGE'];policies=['frozen','expanding','rolling','recency'];colors=['#8396a5','#376cab','#c08732','#16838b']
 summary=pd.read_csv(EV/'three_seed_summary.csv');phase=pd.read_csv(EV/'phase_results_all_seeds.csv');metrics=pd.read_csv(EV/'independently_audited_metrics.csv')
 fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(3)
 for i,(policy,color) in enumerate(zip(policies,colors)):
  rows=summary[summary.policy==policy].set_index('model').loc[models];pos=x+(i-1.5)*.19
  ax.bar(pos,rows.AP_mean,width=.18,yerr=rows.AP_sd,capsize=3,label=policy,color=color)
 ax.set_xticks(x,models);ax.set_ylim(0,.5);ax.set_ylabel('Mean post-shift average precision');ax.legend(ncol=4,frameon=False,loc='upper right');ax.set_title('Three-seed results on steps 44–49');ax.text(0,-.20,'Error bars: sample SD across seeds 0, 1 and 2. Seed 0 also appears in Deliverable 2.',transform=ax.transAxes,fontsize=9)
 for ext in ['png','pdf']:fig.savefig(OUT/f'three_seed_comparison.{ext}',dpi=180)
 plt.close(fig)
 gains=pd.read_csv(EV/'paired_adaptation_gains.csv');fig,ax=plt.subplots(figsize=(7,4))
 for j,model in enumerate(models):
  values=gains[gains.model==model].sort_values('seed').recency_minus_expanding.to_numpy();ax.scatter(j+np.array([-.06,0,.06]),values,s=48,color=colors[j+1],label=model);ax.errorbar(j,values.mean(),yerr=values.std(ddof=1),fmt='_',markersize=22,capsize=7,color='#152f43')
 ax.axhline(0,color='#8493a0',ls='--');ax.set_xticks(range(3),models);ax.set_ylabel('AP gain: recency minus expanding');ax.set_title('Paired adaptation gains within each seed');ax.text(0,-.2,'Dots: individual seeds. Black markers: mean ± sample SD, not a confidence interval.',transform=ax.transAxes,fontsize=8.5)
 for ext in ['png','pdf']:fig.savefig(OUT/f'paired_recency_gains.{ext}',dpi=180)
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
 for ax,model in zip(axes,models):
  for seed,color in zip(range(3),['#8193a4','#16838b','#376cab']):
   g=metrics[(metrics.model==model)&(metrics.policy=='recency')&(metrics.seed==seed)].sort_values('time_step');ax.plot(g.time_step,g.average_precision,'o-',ms=3,color=color,label=f'Seed {seed}')
  ax.axvline(43,color='#8b7255',ls='--');ax.set_title(model);ax.set_xlabel('Time step');ax.set_xticks([35,39,43,47,49]);ax.set_ylim(0,1.02)
 axes[0].set_ylabel('Average precision');axes[-1].legend(frameon=False);fig.suptitle('Recency weighting: temporal performance across seeds',y=1.03)
 for ext in ['png','pdf']:fig.savefig(OUT/f'temporal_recency_seeds.{ext}',dpi=180)
 plt.close(fig)
 # Preserve seed differences rather than pool stochastic repetitions as new examples.
 confusion=pd.read_csv(EV/'confusion_counts_by_phase.csv');fig,axes=plt.subplots(1,3,figsize=(10,3.5))
 for ax,model in zip(axes,models):
  r=confusion[(confusion.model==model)&(confusion.policy=='recency')&(confusion.seed==0)&(confusion.phase=='after')].iloc[0]
  a=np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int);ax.imshow(np.log1p(a),cmap='Blues')
  for (i,j),v in np.ndenumerate(a):ax.text(j,i,str(v),ha='center',va='center',color='white' if np.log1p(v)>np.log1p(a.max())*.6 else '#15364a')
  ax.set_xticks([0,1],['Below threshold','Flagged'],fontsize=8);ax.set_yticks([0,1],['Licit','Illicit']);ax.set_title(model);ax.set_xlabel('Predicted decision')
 axes[0].set_ylabel('Observed label');fig.suptitle('Original seed-0 recency confusion counts after the shift\nRaw counts annotated; logarithmic color scale. Seeds are not pooled.',fontsize=11,y=1.08)
 for ext in ['png','pdf']:fig.savefig(OUT/f'confusion_seed0_recency.{ext}',dpi=180)
 plt.close(fig)
 print('Saved four PNG/PDF figures to',OUT)
if __name__=='__main__':main()
