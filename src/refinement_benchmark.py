"""Paired warm-callback comparison. Excludes browser/network/startup latency."""
from pathlib import Path
import json,time,platform
import numpy as np
from src import service,service_v1
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=lambda:service_v1.replay('XGBoost','recency',49,10)
 new=lambda:service.replay('XGBoost','recency',49,10,0)
 a,b=old(),new()
 for key in ['average_precision','f1','precision','recall','precision_at_k','recall_at_k']:
  np.testing.assert_allclose(a['metrics'][key],b['metrics'][key],atol=1e-12)
 assert [r['txId'] for r in a['queue']]==[r['txId'] for r in b['queue']]
 for _ in range(3):old();new()
 times={'deliverable2':[],'deliverable3':[]}
 for i in range(30):
  functions=[('deliverable2',old),('deliverable3',new)]
  if i%2:functions.reverse()
  for label,fn in functions:
   start=time.perf_counter();fn();times[label].append((time.perf_counter()-start)*1000)
 result={'workload':'XGBoost/recency/seed0/step49/10% known-label review queue','scope':'Warm Python callback only; no browser, HTTP, startup or network latency. Same process/hardware, alternating order; not a usability study.','repetitions':30,'warmup_per_version':3,'python':platform.python_version(),'processor':platform.machine(),'verified_metrics_and_queue_equal':True}
 for k,v in times.items():result[k]={'median_ms':float(np.median(v)),'p95_ms':float(np.quantile(v,.95)),'samples_ms':v}
 result['median_speedup']=result['deliverable2']['median_ms']/result['deliverable3']['median_ms']
 out=ROOT/'results/deliverable3/evaluation/backend_latency.json';out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in times},indent=2));print('D2 median ms',result['deliverable2']['median_ms'],'D3 median ms',result['deliverable3']['median_ms'])
if __name__=='__main__':main()
