import csv,io,unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from ui.app import app
from src import service
class RefinedInterfaceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.client=TestClient(app)
 def test_all_540_selections_match_audit(self):
  metrics=service._metrics()
  for r in metrics.itertuples():
   result=service.replay(r.model,r.policy,int(r.time_step),10,int(r.seed))['metrics']
   for key in ['average_precision','f1','precision','recall','tn','fp','fn','tp']:
    self.assertAlmostEqual(result[key],getattr(r,key),places=10)
 def test_real_example_single_and_csv_agree(self):
  example=self.client.get('/api/example').json();out=self.client.post('/api/predict',json={'features':example['features']})
  self.assertEqual(out.status_code,200);self.assertAlmostEqual(out.json()['score'],example['expected_score'],places=6)
  template=self.client.get('/api/template').text;batch=self.client.post('/api/batch',json={'csv_text':template})
  self.assertEqual(batch.status_code,200);self.assertEqual(batch.json()['count'],1);self.assertAlmostEqual(batch.json()['rows'][0]['score'],out.json()['score'],places=10)
 def test_reordered_csv_and_input_errors(self):
  ex=service.example()['features'];text=io.StringIO();writer=csv.writer(text);writer.writerow(service.COLUMNS[::-1]);writer.writerow(ex[::-1])
  r=self.client.post('/api/batch',json={'csv_text':text.getvalue()});self.assertEqual(r.status_code,200);self.assertAlmostEqual(r.json()['rows'][0]['score'],service.live(ex)['score'],places=10)
  for features in [[0]*92,[True]*93,[None]*93,['0']*93]:self.assertEqual(self.client.post('/api/predict',json={'features':features}).status_code,400)
  for text in ['', 'a,b\n1,2\n',service.template_csv().replace('local_01','local_00',1),service.template_csv().replace(str(ex[0]),'NaN',1)]:self.assertEqual(self.client.post('/api/batch',json={'csv_text':text}).status_code,400)
  lines=service.template_csv().splitlines();oversized=lines[0]+'\n'+(lines[1]+'\n')*201
  self.assertEqual(self.client.post('/api/batch',json={'csv_text':oversized}).status_code,400)
 def test_queue_export_uses_exact_selection(self):
  r=self.client.get('/api/replay?model=GraphSAGE&policy=recency&seed=1&step=49&budget=20');self.assertEqual(r.status_code,200)
  export=self.client.get('/api/export?model=GraphSAGE&policy=recency&seed=1&step=49&budget=20');self.assertEqual(export.status_code,200)
  frame=pd.read_csv(io.StringIO(export.text));self.assertEqual(len(frame),r.json()['metrics']['review_k']);self.assertTrue(frame.score.is_monotonic_decreasing)
  self.assertEqual(frame.txId.head(100).tolist(),[v['txId'] for v in r.json()['queue']])
 def test_controls_summary_and_low_support(self):
  self.assertEqual(self.client.get('/').status_code,200)
  for query in ['seed=5','step=50','budget=0','model=Unknown','policy=bad']:
   self.assertEqual(self.client.get('/api/replay?'+query).status_code,400)
  r=self.client.get('/api/replay?step=46').json();self.assertIn('low_support_warning',r['metrics'])
  summary=self.client.get('/api/comparison').json();self.assertEqual(len(summary['rows']),12);self.assertTrue(all(r['seeds']==3 for r in summary['rows']))
if __name__=='__main__':unittest.main()
