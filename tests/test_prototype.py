"""Run: python -m unittest discover -s tests -v (no GPU or raw data required)."""
import unittest
import numpy as np
from fastapi.testclient import TestClient
from src.service import replay,live
from src.inference import PortableXGBoost
from ui.app import app

class PrototypeTests(unittest.TestCase):
    def test_replay_all_combinations(self):
        for model in ['XGBoost','MLP','GraphSAGE']:
            for policy in ['frozen','expanding','rolling','recency']:
                for step in range(35,50):
                    d=replay(model,policy,step,10)
                    self.assertGreater(d['metrics']['labeled'],0)
                    self.assertEqual(d['metrics']['seed'],0)
    def test_model_arithmetic(self):
        # Independent two-leaf tree fixture: tests branch direction and sigmoid.
        predictor=PortableXGBoost.__new__(PortableXGBoost)
        predictor.columns=['x']*93;predictor.base_margin=np.float32(0)
        predictor.trees=[{'left_children':[1,-1,-1], 'right_children':[2,-1,-1],
                          'split_indices':[0,0,0], 'split_conditions':[0,-1,1]}]
        values=np.zeros((2,93),dtype=np.float32);values[0,0]=-1
        np.testing.assert_allclose(predictor.predict(values),[1/(1+np.exp(1)),1/(1+np.exp(-1))])
    def test_http_and_errors(self):
        with TestClient(app) as client:
            self.assertEqual(client.get('/').status_code,200)
            self.assertEqual(client.get('/api/replay?step=46&model=GraphSAGE&policy=recency').status_code,200)
            self.assertIn('low_support_warning',client.get('/api/replay?step=46').json()['metrics'])
            self.assertEqual(client.get('/api/replay?policy=invalid').status_code,400)
            self.assertEqual(client.post('/api/predict',json={'features':[0]*92}).status_code,400)
            result=client.post('/api/predict',json={'features':[0]*93})
            self.assertEqual(result.status_code,200)
            self.assertEqual(result.json()['predicted_label'],0)
            self.assertGreater(result.json()['score'],0)
            self.assertLess(result.json()['score'],1)

if __name__=='__main__':unittest.main()
