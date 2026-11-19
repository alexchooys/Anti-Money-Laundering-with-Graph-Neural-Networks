"""Restricted portable evaluator for the supplied numeric binary XGBoost model.

This is not a general XGBoost implementation. It rejects unsupported models.
No labels, IDs or time steps enter the live prediction calculation.
"""
from pathlib import Path
import json
import numpy as np


class PortableXGBoost:
    def __init__(self, path):
        self.path = Path(path)
        payload = json.loads(self.path.read_text())
        learner = payload['learner']
        if learner['objective']['name'] != 'binary:logistic':
            raise ValueError('Only binary:logistic is supported.')
        booster = learner['gradient_booster']
        if booster['name'] != 'gbtree':
            raise ValueError('Only numeric gbtree models are supported.')
        self.columns = learner['feature_names']
        if len(self.columns) != 93:
            raise ValueError('Expected the original 93-predictor model.')
        self.trees = booster['model']['trees']
        if any(any(t['split_type']) or t['categories'] for t in self.trees):
            raise ValueError('Categorical splits are not supported.')
        base = json.loads(learner['learner_model_param']['base_score'])
        base = float(base[0] if isinstance(base, list) else base)
        if not 0 < base < 1:
            raise ValueError('Base score must be a probability.')
        self.base_margin = np.float32(np.log(base / (1 - base)))

    def predict(self, rows):
        values = np.asarray(rows, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.columns):
            raise ValueError('Provide exactly 93 numeric feature values per row.')
        if len(values) > 1000 or not np.isfinite(values).all():
            raise ValueError('At most 1000 rows; every value must be finite.')
        margins = np.full(len(values), self.base_margin, dtype=np.float32)
        for tree in self.trees:
            for row_index, row in enumerate(values):
                node = 0
                while tree['left_children'][node] != -1:
                    feature = tree['split_indices'][node]
                    cutoff = np.float32(tree['split_conditions'][node])
                    node = (tree['left_children'][node] if row[feature] < cutoff
                            else tree['right_children'][node])
                margins[row_index] += np.float32(tree['split_conditions'][node])
        return 1.0 / (1.0 + np.exp(-margins.astype(np.float64)))
