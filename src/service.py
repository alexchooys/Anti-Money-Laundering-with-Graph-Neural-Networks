"""Shared interface callbacks. Saved replay is explicitly separate from live inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from src.inference import PortableXGBoost

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
PREFIXES = {'XGBoost': 'xgb', 'MLP': 'mlp', 'GraphSAGE': 'graphsage'}
POLICIES = ['frozen', 'expanding', 'rolling', 'recency']


def _clean(value):
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list): return [_clean(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def replay(model='XGBoost', policy='recency', step=49, budget=10):
    if model not in PREFIXES or policy not in POLICIES:
        raise ValueError('Choose an available model and policy.')
    if not 35 <= step <= 49 or not 1 <= budget <= 100:
        raise ValueError('Step must be 35-49; review budget must be 1-100%.')
    prefix = PREFIXES[model]
    path = RESULTS / (f'{prefix}_frozen_test_predictions.csv' if policy == 'frozen'
                      else f'{prefix}_adaptation_seed0_predictions.csv')
    data = pd.read_csv(path)
    if policy != 'frozen': data = data[data.policy == policy]
    subset = data[data.time_step == step].copy()
    if subset.empty: raise ValueError('No saved predictions for this selection.')
    y, scores, pred = subset.y.to_numpy(), subset.score.to_numpy(), subset.prediction.to_numpy()
    k = max(1, int(np.ceil(len(y) * budget / 100)))
    ranked = subset.sort_values('score', ascending=False, kind='stable').head(k)
    info = {'mode': 'saved prediction replay; no model refit', 'model': model,
            'policy': policy, 'seed': 0, 'time_step': step, 'labeled': len(y),
            'illicit': int(y.sum()), 'average_precision': average_precision_score(y, scores),
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0),
            'f1': f1_score(y, pred, zero_division=0), 'review_k': k,
            'precision_at_k': ranked.y.mean(), 'recall_at_k': ranked.y.sum()/y.sum(),
            'scores_are_calibrated_probabilities': False,
            'warning': 'Research only. Metrics apply to known-label transactions; no automatic sanctions.'}
    if int(y.sum()) < 10: info['low_support_warning'] = 'Fewer than 10 illicit nodes; metrics are unstable.'
    history = pd.read_csv(RESULTS / 'audited_metrics_seed0.csv')
    history = history[(history.model == model) & (history.policy == policy)]
    return _clean({'metrics':info, 'queue':ranked.head(100).to_dict(orient='records'),
                   'history':history[['time_step','average_precision','f1','n_illicit']].to_dict(orient='records')})


def live(values):
    if isinstance(values, str): values = json.loads(values)
    row = np.asarray(values, dtype=np.float32)
    if row.ndim != 1 or row.shape[0] != 93 or not np.isfinite(row).all():
        raise ValueError('Enter one JSON array containing exactly 93 finite numbers.')
    model_path = RESULTS / 'xgb_frozen_seed0.json'
    predictor = PortableXGBoost(model_path)
    try:
        import xgboost as xgb
    except ImportError:
        score = float(predictor.predict(row)[0])
        engine = 'restricted numeric JSON-tree evaluator; parity verified on 50 real test rows'
    else:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        booster.set_param({'device': 'cpu'})
        score = float(booster.predict(xgb.DMatrix(row[None, :], feature_names=predictor.columns))[0])
        engine = 'native XGBoost CPU inference'
    settings = json.loads((RESULTS / 'xgb_frozen_run_info.json').read_text())
    threshold = float(settings['threshold'])
    return {'mode':'live model inference; frozen XGBoost, seed 0', 'score':score,
            'threshold':threshold, 'predicted_label':int(score >= threshold),
            'display_label':'flag for human review' if score >= threshold else 'below review threshold',
            'input_features':93, 'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'engine':engine,
            'note':'Weighted-model risk score, not calibrated probability or proof of illegal activity.'}
