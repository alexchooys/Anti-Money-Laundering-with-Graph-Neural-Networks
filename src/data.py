"""Validated loading for the original Elliptic CSV schema; no automatic fallback."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def load_elliptic(folder):
    folder = Path(folder)
    features = pd.read_csv(folder / 'elliptic_txs_features.csv', header=None)
    if features.shape[1] != 167:
        raise ValueError('Expected txId plus 166 features (including time step).')
    features.columns = ['txId', 'time_step'] + [f'local_{i:02}' for i in range(93)] + [f'aggregate_{i:02}' for i in range(72)]
    classes = pd.read_csv(folder / 'elliptic_txs_classes.csv', dtype={'class': str})
    edges = pd.read_csv(folder / 'elliptic_txs_edgelist.csv')
    if not {'txId', 'class'}.issubset(classes.columns):
        raise ValueError('Expected class columns txId,class.')
    if list(edges.columns) != ['txId1', 'txId2']:
        raise ValueError('Expected edge columns txId1,txId2.')
    for frame in (features, classes):
        if frame.txId.duplicated().any():
            raise ValueError('Duplicate transaction IDs.')
    if set(features.txId) != set(classes.txId):
        raise ValueError('Feature and class IDs must match exactly.')
    nodes = features.merge(classes, on='txId', validate='one_to_one')
    if not set(nodes['class']).issubset({'1', '2', 'unknown'}):
        raise ValueError('Unexpected label. Original schema: 1 illicit, 2 licit, unknown.')
    numeric = nodes.drop(columns=['class']).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError('Missing/non-finite numeric data; investigate before training.')
    if not (nodes.time_step.between(1, 49) & (nodes.time_step % 1 == 0)).all():
        raise ValueError('Time steps must be integers from 1 to 49.')
    ids = set(nodes.txId)
    if not set(edges.to_numpy().ravel()).issubset(ids):
        raise ValueError('Edge endpoint absent from feature table.')
    if edges.duplicated().any():
        raise ValueError('Duplicate edges; review rather than silently removing.')
    times = nodes.set_index('txId').time_step
    if not edges.txId1.map(times).eq(edges.txId2.map(times)).all():
        raise ValueError('Cross-time edges found: revise snapshot isolation protocol.')
    nodes['y'] = nodes['class'].map({'1': 1, '2': 0}).fillna(-1).astype(int)
    return nodes, edges


def temporal_counts(nodes):
    counts = pd.crosstab(nodes.time_step, nodes['class']).reindex(index=range(1, 50), columns=['1', '2', 'unknown'], fill_value=0)
    counts.columns = ['illicit', 'licit', 'unknown']
    counts.index.name = 'time_step'
    counts['labeled'] = counts.illicit + counts.licit
    counts['illicit_share_labeled'] = counts.illicit.div(counts.labeled.replace(0, np.nan))
    return counts


def summarize(nodes, edges):
    counts = temporal_counts(nodes)
    return {'nodes': len(nodes), 'edges': len(edges), 'label_counts': nodes['class'].value_counts().to_dict(),
            'validation_31_34_illicit': int(counts.loc[31:34, 'illicit'].sum()),
            'validation_31_34_licit': int(counts.loc[31:34, 'licit'].sum()),
            'unlabeled_excluded_from_metrics': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--output-dir', default='results/real')
    args = parser.parse_args()
    nodes, edges = load_elliptic(args.data_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    temporal_counts(nodes).to_csv(out / 'counts_by_step.csv')
    (out / 'data_summary.json').write_text(json.dumps(summarize(nodes, edges), indent=2))
    print(json.dumps(summarize(nodes, edges), indent=2))
