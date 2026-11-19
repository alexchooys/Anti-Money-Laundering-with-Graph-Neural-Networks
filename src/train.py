"""Reproduce one frozen model with the settings used in the executed notebook.

The executed notebook remains the reference for the full 12-cell adaptation
study. This CLI writes new files and never overwrites the submitted results.
"""
from pathlib import Path
import argparse
import copy
import json
import random
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from src.data import load_elliptic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['XGBoost', 'MLP', 'GraphSAGE'], default='XGBoost')
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--output-dir', default='results_reproduced')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out / f'{args.family}_seed{args.seed}_run.json').exists():
        raise FileExistsError('Use a new output directory to preserve the previous run.')
    nodes, edges = load_elliptic(args.data_dir)
    cols = [c for c in nodes if c.startswith('local_')]
    train = nodes.time_step.between(1, 30) & nodes.y.ge(0)
    val = nodes.time_step.between(31, 34) & nodes.y.ge(0)
    test = nodes.time_step.between(35, 49) & nodes.y.ge(0)
    weight = float((nodes.loc[train, 'y'] == 0).sum() / (nodes.loc[train, 'y'] == 1).sum())
    random.seed(args.seed); np.random.seed(args.seed)
    selected_epoch = None
    start = time.perf_counter()
    if args.family == 'XGBoost':
        from xgboost import XGBClassifier, DMatrix
        model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=.05,
            tree_method='hist', device=args.device, scale_pos_weight=weight,
            objective='binary:logistic', eval_metric='logloss', random_state=args.seed, n_jobs=2)
        model.fit(nodes.loc[train, cols].astype('float32'), nodes.loc[train, 'y'])
        val_scores = model.get_booster().predict(DMatrix(nodes.loc[val, cols].astype('float32')))
        test_scores = model.get_booster().predict(DMatrix(nodes.loc[test, cols].astype('float32')))
        model.save_model(str(out / f'XGBoost_seed{args.seed}.json'))
    else:
        import torch
        from torch import nn
        import joblib
        from src.models import TransactionMLP, TransactionGraphSAGE
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            if not torch.cuda.is_available(): raise RuntimeError('CUDA required; or choose --device cpu.')
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        scaler = StandardScaler().fit(nodes.loc[train, cols].astype('float32'))
        def partition(a, b):
            frame = nodes[nodes.time_step.between(a, b)]
            x = torch.from_numpy(scaler.transform(frame[cols].astype('float32')).astype('float32')).to(args.device)
            y = torch.tensor(frame.y.to_numpy(), dtype=torch.long, device=args.device)
            index = pd.Series(np.arange(len(frame)), index=frame.txId)
            e = edges[edges.txId1.isin(index.index) & edges.txId2.isin(index.index)]
            edge = torch.tensor(np.stack([e.txId1.map(index).to_numpy(), e.txId2.map(index).to_numpy()]), dtype=torch.long, device=args.device)
            return x, y, edge
        train_part, val_part = partition(1, 30), partition(31, 34)
        model = (TransactionMLP() if args.family == 'MLP' else TransactionGraphSAGE()).to(args.device)
        def scores(part):
            x, y, edge = part
            logits = model(x) if args.family == 'MLP' else model(x, edge)
            return logits[y >= 0]
        # MLP training excludes unknown nodes, matching the original dropout batch.
        if args.family == 'MLP':
            x, y, edge = train_part
            train_part = (x[y >= 0], y[y >= 0], edge)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=args.device))
        best_ap, patience, best_state = -np.inf, 0, None
        history = []
        for epoch in range(1, 201):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = criterion(scores(train_part), train_part[1][train_part[1] >= 0].float())
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward(); optimizer.step(); model.eval()
            with torch.no_grad(): val_scores = torch.sigmoid(scores(val_part)).cpu().numpy()
            ap = average_precision_score(nodes.loc[val, 'y'], val_scores)
            history.append({'epoch':epoch, 'loss':loss.item(), 'validation_AP':ap})
            if ap > best_ap + 1e-5:
                best_ap, selected_epoch, patience = ap, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else: patience += 1
            if patience >= 25: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(scores(val_part)).cpu().numpy()
            test_scores = torch.sigmoid(scores(partition(35, 49))).cpu().numpy()
        torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'selected_epoch':selected_epoch},out/f'{args.family}_seed{args.seed}.pt')
        joblib.dump(scaler, out / f'{args.family}_scaler.joblib')
        pd.DataFrame(history).to_csv(out / f'{args.family}_history.csv', index=False)
    grid = np.linspace(.05, .95, 19)
    threshold = float(max(grid, key=lambda t:(f1_score(nodes.loc[val,'y'],val_scores>=t),t)))
    predictions = nodes.loc[test, ['txId','time_step','y']].copy()
    predictions['score'] = test_scores; predictions['prediction'] = (test_scores >= threshold).astype(int)
    predictions.to_csv(out / f'{args.family}_seed{args.seed}_predictions.csv',index=False)
    info = {'family':args.family,'seed':args.seed,'device':args.device,'threshold':threshold,
            'selected_epoch':selected_epoch,'validation_AP':float(average_precision_score(nodes.loc[val,'y'],val_scores)),
            'elapsed_train_validate_infer_seconds':time.perf_counter()-start}
    (out / f'{args.family}_seed{args.seed}_run.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__': main()
