"""
Train the fraud model.

The important idea here is that the velocity features the API computes from
Redis at serve time must be reproduced offline, point-in-time correctly, for
training. If you train on features computed differently from how they are
computed in production, you get train/serve skew: the model is optimised for
numbers it will never actually receive.

So instead of vectorised group aggregates over the whole dataset (which peek
at the future), we replay the transaction log in timestamp order and compute
each row's features from the rows before it only. It is slower, and it is
the only version that is correct.

Usage:
    python train.py --csv data/transactions.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from app.features import WINDOWS

CATEGORICAL = ["merchant_category", "country"]


def replay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk the log in time order, computing per-card velocity features from
    strictly prior events. Mirrors app/features.py exactly.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    history: dict[str, deque] = defaultdict(deque)  # card -> (ts, amount)
    running: dict[str, list] = defaultdict(lambda: [0, 0.0])  # card -> [n, sum]
    merchants: dict[str, set] = defaultdict(set)
    devices: dict[str, set] = defaultdict(set)

    rows = []
    for t in df.itertuples(index=False):
        card, ts, amt = t.card_id, float(t.timestamp), float(t.amount)
        h = history[card]

        # Evict events older than the widest window, same as the Redis
        # ZREMRANGEBYSCORE on the read path.
        while h and h[0][0] < ts - max(WINDOWS):
            h.popleft()

        counts = {w: 0 for w in WINDOWS}
        sums = {60: 0.0, 300: 0.0, 3600: 0.0}
        for ets, eamt in h:
            age = ts - ets
            for w in WINDOWS:
                if age <= w:
                    counts[w] += 1
            for w in sums:
                if age <= w:
                    sums[w] += eamt

        n, total = running[card]
        mean = total / n if n else 0.0
        ratio = min(amt / mean, 100.0) if mean > 0 else 1.0
        since_last = (ts - h[-1][0]) if h else 86400.0 * 7

        dt = pd.Timestamp(ts, unit="s", tz="UTC")
        rows.append(
            {
                "amount": amt,
                "hour_of_day": dt.hour,
                "is_night": int(dt.hour < 6 or dt.hour >= 23),
                "txn_count_60s": counts[60],
                "txn_count_300s": counts[300],
                "txn_count_3600s": counts[3600],
                "txn_count_86400s": counts[86400],
                "amount_sum_60s": sums[60],
                "amount_sum_300s": sums[300],
                "amount_sum_3600s": sums[3600],
                "amount_vs_card_mean": ratio,
                "seconds_since_last_txn": since_last,
                "is_new_merchant_for_card": int(t.merchant not in merchants[card]),
                "is_new_device_for_card": int(t.device_id not in devices[card]),
            }
        )

        # Update state only after the row's features are frozen.
        h.append((ts, amt))
        running[card] = [n + 1, total + amt]
        merchants[card].add(t.merchant)
        devices[card].add(t.device_id)

    feats = pd.DataFrame(rows)
    for col in CATEGORICAL:
        feats[col] = df[col].values
    feats["label"] = df["label"].values
    feats["timestamp"] = df["timestamp"].values
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/transactions.csv")
    ap.add_argument("--out", default="models/model.joblib")
    ap.add_argument("--alert-budget", type=float, default=0.005,
                    help="fraction of transactions you can afford to review")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    print(f"{len(raw):,} transactions, {raw.label.mean():.3%} fraud")

    data = replay_features(raw)

    # Time-based split. A random split leaks: fraud arrives in bursts, so a
    # shuffled test set contains transactions from the same attack the model
    # saw in training, and your metrics come out flattering and fake.
    cut = data.timestamp.quantile(0.8)
    train = data[data.timestamp <= cut].copy()
    test = data[data.timestamp > cut].copy()
    print(f"train {len(train):,} / test {len(test):,}")

    categories = {}
    for col in CATEGORICAL:
        cats = sorted(train[col].astype(str).unique())
        categories[col] = {c: i for i, c in enumerate(cats)}
        for split in (train, test):
            split[col] = split[col].astype(str).map(categories[col]).fillna(-1).astype(int)

    drop = ["label", "timestamp"]
    feature_names = [c for c in train.columns if c not in drop]
    X_tr, y_tr = train[feature_names], train.label
    X_te, y_te = test[feature_names], test.label

    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        # Do NOT oversample or SMOTE. Class weighting preserves calibrated
        # probabilities, which the threshold tuning below depends on.
        scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, categorical_feature=CATEGORICAL)

    prob = model.predict_proba(X_te)[:, 1]
    pr_auc = average_precision_score(y_te, prob)
    roc = roc_auc_score(y_te, prob)

    # The metric that matters: at an alert budget you can actually staff,
    # what fraction of fraud do you catch?
    threshold = float(np.quantile(prob, 1 - args.alert_budget))
    flagged = prob >= threshold
    recall = y_te[flagged].sum() / max(y_te.sum(), 1)
    precision = y_te[flagged].mean() if flagged.sum() else 0.0

    print(f"\nPR-AUC        {pr_auc:.4f}")
    print(f"ROC-AUC       {roc:.4f}   (ignore this, it flatters imbalanced data)")
    print(f"\nAt a {args.alert_budget:.1%} review rate (threshold {threshold:.3f}):")
    print(f"  recall      {recall:.1%}  <- the resume number")
    print(f"  precision   {precision:.1%}")
    print(f"\nSet REVIEW_THRESHOLD={threshold:.3f} in the API env.")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "categories": categories,
            "metrics": {
                "pr_auc": pr_auc,
                "recall_at_budget": float(recall),
                "precision_at_budget": float(precision),
                "threshold": threshold,
                "alert_budget": args.alert_budget,
            },
        },
        args.out,
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
