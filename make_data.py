"""
Generate a synthetic labelled transaction log so the repo runs end-to-end
before you download anything from Kaggle.

The fraud here is deliberately *velocity-shaped*: compromised cards get a
burst of transactions in a short window, at unfamiliar merchants, on a new
device, with escalating amounts. That is what real card-testing fraud looks
like, and it is exactly the pattern the Redis feature layer exists to catch.

Swap this for IEEE-CIS or PaySim once the pipeline works. Keep the same
column names and nothing downstream changes.

    python make_data.py --out data/transactions.csv --n 200000
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "fuel", "electronics", "travel", "gaming",
              "gift_cards", "food_delivery", "pharmacy", "apparel"]
# Categories fraud rings favour — liquid, resellable, instantly delivered.
FRAUD_FAVOURED = ["gift_cards", "electronics", "gaming"]
COUNTRIES = ["IN", "IN", "IN", "IN", "AE", "US", "SG"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/transactions.csv")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--cards", type=int, default=6_000)
    ap.add_argument("--fraud-rate", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # Each card has its own spending personality, so "5x this card's mean"
    # means something different per card — which is the whole point of a
    # per-card rolling average rather than a global threshold.
    card_ids = [f"card_{i:06d}" for i in range(args.cards)]
    card_mean = {c: float(rng.lognormal(6.2, 0.7)) for c in card_ids}
    card_home_device = {c: f"dev_{rng.integers(0, 90_000):06d}" for c in card_ids}
    card_country = {c: random.choice(COUNTRIES) for c in card_ids}

    start = 1_735_689_600  # 2025-01-01 UTC
    rows = []

    # --- legitimate traffic ---
    #
    # Crucially, some of this is ALSO bursty: people buy three things in a
    # row, retry a declined payment, or get billed twice by a subscription.
    # Without these, "more than one transaction in a minute" perfectly
    # separates fraud from legit, the model scores PR-AUC 1.00, and the
    # project is worthless — a metric that good means the data leaked.
    n_legit = int(args.n * (1 - args.fraud_rate))
    made = 0
    while made < n_legit:
        card = random.choice(card_ids)
        day = int(rng.integers(0, 90))
        # Diurnal pattern: most spend during waking hours.
        hour = int(np.clip(rng.normal(14, 4.5), 0, 23.99))
        t0 = start + day * 86400 + hour * 3600 + int(rng.integers(0, 3600))

        # 18% of sessions are multi-transaction, same as real card traffic.
        session = int(rng.integers(2, 6)) if rng.random() < 0.18 else 1
        # Occasionally a legit user is on a new phone, abroad, at 3am.
        device = (
            card_home_device[card]
            if rng.random() < 0.90
            else f"dev_{rng.integers(0, 999_999):06d}"
        )
        country = card_country[card] if rng.random() < 0.95 else random.choice(COUNTRIES)

        for k in range(session):
            if made >= n_legit:
                break
            # Legit spend is mostly near the card's mean but has a fat tail:
            # the occasional flight or laptop overlaps fraud amounts.
            mult = float(rng.uniform(3.0, 12.0)) if rng.random() < 0.04 else \
                float(rng.lognormal(0.0, 0.55))
            rows.append(
                {
                    "txn_id": "",
                    "timestamp": t0 + k * int(rng.integers(15, 240)),
                    "card_id": card,
                    "amount": round(max(20.0, card_mean[card] * mult), 2),
                    "merchant": f"m_{rng.integers(0, 4000):04d}",
                    "merchant_category": random.choice(CATEGORIES),
                    "device_id": device,
                    "country": country,
                    "label": 0,
                }
            )
            made += 1

    # --- fraud ---
    #
    # A mix of patterns, not one signature. If every fraud looks the same the
    # model memorises that one shape and the metric lies about how it would
    # do against an attacker who changes tactics.
    n_fraud = args.n - n_legit
    made = 0
    while made < n_fraud:
        card = random.choice(card_ids)
        day = int(rng.integers(0, 90))
        roll = rng.random()

        if roll < 0.55:
            # Card testing: fast burst, fresh device, overnight.
            burst = int(rng.integers(3, 9))
            gap = (8, 70)
            hour = int(np.clip(rng.normal(2.5, 3.5), 0, 23.99))
            device = f"dev_{rng.integers(0, 999_999):06d}"
        elif roll < 0.85:
            # Patient fraud: spread over hours to duck velocity rules.
            burst = int(rng.integers(2, 5))
            gap = (1800, 9000)
            hour = int(rng.integers(0, 24))
            device = f"dev_{rng.integers(0, 999_999):06d}"
        else:
            # Single quiet transaction on the real device — the hard case
            # that no velocity feature can catch. Keeps recall honest.
            burst = 1
            gap = (60, 600)
            hour = int(rng.integers(0, 24))
            device = card_home_device[card]

        t0 = start + day * 86400 + hour * 3600
        country = random.choice(COUNTRIES) if rng.random() < 0.5 else card_country[card]

        for k in range(burst):
            if made >= n_fraud:
                break
            # Escalating amounts, but overlapping the legit fat tail.
            mult = 0.2 if (k == 0 and burst > 2) else float(rng.uniform(1.5, 9.0))
            rows.append(
                {
                    "txn_id": "",
                    "timestamp": t0 + k * int(rng.integers(*gap)),
                    "card_id": card,
                    "amount": round(max(20.0, card_mean[card] * mult), 2),
                    "merchant": f"m_{rng.integers(0, 4000):04d}",
                    "merchant_category": random.choice(FRAUD_FAVOURED)
                    if rng.random() < 0.55
                    else random.choice(CATEGORIES),
                    "device_id": device,
                    "country": country,
                    "label": 1,
                }
            )
            made += 1

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["txn_id"] = [f"txn_{i:08d}" for i in range(len(df))]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} rows, {df.label.mean():.3%} fraud -> {args.out}")


if __name__ == "__main__":
    main()
