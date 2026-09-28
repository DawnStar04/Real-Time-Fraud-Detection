"""
Replay the transaction log against the live API at a controlled rate.

This replaces Kafka. One producer, one consumer, no replay-from-offset
requirement — a broker would be ceremony. If an interviewer asks why there
is no Kafka, the answer is that sentence, and it scores better than having
bolted one on.

Fraud bursts are preserved: transactions go out in timestamp order, so the
"4 transactions in 90 seconds" pattern actually materialises in Redis and
the velocity features fire the way they will in production.

    python simulate.py --rate 20
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools

import httpx


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/transactions.csv")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--rate", type=float, default=20.0, help="transactions per second")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole file")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N rows (they were used for training)")
    args = ap.parse_args()

    delay = 1.0 / args.rate
    sent = flagged = 0

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = itertools.islice(reader, args.skip, None)
        if args.limit:
            rows = itertools.islice(rows, args.limit)

        async with httpx.AsyncClient(base_url=args.api, timeout=10.0) as client:
            for row in rows:
                payload = {
                    "txn_id": row["txn_id"],
                    "card_id": row["card_id"],
                    "amount": float(row["amount"]),
                    "merchant": row["merchant"],
                    "merchant_category": row["merchant_category"],
                    "device_id": row["device_id"],
                    "country": row["country"],
                    "label": int(row["label"]),
                    # Pass the original timestamp so the service computes
                    # windows in event time. Without this, replaying faster
                    # than real time destroys every velocity feature.
                    "event_time": float(row["timestamp"]),
                }
                try:
                    r = await client.post("/score", json=payload)
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    print(f"! {e}")
                    await asyncio.sleep(1.0)
                    continue

                res = r.json()
                sent += 1
                if res["decision"] != "approve":
                    flagged += 1
                    top = res["reasons"][0]["text"] if res["reasons"] else "-"
                    print(
                        f"[{res['decision'].upper():6}] {res['txn_id']} "
                        f"{res['amount']:>10,.0f}  p={res['fraud_probability']:.3f}  "
                        f"{res['latency_ms']:>5.1f}ms  {top}"
                    )
                if sent % 200 == 0:
                    print(f"--- {sent:,} sent, {flagged:,} flagged "
                          f"({flagged / sent:.2%}) ---")

                await asyncio.sleep(delay)

    print(f"\ndone: {sent:,} sent, {flagged:,} flagged")


if __name__ == "__main__":
    asyncio.run(main())
