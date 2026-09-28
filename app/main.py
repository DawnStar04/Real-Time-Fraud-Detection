"""
The scoring service.

One endpoint does the real work: POST /score takes a transaction, enriches
it with velocity features from Redis, scores it, explains it, and returns
the decision. Everything is in-process and synchronous because the whole
budget is ~80ms and a queue hop would eat a chunk of that for no benefit at
this volume.

The order inside /score matters and is the single most important detail in
this file:

    read features -> score -> explain -> record the transaction

Recording last means a transaction never contributes to its own features.
Get this backwards and your offline PR-AUC looks superb while the live
service quietly degrades.
"""

from __future__ import annotations

import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .explain import Explainer
from .features import FeatureStore
from .schemas import ScoreResponse, Transaction

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Decision thresholds. REVIEW_THRESHOLD is the number you tune to hit your
# alert budget — see README, "picking the threshold". It is not 0.5; a 0.5
# cutoff on a 0.2%-positive problem approves almost everything.
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.35"))
BLOCK_THRESHOLD = float(os.getenv("BLOCK_THRESHOLD", "0.85"))

state: dict = {}

# Recent alerts, kept in memory purely so the dashboard has something to
# poll. A real deployment writes these to Postgres; for a demo an in-process
# ring buffer keeps the whole thing to two containers.
ALERTS: deque = deque(maxlen=200)
COUNTERS = {"scored": 0, "review": 0, "block": 0, "caught": 0, "missed": 0}
LATENCIES: deque = deque(maxlen=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(MODEL_PATH)
    state["model"] = bundle["model"]
    state["feature_names"] = bundle["feature_names"]
    state["categories"] = bundle.get("categories", {})
    state["explainer"] = Explainer(bundle["model"], bundle["feature_names"])
    state["store"] = FeatureStore(REDIS_URL)
    yield
    state.clear()


app = FastAPI(title="Real-time fraud detection", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_row(txn: Transaction, velocity: dict, now: float) -> pd.DataFrame:
    """Assemble the exact feature vector the model was trained on."""
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    row = {
        "amount": txn.amount,
        "hour_of_day": dt.hour,
        "is_night": int(dt.hour < 6 or dt.hour >= 23),
        **velocity,
    }

    # Categoricals are encoded as the integer codes recorded at training
    # time. Unseen values map to -1, which LightGBM handles natively — this
    # is why we save the category mapping alongside the model rather than
    # re-fitting an encoder at serve time.
    for col, mapping in state["categories"].items():
        raw = getattr(txn, col, None)
        row[col] = mapping.get(raw, -1)

    # Reindex to the training column order. Silent column reordering is a
    # classic source of "the model works in the notebook but not the API".
    return pd.DataFrame([row]).reindex(columns=state["feature_names"], fill_value=0)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "redis": state["store"].ping() if "store" in state else False,
        "model_loaded": "model" in state,
    }


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    if "model" not in state:
        raise HTTPException(503, "model not loaded")

    started = time.perf_counter()
    # Event time when the producer supplies one, processing time otherwise.
    # Replaying a historical log at 80 txn/sec against wall-clock windows
    # squashes every card's history into the same second, so the features the
    # model sees bear no resemblance to the ones it trained on.
    now = txn.event_time if txn.event_time is not None else time.time()
    store: FeatureStore = state["store"]

    velocity = store.compute(
        card_id=txn.card_id,
        amount=txn.amount,
        merchant=txn.merchant,
        device_id=txn.device_id,
        now=now,
    ).as_dict()

    row = build_row(txn, velocity, now)
    prob = float(state["model"].predict_proba(row)[0][1])
    reasons = state["explainer"].top_reasons(row, k=5)

    # Write the event only after it has been scored.
    store.record(txn.card_id, txn.amount, txn.merchant, txn.device_id, now=now)

    if prob >= BLOCK_THRESHOLD:
        decision = "block"
    elif prob >= REVIEW_THRESHOLD:
        decision = "review"
    else:
        decision = "approve"

    latency_ms = (time.perf_counter() - started) * 1000
    LATENCIES.append(latency_ms)

    COUNTERS["scored"] += 1
    if decision != "approve":
        COUNTERS[decision] += 1
    if txn.label is not None:
        if txn.label == 1 and decision != "approve":
            COUNTERS["caught"] += 1
        elif txn.label == 1:
            COUNTERS["missed"] += 1

    resp = ScoreResponse(
        txn_id=txn.txn_id,
        card_id=txn.card_id,
        amount=txn.amount,
        fraud_probability=round(prob, 4),
        decision=decision,
        reasons=reasons,
        latency_ms=round(latency_ms, 2),
    )

    if decision != "approve":
        ALERTS.appendleft(
            {
                **resp.model_dump(),
                "merchant": txn.merchant,
                "ts": dt_iso(now),
                "actual_label": txn.label,
            }
        )
    return resp


@app.get("/alerts")
def alerts(limit: int = 50):
    """Live alert feed for the dashboard."""
    return {"alerts": list(ALERTS)[:limit]}


@app.get("/stats")
def stats():
    n = COUNTERS["scored"] or 1
    flagged = COUNTERS["review"] + COUNTERS["block"]
    frauds = COUNTERS["caught"] + COUNTERS["missed"]
    lat = sorted(LATENCIES)
    return {
        **COUNTERS,
        "alert_rate": round(flagged / n, 4),
        # Only meaningful when the simulator replays labelled data.
        "recall": round(COUNTERS["caught"] / frauds, 4) if frauds else None,
        "p50_latency_ms": round(lat[len(lat) // 2], 2) if lat else None,
        "p95_latency_ms": round(lat[int(len(lat) * 0.95)], 2) if lat else None,
    }


def dt_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
