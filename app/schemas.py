"""Request/response contracts for the scoring service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    txn_id: str
    card_id: str
    amount: float = Field(gt=0)
    merchant: str
    merchant_category: str = "unknown"
    device_id: str = "unknown"
    country: str = "IN"
    # Unix timestamp of when the transaction actually happened. Optional; the
    # service falls back to wall-clock time. This is the event-time vs
    # processing-time distinction, and it matters for more than replay: if
    # events arrive out of order or delayed, windows computed from "now" are
    # wrong. Trust the event's own clock when it has one.
    event_time: float | None = None
    # Optional: the simulator replays historical data, so it can pass the
    # original label through for live metric tracking on the dashboard.
    label: int | None = None


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float
    # Human-readable version of the same thing, for the UI.
    text: str


class ScoreResponse(BaseModel):
    txn_id: str
    card_id: str
    amount: float
    fraud_probability: float
    decision: str  # "approve" | "review" | "block"
    reasons: list[Reason]
    latency_ms: float
