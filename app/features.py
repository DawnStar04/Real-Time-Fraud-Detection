"""
Real-time feature store backed by Redis.

This is the part of the project that is actually hard, and the part most
portfolio fraud projects skip. A model trained offline sees a static row.
A fraud service sees a transaction arriving *now* and has to answer
questions about the recent past in single-digit milliseconds:

    - how many transactions has this card made in the last 60 seconds?
    - how much has it spent in the last 5 minutes?
    - is this amount unusual compared to the card's rolling average?
    - have we ever seen this device / merchant for this card before?

We answer these with Redis sorted sets keyed by card, scored by timestamp.
A sliding window is then just ZRANGEBYSCORE + ZREMRANGEBYSCORE, which is
O(log N) and stays flat as history grows.

IMPORTANT (leakage): every feature here is computed from events strictly
BEFORE the current transaction. We read the window first, then write the
current event. Doing it the other way round inflates your offline metrics
and the model collapses in production.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any

import redis


# Sliding windows we maintain, in seconds.
WINDOWS = (60, 300, 3600, 86400)

# How long to retain per-card event history. Anything older than the widest
# window is dead weight, plus a small buffer.
HISTORY_TTL = 86400 * 2


@dataclass
class VelocityFeatures:
    """Features derived from the card's own recent history."""

    txn_count_60s: int
    txn_count_300s: int
    txn_count_3600s: int
    txn_count_86400s: int

    amount_sum_60s: float
    amount_sum_300s: float
    amount_sum_3600s: float

    # Ratio of this amount to the card's rolling mean. 1.0 means "typical".
    # Capped so a brand-new card with a tiny history can't produce inf.
    amount_vs_card_mean: float

    # Seconds since this card's previous transaction. Large sentinel when
    # this is the first transaction we have ever seen for the card.
    seconds_since_last_txn: float

    # First-time flags. These are strong signals in practice: fraud rings
    # burn through new device/merchant combinations fast.
    is_new_merchant_for_card: int
    is_new_device_for_card: int

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


class FeatureStore:
    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self.r = redis.from_url(url, decode_responses=True)

    # ---------- key layout ----------

    @staticmethod
    def _txn_key(card_id: str) -> str:
        """Sorted set: member = "<ts>:<amount>", score = unix ts."""
        return f"card:{card_id}:txns"

    @staticmethod
    def _stats_key(card_id: str) -> str:
        """Hash holding running count + sum, for a cheap rolling mean."""
        return f"card:{card_id}:stats"

    @staticmethod
    def _merchants_key(card_id: str) -> str:
        return f"card:{card_id}:merchants"

    @staticmethod
    def _devices_key(card_id: str) -> str:
        return f"card:{card_id}:devices"

    # ---------- read path ----------

    def compute(
        self,
        card_id: str,
        amount: float,
        merchant: str,
        device_id: str,
        now: float | None = None,
    ) -> VelocityFeatures:
        """
        Compute velocity features for a transaction arriving at `now`.

        Read-only with respect to the card's history: call `record()` after
        scoring to add this transaction. Keeping the two apart is what
        prevents the current event leaking into its own features.
        """
        now = now if now is not None else time.time()
        key = self._txn_key(card_id)

        pipe = self.r.pipeline()

        # Drop anything that has fallen out of the widest window. Doing this
        # on the read path keeps the sorted sets from growing unbounded
        # without needing a separate janitor process.
        pipe.zremrangebyscore(key, 0, now - max(WINDOWS))

        for w in WINDOWS:
            pipe.zcount(key, now - w, now)

        # Pull raw members for the amount-sum windows. We only need the
        # narrower windows here; summing a day of history per request would
        # defeat the point.
        for w in (60, 300, 3600):
            pipe.zrangebyscore(key, now - w, now)

        pipe.hgetall(self._stats_key(card_id))
        pipe.zrevrange(key, 0, 0, withscores=True)  # most recent txn
        pipe.sismember(self._merchants_key(card_id), merchant)
        pipe.sismember(self._devices_key(card_id), device_id)

        res = pipe.execute()

        # Unpack in the order queued above.
        idx = 1  # res[0] is the zremrangebyscore result
        counts = res[idx : idx + len(WINDOWS)]
        idx += len(WINDOWS)
        amount_members = res[idx : idx + 3]
        idx += 3
        stats: dict[str, str] = res[idx] or {}
        idx += 1
        last = res[idx]
        idx += 1
        seen_merchant = bool(res[idx])
        idx += 1
        seen_device = bool(res[idx])

        sums = [self._sum_amounts(members) for members in amount_members]

        # Rolling mean from the running count/sum hash rather than from the
        # window, so a card's "normal" reflects its whole history, not just
        # the last hour.
        n = float(stats.get("count", 0) or 0)
        total = float(stats.get("sum", 0) or 0)
        mean = (total / n) if n > 0 else 0.0
        if mean > 0:
            ratio = min(amount / mean, 100.0)
        else:
            # No history: treat as typical rather than as a 100x outlier,
            # otherwise every card's first transaction looks like fraud.
            ratio = 1.0

        if last:
            _, last_ts = last[0]
            since_last = max(0.0, now - float(last_ts))
        else:
            since_last = 86400.0 * 7  # sentinel: effectively "never"

        return VelocityFeatures(
            txn_count_60s=int(counts[0]),
            txn_count_300s=int(counts[1]),
            txn_count_3600s=int(counts[2]),
            txn_count_86400s=int(counts[3]),
            amount_sum_60s=sums[0],
            amount_sum_300s=sums[1],
            amount_sum_3600s=sums[2],
            amount_vs_card_mean=ratio,
            seconds_since_last_txn=since_last,
            is_new_merchant_for_card=0 if seen_merchant else 1,
            is_new_device_for_card=0 if seen_device else 1,
        )

    @staticmethod
    def _sum_amounts(members: list[str]) -> float:
        total = 0.0
        for m in members:
            try:
                total += float(m.split(":", 1)[1])
            except (IndexError, ValueError):
                continue
        return total

    # ---------- write path ----------

    def record(
        self,
        card_id: str,
        amount: float,
        merchant: str,
        device_id: str,
        now: float | None = None,
    ) -> None:
        """Append this transaction to the card's history. Call AFTER scoring."""
        now = now if now is not None else time.time()
        key = self._txn_key(card_id)

        pipe = self.r.pipeline()
        # Member must be unique or the sorted set silently dedupes two
        # identical amounts landing in the same millisecond.
        pipe.zadd(key, {f"{now:.6f}:{amount}": now})
        pipe.expire(key, HISTORY_TTL)

        pipe.hincrby(self._stats_key(card_id), "count", 1)
        pipe.hincrbyfloat(self._stats_key(card_id), "sum", amount)
        pipe.expire(self._stats_key(card_id), HISTORY_TTL)

        pipe.sadd(self._merchants_key(card_id), merchant)
        pipe.expire(self._merchants_key(card_id), HISTORY_TTL)

        pipe.sadd(self._devices_key(card_id), device_id)
        pipe.expire(self._devices_key(card_id), HISTORY_TTL)

        pipe.execute()

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except redis.RedisError:
            return False
