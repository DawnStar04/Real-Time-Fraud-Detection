"""
SHAP explanations, turned into sentences a fraud analyst can read.

Two things worth knowing:

1. For tree models, `shap.TreeExplainer` is exact and fast — microseconds
   per row, not the sampling-based approximation KernelExplainer uses. That
   is the main reason this project uses LightGBM rather than a neural net:
   per-transaction explanation stays inside the latency budget.

2. Raw SHAP values are in log-odds space, not probability. We keep the raw
   value for the waterfall chart (where it must sum correctly) but never
   present it as "this feature added 12% risk", which would be wrong.
"""

from __future__ import annotations

import shap

# Feature name -> template for the analyst-facing sentence. {v} is the
# feature's value for this transaction.
PHRASING = {
    "amount": "Transaction amount is {v:,.0f}",
    "amount_vs_card_mean": "Amount is {v:.1f}x this card's average spend",
    "txn_count_60s": "{v:.0f} transactions from this card in the last 60 seconds",
    "txn_count_300s": "{v:.0f} transactions from this card in the last 5 minutes",
    "txn_count_3600s": "{v:.0f} transactions from this card in the last hour",
    "txn_count_86400s": "{v:.0f} transactions from this card in the last 24 hours",
    "amount_sum_60s": "{v:,.0f} spent by this card in the last 60 seconds",
    "amount_sum_300s": "{v:,.0f} spent by this card in the last 5 minutes",
    "amount_sum_3600s": "{v:,.0f} spent by this card in the last hour",
    "seconds_since_last_txn": "{v:,.0f} seconds since this card's previous transaction",
    "is_new_merchant_for_card": "Merchant never seen before for this card",
    "is_new_device_for_card": "Device never seen before for this card",
    "hour_of_day": "Transaction at hour {v:.0f}",
    "is_night": "Transaction during overnight hours",
}

# Flags only make sense as a sentence when they are set.
FLAGS = {"is_new_merchant_for_card", "is_new_device_for_card", "is_night"}


class Explainer:
    def __init__(self, model, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def top_reasons(self, row, k: int = 5) -> list[dict]:
        """
        Return the k features pushing this transaction most towards fraud.

        We deliberately return only positive contributions. An analyst
        reviewing an alert wants to know why it fired; the features arguing
        for innocence are noise in that workflow (they belong in the full
        waterfall view, which the UI renders from the same values).
        """
        values = self.explainer.shap_values(row)

        # LightGBM binary returns either (n, f) or a list of two arrays
        # depending on version. Normalise to a single row of per-feature
        # contributions for the positive class.
        if isinstance(values, list):
            values = values[1]
        contribs = values[0] if getattr(values, "ndim", 1) > 1 else values

        ranked = sorted(
            zip(self.feature_names, row.iloc[0].tolist(), contribs),
            key=lambda t: t[2],
            reverse=True,
        )

        out = []
        for name, value, contrib in ranked:
            if contrib <= 0:
                break
            if name in FLAGS and not value:
                continue  # "device NOT new" is not a reason to flag
            out.append(
                {
                    "feature": name,
                    "value": float(value),
                    "contribution": round(float(contrib), 4),
                    "text": self._phrase(name, float(value)),
                }
            )
            if len(out) == k:
                break
        return out

    @staticmethod
    def _phrase(name: str, value: float) -> str:
        template = PHRASING.get(name)
        if template is None:
            return f"{name} = {value:g}"
        return template.format(v=value)
