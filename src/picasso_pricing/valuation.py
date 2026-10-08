"""Turn the fitted models into per-work fair values and compare with ask prices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import BASE_PERIOD, parse_inches_area_cm2
from .models import FittedModels

INTERVAL = 0.80  # width of the prediction interval reported for a single sale


def _baseline_row(work: pd.Series, comps: pd.DataFrame, tier: str = "A_major") -> pd.DataFrame:
    """Valuation scenario: today, at the given house tier, regular impression, typical size."""
    if len(comps):
        log_area = float(comps["log_area"].median())
    else:
        log_area = float(np.log(parse_inches_area_cm2(work["image_in"])))
    return pd.DataFrame(
        [
            {
                "bloch": work["bloch"],
                "period": BASE_PERIOD,
                "tier": tier,
                "special": 0,
                "log_area_dev": 0.0,
                "log_area": log_area,
                "medium_group": work["medium_group"],
                "era": work["era"],
                "series_group": work["series_group"],
            }
        ]
    )


def verdict(ratio: float) -> str:
    if np.isnan(ratio):
        return "characteristics-only estimate"
    if ratio <= 1.0:
        return "at or below fair value"
    if ratio <= 1.3:
        return "close to fair value"
    if ratio <= 2.0:
        return "typical dealer premium"
    if ratio <= 3.0:
        return "high"
    return "very high"


def value_works(
    models: FittedModels, df: pd.DataFrame, inventory: pd.DataFrame
) -> pd.DataFrame:
    """One row per print: fair value, interval, ask/fair ratio and P(auction >= ask)."""
    alpha = 1 - INTERVAL
    z = norm.ppf(1 - alpha / 2)
    rows = []
    for _, work in inventory[inventory["medium_group"] != "Drawing"].iterrows():
        comps = df[df["bloch"] == work["bloch"]]
        x = _baseline_row(work, comps)
        pred_a = models.a.get_prediction(x).summary_frame(alpha=alpha).iloc[0]
        rec = {
            "bloch": work["bloch"],
            "title": work["title"],
            "medium_group": work["medium_group"],
            "ask_usd": work["ask_usd"],
            "n_sales": len(comps),
            "fair_value": np.nan,
            "fair_low": np.nan,
            "fair_high": np.nan,
            "ask_to_fair": np.nan,
            "p_auction_ge_ask": np.nan,
            "fair_value_specialist_house": np.nan,
            "characteristics_value": np.exp(pred_a["mean"]),
            "identity_premium": np.nan,
        }
        if len(comps):
            pred_b = models.b.get_prediction(x).summary_frame(alpha=alpha).iloc[0]
            mean = pred_b["mean"]
            sd = (pred_b["obs_ci_upper"] - mean) / z
            spec = _baseline_row(work, comps, tier="B_specialist")
            rec.update(
                fair_value=np.exp(mean),
                fair_low=np.exp(pred_b["obs_ci_lower"]),
                fair_high=np.exp(pred_b["obs_ci_upper"]),
                ask_to_fair=work["ask_usd"] / np.exp(mean),
                p_auction_ge_ask=1 - norm.cdf((np.log(work["ask_usd"]) - mean) / sd),
                fair_value_specialist_house=np.exp(models.b.predict(spec).iloc[0]),
                identity_premium=np.exp(mean - pred_a["mean"]),
            )
        rec["verdict"] = verdict(rec["ask_to_fair"])
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("ask_to_fair", ascending=False, na_position="last")
