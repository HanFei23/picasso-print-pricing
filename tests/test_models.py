"""Check that the regressions recover known effects from synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picasso_pricing.models import cross_validate, FORMULA_B, fit_models  # noqa: E402

PERIODS = ["2010-14", "2015-19", "2023-26"]


def synthetic_sales(n_per_work: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    works = {
        "W1": ("Etching", "1930s", "none", 10.0),
        "W2": ("Etching", "1930s", "Vollard", 9.0),
        "W3": ("Linocut", "1960s", "none", 11.0),
        "W4": ("Lithograph", "1960s", "none", 8.5),
    }
    rows = []
    for bloch, (medium, era, series, base) in works.items():
        for _ in range(n_per_work):
            tier = rng.choice(["A_major", "B_specialist"])
            rows.append(
                {
                    "bloch": bloch,
                    "medium_group": medium,
                    "era": era,
                    "series_group": series,
                    "period": rng.choice(PERIODS),
                    "tier": tier,
                    "special": int(rng.random() < 0.1),
                    "log_area": np.log(1500) + rng.normal(0, 0.05),
                    "log_area_dev": rng.normal(0, 0.05),
                    # True model: major houses are 0.2 log points dearer.
                    "log_price": base + (0.2 if tier == "A_major" else 0.0) + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


def test_model_b_recovers_house_effect():
    models = fit_models(synthetic_sales())
    coef = models.b.params["C(tier, Treatment('A_major'))[T.B_specialist]"]
    assert coef == pytest.approx(-0.2, abs=0.05)
    assert models.b.rsquared > 0.95


def test_cross_validation_error_is_small_on_clean_data():
    df = synthetic_sales()
    err = cross_validate(df, FORMULA_B)
    assert 0 < err < 0.15
