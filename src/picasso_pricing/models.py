"""Hedonic regressions for Picasso print auction prices.

Model B (main): work fixed effects + shared controls. Answers "what would this
exact print fetch today, at a major house, as a regular numbered impression?"

Model A (characteristics only): the model does not know which work a lot is.
Used to value works with no auction history and to measure how much of a
work's price comes from its identity rather than its characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .features import BASE_PERIOD

CONTROLS = (
    f"C(period, Treatment('{BASE_PERIOD}'))"
    " + C(tier, Treatment('A_major'))"
    " + special"
)
FORMULA_B = f"log_price ~ C(bloch) + log_area_dev + {CONTROLS}"
FORMULA_A = (
    "log_price ~ C(medium_group) + C(era) + C(series_group, Treatment('none'))"
    f" + log_area + {CONTROLS}"
)


@dataclass
class FittedModels:
    a: RegressionResultsWrapper
    b: RegressionResultsWrapper


def fit_models(df: pd.DataFrame) -> FittedModels:
    """OLS with heteroskedasticity-robust (HC1) standard errors."""
    a = smf.ols(FORMULA_A, data=df).fit(cov_type="HC1")
    b = smf.ols(FORMULA_B, data=df).fit(cov_type="HC1")
    return FittedModels(a=a, b=b)


def cross_validate(
    df: pd.DataFrame, formula: str, k: int = 5, seed: int = 7
) -> float:
    """K-fold CV. Returns the median absolute error as a % of price.

    Test rows whose work never appears in the training fold are skipped,
    because a fixed-effects model cannot predict an unseen work.
    """
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(df)), k)
    errors: list[float] = []
    for test_idx in folds:
        train = df.drop(df.index[test_idx])
        test = df.iloc[test_idx]
        test = test[test["bloch"].isin(train["bloch"].unique())]
        model = smf.ols(formula, data=train).fit()
        errors.extend(np.abs(model.predict(test) - test["log_price"]))
    return float(np.exp(np.median(errors)) - 1)


def effect_table(model: RegressionResultsWrapper, pattern: str) -> pd.DataFrame:
    """Coefficients matching `pattern`, with the implied % price effect."""
    params = model.params.filter(regex=pattern)
    return pd.DataFrame(
        {
            "coef": params,
            "std_err": model.bse[params.index],
            "p_value": model.pvalues[params.index],
            "price_effect": np.exp(params) - 1,
        }
    )
