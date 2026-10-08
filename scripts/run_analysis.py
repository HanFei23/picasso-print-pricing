"""Run the full analysis: fit both models, value every print, save outputs.

Usage:
    python scripts/run_analysis.py [--data data] [--out outputs]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picasso_pricing import (  # noqa: E402
    build_model_frame,
    cross_validate,
    effect_table,
    fit_models,
    load_comps,
    load_inventory,
    value_works,
)
from picasso_pricing.models import FORMULA_A, FORMULA_B  # noqa: E402


def plot_ratios(valuation: pd.DataFrame, path: Path) -> None:
    v = valuation.dropna(subset=["ask_to_fair"]).sort_values("ask_to_fair")
    colors = ["#2e7d32" if r <= 1.3 else "#c62828" if r > 2 else "#9e9e9e" for r in v["ask_to_fair"]]
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(v["bloch"], v["ask_to_fair"], color=colors)
    ax.axvline(1, color="black", lw=0.8)
    ax.axvline(2, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Dealer ask / model fair value (x)")
    ax.set_title("John Szoke Booth 412: ask price vs auction fair value")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--out", default="outputs", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    inventory = load_inventory(args.data / "booth_inventory.csv")
    comps = load_comps(args.data / "auction_comps.csv")
    df = build_model_frame(comps, inventory)
    models = fit_models(df)

    fit = pd.DataFrame(
        {
            "model_B_work_fixed_effects": [
                int(models.b.nobs), models.b.rsquared, models.b.rsquared_adj,
                cross_validate(df, FORMULA_B),
            ],
            "model_A_characteristics": [
                int(models.a.nobs), models.a.rsquared, models.a.rsquared_adj,
                cross_validate(df, FORMULA_A),
            ],
        },
        index=["n_sales", "r_squared", "adj_r_squared", "cv_median_abs_error"],
    )
    shared = effect_table(models.b, r"period|tier|special|area")
    traits = effect_table(models.a, r"medium_group|era|series_group|log_area")
    valuation = value_works(models, df, inventory)

    priced = valuation.dropna(subset=["fair_value"])
    total_ratio = priced["ask_usd"].sum() / priced["fair_value"].sum()

    fit.to_csv(args.out / "model_fit.csv")
    shared.to_csv(args.out / "shared_effects_model_B.csv")
    traits.to_csv(args.out / "characteristic_effects_model_A.csv")
    valuation.to_csv(args.out / "valuation.csv", index=False)
    with pd.ExcelWriter(args.out / "report.xlsx") as xl:
        valuation.to_excel(xl, sheet_name="valuation", index=False)
        fit.to_excel(xl, sheet_name="model_fit")
        shared.to_excel(xl, sheet_name="shared_effects")
        traits.to_excel(xl, sheet_name="characteristic_effects")
        df.to_excel(xl, sheet_name="model_data", index=False)
    plot_ratios(valuation, args.out / "ask_vs_fair_value.png")

    pd.set_option("display.width", 160)
    print(f"Sales used: {len(df)} across {df['bloch'].nunique()} works\n")
    print(fit.round(3).to_string(), "\n")
    print(f"Booth total: ask ${priced['ask_usd'].sum():,.0f} vs fair ${priced['fair_value'].sum():,.0f}"
          f" -> {total_ratio:.2f}x\n")
    cols = ["bloch", "ask_usd", "n_sales", "fair_value", "fair_low", "fair_high",
            "ask_to_fair", "p_auction_ge_ask", "verdict"]
    print(valuation[cols].round(2).to_string(index=False))
    print(f"\nOutputs written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
