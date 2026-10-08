import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picasso_pricing.features import (  # noqa: E402
    SPECIAL_EDITION,
    build_model_frame,
    house_tier,
    parse_area_cm2,
    parse_inches_area_cm2,
    series_group,
)
from picasso_pricing.valuation import verdict  # noqa: E402


@pytest.mark.parametrize(
    "house, tier",
    [
        ("Sotheby's New York", "A_major"),
        ("Christie's Online", "online_major"),
        ("Phillips de Pury & Company New York", "A_major"),
        ("Galerie Kornfeld Bern", "B_specialist"),
        ("Bonhams New York", "B_specialist"),
        ("Mainichi Auction", "C_other"),
    ],
)
def test_house_tier(house, tier):
    assert house_tier(house) == tier


def test_parse_area_cm2():
    assert parse_area_cm2("49.5 x 69.2") == pytest.approx(49.5 * 69.2)
    assert math.isnan(parse_area_cm2(""))
    assert math.isnan(parse_area_cm2("365 x 270"))  # mm recorded as cm


def test_parse_inches_area_cm2():
    # 15 3/8 x 10 inches
    assert parse_inches_area_cm2("15 3/8 x 10") == pytest.approx(15.375 * 10 * 6.4516)
    assert parse_inches_area_cm2("11 5/8 x 14 3/8 (29.5 x 36.5 cm)") == pytest.approx(
        11.625 * 14.375 * 6.4516
    )


@pytest.mark.parametrize(
    "edition, special",
    [("E.A.", True), ("BAT aside from ed.50", True), ("H.C B/C", True),
     ("34/50", False), ("ed. 50", False), ("", False)],
)
def test_special_edition(edition, special):
    assert bool(SPECIAL_EDITION.search(edition)) is special


def test_series_group():
    assert series_group("Suite Vollard, /50") == "Vollard"
    assert series_group("Suite 347 / book page") == "347"
    assert series_group("Series 156") == "156"
    assert series_group("") == "none"


def test_verdict_bands():
    assert verdict(0.8) == "at or below fair value"
    assert verdict(1.2) == "close to fair value"
    assert verdict(1.8) == "typical dealer premium"
    assert verdict(2.5) == "high"
    assert verdict(4.0) == "very high"
    assert verdict(float("nan")) == "characteristics-only estimate"


def test_build_model_frame_filters_and_features():
    inventory = pd.DataFrame(
        {"bloch": ["B1"], "medium_group": ["Etching"], "era": ["1930s"], "series_group": ["none"]}
    )
    comps = pd.DataFrame(
        {
            "bloch": ["B1"] * 4,
            "size_cm": ["30 x 40", "", "30 x 40", "30 x 40"],
            "edition": ["10/50", "E.A.", "", ""],
            "sale_date": pd.to_datetime(["2024-05-01", "2010-01-01", "1990-01-01", "2024-01-01"]),
            "auction_house": ["Sotheby's London", "Kornfeld", "Other", "Christie's"],
            "status": ["Sold", "Sold", "Bought In", "Sold"],
            "price_usd": [1000.0, 2000.0, np.nan, 500.0],
            "exclude": [0, 0, 0, 1],
        }
    )
    df = build_model_frame(comps, inventory)
    assert len(df) == 2  # bought-in and excluded rows dropped
    assert df["area"].tolist() == [1200.0, 1200.0]  # missing size filled with work median
    assert df["special"].tolist() == [0, 1]
    assert df["period"].tolist() == ["2023-26", "2010-14"]
    assert df["log_price"].iloc[0] == pytest.approx(np.log(1000))
