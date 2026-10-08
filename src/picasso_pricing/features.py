"""Load the raw CSVs and turn auction records into model-ready features."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Sale-date buckets. The most recent bucket is the valuation baseline.
PERIOD_BINS = [0, 1994, 1999, 2004, 2009, 2014, 2019, 2022, 2100]
PERIOD_LABELS = ["<1995", "1995-99", "2000-04", "2005-09", "2010-14", "2015-19", "2020-22", "2023-26"]
BASE_PERIOD = PERIOD_LABELS[-1]

# Creation-date buckets (year the print was made).
ERA_BINS = [0, 1929, 1939, 1959, 1969, 2100]
ERA_LABELS = ["1920s", "1930s", "1940-50s", "1960s", "1970s"]

MAJOR_HOUSES = re.compile(r"Sotheby|Christie|Phillips", re.I)
SPECIALIST_HOUSES = re.compile(
    r"Bonhams|Kornfeld|Grisebach|Lempertz|Ketterer|Artcurial|Swann|Bassenge|"
    r"Hauswedell|Koller|Dorotheum|Bukowskis|Heffel",
    re.I,
)
# Proofs and other impressions outside the numbered edition.
SPECIAL_EDITION = re.compile(
    r"E\.?A|épreuve|proof|BAT|H\.?C|TP|aside|artist|unrecorded|rinc", re.I
)
MAX_PLAUSIBLE_CM = 150  # larger values are mm or framed sizes recorded by mistake


def load_inventory(path: str | Path) -> pd.DataFrame:
    """Gallery booth price list, one row per work."""
    inv = pd.read_csv(path)
    inv["medium_group"] = inv["medium_group"].str.split(" (", regex=False).str[0]
    inv["series_group"] = inv["series"].fillna("").map(series_group)
    inv["era"] = pd.cut(inv["year"], ERA_BINS, labels=ERA_LABELS).astype(str)
    return inv


def load_comps(path: str | Path) -> pd.DataFrame:
    """Artnet auction records already matched to a booth work (one row per lot)."""
    comps = pd.read_csv(path, parse_dates=["sale_date"])
    comps["edition"] = comps["edition"].fillna("")
    comps["note"] = comps["note"].fillna("")
    return comps


def series_group(series: str) -> str:
    if "Vollard" in series:
        return "Vollard"
    if "347" in series:
        return "347"
    if "156" in series:
        return "156"
    return "none"


def house_tier(house: str) -> str:
    """Bucket auction houses: major (live), major (online), specialist, other."""
    if MAJOR_HOUSES.search(house):
        return "online_major" if "Online" in house else "A_major"
    if SPECIALIST_HOUSES.search(house):
        return "B_specialist"
    return "C_other"


def parse_area_cm2(size: str) -> float:
    """'49.5 x 69.2' -> 3425.4. Returns NaN when missing or implausible."""
    nums = re.findall(r"[\d.]+", size or "")
    if len(nums) < 2:
        return np.nan
    a, b = float(nums[0]), float(nums[1])
    if a > MAX_PLAUSIBLE_CM or b > MAX_PLAUSIBLE_CM:
        return np.nan
    return a * b


def parse_inches_area_cm2(size: str) -> float:
    """'15 3/8 x 10' (inches, optional fractions) -> area in cm²."""
    head = (size or "").split("(")[0]
    parts = re.findall(r"(\d+)(?: (\d+)/(\d+))?", head)[:2]
    if len(parts) < 2:
        return np.nan
    dims = [int(w) + (int(n) / int(d) if n else 0) for w, n, d in parts]
    return dims[0] * dims[1] * 2.54**2


def build_model_frame(comps: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Keep sold, non-excluded lots and add the regression features."""
    df = comps[
        (comps["status"] == "Sold") & comps["price_usd"].notna() & (comps["exclude"] == 0)
    ].copy()
    work = inventory.set_index("bloch")

    df["area"] = df["size_cm"].fillna("").map(parse_area_cm2)
    df["area"] = df.groupby("bloch")["area"].transform(lambda s: s.fillna(s.median()))
    df["log_area"] = np.log(df["area"])
    # Size relative to the typical record for the same work (captures sheet vs image noise).
    df["log_area_dev"] = df["log_area"] - df.groupby("bloch")["log_area"].transform("median")

    df["special"] = df["edition"].str.contains(SPECIAL_EDITION).astype(int)
    df["tier"] = df["auction_house"].map(house_tier)
    df["period"] = pd.cut(
        df["sale_date"].dt.year, PERIOD_BINS, labels=PERIOD_LABELS
    ).astype(str)

    df["medium_group"] = df["bloch"].map(work["medium_group"])
    df["era"] = df["bloch"].map(work["era"])
    df["series_group"] = df["bloch"].map(work["series_group"])
    df["log_price"] = np.log(df["price_usd"])
    return df.reset_index(drop=True)
