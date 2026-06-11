"""
data_pipeline.py
----------------
Turns raw dealership sales records into a clean weekly panel at the
region level. This is the foundation every other layer builds on.

What is REAL in the source data (Kaggle "Car Sales Report", missionjee):
    - Date of each sale (daily, Jan 2022 - Dec 2023)
    - Dealer_Region (7 real regions)
    - Price ($) per sale
    - Company / Model / Body Style
    - Customer Annual Income, Gender

What we DERIVE (and why it is honest):
    - We aggregate real sales into weekly counts and revenue per region.
      No values are invented; we are only grouping real transactions.
    - "demand" here means realized sales volume. We forecast demand and
      allocate a demo fleet toward it. We do NOT fabricate a demo-drive
      or conversion number, because the raw data does not contain one.

Output: a tidy panel with one row per (region, week).
"""

import pandas as pd
import numpy as np

import os

# Resolve the dataset whether running from the repo (data/car_sales_raw.csv)
# or in the original build sandbox. Repo-relative path wins so the app works
# on Streamlit Cloud and on a fresh clone.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DATA = os.path.join(_HERE, "..", "data", "car_sales_raw.csv")
_FALLBACK = "/mnt/user-data/uploads/Car_Sales_xlsx_-_car_data.csv"
RAW_PATH = _REPO_DATA if os.path.exists(_REPO_DATA) else _FALLBACK


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load and clean the raw sales file."""
    df = pd.read_csv(path)
    # Column names in the file carry stray whitespace ("Dealer_No ").
    df.columns = [c.strip() for c in df.columns]
    # Dates are m/d/Y strings.
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    # Company has a stray encoding in a couple rows; strip whitespace.
    df["Company"] = df["Company"].astype(str).str.strip()
    df["Body Style"] = df["Body Style"].astype(str).str.strip()
    df["Dealer_Region"] = df["Dealer_Region"].astype(str).str.strip()
    return df


def build_weekly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate real sales into a weekly region-level panel.

    Weekly (not daily) because daily sales are sparse and noisy at the
    region level; weekly is the natural planning cadence for fleet moves
    and gives the model a cleaner seasonal signal.
    """
    df = df.copy()
    # Period 'W' -> start-of-week timestamp, a stable weekly index.
    df["week"] = df["Date"].dt.to_period("W").dt.start_time

    panel = (
        df.groupby(["Dealer_Region", "week"])
        .agg(
            units_sold=("Car_id", "count"),
            revenue=("Price ($)", "sum"),
            avg_price=("Price ($)", "mean"),
            avg_income=("Annual Income", "mean"),
        )
        .reset_index()
    )

    # Make the panel rectangular: every region must have every week, so
    # the time series has no holes for the model or the decomposition.
    all_weeks = panel["week"].sort_values().unique()
    all_regions = panel["Dealer_Region"].unique()
    full_index = pd.MultiIndex.from_product(
        [all_regions, all_weeks], names=["Dealer_Region", "week"]
    )
    panel = (
        panel.set_index(["Dealer_Region", "week"])
        .reindex(full_index)
        .reset_index()
    )
    # Weeks with no sales in a region -> 0 units / 0 revenue (a true zero,
    # not missing). Price/income left as NaN then filled forward per region.
    panel["units_sold"] = panel["units_sold"].fillna(0).astype(int)
    panel["revenue"] = panel["revenue"].fillna(0.0)
    panel = panel.sort_values(["Dealer_Region", "week"])
    panel["avg_price"] = panel.groupby("Dealer_Region")["avg_price"].ffill().bfill()
    panel["avg_income"] = panel.groupby("Dealer_Region")["avg_income"].ffill().bfill()

    # Trim the first and last partial weeks (period 'W' can clip the edges),
    # so every region starts and ends on a fully-populated week.
    wk_counts = panel.groupby("week")["units_sold"].sum()
    good_weeks = wk_counts[wk_counts > 0].index
    panel = panel[panel["week"].isin(good_weeks)].reset_index(drop=True)

    return panel


def get_panel() -> pd.DataFrame:
    """Convenience: raw -> clean weekly panel in one call."""
    return build_weekly_panel(load_raw())


if __name__ == "__main__":
    p = get_panel()
    print("Panel shape:", p.shape)
    print("Regions:", sorted(p["Dealer_Region"].unique()))
    print("Weeks:", p["week"].nunique(),
          "(", p["week"].min().date(), "->", p["week"].max().date(), ")")
    print(p.head(10).to_string(index=False))
