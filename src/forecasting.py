"""
forecasting.py
--------------
Forecasts next-week sales demand for each region, then exposes the
per-region forecast that the optimizer consumes.

Design choices an interviewer will ask about, and the answers:

1. Why lag features?
   Demand is autocorrelated: last week and the recent average are the
   strongest honest predictors of next week. We give the model lag_1,
   lag_2, lag_4 and a rolling 4-week mean.

2. Why encode week-of-year with sin/cos?
   The data has a real yearly cycle (Sep/Nov/Dec peaks repeat in both
   2022 and 2023). Raw week number (1..52) would tell the model that
   week 52 and week 1 are far apart when they are actually adjacent.
   sin/cos of the week angle makes the calendar wrap around correctly.

3. Why a TIME-BASED split, not random?
   This is a forecast. If we shuffled rows, the model could "peek" at
   future weeks to predict past ones (leakage) and report a fake-good
   score. We train on the earliest ~80% of weeks and test on the most
   recent ~20%, which is how the model would really be used.

4. Why XGBoost?
   Tabular, non-linear interactions (region x season x recent trend),
   robust with modest data. We report a holdout MAE / MAPE so the
   accuracy claim is honest, and we compare against a naive
   "next week = last week" baseline so the model has to earn its keep.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_pipeline import get_panel


FEATURES = [
    "lag_1", "lag_2", "lag_4", "roll4_mean",
    "woy_sin", "woy_cos", "month",
    "avg_price", "avg_income", "region_code",
]
TARGET = "units_sold"


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and seasonal features per region."""
    df = panel.sort_values(["Dealer_Region", "week"]).copy()
    g = df.groupby("Dealer_Region")["units_sold"]

    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_4"] = g.shift(4)
    # rolling mean of the 4 weeks BEFORE the current one (shift first to
    # avoid including the current week = no leakage).
    df["roll4_mean"] = g.shift(1).rolling(4).mean().reset_index(level=0, drop=True)

    woy = df["week"].dt.isocalendar().week.astype(int)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    df["month"] = df["week"].dt.month

    df["region_code"] = df["Dealer_Region"].astype("category").cat.codes

    df = df.dropna(subset=["lag_1", "lag_2", "lag_4", "roll4_mean"]).reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split by calendar time: earliest weeks train, latest weeks test."""
    weeks = np.sort(df["week"].unique())
    cut = weeks[int(len(weeks) * (1 - test_frac))]
    train = df[df["week"] < cut]
    test = df[df["week"] >= cut]
    return train, test, cut


def train_and_evaluate(df: pd.DataFrame):
    """Train XGBoost, evaluate on the held-out recent weeks."""
    train, test, cut = time_split(df)

    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=2,
    )
    model.fit(train[FEATURES], train[TARGET])

    pred = model.predict(test[FEATURES])
    pred = np.clip(pred, 0, None)  # demand can't be negative

    mae = mean_absolute_error(test[TARGET], pred)
    rmse = mean_squared_error(test[TARGET], pred) ** 0.5
    # MAPE on weeks with real volume (avoid divide-by-zero on empty weeks).
    mask = test[TARGET] > 0
    mape = np.mean(np.abs((test[TARGET][mask] - pred[mask]) / test[TARGET][mask])) * 100

    # Naive baseline: predict last week's value (lag_1). The model should beat it.
    naive_mae = mean_absolute_error(test[TARGET], test["lag_1"])

    metrics = {
        "test_start": str(pd.Timestamp(cut).date()),
        "n_train": len(train),
        "n_test": len(test),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_pct": round(mape, 1),
        "naive_MAE": round(naive_mae, 2),
        "improvement_vs_naive_pct": round((naive_mae - mae) / naive_mae * 100, 1),
    }
    return model, metrics, test.assign(pred=pred)


def feature_importance(model) -> pd.DataFrame:
    imp = pd.DataFrame(
        {"feature": FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp


def forecast_next_week(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict the week AFTER the last observed week, for every region.
    Uses each region's most recent rows to build the forward feature row.
    """
    rows = []
    last_week = df["week"].max()
    next_week = last_week + pd.Timedelta(weeks=1)
    woy = int(pd.Timestamp(next_week).isocalendar().week)

    for region, grp in df.groupby("Dealer_Region"):
        grp = grp.sort_values("week")
        recent = grp["units_sold"].values
        feat = {
            "lag_1": recent[-1],
            "lag_2": recent[-2],
            "lag_4": recent[-4],
            "roll4_mean": recent[-4:].mean(),
            "woy_sin": np.sin(2 * np.pi * woy / 52.0),
            "woy_cos": np.cos(2 * np.pi * woy / 52.0),
            "month": next_week.month,
            "avg_price": grp["avg_price"].iloc[-1],
            "avg_income": grp["avg_income"].iloc[-1],
            "region_code": grp["region_code"].iloc[-1],
        }
        pred = float(np.clip(model.predict(pd.DataFrame([feat])[FEATURES])[0], 0, None))
        rows.append({"Dealer_Region": region, "week": next_week,
                     "forecast_units": round(pred, 1)})
    return pd.DataFrame(rows).sort_values("forecast_units", ascending=False).reset_index(drop=True)


def run_all():
    panel = get_panel()
    df = make_features(panel)
    model, metrics, test_pred = train_and_evaluate(df)
    imp = feature_importance(model)
    fc = forecast_next_week(model, df)
    return {"panel": panel, "featured": df, "model": model,
            "metrics": metrics, "importance": imp,
            "forecast": fc, "test_pred": test_pred}


if __name__ == "__main__":
    out = run_all()
    print("=== HOLDOUT METRICS ===")
    for k, v in out["metrics"].items():
        print(f"  {k}: {v}")
    print("\n=== FEATURE IMPORTANCE ===")
    print(out["importance"].to_string(index=False))
    print("\n=== NEXT-WEEK FORECAST BY REGION ===")
    print(out["forecast"].to_string(index=False))
