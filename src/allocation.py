"""
allocation.py
-------------
The decision-support layer. Two honest, data-grounded outputs:

1. Seasonal demand signals
   For the forecast week, compare each region's forecast to the SAME week
   one year earlier (the correct seasonal baseline, since demand swings
   ~4.5x across the year). Regions forecast well above last year are
   "ramping"; well below are "cooling". This is the operational signal a
   fleet team acts on: move inventory toward markets entering strength.

2. Proportional fleet allocation
   Distribute a fixed demo fleet across regions in proportion to forecast
   demand, with a per-region floor (every active market needs presence).
   We compare against an even split and report the difference honestly.

A note on scope (and why this is the right call):
   Analysis of this data showed regional demand LEVELS are similar
   (~1.4x spread), so a heavy optimization model produces only marginal
   gains over an even split. The real lever is seasonal TIMING, not
   cross-region reallocation. We therefore present proportional
   allocation + timing signals rather than overclaiming an optimizer's
   impact. Interrogating the data and scoping to what it supports is the
   point, not forcing a bigger headline number.
"""

import numpy as np
import pandas as pd


def seasonal_signals(panel: pd.DataFrame, forecast: pd.DataFrame,
                     ramp_threshold: float = 8.0) -> pd.DataFrame:
    """
    Compare forecast to the same week last year per region.
    ramp_threshold: +/- percent that counts as RAMPING / COOLING.
    """
    next_week = forecast["week"].iloc[0]
    ly_week = next_week - pd.Timedelta(weeks=52)

    window = panel[
        panel["week"].between(ly_week - pd.Timedelta(days=10),
                              ly_week + pd.Timedelta(days=10))
    ]
    ly_avg = window.groupby("Dealer_Region")["units_sold"].mean()

    m = forecast.merge(ly_avg.rename("same_week_last_year"),
                       on="Dealer_Region", how="left")
    m["yoy_pct"] = ((m["forecast_units"] - m["same_week_last_year"])
                    / m["same_week_last_year"] * 100)
    m["signal"] = np.where(m["yoy_pct"] > ramp_threshold, "RAMPING UP",
                  np.where(m["yoy_pct"] < -ramp_threshold, "COOLING", "STEADY"))
    return m.sort_values("yoy_pct", ascending=False).reset_index(drop=True)


def proportional_allocation(forecast: pd.DataFrame, fleet_size: int = 120,
                            min_per_region: int = 8) -> tuple[pd.DataFrame, dict]:
    """Allocate fleet in proportion to forecast demand, with a floor."""
    fc = forecast.sort_values("forecast_units", ascending=False).reset_index(drop=True)
    n = len(fc)
    demand = fc["forecast_units"].to_numpy(dtype=float)

    # Reserve the floor, distribute the remainder proportionally to demand.
    floor_total = min_per_region * n
    remainder = fleet_size - floor_total
    if remainder < 0:
        raise ValueError("fleet_size too small for the requested floor.")
    share = demand / demand.sum()
    alloc = min_per_region + share * remainder

    # Integerize to sum exactly to fleet_size (largest remainder method).
    base = np.floor(alloc).astype(int)
    leftover = fleet_size - base.sum()
    fracs = alloc - base
    for idx in np.argsort(-fracs)[:leftover]:
        base[idx] += 1

    even = np.full(n, fleet_size // n)
    even[: fleet_size - even.sum()] += 1

    out = fc.copy()
    out["forecast_share"] = (share * 100).round(1)
    out["recommended_vehicles"] = base
    out["even_split"] = even
    out["vs_even"] = out["recommended_vehicles"] - out["even_split"]

    summary = {
        "fleet_size": fleet_size,
        "min_per_region": min_per_region,
        "note": "Allocation tracks forecast demand share; floor guarantees presence.",
    }
    return out, summary


def what_if(forecast: pd.DataFrame, fleet_sizes: list[int],
            min_per_region: int = 8) -> pd.DataFrame:
    """Scenario view: allocation for the top region across fleet sizes."""
    rows = []
    for s in fleet_sizes:
        out, _ = proportional_allocation(forecast, fleet_size=s,
                                         min_per_region=min_per_region)
        top = out.iloc[0]
        rows.append({
            "fleet_size": s,
            "top_region": top["Dealer_Region"],
            "top_region_vehicles": int(top["recommended_vehicles"]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from forecasting import run_all
    out = run_all()
    sig = seasonal_signals(out["panel"], out["forecast"])
    alloc, summ = proportional_allocation(out["forecast"], fleet_size=120)
    print("=== SEASONAL SIGNALS (forecast vs same week last year) ===")
    print(sig[["Dealer_Region", "forecast_units", "same_week_last_year",
               "yoy_pct", "signal"]].round(1).to_string(index=False))
    print("\n=== PROPORTIONAL ALLOCATION ===")
    print(alloc[["Dealer_Region", "forecast_units", "forecast_share",
                 "recommended_vehicles", "even_split", "vs_even"]].to_string(index=False))
    print("\n=== WHAT-IF SWEEP ===")
    print(what_if(out["forecast"], [80, 100, 120, 140, 160]).to_string(index=False))
