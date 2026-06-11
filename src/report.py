"""
report.py
---------
Generates a plain-English weekly briefing from the forecast, seasonal
signals, and allocation. This is the "automated report generator" a fleet
manager would receive each week: what's coming, what's changing, and the
recommended move, in language a non-analyst can act on.
"""

import pandas as pd


def generate_briefing(metrics: dict, signals: pd.DataFrame,
                      allocation: pd.DataFrame, fleet_size: int) -> str:
    week = pd.Timestamp(signals["week"].iloc[0]).date()

    ramping = signals[signals["signal"] == "RAMPING UP"]
    cooling = signals[signals["signal"] == "COOLING"]
    total_fc = signals["forecast_units"].sum()

    top = allocation.iloc[0]
    movers = allocation[allocation["vs_even"] != 0].sort_values("vs_even", ascending=False)

    lines = []
    lines.append(f"FLEET DEMAND BRIEFING  |  Week of {week}")
    lines.append("=" * 52)
    lines.append("")
    lines.append(f"Projected demand across all 7 markets: ~{total_fc:.0f} units "
                 f"this week.")
    lines.append(f"Forecast model: XGBoost, holdout MAE {metrics['MAE']} units "
                 f"({metrics['improvement_vs_naive_pct']}% better than a naive "
                 f"last-week guess).")
    lines.append("")

    # Seasonal movement
    if len(ramping):
        names = ", ".join(f"{r.Dealer_Region} (+{r.yoy_pct:.0f}% YoY)"
                          for r in ramping.head(3).itertuples())
        lines.append(f"RAMPING UP vs last year: {names}.")
        lines.append("  -> These markets are entering seasonal strength. "
                     "Prioritize demo availability here.")
    if len(cooling):
        names = ", ".join(f"{r.Dealer_Region} ({r.yoy_pct:.0f}% YoY)"
                          for r in cooling.head(3).itertuples())
        lines.append(f"COOLING vs last year: {names}.")
        lines.append("  -> Candidates to pull vehicles from if supply is tight.")
    if not len(ramping) and not len(cooling):
        lines.append("Demand is steady vs last year across all markets.")
    lines.append("")

    # Allocation recommendation
    lines.append(f"RECOMMENDED ALLOCATION (fleet of {fleet_size} demo vehicles):")
    for r in allocation.itertuples():
        tag = ""
        if r.vs_even > 0:
            tag = f"  (+{r.vs_even} vs even split)"
        elif r.vs_even < 0:
            tag = f"  ({r.vs_even} vs even split)"
        lines.append(f"  {r.Dealer_Region:<12} {int(r.recommended_vehicles):>3} "
                     f"vehicles{tag}")
    lines.append("")
    lines.append(f"Highest-demand market this week: {top.Dealer_Region} "
                 f"({top.forecast_share:.0f}% of forecast demand).")
    lines.append("")
    lines.append("Method note: allocation tracks forecast demand share with a "
                 "per-market floor. Regional demand levels are similar, so the "
                 "primary lever is seasonal timing rather than large "
                 "cross-market moves.")
    return "\n".join(lines)


if __name__ == "__main__":
    from forecasting import run_all
    from allocation import seasonal_signals, proportional_allocation
    out = run_all()
    sig = seasonal_signals(out["panel"], out["forecast"])
    alloc, _ = proportional_allocation(out["forecast"], fleet_size=120)
    print(generate_briefing(out["metrics"], sig, alloc, 120))
