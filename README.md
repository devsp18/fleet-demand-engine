# Fleet Demand Forecasting & Allocation Engine

**If a marketing team runs a fixed pool of demo vehicles across markets, where should those vehicles be next week, and which markets are heading into seasonal strength?**

This project forecasts next-week vehicle demand for each market, flags which markets are ramping up or cooling down versus a year ago, and recommends how to distribute a fixed demo fleet. It is the predictive and planning layer that sits on top of a descriptive conversion dashboard.

**Live dashboard:** _(Streamlit Cloud link once deployed)_
**Code:** this repo

---

## The question behind it

A descriptive dashboard tells you what already happened: which markets converted, where the gaps were. The harder operational question is forward-looking. Demand is not flat across the year, and a fleet planned for last quarter is wrong for next quarter. This engine forecasts demand per market, surfaces the seasonal momentum, and turns that into an allocation a planner can act on.

## What is real and what is derived

I want this to be defensible line by line, so the data provenance is explicit.

**Real (from the source data):** the dataset is real dealership sales records from Kaggle (`missionjee/car-sales-report`), about 24,000 transactions across 7 regions, dated daily from January 2022 through December 2023. Sale dates, dealer regions, prices, vehicle make/model/body style, and customer income are all real fields.

**Derived (by aggregation, not invention):** I aggregate those real sales into a weekly panel at the region level (one row per region per week, with units sold, revenue, average price, average income). "Demand" here means realized weekly sales volume. I deliberately do **not** fabricate demo-drive counts or a conversion rate, because the raw data does not contain them, and inventing them would make the forecast indefensible.

## How it works

**1. Data pipeline** (`src/data_pipeline.py`)
Cleans the raw file and builds a rectangular weekly region panel (7 regions x 105 weeks), with true-zero weeks where a region had no sales.

**2. Forecasting** (`src/forecasting.py`)
An XGBoost regressor predicts next-week units per region. Features: recent lags (1, 2, 4 weeks), a rolling 4-week mean, and seasonal encoding (month plus sine/cosine of week-of-year so the calendar wraps correctly). Validation is a **time-based split**: train on the earliest ~80% of weeks, test on the most recent ~20%. No row shuffling, so there is no leakage from future to past. The model is held against a naive "next week equals last week" baseline so it has to earn its accuracy.

**3. Seasonal signals** (`src/allocation.py`)
For the forecast week, each region's forecast is compared to the **same week one year earlier**, the correct baseline given that demand swings roughly 4.5x between seasonal peak and trough. Markets well above last year are ramping; well below are cooling.

**4. Allocation** (`src/allocation.py`)
A fixed demo fleet is distributed in proportion to forecast demand, with a per-market floor so every active market keeps a presence. The recommendation is compared honestly against an even split.

**5. Auto-briefing** (`src/report.py`)
A plain-English weekly summary a planner could read in 20 seconds: projected demand, what is ramping, the recommended allocation, and a stated method note.

**6. Dashboard** (`src/app.py`)
A Streamlit app tying it together, including the model diagnostics (holdout metrics, feature importance, forecast-vs-actual) so the honesty is visible, not hidden.

## Headline results

- **Forecast accuracy:** holdout MAE of about 11 units per week, roughly **18% better than a naive last-week baseline**. A RandomForest cross-check lands in the same place (MAE ~11.6), which says the signal is real and not an artifact of one algorithm.
- **What drives demand:** seasonality (month, week-of-year) and recent momentum (lags) lead the feature importance, matching the visible yearly cycle. Region identity matters least, which leads to the main finding.
- **The honest finding:** regional demand *levels* are similar (about a 1.4x spread top to bottom), but every region has a large *seasonal* swing (~4.5x peak to trough). So the real operational lever is **timing**, moving the fleet toward markets entering their seasonal peak, not large permanent reallocations between markets. The engine is scoped to what the data supports rather than overclaiming an optimizer's impact.

## Why it is scoped this way

I tried a heavier constrained optimizer first. On this data it produced only marginal gains over an even split, for a real reason: the markets are similar in scale, so an even split is already close to optimal. Rather than tune constants until a bigger number appeared, I scoped the engine to the lever the data actually supports (seasonal timing) and said so plainly. Interrogating the data and matching the method to it is the point.

## Run it locally

```bash
pip install -r requirements.txt
cd src
streamlit run app.py
```

## Stack

Python, pandas, XGBoost, scikit-learn, statsmodels, SciPy, Plotly, Streamlit.

## Note

Built as a portfolio project. The data is real automotive sales reshaped into fleet-operations terms; the forecasting and allocation methods are production-style but the numbers describe this dataset, not any specific company's fleet.
