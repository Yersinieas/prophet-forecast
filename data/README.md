# Data Folder

Place your time-series CSV datasets here.

## Required Format

Your CSV must have exactly these two columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `ds` | string | Datestamp in YYYY-MM-DD format | `2024-01-15` |
| `y` | float | The numeric value to forecast | `142.5` |

## Sample Data

`sample_data.csv` is included as a reference — it contains 3 years of synthetic daily data with trend and seasonality baked in so you can run the model immediately without needing a real dataset.

## Suggested Free Datasets to Try

- **Air Passengers** (classic): monthly airline passengers 1949-1960
  - Search: "AirPassengers dataset CSV"
- **Singapore Weather** (relevant to you): daily temperature/rainfall from data.gov.sg
  - https://data.gov.sg/datasets?topics=environment
- **Electricity Demand**: hourly/daily power usage time series
- **Stock Prices**: Yahoo Finance daily OHLCV data via `yfinance` library

## Note on Large Files

The `.gitignore` is set to ignore large CSVs (except `sample_data.csv`).
Do not commit files larger than ~5MB to this repo.
