# Singapore Rainfall Forecasting with Prophet

A time series forecasting project using Meta's Prophet library to model and evaluate daily rainfall in Singapore, using real historical weather data pulled from a public API.

## What this project does

- Pulls real daily rainfall data for Singapore (2010–2024) from the [Open-Meteo Historical Weather API](https://open-meteo.com/) — no API key required
- Fits a Prophet model that decomposes rainfall into trend, weekly seasonality, and yearly seasonality
- Backtests the model on a held-out period and compares it against two naive baselines
- Reports an honest, nuanced result rather than a single headline accuracy number

## Data

| | |
|---|---|
| Source | Open-Meteo Archive API (ERA5 / ERA5-Land reanalysis) |
| Location | Singapore (1.3521° N, 103.8198° E) |
| Variable | Daily total precipitation (mm) |
| Range | 2010-01-01 to 2024-12-31 |

The data is a reanalysis product — a blend of real ground stations, satellite observations, and weather models into a gap-free historical record. It's real measured-and-modeled weather, not synthetic data.

## Method

1. Load daily rainfall as a `ds` (date) / `y` (value) dataframe — Prophet's required format
2. Split into training and held-out test periods
3. Fit `Prophet()` on the training data only
4. Forecast the test period and compare predictions against actual rainfall
5. Compare Prophet's error against two naive baselines:
   - **Naive (mean):** predict the historical average every day
   - **Seasonal-naive:** predict the historical average for that calendar day-of-year

## Results

| Model | MAE (mm) | RMSE (mm) |
|---|---|---|
| Prophet | 7.07 | **8.64** |
| Naive (mean) | **6.97** | 10.38 |
| Seasonal-naive | 7.05 | 10.58 |

**Key finding:** Prophet does not reduce typical-day error compared to naive guessing — MAE is essentially tied across all three approaches, since day-to-day rainfall is dominated by short-term storm noise that no smooth seasonal curve can predict. However, Prophet reduces RMSE by ~17–18% relative to both baselines, meaning it specifically improves predictions during high-rainfall events. This is consistent with its yearly seasonality component correctly anticipating Singapore's November–January monsoon season, even though it can't predict individual storms.

## What I'd improve next

- Use a smoothed climatology (±7-day window) for a fairer seasonal-naive baseline — the current one only averages ~15 samples per calendar day, which is fairly noisy
- Add monsoon season as a custom regressor instead of relying solely on built-in yearly seasonality
- Break error down by month to directly confirm Prophet's RMSE advantage is concentrated in wetter months
- Try `seasonality_mode='multiplicative'` to see if it better captures how much rainfall varies between dry and wet months

## Setup

```powershell
git clone https://github.com/Yersinieas/prophet-forecast.git
cd prophet-forecast
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook notebooks/01_prophet_basics.ipynb
```

## Stack

Python · pandas · Prophet · matplotlib · scikit-learn · Open-Meteo API

## Notes

`requirements.txt` uses minimum-version pins (`>=`) rather than exact pins (`==`). Exact pins on `pandas`/`prophet`/`scikit-learn` predated Python 3.14 and broke the install with C-extension compile errors — pinning a floor instead of an exact version fixed it without needing to downgrade Python.