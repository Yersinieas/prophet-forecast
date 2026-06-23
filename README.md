# Prophet Forecast

A time-series forecasting project using [Facebook Prophet](https://facebook.github.io/prophet/), built as part of a learning journey into ML, data analysis, and AI systems.

This project covers:
- Loading and preparing time-series data
- Fitting a Prophet model and forecasting future values
- Visualising trends, seasonality, and uncertainty intervals
- Evaluating model accuracy with train/test splits
- Scenario planning: "what if" comparisons under different assumptions

## Project Structure

```
prophet-forecast/
├── data/               # Raw and processed datasets (CSV)
├── notebooks/          # Jupyter notebooks (step-by-step walkthrough)
├── src/                # Clean Python scripts
│   └── forecast.py     # Main forecasting script
├── outputs/            # Saved plots and forecast CSVs
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/Yersinieas/prophet-forecast.git
cd prophet-forecast

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook notebooks/01_prophet_basics.ipynb
```

## Dataset

Place your CSV file inside the `data/` folder. It must have two columns:
- `ds` — datestamp (e.g. `2024-01-01`)
- `y`  — the numeric value to forecast (e.g. sales, temperature, traffic)

A sample dataset is included in `data/sample_data.csv`.

## Dependencies

See `requirements.txt`. Key libraries:
- `prophet` — core forecasting model
- `pandas` — data wrangling
- `matplotlib` — plotting
- `scikit-learn` — evaluation metrics
- `jupyter` — interactive notebooks

## What I Learned

- How Prophet decomposes time series into trend + seasonality + holidays
- How to evaluate a forecasting model (MAE, RMSE)
- How to test "what if" scenarios by adjusting future regressors
- How to structure a clean, reproducible ML project

## Next Steps

- Add external regressors (e.g. weather, events)
- Build a simple Streamlit dashboard for interactive forecasting
- Connect forecasts to an AI agent for natural-language scenario queries

## Author

Yersinieas — Mechanical Engineering @ NTU Singapore  
Building at the intersection of engineering, data, and AI.
