# The Machine v2.0

The Machine v2.0 is a Chicago crime-risk intelligence dashboard built with Python and Streamlit. It combines predictive modeling, tactical mapping, precinct logistics, explainable AI, and a cyber-inspired interface into a single operational view.

## What the app does

- Predicts relative risk by Chicago risk zone
- Renders a dark tactical map with heat intensity
- Shows nearby precincts and estimated response time
- Explains the prediction with feature-level contribution analysis
- Displays an hourly trend for the selected zone
- Uses a live Chicago clock in the sidebar
- Supports a payday heuristic for tactical scenario testing

## Product Overview

The dashboard is designed as a compact intelligence layer rather than a generic analytics page. It is optimized to help an analyst or operator quickly answer three questions:

1. Where is the highest risk right now?
2. Why is that zone elevated?
3. What resources are closest to the target area?

## How the system is organized

```text
The_Machine_Project_v2/
├─ app.py
├─ data_pipeline.py
├─ model_engine.py
├─ feature_schema.py
├─ requirements.txt
├─ centroids.csv
├─ processed_crime_data.csv
├─ scaler.pkl
└─ the_machine_model.pkl
```

### `data_pipeline.py`
Fetches and prepares the source data, engineers spatial and temporal features, and writes the processed dataset.

### `model_engine.py`
Trains the scikit-learn model and saves the fitted model and scaler as artifacts.

### `app.py`
Loads the saved artifacts and renders the Streamlit dashboard.

## Feature Schema

The model input must remain in this exact order:

```python
['Precipitation', 'Risk_Zone', 'Temperature', 'Hour', 'DayOfWeek', 'Month', 'Is_Holiday', 'Spatial_Lag']
```

Do not change this order unless you retrain the model and scaler together.

## Local Run

### 1. Create a virtual environment

#### Windows PowerShell
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

## Rebuild the artifacts

If you want to regenerate the processed data, scaler, and model from scratch:

```bash
python data_pipeline.py
python model_engine.py
```

## Troubleshooting

### The app opens but shows no results
- Click **Analyze & Predict Risk**.
- Confirm that `the_machine_model.pkl`, `scaler.pkl`, and `centroids.csv` exist.
- Check that `processed_crime_data.csv` is present if the app expects it.

### Feature order mismatch
If you see a feature-name error, the input order must still be exactly:

```python
['Precipitation', 'Risk_Zone', 'Temperature', 'Hour', 'DayOfWeek', 'Month', 'Is_Holiday', 'Spatial_Lag']
```


## Dependencies

This project uses:

- streamlit
- pandas
- numpy
- scikit-learn
- joblib
- folium
- streamlit-folium
- holidays
- pytz
- plotly
- requests

## Disclaimer

This project is a technical prototype for research and demonstration. It should not be used as the sole basis for real-world law-enforcement decisions without validation, oversight, and ethical review.
