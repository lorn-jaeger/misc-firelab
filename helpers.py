from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

def correlations(data, min=None, max=None, sat_cols=None):

    base_col = "Sample Measurement"

    if sat_cols is None:
        sat_cols = ["CONUS", "CAMS", "MERRA2", "MERRA2R"]
    results = {}

    if min is None:
        min = 2 
    if max is None:
        max = 1000

    for col in sat_cols:
        subset = data[[base_col, col]].dropna()
        subset = subset[
            (subset[base_col].between(0.1, 1000)) &
            (subset[col].between(0.1, 1000))
        ]

        if subset.empty:
            results[col] = {
                "Pearson": np.nan,
                "Log Pearson": np.nan,
                "Spearman": np.nan,
                "RMSE": np.nan,
                "Bias": np.nan,
                "Slope": np.nan,
            }
            continue

        log_subset = np.log(subset)

        x = subset[col].values.reshape(-1, 1)
        y = subset[base_col].values

        model = LinearRegression().fit(x, y)
        slope = model.coef_[0]

        bias = np.mean(y - x.flatten())

        mse = mean_squared_error(y, x.flatten())
        rmse = np.sqrt(mse)

        pearson = subset[base_col].corr(subset[col], method="pearson")
        log_pearson = log_subset[base_col].corr(log_subset[col], method="pearson")
        spearman = spearmanr(subset[base_col], subset[col], nan_policy="omit").correlation

        results[col] = {
            "Pearson": pearson,
            "Log Pearson": log_pearson,
            "Spearman": spearman,
            "RMSE": rmse,
            "Bias": bias,
            "Slope": slope,
            "Count": len(subset),
        }

    df = pd.DataFrame(results).T

    if df.empty:
            print(f"No valid data")
    else:
        with pd.option_context('display.precision', 3, 'display.width', None, 'display.max_columns', None):
            print(df.to_string(index=True))




import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

def yearly_correlations(data, source, min_val=None, max_val=None):
    base_col = "Sample Measurement"

    if min_val is None:
        min_val = 2 
    if max_val is None:
        max_val = 1000

    data["Time"] = pd.to_datetime(data["Time"])
    years = sorted(data["Time"].dt.year.unique())
    rows = []

    for year in years:
        year_data = data[data["Time"].dt.year == year]
        subset = year_data[[base_col, source]].dropna()
        subset = subset[
            (subset[base_col].between(min_val, 1000)) &
            (subset[source].between(min_val, 1000))
        ]

        if subset.empty:
            continue

        log_subset = np.log(subset)
        x = subset[source].values.reshape(-1, 1)
        y = subset[base_col].values

        model = LinearRegression().fit(x, y)
        slope = model.coef_[0]
        bias = np.mean(y - x.flatten())
        rmse = np.sqrt(mean_squared_error(y, x.flatten()))
        pearson = subset[base_col].corr(subset[source], method="pearson")
        log_pearson = log_subset[base_col].corr(log_subset[source], method="pearson")
        spearman = spearmanr(subset[base_col], subset[source], nan_policy="omit").correlation

        row = {
            "Year": year,
            "Pearson": pearson,
            "Log Pearson": log_pearson,
            "Spearman": spearman,
            "RMSE": rmse,
            "Bias": bias,
            "Slope": slope,
            "Count": len(subset),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"No valid data for source: {source}")
    else:
        print(f"\nMetrics for source: {source}")
        with pd.option_context('display.precision', 3, 'display.width', None, 'display.max_columns', None):
            print(df.to_string(index=False))

