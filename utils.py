import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import torch
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense

NUM_COLS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

def correlation_matrix(df, numeric_cols):
    """
    Plot correlation heatmaps of the target variable against all other variables for each category.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): The list of numeric columns to be used for correlation.
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))  # Fixed variable name `ax`

    corr = df[numeric_cols].corr()
    
    sns.heatmap(
        corr,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        ax=ax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    
    plt.suptitle('Correlation Matrix of AAPL stock data', fontsize=15)
    plt.show()

P = 0
D = 1
Q = 0

from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

P = 0
D = 1
Q = 0

def get_arima_forecasts(df, formatted_date, target_col,
                        start_date, end_date, context_len=32):
    """
    Forecast stock value with ARIMA.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    formatted_date (str): Date column name.
    target_col (str): The target variable to correlate with other variables.
    start_date (str): Lower bound for normal data.
    end_date (str): Upper bound for normal data. Normal data is used as training
    resource for ARIMA model.
    context_len (int, Default=64): number of points to make prediction on.

    Returns: a dictionary containing 'train_data','test_data',
    'forecasts', 'actuals', 'timestamps' (date), and 'forecast_timestamps'.
    """
    
    def get_arma_forecast(context, p=P, r=D, q=Q):
        arma = ARIMA(context, order=(p, r, q)).fit(
            method_kwargs={"warn_convergence": False}
        )
        predict_cat = arma.forecast(steps=1)[0]  # Predicting the next step
        return predict_cat

    df[formatted_date] = pd.to_datetime(df[formatted_date])
    predictions = {}

    # Extract normal data for training
    normal_data = df[
        (df[formatted_date] >= start_date) & (df[formatted_date] < end_date)
    ]
    train_data = normal_data[[target_col]].values

    # Scale the data using the same scaler for training and test data
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    
    # Test data (future data after end_date)
    test_data = df[df[formatted_date] >= end_date][[target_col]].values
    scaled_test_data = np.array(scaler.transform(test_data))

    forecast_len = 1  # Predicting the next value
    arma_summary = []

    # Use the training data for fitting the ARIMA model
    for idx in range(len(scaled_train_data) - context_len - forecast_len + 1):
        context = scaled_train_data[idx : idx + context_len]
        forecast = get_arma_forecast(context)  # Forecasting on training data
        # arma_summary.append((context, scaled_train_data[idx + context_len], forecast))

    # Now, use the trained ARIMA model to predict on the test data
    for idx in range(len(scaled_test_data) - context_len - forecast_len + 1):
        context = scaled_test_data[idx : idx + context_len]
        forecast = get_arma_forecast(context)
        arma_summary.append((context, scaled_test_data[idx + context_len], forecast))

    # Extract actual and forecast values
    arma_actual = [i[1][0] for i in arma_summary]
    arma_forecast = [i[2] for i in arma_summary]

    # Inverse transform the scaled data for evaluation
    inv_train_data = scaler.inverse_transform(scaled_train_data)
    inv_test_data = scaler.inverse_transform(scaled_test_data)
    inv_arma_forecast = scaler.inverse_transform(np.array(arma_forecast).reshape(-1, 1))

    predictions["results"] = {
        "train_data": inv_train_data,
        "test_data": inv_test_data,
        "forecasts": inv_arma_forecast,
        "actuals": arma_actual,
        "timestamps": df[formatted_date],
        "forecast_timestamps": df[formatted_date][
            len(train_data) + context_len: len(train_data) + context_len + len(arma_forecast)
            ],
    }

    # Calculate and print MSE, MAD, MAPE
    if len(arma_actual) > 0 and len(arma_forecast) > 0:
        actual_array = np.array(arma_actual)
        forecast_array = np.array(arma_forecast)
        mape = np.mean(np.abs((actual_array - forecast_array) / actual_array))
        print(f"MSE: {np.round(np.mean((actual_array - forecast_array)**2), 5)}")
        print(f"MAD: {np.mean(np.abs(actual_array - forecast_array))}")
        print(f"MAPE: {mape * 100:.2f}%")
    else:
        raise ValueError(
            "No forecasts were generated. Check your data and sliding window configuration."
        )

    return pd.DataFrame(predictions)

def plot_arima_forecasts(predictions, save_dir="", dset_name="APPL", save_plot=False):
    """
    Plot actual vs predicted sales demand values using ARIMA

    Parameters:
    predictions (dict or DataFrame): a dictionary containing forecasted demand values by ARIMA
    save_dir: directory to save figure
    dset_name: the name of the dataset, used for naming figure files
    save_plot (Default=True): option to save plot or not
    """
    data = predictions["results"]
    train_data = data["train_data"]
    test_data = data["test_data"]
    forecasts = data["forecasts"]
    timestamps = data["timestamps"]
    forecast_timestamps = data["forecast_timestamps"]

    # Create a continuous time series for the actual data
    full_data = np.concatenate((train_data, test_data))

    # Plot actual data
    plt.figure(figsize=(16, 6))
    plt.plot(
        timestamps[len(timestamps) - len(forecast_timestamps):],
        full_data[len(timestamps) - len(forecast_timestamps):],
        label="Actual"
    )

    # Plot forecasted values
    plt.plot(
        timestamps[len(timestamps) - len(forecast_timestamps):],
        forecasts,
        label="Forecast",
        linestyle="dashed",
        color="red"
    )

    plt.title(f"Actual vs Forecasted Payment Values for '{dset_name}' using ARIMA")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()

    if save_plot:
        plot_filename = f"arima_forecast_{dset_name}.png"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, plot_filename))