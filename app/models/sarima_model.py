from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, Tuple
import numpy as np
from .metrics import calculate_rmse
from scipy import stats

def sarima_forecast(train_data: np.array, forecast_periods: int) -> Tuple[dict, float]:
    """Forecast using SARIMA"""
    try:
        # Data preprocessing
        # Remove outliers using z-score
        z_scores = np.abs(stats.zscore(train_data))
        train_data_clean = train_data[z_scores < 3]
        
        # Fit SARIMA model with simpler parameters
        model = SARIMAX(
            train_data_clean,
            order=(1, 0, 1),          # Simpler ARMA structure, no differencing
            seasonal_order=(0, 1, 1, 7),  # Simple seasonal structure with weekly seasonality
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Fit the model with robust optimization
        fitted = model.fit(disp=False, method='powell', maxiter=100)
        
        # Generate forecast
        forecast = fitted.forecast(steps=forecast_periods)
        
        # Calculate confidence intervals
        forecast_conf = fitted.get_forecast(steps=forecast_periods)
        conf_int = forecast_conf.conf_int(alpha=0.1)  # 90% confidence interval
        
        # Extract confidence intervals (conf_int is already a numpy array)
        lower_bound = np.array(conf_int)[:, 0]  # First column for lower bound
        upper_bound = np.array(conf_int)[:, 1]  # Second column for upper bound
        
        # Calculate RMSE on cleaned data
        rmse = calculate_rmse(
            train_data_clean,
            fitted.get_prediction(start=0, end=len(train_data_clean)-1).predicted_mean
        )
        
        return {
            'predictions': forecast.round(2).tolist(),
            'upper_bound': upper_bound.round(2).tolist(),
            'lower_bound': lower_bound.round(2).tolist()
        }, rmse
        
    except Exception as e:
        print(f"SARIMA model error: {str(e)}")
        # Return a simple moving average forecast as fallback
        ma_predictions = np.array([np.mean(train_data[-7:])] * forecast_periods)
        ma_std = np.std(train_data[-7:])
        
        return {
            'predictions': ma_predictions.round(2).tolist(),
            'upper_bound': (ma_predictions + 1.96 * ma_std).round(2).tolist(),
            'lower_bound': (ma_predictions - 1.96 * ma_std).round(2).tolist()
        }, float('inf')  # High RMSE to ensure this is not selected as best model if it fails 