import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from .prophet_model import prophet_forecast
from .sarima_model import sarima_forecast
from .xgboost_model import xgboost_forecast
from .lstm_model import lstm_forecast

def prepare_data_for_ml(df: pd.DataFrame, n_steps: int = 7) -> Tuple[np.array, np.array]:
    """Prepare data for ML models (XGBoost and LSTM)"""
    if len(df) <= n_steps:
        raise ValueError(f"Not enough data points. Need more than {n_steps} points.")
        
    X, y = [], []
    for i in range(len(df) - n_steps):
        X.append(df[i:(i + n_steps)])
        y.append(df[i + n_steps])
    return np.array(X), np.array(y)

def perform_forecasting(data: Dict[str, Any], forecast_periods: int = 30) -> Dict[str, Any]:
    """Perform forecasting using multiple models and select the best one"""
    try:
        df = pd.DataFrame(data)
        
        if not all(col in df.columns for col in ['date', 'profit']):
            return {"error": "Data must contain 'date' and 'profit' columns"}
            
        if len(df) < 14:  # Require at least 14 data points
            return {"error": "Need at least 14 data points for forecasting"}
        
        # Prepare data
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['date']),
            'y': df['profit']
        })
        
        profit_series = np.array(df['profit'])
        scaler = MinMaxScaler()
        scaled_profit = scaler.fit_transform(profit_series.reshape(-1, 1)).flatten()
        
        # Initialize containers for results
        forecasts = {}
        rmse_scores = {}
        
        # Try each model separately
        for model_name, model_func, args in [
            ('prophet', prophet_forecast, (prophet_df, forecast_periods)),
            ('sarima', sarima_forecast, (profit_series, forecast_periods)),
            ('xgboost', xgboost_forecast, (prepare_data_for_ml(scaled_profit)[0], 
                                         prepare_data_for_ml(scaled_profit)[1],
                                         scaled_profit[-7:],
                                         forecast_periods)),
            ('lstm', lstm_forecast, (prepare_data_for_ml(scaled_profit)[0].reshape(-1, 7, 1),
                                   prepare_data_for_ml(scaled_profit)[1],
                                   scaled_profit[-7:].reshape(-1, 1),
                                   forecast_periods))
        ]:
            try:
                result, rmse = model_func(*args)
                forecasts[model_name] = result
                rmse_scores[model_name] = rmse
            except Exception as e:
                print(f"{model_name} model failed: {str(e)}")
        
        if not rmse_scores:
            return {"error": "All models failed to process the data"}
            
        # Select best model
        best_model = min(rmse_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "best_model": best_model,
            "rmse_score": rmse_scores[best_model],
            "forecast": forecasts[best_model],
            "forecast_dates": [
                (datetime.strptime(df['date'].iloc[-1], '%Y-%m-%d') + timedelta(days=i+1)).strftime('%Y-%m-%d')
                for i in range(forecast_periods)
            ]
        }
        
    except Exception as e:
        return {"error": str(e)} 