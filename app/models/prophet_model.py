from prophet import Prophet
from typing import Dict, Tuple
import pandas as pd
from .metrics import calculate_rmse

def prophet_forecast(train_df: pd.DataFrame, forecast_periods: int) -> Tuple[dict, float]:
    """Forecast using Prophet"""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    model.fit(train_df)
    future_dates = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future_dates)
    
    rmse = calculate_rmse(
        train_df['y'].values,
        forecast['yhat'][:len(train_df)].values
    )
    
    return {
        'dates': forecast['ds'].tail(forecast_periods).dt.strftime('%Y-%m-%d').tolist(),
        'predictions': forecast['yhat'].tail(forecast_periods).round(2).tolist(),
        'upper_bound': forecast['yhat_upper'].tail(forecast_periods).round(2).tolist(),
        'lower_bound': forecast['yhat_lower'].tail(forecast_periods).round(2).tolist()
    }, rmse 