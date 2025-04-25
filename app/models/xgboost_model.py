from xgboost import XGBRegressor
from typing import Dict, Tuple
import numpy as np
from .metrics import calculate_rmse

def xgboost_forecast(X_train: np.array, y_train: np.array, 
                    last_sequence: np.array, forecast_periods: int) -> Tuple[dict, float]:
    """Forecast using XGBoost"""
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(forecast_periods):
        pred = model.predict(current_sequence.reshape(1, -1))[0]
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = pred
    
    rmse = calculate_rmse(
        y_train,
        model.predict(X_train.reshape(X_train.shape[0], -1))
    )
    
    return {
        'predictions': np.array(predictions).round(2).tolist()
    }, rmse 