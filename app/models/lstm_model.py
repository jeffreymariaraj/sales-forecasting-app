from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from typing import Dict, Tuple
import numpy as np
from .metrics import calculate_rmse
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def lstm_forecast(X_train: np.array, y_train: np.array, 
                 last_sequence: np.array, forecast_periods: int) -> Tuple[dict, float]:
    """Forecast using LSTM"""
    # Define input shape
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    # LSTM layer
    x = LSTM(50, activation='relu')(inputs)
    
    # Output layer
    outputs = Dense(1)(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(X_train, y_train, epochs=50, verbose=0)
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(forecast_periods):
        pred = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)[0][0]
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = pred
    
    rmse = calculate_rmse(
        y_train,
        model.predict(X_train, verbose=0).flatten()
    )
    
    return {
        'predictions': np.array(predictions).round(2).tolist()
    }, rmse 