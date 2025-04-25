from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
from database import DatabaseConnection
from scipy import stats
from pandas.api.types import is_numeric_dtype
import ast
import sys
import logging

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("=== Flask App Starting ===")
app = Flask(__name__)

def prepare_data_for_ml(df: pd.DataFrame, n_steps: int = 7) -> Tuple[np.array, np.array]:
    """Prepare data for ML models (XGBoost and LSTM)"""
    if len(df) <= n_steps:
        raise ValueError(f"Not enough data points. Need more than {n_steps} points.")
        
    X, y = [], []
    for i in range(len(df) - n_steps):
        X.append(df[i:(i + n_steps)])
        y.append(df[i + n_steps])
    return np.array(X), np.array(y)

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
    
    # Only return predictions
    predictions = forecast['yhat'].tail(forecast_periods).round(2).tolist()
    return {'predictions': predictions}, rmse

def sarima_forecast(train_data: np.array, forecast_periods: int) -> Tuple[dict, float]:
    """Forecast using SARIMA"""
    try:
        # Data preprocessing
        z_scores = np.abs(stats.zscore(train_data))
        train_data_clean = train_data[z_scores < 3]
        
        model = SARIMAX(
            train_data_clean,
            order=(1, 0, 1),
            seasonal_order=(0, 1, 1, 7)
        )
        
        fitted = model.fit(disp=False, method='powell', maxiter=100)
        forecast = fitted.forecast(steps=forecast_periods)
        
        rmse = calculate_rmse(
            train_data_clean,
            fitted.get_prediction(start=0, end=len(train_data_clean)-1).predicted_mean
        )
        
        return {
            'predictions': forecast.round(2).tolist()
        }, rmse
        
    except Exception as e:
        print(f"SARIMA model error: {str(e)}")
        # Return a simple moving average forecast as fallback
        ma_predictions = np.array([np.mean(train_data[-7:])] * forecast_periods)
        return {
            'predictions': ma_predictions.round(2).tolist()
        }, float('inf')

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

def lstm_forecast(X_train: np.array, y_train: np.array, 
                 last_sequence: np.array, forecast_periods: int) -> Tuple[dict, float]:
    """Forecast using LSTM"""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
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

def format_response(actual_data: pd.DataFrame, forecast_data: dict, 
                   chart_title: str = "Sales Forecast Analysis") -> dict:
    """Format the response according to Highcharts requirements"""
    
    # Prepare actual data
    actual_sales = {}
    for idx, (date, value) in enumerate(zip(actual_data['date'], actual_data['profit']), 1):
        actual_sales[str(idx)] = {
            "data": float(value),
            "style": "color: #2f7ed8"  # Blue color for actual data
        }
    
    # Prepare forecast data
    forecast_sales = {}
    start_idx = len(actual_data) + 1
    
    # Handle both direct predictions and dictionary format
    predictions = (forecast_data['predictions'] 
                  if isinstance(forecast_data, dict) 
                  else forecast_data)
    
    for idx, value in enumerate(predictions, start_idx):
        forecast_sales[str(idx)] = {
            "data": float(value),
            "style": "color: #f45b5b"  # Red color for forecast
        }
    
    # Construct the full response
    response = {
        "config": {
            "chartTitle": chart_title,
            "chartConfig": {
                "Actual Sales": {
                    "type": "line",
                    "rightAxis": False
                },
                "Forecast": {
                    "type": "line",
                    "rightAxis": False
                }
            },
            "isScatteredPlot": False
        },
        "data": {
            "Actual Sales": actual_sales,
            "Forecast": forecast_sales
        }
    }
    
    return response

def perform_forecasting(data: Dict[str, Any], forecast_periods: int = 30) -> Dict[str, Any]:
    """Perform forecasting using multiple models and select the best one"""
    try:
        df = pd.DataFrame(data)
        
        if not all(col in df.columns for col in ['date', 'profit']):
            return {"error": "Data must contain 'date' and 'profit' columns"}
            
        if len(df) < 14:
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
        
        # Try each model separately and catch individual failures
        try:
            # Prophet
            prophet_result, prophet_rmse = prophet_forecast(prophet_df, forecast_periods)
            forecasts['prophet'] = prophet_result['predictions']  # Store only predictions
            rmse_scores['prophet'] = prophet_rmse
        except Exception as e:
            print(f"Prophet model failed: {str(e)}")
            
        try:
            # SARIMA
            sarima_result, sarima_rmse = sarima_forecast(profit_series, forecast_periods)
            forecasts['sarima'] = sarima_result['predictions']  # Store only predictions
            rmse_scores['sarima'] = sarima_rmse
        except Exception as e:
            print(f"SARIMA model failed: {str(e)}")
        
        try:
            # Prepare data for ML models
            X, y = prepare_data_for_ml(scaled_profit)
            last_sequence = scaled_profit[-7:]  # Last 7 days for prediction
            
            # XGBoost
            xgb_result, xgb_rmse = xgboost_forecast(X, y, last_sequence, forecast_periods)
            forecasts['xgboost'] = {
                'predictions': scaler.inverse_transform(
                    np.array(xgb_result['predictions']).reshape(-1, 1)
                ).flatten().round(2).tolist()
            }
            rmse_scores['xgboost'] = xgb_rmse
        except Exception as e:
            print(f"XGBoost model failed: {str(e)}")
            
        try:
            # LSTM
            lstm_result, lstm_rmse = lstm_forecast(
                X.reshape(X.shape[0], X.shape[1], 1), 
                y, 
                last_sequence.reshape(-1, 1), 
                forecast_periods
            )
            forecasts['lstm'] = {
                'predictions': scaler.inverse_transform(
                    np.array(lstm_result['predictions']).reshape(-1, 1)
                ).flatten().round(2).tolist()
            }
            rmse_scores['lstm'] = lstm_rmse
        except Exception as e:
            print(f"LSTM model failed: {str(e)}")
        
        if not rmse_scores:
            return {"error": "All models failed to process the data"}
            
        # Select best model
        best_model = min(rmse_scores.items(), key=lambda x: x[1])[0]
        
        # Format the response for Highcharts
        response = format_response(
            actual_data=df,
            forecast_data=forecasts[best_model],  # Pass predictions directly
            chart_title=f"Sales Forecast Analysis (using {best_model.upper()})"
        )
        
        # Add additional metadata
        response["metadata"] = {
            "model_used": best_model,
            "rmse_score": rmse_scores[best_model],
            "forecast_periods": forecast_periods
        }
        
        return response
        
    except Exception as e:
        return {"error": str(e)}

def calculate_rmse(actual: np.array, predicted: np.array) -> float:
    """Calculate Root Mean Square Error"""
    # Add small epsilon to avoid potential numerical issues
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse + 1e-10)  # Add small constant to avoid sqrt(0)
    return round(rmse, 4)  # Round to 4 decimal places for better precision

def apply_filter_safely(df: pd.DataFrame, filter_query: str) -> pd.DataFrame:
    """
    Safely apply a filter query to DataFrame
    Example filters:
    - "profit > 1000"
    - "profit > 1000 and profit < 2000"
    - "date > '2023-01-15'"
    """
    try:
        if not filter_query or filter_query.strip() == '':
            return df
            
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Validate filter query using ast to prevent code injection
        ast.parse(filter_query)
        
        # Apply filter and check if result is not empty
        filtered_df = df.query(filter_query)
        if len(filtered_df) < 14:
            raise ValueError("Filtered data has less than 14 points")
            
        # Convert date back to string format
        filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')
        return filtered_df
        
    except Exception as e:
        raise ValueError(f"Invalid filter query: {str(e)}")

@app.route('/')
def index():
    """Serve the HTML interface"""
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast_data():
    print("\n=== Forecast endpoint hit ===")
    try:
        print("\n=== Starting forecast request ===")
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get parameters from request
        data_source = request.args.get('source', 'direct')
        filter_query = data.get('filter', '')  # Get filter from request data
        forecast_periods = request.args.get('periods', default=30, type=int)
        print(f"Data source: {data_source}")
        
        if data_source == 'sql':
            try:
                print("\n=== Processing SQL data source ===")
                # Get datamartid and optional table name from request
                datamartid = data.get('datamartid')
                table_name = data.get('table_name')
                
                print(f"Datamartid: {datamartid}")
                print(f"Table name: {table_name}")
                
                if not datamartid:
                    return jsonify({"error": "datamartid is required for SQL source"}), 400
                
                # Initialize database connection
                print("\n=== Initializing database connection ===")
                db = DatabaseConnection()
                
                # Fetch data from database
                print("\n=== Attempting to fetch data ===")
                sql_data = db.fetch_data(datamartid, table_name)
                
                print("\n=== SQL data fetched successfully ===")
                
                # Process the data into required format
                if table_name:
                    table_data = sql_data[table_name]
                else:
                    # Use first table if no specific table requested
                    table_data = next(iter(sql_data.values()))
                
                # Create DataFrame with explicit index
                df = pd.DataFrame(table_data).reset_index(drop=True)
                
                # Ensure date and profit columns exist
                if 'date' not in df.columns or 'profit' not in df.columns:
                    return jsonify({"error": "SQL data must contain 'date' and 'profit' columns"}), 400
                
                processed_data = {
                    'date': df['date'].tolist(),
                    'profit': df['profit'].astype(float).tolist()
                }
                
            except Exception as e:
                return jsonify({"error": f"Database error: {str(e)}"}), 500
                
        else:
            # Handle direct data input
            processed_data = data
            
        # Convert to DataFrame and apply filter
        df = pd.DataFrame(processed_data)
        filtered_df = apply_filter_safely(df, filter_query)
        
        # Convert filtered data back to dict format
        filtered_data = {
            'date': filtered_df['date'].tolist(),
            'profit': filtered_df['profit'].tolist()
        }
        
        results = perform_forecasting(filtered_data, forecast_periods)
        return jsonify(results)
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
