import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Profit Forecasting & Analytics App",
    page_icon="📈",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTitle {
        font-weight: bold;
        color: #0066cc;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 5px;
    }
    .stSelectbox label {
        font-weight: bold;
    }
    .query-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📈 Profit Forecasting & Analytics Dashboard")
st.markdown("---")

# Initialize session state for filtered dataframe
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None

def load_data(file):
    try:
        df = pd.read_csv(file)
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_conversion.isna().all():
                        df[col] = numeric_conversion
            except:
                continue
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def prepare_data(df, target_column):
    try:
        df['DateKey'] = pd.to_datetime(df['DateKey'],format='mixed')
        df = df.dropna(subset=['DateKey'])
        daily_data = df.groupby('DateKey', observed=True).agg({
            target_column: 'sum'
        }).reset_index()
        daily_data = daily_data.sort_values('DateKey')
        daily_data = daily_data.rename(columns={target_column: 'Target'})
        daily_data = daily_data.dropna()
        return daily_data
    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return None

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def lstm_forecast(data, forecast_days, seq_length=30):
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['Target']])
        
        # Split data for evaluation
        train_size = int(len(scaled_data) * 0.8)
        train, test = scaled_data[:train_size], scaled_data[train_size:]
        
        # Create sequences
        X_train, y_train = create_sequences(train, seq_length)
        X_test, y_test = create_sequences(test, seq_length)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build and train model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Evaluate model
        test_predict = model.predict(X_test)
        test_predict = scaler.inverse_transform(test_predict)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse = sqrt(mean_squared_error(y_test_inv, test_predict))
        
        # Generate forecast
        forecast = []
        current_sequence = scaled_data[-seq_length:]
        for _ in range(forecast_days):
            current_sequence_reshaped = current_sequence.reshape((1, seq_length, 1))
            next_pred = model.predict(current_sequence_reshaped, verbose=0)
            forecast.append(next_pred[0, 0])
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        last_date = data['DateKey'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({'DateKey': forecast_dates, 'Forecast': forecast.flatten()})
        
        return forecast_df, rmse
        
    except Exception as e:
        st.error(f"Error in LSTM forecasting: {str(e)}")
        return None, None

def xgboost_forecast(data, forecast_days, seq_length=30):
    try:
        df = data.copy()
        df['Year'] = df['DateKey'].dt.year
        df['Month'] = df['DateKey'].dt.month
        df['DayOfWeek'] = df['DateKey'].dt.dayofweek
        df['DayOfMonth'] = df['DateKey'].dt.day
        
        for i in range(1, seq_length + 1):
            df[f'lag_{i}'] = df['Target'].shift(i)
        
        df = df.dropna()
        features = ['Year', 'Month', 'DayOfWeek', 'DayOfMonth'] + [f'lag_{i}' for i in range(1, seq_length + 1)]
        X = df[features]
        y = df['Target']
        
        # Split data for evaluation
        train_size = int(len(df) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
        model.fit(X_train, y_train)
        
        # Evaluate model
        test_predict = model.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, test_predict))
        
        # Generate forecast
        forecast = []
        current_data = df.iloc[-1:].copy()
        for _ in range(forecast_days):
            next_date = current_data['DateKey'].iloc[-1] + timedelta(days=1)
            new_row = pd.DataFrame({
                'DateKey': [next_date],
                'Year': [next_date.year],
                'Month': [next_date.month],
                'DayOfWeek': [next_date.dayofweek],
                'DayOfMonth': [next_date.day]
            })
            for i in range(1, seq_length + 1):
                new_row[f'lag_{i}'] = current_data['Target'].iloc[-1] if i == 1 else current_data[f'lag_{i-1}'].iloc[-1]
            pred = model.predict(new_row[features])
            forecast.append(pred[0])
            new_row['Target'] = pred[0]
            current_data = pd.concat([current_data, new_row]).iloc[1:].copy()
        
        last_date = data['DateKey'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({'DateKey': forecast_dates, 'Forecast': forecast})
        
        return forecast_df, rmse
        
    except Exception as e:
        st.error(f"Error in XGBoost forecasting: {str(e)}")
        return None, None

def prophet_forecast(data, forecast_days):
    try:
        # Split data for evaluation
        train_size = int(len(data) * 0.8)
        train = data.iloc[:train_size]
        test = data.iloc[train_size:]
        
        prophet_train = train.rename(columns={'DateKey': 'ds', 'Target': 'y'})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_train)
        
        # Evaluate model
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)
        test_forecast = forecast.iloc[train_size:][['ds', 'yhat']]
        rmse = sqrt(mean_squared_error(test['Target'], test_forecast['yhat']))
        
        # Full dataset forecast
        model_full = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model_full.fit(data.rename(columns={'DateKey': 'ds', 'Target': 'y'}))
        future = model_full.make_future_dataframe(periods=forecast_days)
        forecast = model_full.predict(future)
        
        return forecast, rmse
        
    except Exception as e:
        st.error(f"Error in Prophet forecasting: {str(e)}")
        return None, None

def sarima_forecast(data, forecast_days):
    try:
        # Split data for evaluation
        train_size = int(len(data) * 0.8)
        train = data.iloc[:train_size]
        test = data.iloc[train_size:]
        
        model = SARIMAX(train['Target'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)
        
        # Evaluate model
        start = len(train)
        end = len(train) + len(test) - 1
        test_predict = results.predict(start=start, end=end)
        rmse = sqrt(mean_squared_error(test['Target'], test_predict))
        
        # Full dataset forecast
        model_full = SARIMAX(data['Target'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results_full = model_full.fit(disp=False)
        forecast = results_full.get_forecast(steps=forecast_days)
        
        return forecast.predicted_mean, rmse
        
    except Exception as e:
        st.error(f"Error in SARIMA forecasting: {str(e)}")
        return None, None

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    row_to_keep_inclusive = 376234

    # Keep rows up to and including the specified row using .iloc
    df = df.iloc[:row_to_keep_inclusive + 1]
    if df is not None:
        try:
            if st.session_state.filtered_df is None:
                st.session_state.filtered_df = df.copy()
            
            tabs = st.tabs(["🔍 Query Analysis", "📊 Data Analysis", "🔮 Forecasting"])
            
            with tabs[0]:
                
                st.subheader("Search Query")
                query = st.text_area(
                    "Query",
                    height=200,
                    key="main_query",
                    help="Enter your query using Python's query syntax. Example: 'column_name > 100 and other_column < 50'"
                )
    
                if st.button("Execute Query"):
                    try:
                        filtered_df = df.query(query)
                        st.write(f"Found {len(filtered_df)} matching rows")
            
                        # Generate HTML for popup
                        popup_html = f"""
                        <div id="queryPopup" style="
                            position: fixed;
                            top: 50%;
                            left: 50%;
                            transform: translate(-50%, -50%);
                            background: white;
                            padding: 20px;
                            z-index: 1000;
                            box-shadow: 0 0 10px rgba(0,0,0,0.5);
                            max-width: 80%;
                            max-height: 80vh;
                            overflow: auto;
                            border-radius: 10px;
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 10px;">
                                <h3 style="margin: 0;">Query Results ({len(filtered_df)} rows)</h3>
                                <div>
                                    <button onclick="togglePopupSize()" style="margin-right: 5px; padding: 5px 10px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer;">
                                        Toggle Size
                                    </button>
                                    <button onclick="document.getElementById('queryPopup').style.display='none';" 
                            style="padding: 5px 10px; background: #ff4444; color: white; border: none; border-radius: 3px; cursor: pointer;">
                            Close
                                    </button>
                                </div>
                            </div>
                            <div style="max-height: 70vh; overflow: auto;">
                                {filtered_df.head().to_html(index=False, classes='dataframe table table-striped')}
                            </div>
                        </div>
            
                        <script>
                            function togglePopupSize() {{
                                var popup = document.getElementById('queryPopup');
                                if (popup.style.width === '80%') {{
                                    popup.style.width = '95%';
                                    popup.style.height = '95vh';
                                }} else {{
                                    popup.style.width = '80%';
                                    popup.style.height = 'auto';
                                }}
                            }}
                        </script>
            
                        <style>
                            .dataframe {{
                                width: 100%;
                                border-collapse: collapse;
                            }}
                            .dataframe th, .dataframe td {{
                                padding: 8px;
                                text-align: left;
                                border-bottom: 1px solid #ddd;
                            }}
                            .dataframe tr:hover {{
                                background-color: #f5f5f5;
                            }}
                        </style>
                        """
                        st.markdown(popup_html, unsafe_allow_html=True)
            
                    except Exception as e:
                        st.error(f"Error executing query: {str(e)}")

            with tabs[1]:
                st.subheader("Data Analysis Dashboard")
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                categorical_cols = [col for col in categorical_cols if col != 'DateKey']
                
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        selected_category = st.selectbox("Select Category for Analysis", numeric_cols)
                    with col2:
                        category_values = ['All'] + sorted(df[selected_category].unique().tolist())
                        selected_value = st.selectbox(f"Select {selected_category}", category_values)
                    with col3:
                        selected_metric = st.selectbox("Select Metric to Analyze", numeric_cols)
                    
                    filtered_df = df.copy()
                    if selected_value != 'All':
                        filtered_df = filtered_df[filtered_df[selected_category] == selected_value]
                    
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        total_metric = filtered_df[selected_metric].sum()
                        st.metric(f"Total {selected_metric}", f"{total_metric:,.2f}")
                    with metric_cols[1]:
                        avg_metric = filtered_df[selected_metric].mean()
                        st.metric(f"Average {selected_metric}", f"{avg_metric:,.2f}")
                    with metric_cols[2]:
                        max_metric = filtered_df[selected_metric].max()
                        st.metric(f"Maximum {selected_metric}", f"{max_metric:,.2f}")
                    with metric_cols[3]:
                        min_metric = filtered_df[selected_metric].min()
                        st.metric(f"Minimum {selected_metric}", f"{min_metric:,.2f}")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    with viz_col1:
                        category_data = filtered_df.groupby(selected_category)[selected_metric].sum().reset_index()
                        fig = px.bar(category_data, x=selected_category, y=selected_metric, 
                                   title=f'{selected_metric} by {selected_category}')
                        st.plotly_chart(fig, use_container_width=True)
                    with viz_col2:
                        time_data = filtered_df.groupby('DateKey')[selected_metric].sum().reset_index()
                        fig = px.line(time_data, x='DateKey', y=selected_metric, title=f'{selected_metric} Trend Over Time')
                        st.plotly_chart(fig, use_container_width=True)

            with tabs[2]:
                st.subheader("Forecasting Dashboard")
                forecast_query = st.text_area(
                    "Forecasting Filters",
                    height=100,
                    key="forecast_query",
                    help="Enter your query to filter data before forecasting. Example: 'column_name > 100'"
                )

                if st.button("Apply Filters"):
                    try:
                        if forecast_query:
                            st.session_state.filtered_df = df.query(forecast_query)
                        else:
                            st.session_state.filtered_df = df.copy()
                        st.success("Filters applied successfully!")
                        # Reset table preview state when new filters are applied
                        st.session_state.show_table_preview = False
                    except Exception as e:
                        st.error(f"Error applying filters: {str(e)}")
    
                if 'filtered_df' in st.session_state:
                    # Generate styled table HTML
                    table_html = st.session_state.filtered_df.head(100).style\
                        .set_table_styles([
                            {'selector': 'thead', 'props': [('position', 'sticky'), ('top', '0'), ('background', 'white')]},
                            {'selector': 'th', 'props': [('padding', '12px'), ('background', '#f8f9fa')]},
                            {'selector': 'td', 'props': [('padding', '12px')]}
                        ])\
                        .hide(axis='index')\
                        .to_html()
                    comparison_table_html = st.session_state.filtered_df.head(100).style\
                        .set_table_styles([
                            {'selector': 'thead', 'props': [('position', 'sticky'), ('top', '0'), ('background', 'white')]},
                            {'selector': 'th', 'props': [('padding', '12px'), ('background', '#f8f9fa')]},
                            {'selector': 'td', 'props': [('padding', '12px')]}
                        ])\
                        .hide(axis='index')\
                        .to_html()
                    # Create toggle component
                    toggle_component = f"""
<script>
function toggleTable(tableNumber) {{
    const tableDiv = document.getElementById('previewTable' + tableNumber);
    const btn = document.getElementById('toggleBtn' + tableNumber);
    
    if(tableNumber === 1) {{
        if(tableDiv.style.display === 'none' || !tableDiv.style.display) {{
            tableDiv.style.display = 'block';
            btn.innerHTML = '<span style="color: #4CAF50;">▼</span> Hide Main Data';
        }} else {{
            tableDiv.style.display = 'none';
            btn.innerHTML = '<span style="color: #4CAF50;">▶</span> Show Main Data';
        }}
    }} else {{
        if(tableDiv.style.display === 'none' || !tableDiv.style.display) {{
            tableDiv.style.display = 'block';
            btn.innerHTML = '<span style="color: #4CAF50;">▼</span> Hide Table';
        }} else {{
            tableDiv.style.display = 'none';
            btn.innerHTML = '<span style="color: #4CAF50;">▶</span> Show Table';
        }}
    }}
}}

function closeTable(tableNumber) {{
    const tableDiv = document.getElementById('previewTable' + tableNumber);
    const btn = document.getElementById('toggleBtn' + tableNumber);
    tableDiv.style.display = 'none';
    btn.innerHTML = '<span style="color: #4CAF50;">▶</span> ' + 
                    (tableNumber === 1 ? 'Show Main Data' : 'Show Comparison');
}}
</script>

<div style="margin: 1rem 0;">
    <!-- Buttons -->
    <div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px;">
        <button id="toggleBtn1" onclick="toggleTable(1)" 
            style="background:none; border:none; padding:6px; cursor:pointer; color:#4CAF50; text-align:left;">
            <span style="color: #4CAF50;">▶</span> Show Main Data
        </button>

        <button id="toggleBtn2" onclick="toggleTable(2)" 
            style="background:none; border:none; padding:6px; cursor:pointer; color:#4CAF50; text-align:left;">
            <span style="color: #4CAF50;">▶</span> Show Second Table
        </button>
    </div>

    <!-- Tables Container -->
    <div style="display: flex; flex-direction: column; gap: 12px;">
        <!-- Main Data Table -->
        <div id="previewTable1" style="display:none; width:100%; margin:8px 0; border:1px solid #e6e9ef; border-radius:0.4rem; background:white;">
            <div style="display:flex; justify-content:space-between; align-items:center; padding:6px; border-bottom:1px solid #e6e9ef;">
                <div style="font-weight:500;">Main Data</div>
                <button onclick="closeTable(1)" style="background:#ff4444; color:white; border:none; border-radius:2px; padding:3px 8px; cursor:pointer;">
                    × Close
                </button>
            </div>
            <div style="max-height:400px; overflow:auto; font-size:0.6em;">
                {table_html}
            </div>
        </div>

        <!-- Comparison Table -->
        <div id="previewTable2" style="display:none; width:100%; margin:8px 0; border:1px solid #e6e9ef; border-radius:0.4rem; background:white;">
            <div style="display:flex; justify-content:space-between; align-items:center; padding:6px; border-bottom:1px solid #e6e9ef;">
                <div style="font-weight:500;">Second Table</div>
                <button onclick="closeTable(2)" style="background:#ff4444; color:white; border:none; border-radius:2px; padding:3px 8px; cursor:pointer;">
                    × Close
                </button>
            </div>
            <div style="max-height:400px; overflow:auto; font-size:0.6em;">
                {comparison_table_html}
            </div>
        </div>
    </div>
</div>

<style>
.dataframe {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.6em;
}}
.dataframe th {{
    background: #f8f9fa !important;
    padding: 2px 3px;
    text-align: left;
    position: sticky;
    top: 0;
    border-bottom: 1px solid #e6e9ef;
}}
.dataframe td {{
    padding: 2px 3px;
    border-bottom: 0.5px solid #e6e9ef;
}}
.dataframe tr:nth-child(even) {{
    background-color: #f8f8f8;
}}
</style>
                    """
        
                    # Render the component
                    html(toggle_component, height=300)
                st.sidebar.markdown("---")
                st.sidebar.header("Forecasting Parameters")
                numeric_cols = st.session_state.filtered_df.select_dtypes(include=['int64', 'float64']).columns
                target_column = st.sidebar.selectbox("Select Variable to Forecast", numeric_cols)
                forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)
    
                daily_data = prepare_data(st.session_state.filtered_df, target_column)
    
                if daily_data is not None and st.sidebar.button("Generate Forecast"):
                    with st.spinner("Generating forecasts..."):
                        try:
                            model_results = []
                            
                            # Prophet
                            prophet_forecast_df, prophet_rmse = prophet_forecast(daily_data, forecast_days)
                            if prophet_rmse is not None:
                                model_results.append(('Prophet', prophet_rmse, prophet_forecast_df))
                            
                            # SARIMA
                            sarima_forecast_vals, sarima_rmse = sarima_forecast(daily_data, forecast_days)
                            if sarima_rmse is not None:
                                sarima_dates = pd.date_range(start=daily_data['DateKey'].iloc[-1] + timedelta(days=1), periods=forecast_days)
                                sarima_df = pd.DataFrame({'DateKey': sarima_dates, 'Forecast': sarima_forecast_vals})
                                model_results.append(('SARIMA', sarima_rmse, sarima_df))
                            
                            # LSTM
                            lstm_forecast_df, lstm_rmse = lstm_forecast(daily_data, forecast_days)
                            if lstm_rmse is not None:
                                model_results.append(('LSTM', lstm_rmse, lstm_forecast_df))
                            
                            # XGBoost
                            xgb_forecast_df, xgb_rmse = xgboost_forecast(daily_data, forecast_days)
                            if xgb_rmse is not None:
                                model_results.append(('XGBoost', xgb_rmse, xgb_forecast_df))
                            
                            # Find best model
                            if model_results:
                                model_results.sort(key=lambda x: x[1])
                                best_model = model_results[0]
                                
                                # Display RMSE comparison
                                st.subheader("Model Performance Comparison")
                                rmse_df = pd.DataFrame([(name, rmse) for name, rmse, _ in model_results], 
                                                     columns=['Model', 'RMSE'])
                                st.dataframe(rmse_df.style.highlight_min(subset=['RMSE'], color='lightgreen'))
                                # Plot best model forecast
                                st.subheader(f"Best Model Forecast: {best_model[0]} (RMSE: {best_model[1]:.2f})")
                                fig = go.Figure()
                                fig.add_trace(  # ✅ Correct method name
                                    go.Scatter(
                                        x=best_model[2]['DateKey'],
                                        y=best_model[2]['Forecast'],
                                        name='Forecast',
                                        line=dict(color='red')
                                    ))
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                                st.error(f"Error in forecasting: {str(e)}")
                                st.info("Try adjusting your parameters or check your data format")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please make sure your data is in the correct format")
else:
    st.info("Please upload a CSV file to begin analysis and forecasting.")
