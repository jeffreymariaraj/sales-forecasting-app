import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import json
from streamlit.components.v1 import declare_component
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Profit Forecasting & Analytics App",
    page_icon="📈",
    layout="wide"
)

# Add custom CSS with autocomplete styling
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
    .autocomplete {
        position: relative;
        display: inline-block;
    }
    .autocomplete-items {
        position: absolute;
        border: 1px solid #d4d4d4;
        border-bottom: none;
        border-top: none;
        z-index: 99;
        top: 100%;
        left: 0;
        right: 0;
    }
    .autocomplete-items div {
        padding: 10px;
        cursor: pointer;
        background-color: #fff;
        border-bottom: 1px solid #d4d4d4;
    }
    .autocomplete-items div:hover {
        background-color: #e9e9e9;
    }
    .autocomplete-active {
        background-color: DodgerBlue !important;
        color: #ffffff;
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
        # Read CSV and ensure string columns are properly handled
        df = pd.read_csv(file)
        # Convert object columns that should be numeric
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    # Try to convert to numeric, if fails keep as string
                    numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_conversion.isna().all():  # If conversion produced some valid numbers
                        df[col] = numeric_conversion
            except:
                continue
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None
def get_query_suggestions(query_text, column_names):
    """Generate query suggestions based on input text"""
    suggestions = []
    common_operators = [' > ', ' < ', ' >= ', ' <= ', ' == ', ' != ', ' and ', ' or ']
    
    # Add column name suggestions
    if not any(op in query_text for op in common_operators):
        suggestions.extend([col for col in column_names if query_text.lower() in col.lower()])
    
    # Add operator suggestions
    last_word = query_text.split()[-1] if query_text else ""
    if any(col.lower().startswith(last_word.lower()) for col in column_names):
        suggestions.extend(common_operators)
    
    return suggestions

def prepare_data(df, target_column):
    """
    Prepare the data for forecasting
    """
    try:
        # Convert date column to datetime
        df['DateKey'] = pd.to_datetime(df['DateKey'])
        df = df.dropna(subset=['DateKey'])
        
        # Group by date and sum target variable
        daily_data = df.groupby('DateKey', observed=True).agg({
            target_column: 'sum'
        }).reset_index()
        
        # Sort by date
        daily_data = daily_data.sort_values('DateKey')
        daily_data = daily_data.rename(columns={target_column: 'Target'})
        
        # Handle any missing values
        daily_data = daily_data.dropna()
        
        return daily_data
        
    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return None
def create_sequences(data, seq_length):
    """Create sequences for LSTM model"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def lstm_forecast(data, forecast_days, seq_length=30):
    """Generate forecast using LSTM model"""
    try:
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['Target']])
        
        # Create sequences for training
        X, y = create_sequences(scaled_data, seq_length)
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Reshape input for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Train model
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Generate forecast
        forecast = []
        current_sequence = scaled_data[-seq_length:]
        
        for _ in range(forecast_days):
            # Reshape current sequence for prediction
            current_sequence_reshaped = current_sequence.reshape((1, seq_length, 1))
            
            # Get prediction
            next_pred = model.predict(current_sequence_reshaped, verbose=0)
            
            # Append prediction to forecast
            forecast.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Inverse transform predictions
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        
        # Create forecast dates
        last_date = data['DateKey'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'DateKey': forecast_dates,
            'Forecast': forecast.flatten()
        })
        
        return forecast_df
        
    except Exception as e:
        st.error(f"Error in LSTM forecasting: {str(e)}")
        return None

def xgboost_forecast(data, forecast_days, seq_length=30):
    """Generate forecast using XGBoost model"""
    try:
        # Create features for XGBoost
        df = data.copy()
        
        # Add time-based features
        df['Year'] = df['DateKey'].dt.year
        df['Month'] = df['DateKey'].dt.month
        df['DayOfWeek'] = df['DateKey'].dt.dayofweek
        df['DayOfMonth'] = df['DateKey'].dt.day
        
        # Create lag features
        for i in range(1, seq_length + 1):
            df[f'lag_{i}'] = df['Target'].shift(i)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Prepare features and target
        features = ['Year', 'Month', 'DayOfWeek', 'DayOfMonth'] + [f'lag_{i}' for i in range(1, seq_length + 1)]
        X = df[features]
        y = df['Target']
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6
        )
        
        model.fit(X, y)
        
        # Generate forecast
        forecast = []
        current_data = df.iloc[-1:].copy()
        
        for _ in range(forecast_days):
            # Update time features for next day
            next_date = current_data['DateKey'].iloc[-1] + timedelta(days=1)
            new_row = pd.DataFrame({
                'DateKey': [next_date],
                'Year': [next_date.year],
                'Month': [next_date.month],
                'DayOfWeek': [next_date.dayofweek],
                'DayOfMonth': [next_date.day]
            })
            
            # Update lag features
            for i in range(1, seq_length + 1):
                new_row[f'lag_{i}'] = current_data['Target'].iloc[-1] if i == 1 else current_data[f'lag_{i-1}'].iloc[-1]
            
            # Make prediction
            pred = model.predict(new_row[features])
            forecast.append(pred[0])
            
            # Update current data for next iteration
            new_row['Target'] = pred[0]
            current_data = pd.concat([current_data, new_row]).iloc[1:].copy()
        
        # Create forecast dates
        last_date = data['DateKey'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'DateKey': forecast_dates,
            'Forecast': forecast
        })
        
        return forecast_df
        
    except Exception as e:
        st.error(f"Error in XGBoost forecasting: {str(e)}")
        return None

def prophet_forecast(data, forecast_days):
    """Generate forecast using Prophet model"""
    try:
        prophet_df = data.rename(columns={'DateKey': 'ds', 'Target': 'y'})
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.fit(prophet_df)
        future_dates = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future_dates)
        return forecast
    except Exception as e:
        st.error(f"Error in Prophet forecasting: {str(e)}")
        return None

def sarima_forecast(data, forecast_days):
    """Generate forecast using SARIMA model"""
    try:
        sarima_df = data.set_index('DateKey')['Target']
        model = SARIMAX(
            sarima_df,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )
        results = model.fit()
        forecast = results.get_forecast(steps=forecast_days)
        return forecast
    except Exception as e:
        st.error(f"Error in SARIMA forecasting: {str(e)}")
        return None

def plot_all_forecasts(data, prophet_forecast, sarima_forecast, lstm_forecast, xgb_forecast, target_column):
    """Plot all forecasts in a 2x2 grid"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Prophet Forecast", "SARIMA Forecast", "LSTM Forecast", "XGBoost Forecast"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Prophet Plot
    fig.add_trace(
        go.Scatter(x=data['DateKey'], y=data['Target'], name="Historical Data", mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=prophet_forecast['ds'],
            y=prophet_forecast['yhat'],
            name="Prophet Forecast",
            mode='lines',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # SARIMA Plot
    fig.add_trace(
        go.Scatter(
            x=data['DateKey'],
            y=data['Target'],
            name="Historical Data",
            mode='lines',
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=sarima_forecast.predicted_mean.index,
            y=sarima_forecast.predicted_mean,
            name="SARIMA Forecast",
            mode='lines',
            line=dict(color='green')
        ),
        row=1, col=2
    )
    
    # LSTM Plot
    fig.add_trace(
        go.Scatter(
            x=data['DateKey'],
            y=data['Target'],
            name="Historical Data",
            mode='lines',
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=lstm_forecast['DateKey'],
            y=lstm_forecast['Forecast'],
            name="LSTM Forecast",
            mode='lines',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    # XGBoost Plot
    fig.add_trace(
        go.Scatter(
            x=data['DateKey'],
            y=data['Target'],
            name="Historical Data",
            mode='lines',
            showlegend=False
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=xgb_forecast['DateKey'],
            y=xgb_forecast['Forecast'],
            name="XGBoost Forecast",
            mode='lines',
            line=dict(color='purple')
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=1000,
        title_text=f"{target_column} Forecast Comparison",
        showlegend=True
    )
    
    return fig

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    row_to_keep_inclusive = 376234

    # Keep rows up to and including the specified row using .iloc
    df = df.iloc[:row_to_keep_inclusive + 1]
    if df is not None:
        try:
            # Initialize filtered_df in session state if not already done
            if st.session_state.filtered_df is None:
                st.session_state.filtered_df = df.copy()
            # Create tabs for different sections
            tabs = st.tabs(["🔍 Query Analysis", "📊 Data Analysis", "🔮 Forecasting"])
            
            # Query Analysis Section
            with tabs[0]:
                st.subheader("Search Query")
                st.markdown("You can customize the query below:")

                # Get column names for suggestions
                column_names = df.columns.tolist()
                # Create query input box with suggestions
                query = st.text_area(
                    "Query",
                    height=200,
                    key="main_query",
                    help="Enter your query using Python's query syntax. Example: 'column_name > 100 and other_column < 50'"
                )
                
                # Show suggestions
                if query:
                    suggestions = get_query_suggestions(query, column_names)
                    if suggestions:
                        st.markdown("### Suggestions")
                        cols = st.columns(3)
                        for idx, suggestion in enumerate(suggestions):
                            with cols[idx % 3]:
                                if st.button(f"Insert: {suggestion}", key=f"suggest_{idx}"):
                                    # If suggestion is an operator, add spaces around it
                                    if any(op.strip() == suggestion.strip() for op in [' > ', ' < ', ' >= ', ' <= ', ' == ', ' != ', ' and ', ' or ']):
                                        new_query = query + f" {suggestion.strip()} "
                                    else:
                                        new_query = query + suggestion
                                    st.session_state.current_query = new_query
                                    st.rerun()
                
                # Add execute query button
                if st.button("Execute Query"):
                    try:
                        # Execute the query and show results
                        filtered_df = df.query(query)
                        st.write(f"Found {len(filtered_df)} matching rows")
                        st.dataframe(filtered_df.head(5))
                        table_html = filtered_df.to_html(classes=[
                                'table', 
                                'table-striped', 
                                'table-hover', 
                                'table-bordered'
                            ], escape=False)
                        popup_content = f"""
                            <div style="
                                padding: 20px;
                                background-color: white;
                                border-radius: 8px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            ">
                                <style>
                                    .table {{
                                        width: 100%;
                                        border-collapse: collapse;
                                        margin-bottom: 1rem;
                                        background-color: white;
                                    }}
                                    .table th,
                                    .table td {{
                                        padding: 12px;
                                        vertical-align: top;
                                        border: 1px solid #dee2e6;
                                    }}
                                    .table thead th {{
                                        vertical-align: bottom;
                                        background-color: #f8f9fa;
                                        border-bottom: 2px solid #dee2e6;
                                    }}
                                    .table-striped tbody tr:nth-of-type(odd) {{
                                        background-color: rgba(0,0,0,.05);
                                    }}
                                    .table-hover tbody tr:hover {{
                                        background-color: rgba(0,0,0,.075);
                                    }}
                                </style>
                                {table_html}
                            </div>
                        """
                            
                            # Create JavaScript for popup
                        js = f"""
                            <script>
                            function openPopup() {{
                                var w = window.open('', 'Query Results', 'width=800,height=600,scrollbars=yes');
                                w.document.write(`
                                    <html>
                                        <head>
                                            <title>Query Results</title>
                                            <style>
                                                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                                .table {{ border-collapse: collapse; width: 100%; }}
                                                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; }}
                                                .table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                                                .table th {{ 
                                                    padding-top: 12px;
                                                    padding-bottom: 12px;
                                                    text-align: left;
                                                    background-color: #4CAF50;
                                                    color: white;
                                                }}
                                            </style>
                                        </head>
                                        <body>
                                            <h2>Query Results ({len(filtered_df)} rows)</h2>
                                            {table_html}
                                            <button onclick="window.print()">Print</button>
                                            <button onclick="window.close()">Close</button>
                                        </body>
                                    </html>
                                `);
                                w.document.close();
                            }}
                            </script>
                            <button 
                                onclick="openPopup()" 
                                style="
                                    padding: 10px 20px;
                                    background-color: #4CAF50;
                                    color: white;
                                    border: none;
                                    border-radius: 4px;
                                    cursor: pointer;
                                    margin: 10px 0;
                                "
                            >
                                Open Results in Popup
                            </button>
                        """
                            
                        # Display the button that triggers the popup
                        st.components.v1.html(js, height=50)


                    except Exception as e:
                        st.error(f"Error executing query: {str(e)}")
                        st.info("Please check your query syntax and try again")
            
            # Data Analysis Section
            with tabs[1]:
                st.subheader("Data Analysis Dashboard")
                
                # Identify numeric and categorical columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                categorical_cols = [col for col in categorical_cols if col != 'DateKey']
                
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    # First row of controls
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        selected_category = st.selectbox(
                            "Select Category for Analysis",
                            options=numeric_cols
                        )
                    
                    with col2:
                        # Get unique values for selected category
                        category_values = ['All'] + sorted(df[selected_category].unique().tolist())
                        selected_value = st.selectbox(
                            f"Select {selected_category}",
                            options=category_values
                        )
                    
                    with col3:
                        selected_metric = st.selectbox(
                            "Select Metric to Analyze",
                            options=numeric_cols
                        )
                    
                    # Filter data based on selection
                    filtered_df = df.copy()
                    if selected_value != 'All':
                        filtered_df = filtered_df[filtered_df[selected_category] == selected_value]
                    
                    # Key Metrics Row
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
                    
                    # Visualizations
                    st.subheader("Visual Analysis")
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Bar chart of selected category vs metric
                        category_data = filtered_df.groupby(selected_category)[selected_metric].sum().reset_index()
                        fig = px.bar(
                            category_data,
                            x=selected_category,
                            y=selected_metric,
                            title=f'{selected_metric} by {selected_category}',
                            color=selected_metric,
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_col2:
                        # Time series trend
                        time_data = filtered_df.groupby('DateKey')[selected_metric].sum().reset_index()
                        fig = px.line(
                            time_data,
                            x='DateKey',
                            y=selected_metric,
                            title=f'{selected_metric} Trend Over Time'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("No suitable categorical or numeric columns found for analysis.")
            
            # Forecasting Section
            with tabs[2]:
                # Initialize session state for query text
                if "forecast_query" not in st.session_state:
                    st.session_state.forecast_query = ""
                st.subheader("Forecasting Dashboard")
                
                # Create query input box for forecasting with suggestions
                forecast_query = st.text_area(
                    "Forecasting Filters",
                    height=200,
                    key="forecast_query",
                    help="Enter your query to filter data before forecasting. Example: 'column_name > 100'"
                )

                # Show suggestions for forecasting query
                query_text = st.session_state.forecast_query
                if query_text:
                    suggestions = get_query_suggestions(query_text, column_names)
                    if suggestions:
                        selected_suggestion = st.selectbox(
                            "Suggestions",
                            options=[""] + suggestions,  # Add an empty option as default
                            key="forecast_suggestions"
                            )
                        if selected_suggestion:
                            if st.button("Insert Suggestion", key="forecast_suggestions"):
                # Append the selected suggestion to the query
                                st.session_state.forecast_query += selected_suggestion
                                st.experimental_rerun()
                
                # Add filter button
                if st.button("Apply Filters"):
                    try:
                        if forecast_query:
                            st.session_state.filtered_df = df.query(forecast_query)
                            st.success("Filters applied successfully!")
                        else:
                            st.session_state.filtered_df = df.copy()
                    except Exception as e:
                        st.error(f"Error applying filters: {str(e)}")
                        st.info("Please check your query syntax and try again")
                        st.session_state.filtered_df = df.copy()
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(st.session_state.filtered_df.head())
                
                # Forecasting controls
                st.sidebar.markdown("---")
                st.sidebar.header("Forecasting Parameters")
                
                # Select target variable for forecasting
                numeric_cols = st.session_state.filtered_df.select_dtypes(include=['int64', 'float64']).columns
                target_column = st.sidebar.selectbox(
                    "Select Variable to Forecast",
                    options=numeric_cols
                )
                
                forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)
                
                # Prepare data with error handling
                daily_data = prepare_data(st.session_state.filtered_df, target_column)
                
                if daily_data is not None:
                    
                    if st.sidebar.button("Generate Forecast"):
                        with st.spinner("Generating forecasts..."):
                            try:
                # Generate forecasts using filtered data
                                prophet_results = prophet_forecast(daily_data, forecast_days)
                                sarima_results = sarima_forecast(daily_data, forecast_days)
                                lstm_results = lstm_forecast(daily_data, forecast_days)
                                xgb_results = xgboost_forecast(daily_data, forecast_days)
                
                # Plot results
                                fig = plot_all_forecasts(
                                    daily_data,
                                    prophet_results,
                                    sarima_results,
                                    lstm_results,
                                    xgb_results,
                                    target_column
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                # Download section
                                st.subheader("Download Forecasts")
                                col1, col2, col3, col4 = st.columns(4)
                
                                with col1:
                                    prophet_csv = prophet_results[['ds', 'yhat']].to_csv(index=False)
                                    st.download_button(
                        label="Download Prophet Forecast",
                        data=prophet_csv,
                        file_name=f"prophet_forecast_{target_column}.csv",
                        mime="text/csv"
                                    )
                
                                with col2:
                                    sarima_csv = pd.DataFrame({
                                        'date': sarima_results.predicted_mean.index,
                                        'forecast': sarima_results.predicted_mean
                                    }).to_csv(index=False)
                                    st.download_button(
                                        label="Download SARIMA Forecast",
                                        data=sarima_csv,
                                        file_name=f"sarima_forecast_{target_column}.csv",
                                        mime="text/csv"
                                    )
                
                                with col3:
                                    lstm_csv = lstm_results.to_csv(index=False)
                                    st.download_button(
                                        label="Download LSTM Forecast",
                                        data=lstm_csv,
                                        file_name=f"lstm_forecast_{target_column}.csv",
                                        mime="text/csv"
                                    )
                
                                with col4:
                                    xgb_csv = xgb_results.to_csv(index=False)
                                    st.download_button(
                                        label="Download XGBoost Forecast",
                                        data=xgb_csv,
                                        file_name=f"xgb_forecast_{target_column}.csv",
                                        mime="text/csv"
                                    )
                
                            except Exception as e:
                                st.error(f"Error in forecasting: {str(e)}")
                                st.info("Try adjusting your parameters or check your data format")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please make sure your data is in the correct format")
else:
    st.info("Please upload a CSV file to begin analysis and forecasting.")
