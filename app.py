import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
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

def plot_forecasts(data, prophet_forecast, sarima_forecast, target_column):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Prophet Forecast", "SARIMA Forecast"),
        vertical_spacing=0.15
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
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=sarima_forecast.predicted_mean.index,
            y=sarima_forecast.predicted_mean,
            name="SARIMA Forecast",
            mode='lines',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text=f"{target_column} Forecast Comparison"
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
                        selected_suggestion = st.selectbox(
                            "Suggestions",
                            options=suggestions,
                            key="query_suggestions"
                        )
                        if selected_suggestion:
                            if st.button("Insert Suggestion"):
                                query += selected_suggestion
                
                # Add execute query button
                if st.button("Execute Query"):
                    try:
                        # Execute the query and show results
                        filtered_df = df.query(query)
                        st.write(f"Found {len(filtered_df)} matching rows")
                        st.dataframe(filtered_df)
                        
                        # Add download button for filtered results
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="Download Filtered Data",
                            data=csv,
                            file_name="filtered_data.csv",
                            mime="text/csv"
                        )
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
                st.subheader("Forecasting Dashboard")
                
                # Create query input box for forecasting with suggestions
                forecast_query = st.text_area(
                    "Forecasting Filters",
                    height=200,
                    key="forecast_query",
                    help="Enter your query to filter data before forecasting. Example: 'column_name > 100'"
                )
                
                # Show suggestions for forecasting query
                if forecast_query:
                    suggestions = get_query_suggestions(forecast_query, column_names)
                    if suggestions:
                        selected_suggestion = st.selectbox(
                            "Suggestions",
                            options=suggestions,
                            key="forecast_suggestions"
                        )
                        if selected_suggestion:
                            if st.button("Insert Suggestion", key="forecast_suggest_button"):
                                forecast_query += selected_suggestion
                
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
                                
                                # Plot results
                                fig = plot_forecasts(daily_data, prophet_results, sarima_results, target_column)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Download section
                                st.subheader("Download Forecasts")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.download_button(
                                        label="Download Prophet Forecast",
                                        data=prophet_forecast_df.to_csv(index=False),
                                        file_name=f"prophet_forecast_{target_column}.csv",
                                        mime="text/csv"
                                    )
                                
                                with col2:
                                    st.download_button(
                                        label="Download SARIMA Forecast",
                                        data=sarima_forecast_df.to_csv(index=False),
                                        file_name=f"sarima_forecast_{target_column}.csv",
                                        mime="text/csv"
                                    )
                            
                            except Exception as e:
                                st.error(f"Error in forecasting: {str(e)}")
                                st.info("Try adjusting your parameters or check your data format")
                else:
                    st.error("Error in data preparation. Please check your data format.")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please make sure your data is in the correct format")
else:
    st.info("Please upload a CSV file to begin analysis and forecasting.")