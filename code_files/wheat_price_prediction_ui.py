import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Wheat Price Forecaster",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API endpoint
API_ENDPOINT = st.sidebar.text_input("API Endpoint", "http://localhost:8000")

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.2rem;
        color: #3498db;
        margin-top: 1rem;
        font-weight: 500;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-tile {
        background-color: #f1f8ff;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2980b9;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    .highlight {
        background-color: #f9f3d6;
        padding: 5px;
        border-radius: 5px;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #7f8c8d;
        font-size: 0.8rem;
    }
    .plotly-graph {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'api_status' not in st.session_state:
    st.session_state.api_status = None
if 'available_states' not in st.session_state:
    st.session_state.available_states = []
if 'current_state_info' not in st.session_state:
    st.session_state.current_state_info = None
if 'forecast_result' not in st.session_state:
    st.session_state.forecast_result = None
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {}

# Main header
st.markdown('<div class="main-header">🌾 Wheat Price Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predicting wheat prices across India using machine learning</div>', unsafe_allow_html=True)

# Functions to interact with the API
def initialize_api(model_path):
    """Initialize the API with the model file path"""
    endpoint = f"{API_ENDPOINT}/initialize"
    params = {
        "model_path": model_path
    }
    
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        result = response.json()
        st.session_state.initialized = True
        return result
    except Exception as e:
        st.error(f"Error initializing API: {str(e)}")
        st.session_state.initialized = False
        return None

def get_api_status():
    """Get API status"""
    try:
        response = requests.get(f"{API_ENDPOINT}/status")
        response.raise_for_status()
        return response.json()
    except:
        return {"status": "error", "message": "Could not connect to API"}

def get_available_states():
    """Get list of available states"""
    try:
        response = requests.get(f"{API_ENDPOINT}/states")
        response.raise_for_status()
        return response.json()["states"]
    except:
        return []

def get_state_info(state):
    """Get detailed information about a state"""
    try:
        response = requests.get(f"{API_ENDPOINT}/state_info/{state}")
        response.raise_for_status()
        return response.json()
    except:
        return None

def get_forecast(state, features, months):
    """Get forecast for a state"""
    try:
        payload = {
            "state": state,
            "features": features,
            "months": months
        }
        response = requests.post(f"{API_ENDPOINT}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching forecast: {str(e)}")
        return None

# Sidebar - Settings and Setup
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3c/Wheat_field.jpg", use_column_width=True)
    st.markdown("### API Configuration")
    
    # Model Path Configuration
    with st.expander("Model Configuration", expanded=not st.session_state.initialized):
        model_path = st.text_input("Model File Path", "model/all_states_best_model_latest.pkl")
        
        if st.button("Initialize API"):
            with st.spinner("Setting up API connection..."):
                result = initialize_api(model_path)
                if result:
                    st.success("API initialized successfully!")
                    st.session_state.api_status = get_api_status()
                    st.session_state.available_states = get_available_states()

    # API Status
    st.markdown("### API Status")
    if st.button("Refresh Status"):
        st.session_state.api_status = get_api_status()
    
    if st.session_state.api_status:
        status = st.session_state.api_status
        if status["status"] == "ready":
            st.markdown(f"""
            <div class="success-message">
                API is ready<br>
                Models: {status.get('models_loaded', 'N/A')}<br>
                States: {status.get('available_states', 'N/A')}<br>
                Features: {status.get('feature_count', 'N/A')}<br>
                Created: {status.get('creation_date', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="error-message">
                API is not ready<br>
                Status: {status.get('status', 'unknown')}<br>
                Message: {status.get('message', 'No message')}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("API status unknown. Please initialize or refresh status.")
    
    # About section
    st.markdown("### About")
    st.markdown("""
    This application uses machine learning models to forecast wheat prices across different states in India. 
    The models are trained on historical data and loaded from a local pickle file.
    
    **Model Types:**
    - Random Forest
    - XGBoost
    - Holt-Winters
    
    Made with ❤️ using Streamlit and FastAPI
    """)

# Only show main content if API is initialized
if not st.session_state.initialized:
    st.warning("Please initialize the API with a model file in the sidebar first")
else:
    # Main app tabs
    tab1, tab2, tab3 = st.tabs(["Forecast", "State Analysis", "Model Insights"])
    
    with tab1:
        st.markdown('<div class="section-header">Generate Wheat Price Forecasts</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # State selection and parameters
            selected_state = st.selectbox(
                "Select State", 
                options=st.session_state.available_states,
                index=0 if st.session_state.available_states else None
            )
            
            forecast_months = st.slider("Forecast Months", 1, 36, 12)
            
            st.markdown("### Advanced Features")
            st.markdown("Optionally override default feature values:")
            
            if selected_state and selected_state != st.session_state.get('last_selected_state'):
                # Get state info when selection changes
                state_info = get_state_info(selected_state)
                st.session_state.current_state_info = state_info
                st.session_state.last_selected_state = selected_state
                # Reset feature values
                st.session_state.feature_values = {}
            
            # Feature customization
            if st.session_state.current_state_info:
                with st.expander("Customize Features"):
                    # Group features into categories
                    economic_features = [f for f in st.session_state.current_state_info["features"] 
                                         if f["name"] in ["MSP_Wheat_KG", "CPI", "diesel_price", "Rainfall"]]
                    
                    # Show only the most important features or those with clear descriptions
                    if economic_features:
                        st.markdown("##### Economic Indicators")
                        for feature in economic_features:
                            current_value = st.session_state.feature_values.get(feature["name"], feature["average_value"])
                            new_value = st.number_input(
                                f"{feature['name']} ({feature['description']})",
                                value=float(current_value),
                                step=0.1
                            )
                            st.session_state.feature_values[feature["name"]] = new_value
            
            # Execute forecast button
            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    forecast_result = get_forecast(
                        selected_state, 
                        st.session_state.feature_values, 
                        forecast_months
                    )
                    if forecast_result:
                        st.session_state.forecast_result = forecast_result
                        st.success("Forecast generated successfully!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Display forecast results
            if st.session_state.forecast_result:
                result = st.session_state.forecast_result
                
                st.markdown(f'<div class="section-header">Forecast for {result["state"]}</div>', unsafe_allow_html=True)
                
                # Create a DataFrame for the forecast
                forecast_df = pd.DataFrame(result["forecasts"])
                forecast_df["date"] = pd.to_datetime(forecast_df["date"])
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-tile">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{result["model_type"].title()}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Model Type</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-tile">', unsafe_allow_html=True)
                    avg_price = forecast_df["forecast_price"].mean()
                    st.markdown(f'<div class="metric-value">₹{avg_price:.2f}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Average Price (per kg)</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-tile">', unsafe_allow_html=True)
                    price_change = ((forecast_df["forecast_price"].iloc[-1] / forecast_df["forecast_price"].iloc[0]) - 1) * 100
                    change_color = "green" if price_change >= 0 else "red"
                    st.markdown(f'<div class="metric-value" style="color:{change_color}">{price_change:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Projected Change</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Create and display the forecast chart
                st.markdown('<div class="plotly-graph">', unsafe_allow_html=True)
                fig = px.line(
                    forecast_df, 
                    x="date", 
                    y="forecast_price",
                    labels={"date": "Date", "forecast_price": "Wheat Price (₹ per kg)"},
                    title=f"Wheat Price Forecast for {result['state']}",
                )
                
                # Improve the chart appearance
                fig.update_layout(
                    plot_bgcolor="white",
                    font=dict(family="Arial, sans-serif", size=14),
                    title=dict(font=dict(size=24, color="#2c3e50")),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=60, b=40),
                    hovermode="x unified",
                )
                
                fig.update_traces(
                    line=dict(width=3, color="#3498db"),
                    hovertemplate="<b>Date:</b> %{x|%B %Y}<br><b>Price:</b> ₹%{y:.2f} per kg<extra></extra>"
                )
                
                # Add a trend area
                fig.add_traces(
                    go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["forecast_price"],
                        fill='tozeroy',
                        fillcolor='rgba(52, 152, 219, 0.1)',
                        line=dict(color='rgba(0,0,0,0)'),
                        showlegend=False,
                        hoverinfo='none'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display forecast data table
                with st.expander("View Forecast Data"):
                    display_df = forecast_df.copy()
                    display_df["date"] = display_df["date"].dt.strftime("%B %Y")
                    display_df.columns = ["Month", "Price (₹ per kg)"]
                    st.dataframe(display_df, use_container_width=True)
                
                # Download forecast as CSV
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv,
                    file_name=f"wheat_forecast_{result['state'].lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                )
    
    with tab2:
        st.markdown('<div class="section-header">State-wise Analysis</div>', unsafe_allow_html=True)
        
        # State selection
        selected_analysis_state = st.selectbox(
            "Select State for Analysis", 
            options=st.session_state.available_states,
            index=0 if st.session_state.available_states else None,
            key="analysis_state"
        )
        
        if selected_analysis_state:
            # Get state info
            if selected_analysis_state != st.session_state.get('last_analysis_state'):
                with st.spinner("Loading state data..."):
                    state_info = get_state_info(selected_analysis_state)
                    st.session_state.analysis_state_info = state_info
                    st.session_state.last_analysis_state = selected_analysis_state
            
            if st.session_state.get('analysis_state_info'):
                state_info = st.session_state.analysis_state_info
                
                # Display state information
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"### {state_info['state']} Overview")
                    
                    st.markdown(f"**Best Model:** {state_info['best_model'].title()}")
                    
                    # Display metrics
                    if 'metrics' in state_info:
                        st.markdown("#### Model Performance")
                        metrics = state_info['metrics']
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("RMSE", f"{metrics.get('test_rmse', 'N/A'):.4f}")
                            st.metric("MAE", f"{metrics.get('test_mae', 'N/A'):.4f}")
                        with col_b:
                            st.metric("MAPE", f"{metrics.get('test_mape', 'N/A'):.2f}%")
                            st.metric("R²", f"{metrics.get('test_r2', 'N/A'):.4f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    # Feature importance visualization
                    if 'features' in state_info:
                        features = state_info['features']
                        
                        # Filter features that have importance values
                        important_features = [f for f in features if f.get('importance') is not None]
                        
                        if important_features:
                            # Sort by importance
                            important_features.sort(key=lambda x: x.get('importance', 0), reverse=True)
                            
                            # Take top 10
                            top_features = important_features[:10]
                            
                            # Create dataframe for visualization
                            imp_df = pd.DataFrame({
                                'Feature': [f['name'] for f in top_features],
                                'Importance': [f['importance'] for f in top_features],
                                'Description': [f['description'] for f in top_features]
                            })
                            
                            # Create horizontal bar chart
                            st.markdown('<div class="plotly-graph">', unsafe_allow_html=True)
                            fig = px.bar(
                                imp_df,
                                y='Feature',
                                x='Importance',
                                title=f"Feature Importance for {state_info['state']}",
                                orientation='h',
                                color='Importance',
                                color_continuous_scale='Viridis',
                                hover_data=['Description']
                            )
                            
                            fig.update_layout(
                                yaxis={'categoryorder':'total ascending'},
                                plot_bgcolor="white",
                                font=dict(family="Arial, sans-serif", size=14),
                                title=dict(font=dict(size=20, color="#2c3e50")),
                                coloraxis_showscale=False,
                                margin=dict(l=20, r=20, t=60, b=20),
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info("No feature importance data available for this state's model.")
                
                # Feature details
                st.markdown("### Key Features Explained")
                if 'features' in state_info:
                    features = state_info['features']
                    
                    # Group features into meaningful categories
                    categories = {
                        "Economic Indicators": ["MSP_Wheat_KG", "CPI", "diesel_price", "Diesel ROC", "Wheat ROC"],
                        "Environmental Factors": ["Rainfall"],
                        "Time Components": ["year", "month_num", "quarter", "month_sin", "month_cos"],
                        "Historical Patterns": ["lag_1m", "lag_3m", "lag_6m", "lag_12m", "rolling_mean_3m", "rolling_mean_6m"],
                        "Volatility Measures": ["rolling_std_3m", "rolling_std_6m", "roc_1m", "roc_3m"]
                    }
                    
                    # Display features by category in expandable sections
                    for category, feature_names in categories.items():
                        category_features = [f for f in features if f['name'] in feature_names]
                        
                        if category_features:
                            with st.expander(category, expanded=category=="Economic Indicators"):
                                for feature in category_features:
                                    importance = feature.get('importance')
                                    importance_str = f" (Importance: {importance:.4f})" if importance is not None else ""
                                    
                                    st.markdown(f"**{feature['name']}**{importance_str}")
                                    st.markdown(f"{feature['description']}")
                                    st.markdown("---")
    
    with tab3:
        st.markdown('<div class="section-header">Model Insights & Comparisons</div>', unsafe_allow_html=True)
        
        # Get all available states that have forecasts
        if st.session_state.api_status and st.session_state.api_status.get("status") == "ready":
            states = st.session_state.available_states
            
            if not states:
                st.warning("No states available for comparison")
            else:
                # Multi-state comparison section
                st.markdown("### Compare States")
                
                # State selection
                selected_states = st.multiselect(
                    "Select States to Compare",
                    options=states,
                    default=[states[0]] if states else []
                )
                
                if selected_states:
                    # Collect data for selected states
                    comparison_data = []
                    
                    with st.spinner("Gathering state data..."):
                        for state in selected_states:
                            state_info = get_state_info(state)
                            if state_info and 'metrics' in state_info:
                                model_type = state_info['best_model']
                                metrics = state_info['metrics']
                                
                                comparison_data.append({
                                    'State': state,
                                    'Model': model_type.title(),
                                    'RMSE': metrics.get('test_rmse', None),
                                    'MAE': metrics.get('test_mae', None),
                                    'MAPE': metrics.get('test_mape', None),
                                    'R²': metrics.get('test_r2', None)
                                })
                    
                    if comparison_data:
                        # Create DataFrame
                        comp_df = pd.DataFrame(comparison_data)
                        
                        # Create visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Model type distribution
                            model_counts = comp_df['Model'].value_counts().reset_index()
                            model_counts.columns = ['Model', 'Count']
                            
                            st.markdown('<div class="plotly-graph">', unsafe_allow_html=True)
                            fig = px.pie(
                                model_counts,
                                values='Count',
                                names='Model',
                                title='Best Model Distribution',
                                hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            
                            fig.update_layout(
                                font=dict(family="Arial, sans-serif", size=14),
                                title=dict(font=dict(size=18, color="#2c3e50")),
                                margin=dict(l=20, r=20, t=60, b=20),
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            # Error metrics comparison
                            st.markdown('<div class="plotly-graph">', unsafe_allow_html=True)
                            fig = px.bar(
                                comp_df,
                                x='State',
                                y='RMSE',
                                color='Model',
                                title='Model Error Comparison (RMSE)',
                                color_discrete_sequence=px.colors.qualitative.Safe
                            )
                            
                            fig.update_layout(
                                plot_bgcolor="white",
                                font=dict(family="Arial, sans-serif", size=14),
                                title=dict(font=dict(size=18, color="#2c3e50")),
                                margin=dict(l=20, r=20, t=60, b=20),
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # R² comparison
                        st.markdown('<div class="plotly-graph">', unsafe_allow_html=True)
                        fig = px.bar(
                            comp_df,
                            x='State',
                            y='R²',
                            color='Model',
                            title='Model Performance Comparison (R²)',
                            color_discrete_sequence=px.colors.qualitative.Vivid
                        )
                        
                        fig.update_layout(
                            plot_bgcolor="white",
                            font=dict(family="Arial, sans-serif", size=14),
                            title=dict(font=dict(size=18, color="#2c3e50")),
                            margin=dict(l=20, r=20, t=60, b=20),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display comparison table
                        st.markdown("### Detailed Metrics")
                        st.dataframe(comp_df, use_container_width=True)
                
                # Generate comparative forecasts
                st.markdown("### Multi-State Forecast Comparison")
                
                forecast_states = st.multiselect(
                    "Select States for Forecast Comparison",
                    options=states,
                    default=[states[0]] if states else [],
                    key="forecast_comparison_states"
                )
                
                forecast_months = st.slider(
                    "Forecast Months", 
                    1, 36, 12,
                    key="comparison_months"
                )
                
                if forecast_states and st.button("Generate Comparative Forecasts"):
                    with st.spinner("Generating forecasts for multiple states..."):
                        all_forecasts = []
                        
                        for state in forecast_states:
                            forecast = get_forecast(state, {}, forecast_months)
                            if forecast:
                                # Extract forecasts and add state info
                                for entry in forecast["forecasts"]:
                                    entry["state"] = forecast["state"]
                                    entry["model_type"] = forecast["model_type"]
                                    all_forecasts.append(entry)
                        
                        if all_forecasts:
                            # Create DataFrame with all forecasts
                            forecast_df = pd.DataFrame(all_forecasts)
                            forecast_df["date"] = pd.to_datetime(forecast_df["date"])
                            
                            # Create comparative visualization
                            st.markdown('<div class="plotly-graph">', unsafe_allow_html=True)
                            fig = px.line(
                                forecast_df,
                                x="date",
                                y="forecast_price",
                                color="state",
                                labels={"date": "Date", "forecast_price": "Wheat Price (₹ per kg)", "state": "State"},
                                title="Multi-State Wheat Price Forecast Comparison",
                                hover_data=["model_type"]
                            )
                            
                            fig.update_layout(
                                plot_bgcolor="white",
                                font=dict(family="Arial, sans-serif", size=14),
                                title=dict(font=dict(size=24, color="#2c3e50")),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(l=40, r=40, t=60, b=40),
                                hovermode="x unified",
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # State with highest predicted prices
                            pivoted_df = forecast_df.pivot_table(
                                index="date", 
                                columns="state", 
                                values="forecast_price"
                            )
                            
                            avg_prices = pivoted_df.mean().sort_values(ascending=False)
                            highest_state = avg_prices.index[0]
                            highest_price = avg_prices.iloc[0]
                            
                            st.markdown(f"### Key Insights")
                            st.markdown(f"- **{highest_state}** is predicted to have the highest average wheat prices (₹{highest_price:.2f}/kg)")
                            
                            # Price gap analysis
                            if len(avg_prices) > 1:
                                lowest_state = avg_prices.index[-1]
                                lowest_price = avg_prices.iloc[-1]
                                price_gap = highest_price - lowest_price
                                price_gap_pct = (price_gap / lowest_price) * 100
                                
                                st.markdown(f"- Price gap between highest ({highest_state}) and lowest ({lowest_state}) is ₹{price_gap:.2f}/kg ({price_gap_pct:.1f}%)")
                            
                            # Trend analysis for each state
                            st.markdown("- **Price Trends:**")
                            for state in pivoted_df.columns:
                                first_price = pivoted_df[state].iloc[0]
                                last_price = pivoted_df[state].iloc[-1]
                                change_pct = ((last_price / first_price) - 1) * 100
                                trend = "increase" if change_pct > 0 else "decrease"
                                
                                st.markdown(f"  - **{state}**: {abs(change_pct):.1f}% {trend} over the forecast period")
                            
                            # Download option
                            csv = forecast_df.to_csv(index=False)
                            st.download_button(
                                label="Download Comparative Forecast CSV",
                                data=csv,
                                file_name="wheat_forecast_comparison.csv",
                                mime="text/csv",
                            )

# Footer
st.markdown('<div class="footer">© 2025 Wheat Price Forecaster | Powered by FastAPI and Streamlit</div>', unsafe_allow_html=True)
                