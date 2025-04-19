from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

app = FastAPI(
    title="Wheat Price Prediction API",
    description="API for predicting wheat prices in different states using trained models stored locally",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables to store model data
models = None
metadata = None
available_states = None
is_initialized = False

# Pydantic models for API requests and responses
class PredictionInput(BaseModel):
    state: str = Field(..., description="State name for prediction")
    features: Dict[str, float] = Field({}, description="Optional feature values to override defaults")
    months: int = Field(12, description="Number of months to forecast", ge=1, le=36)

class PredictionResult(BaseModel):
    state: str
    forecasts: List[Dict[str, Any]]
    model_type: str
    metrics: Dict[str, float]

class FeatureInfo(BaseModel):
    name: str
    description: str
    average_value: float
    importance: Optional[float] = None

class StateInfo(BaseModel):
    state: str
    best_model: str
    metrics: Dict[str, float]
    features: List[FeatureInfo]

class StatesList(BaseModel):
    states: List[str]

# Feature descriptions for documentation
FEATURE_DESCRIPTIONS = {
    'MSP_Wheat_KG': 'Minimum Support Price for wheat per KG',
    'CPI': 'Consumer Price Index',
    'diesel_price': 'Price of diesel per liter',
    'Diesel ROC': 'Rate of change in diesel prices',
    'Wheat ROC': 'Rate of change in wheat prices',
    'Diesel / Wheat Price Ratio': 'Ratio of diesel price to wheat price',
    'Rainfall': 'Average rainfall in mm',
    'year': 'Year of prediction',
    'month_num': 'Month number (1-12)',
    'quarter': 'Quarter of the year (1-4)',
    'month_sin': 'Sine transform of month for cyclical features',
    'month_cos': 'Cosine transform of month for cyclical features',
    'quarter_sin': 'Sine transform of quarter for cyclical features',
    'quarter_cos': 'Cosine transform of quarter for cyclical features',
    'lag_1m': 'Wheat price 1 month ago',
    'lag_3m': 'Wheat price 3 months ago',
    'lag_6m': 'Wheat price 6 months ago',
    'lag_12m': 'Wheat price 12 months ago',
    'rolling_mean_3m': '3-month rolling average of wheat prices',
    'rolling_mean_6m': '6-month rolling average of wheat prices',
    'rolling_std_3m': '3-month rolling standard deviation of wheat prices',
    'rolling_std_6m': '6-month rolling standard deviation of wheat prices',
    'roc_1m': '1-month rate of change in wheat prices',
    'roc_3m': '3-month rate of change in wheat prices',
    'MSP_to_retail_ratio': 'Ratio of MSP to retail price of wheat'
}

def check_initialization():
    """Check if the API is initialized with models"""
    if not is_initialized:
        raise HTTPException(status_code=400, detail="API not initialized. Please call /initialize endpoint first.")
    return True

def load_models_from_file(file_path):
    """Load models from local pickle file"""
    global models, metadata, available_states, is_initialized
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise ValueError(f"Model file not found at: {file_path}")
        
        # Load the model
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
            
        # Extract components
        models = model_data['models']
        metadata = model_data['metadata']
        available_states = list(models.keys())
        is_initialized = True
        
        return {
            'models_loaded': len(models),
            'available_states': available_states,
            'feature_count': len(metadata['feature_cols'])
        }
        
    except Exception as e:
        raise ValueError(f"Error loading models: {str(e)}")

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "api": "Wheat Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "initialized": is_initialized,
        "endpoints": [
            {"path": "/initialize", "method": "GET", "description": "Initialize the API with model file"},
            {"path": "/status", "method": "GET", "description": "Get API status and loaded models information"},
            {"path": "/states", "method": "GET", "description": "Get list of available states for prediction"},
            {"path": "/state_info/{state}", "method": "GET", "description": "Get detailed information about a state"},
            {"path": "/predict", "method": "POST", "description": "Generate price forecasts for a state"}
        ]
    }

@app.get("/initialize", tags=["Configuration"])
async def initialize(model_path: str = "model/all_states_best_model_latest.pkl"):
    """
    Initialize the API with the model file
    
    - **model_path**: Path to the model pickle file (default: 'model/all_states_best_model_latest.pkl')
    """
    try:
        result = load_models_from_file(model_path)
        return {
            "status": "success",
            "message": f"Successfully loaded {result['models_loaded']} models with {result['feature_count']} features",
            "loaded": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", tags=["Info"])
async def status():
    """Get API status including loaded models information"""
    global models, metadata, available_states, is_initialized
    
    if not is_initialized:
        return {
            "status": "not_initialized",
            "message": "API not initialized. Please call /initialize endpoint first."
        }
    
    return {
        "status": "ready",
        "models_loaded": len(models),
        "available_states": len(available_states),
        "feature_count": len(metadata['feature_cols']),
        "creation_date": metadata.get('creation_date', 'unknown')
    }

@app.get("/states", response_model=StatesList, tags=["States"])
async def get_states(_=Depends(check_initialization)):
    """Get list of available states for prediction"""
    global available_states
    
    return {"states": [state.title() for state in available_states]}

@app.get("/state_info/{state}", response_model=StateInfo, tags=["States"])
async def get_state_info(state: str, _=Depends(check_initialization)):
    """
    Get detailed information about a state's model and features
    
    - **state**: State name to get information for
    """
    global models, metadata, available_states
    
    state = state.lower()
    if state not in available_states:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found. Available states: {', '.join(available_states)}")
    
    # Get model details
    model_details = metadata['model_details'][state]
    best_model = model_details['best_model']
    metrics = model_details['metrics']
    
    # Get feature importance if available
    feature_importance = {}
    if model_details.get('feature_importance'):
        for feature_data in model_details['feature_importance'].get('records', []):
            feature_importance[feature_data['Feature']] = feature_data['Importance']
    
    # Create feature info objects
    features = []
    for feature in metadata['feature_cols']:
        # Calculate an average value (this would normally come from your data analysis)
        avg_value = 0.0
        if feature == 'year':
            avg_value = datetime.now().year
        elif feature == 'month_num':
            avg_value = datetime.now().month
        elif feature == 'quarter':
            avg_value = (datetime.now().month - 1) // 3 + 1
        
        features.append(FeatureInfo(
            name=feature,
            description=FEATURE_DESCRIPTIONS.get(feature, "No description available"),
            average_value=avg_value,
            importance=feature_importance.get(feature)
        ))
    
    return StateInfo(
        state=state.title(),
        best_model=best_model,
        metrics=metrics,
        features=features
    )

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(prediction_input: PredictionInput, _=Depends(check_initialization)):
    """
    Generate price forecasts for a specific state
    
    - **state**: State name to predict for
    - **features**: Dictionary of feature values to override defaults
    - **months**: Number of months to forecast (1-36)
    """
    global models, metadata, available_states
    
    state = prediction_input.state.lower()
    if state not in available_states:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found. Available states: {', '.join(available_states)}")
    
    # Get the model and model type
    model = models[state]
    model_type = metadata['model_details'][state]['best_model']
    
    # Generate forecast
    try:
        # Initialize forecast dates
        forecast_dates = [datetime.now() + timedelta(days=30*i) for i in range(prediction_input.months)]
        
        forecasts = []
        
        if model_type == 'holt_winters':
            # For Holt-Winters, we can directly forecast
            predictions = model.forecast(steps=prediction_input.months)
            
            # Convert to list of dictionaries
            for i, date in enumerate(forecast_dates):
                forecasts.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'forecast_price': float(predictions[i]) if i < len(predictions) else None
                })
        else:
            # For RF and XGBoost, we need to build features for each forecast period
            current_features = {}
            
            # Initialize with user-provided features
            current_features.update(prediction_input.features)
            
            # Add default values for missing features
            for feature in metadata['feature_cols']:
                if feature not in current_features:
                    if feature == 'year':
                        current_features[feature] = datetime.now().year
                    elif feature == 'month_num':
                        current_features[feature] = datetime.now().month
                    elif feature == 'quarter':
                        current_features[feature] = (datetime.now().month - 1) // 3 + 1
                    elif feature == 'month_sin':
                        current_features[feature] = np.sin(2 * np.pi * datetime.now().month / 12)
                    elif feature == 'month_cos':
                        current_features[feature] = np.cos(2 * np.pi * datetime.now().month / 12)
                    elif feature == 'quarter_sin':
                        quarter = (datetime.now().month - 1) // 3 + 1
                        current_features[feature] = np.sin(2 * np.pi * quarter / 4)
                    elif feature == 'quarter_cos':
                        quarter = (datetime.now().month - 1) // 3 + 1
                        current_features[feature] = np.cos(2 * np.pi * quarter / 4)
                    else:
                        # For other features, use zero as default
                        current_features[feature] = 0.0
            
            # Generate predictions month by month
            for i, date in enumerate(forecast_dates):
                # Update time features for this month
                current_features['year'] = date.year
                current_features['month_num'] = date.month
                current_features['quarter'] = (date.month - 1) // 3 + 1
                current_features['month_sin'] = np.sin(2 * np.pi * date.month / 12)
                current_features['month_cos'] = np.cos(2 * np.pi * date.month / 12)
                current_features['quarter_sin'] = np.sin(2 * np.pi * current_features['quarter'] / 4)
                current_features['quarter_cos'] = np.cos(2 * np.pi * current_features['quarter'] / 4)
                
                # Create a DataFrame for prediction
                X = pd.DataFrame([{f: current_features.get(f, 0) for f in metadata['feature_cols']}])
                
                # Make prediction
                prediction = model.predict(X)[0]
                
                # Add to forecasts
                forecasts.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'forecast_price': float(prediction)
                })
                
                # Update lag features for next prediction if possible
                if i < prediction_input.months - 1:  # Not needed for the last prediction
                    current_features['lag_1m'] = prediction
                    if i >= 2:
                        current_features['lag_3m'] = forecasts[i-2]['forecast_price']
                    if i >= 5:
                        current_features['lag_6m'] = forecasts[i-5]['forecast_price']
                    if i >= 11:
                        current_features['lag_12m'] = forecasts[i-11]['forecast_price']
        
        return PredictionResult(
            state=state.title(),
            forecasts=forecasts,
            model_type=model_type,
            metrics=metadata['model_details'][state]['metrics']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Create the model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    # Check if model file exists
    default_model_path = "model/all_states_best_model_latest.pkl"
    if os.path.exists(default_model_path):
        try:
            # Try to initialize with default model file
            load_models_from_file(default_model_path)
            print(f"Successfully initialized with default model: {default_model_path}")
        except Exception as e:
            print(f"Warning: Could not initialize with default model: {str(e)}")
            print("API will start uninitialized. Call /initialize endpoint to load a model.")
    else:
        print(f"Default model not found at: {default_model_path}")
        print("API will start uninitialized. Call /initialize endpoint to load a model.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)