Wheat Price Forecaster
A comprehensive system for predicting wheat prices across different states in India using machine learning models.
Overview
This project provides an end-to-end solution for wheat price forecasting using various machine learning models including Random Forest, XGBoost, and Holt-Winters. The system consists of a backend API built with FastAPI, a frontend dashboard created with Streamlit, and MLflow for experiment tracking.
Features

Price Forecasting: Generate forecasts for wheat prices in different states
Model Comparison: Compare various model performance metrics across states
Feature Importance Analysis: Understand which factors most influence wheat prices
Interactive Visualizations: Explore forecasts and trends through intuitive charts
State-wise Analysis: Detailed analysis of model performance by state

System Architecture
The system consists of three main components:

FastAPI Backend: Serves predictions and model metadata
Streamlit Frontend: Provides an intuitive user interface for interacting with models
MLflow Server: Tracks model performance and experiments

Installation
Prerequisites

Python 3.7+
pip
virtualenv
git

Setup Script
You can use the provided deployment script to automatically set up the entire system:
bashchmod +x deploy_wheat_prediction.sh
./deploy_wheat_prediction.sh
Manual Installation
If you prefer to set up manually:

Clone the repository:

bashgit clone https://github.com/the-way-of-learning/wheat_price_prediction.git

Set up backend:

bashmkdir -p ~/wheat-app/backend/model
cp ~/wheat-repo/wheat_price_prediction_api.py ~/wheat-app/backend/
cd ~/wheat-app/backend
python3 -m virtualenv venv
source venv/bin/activate
pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib statsmodels matplotlib seaborn plotly mlflow boto3

Set up frontend:

bashmkdir -p ~/wheat-app/frontend
cp ~/wheat-repo/wheat_price_prediction_ui.py ~/wheat-app/frontend/
cd ~/wheat-app/frontend
python3 -m virtualenv venv
source venv/bin/activate
pip install streamlit pandas numpy plotly requests

Download model file:

bashaws s3 cp s3://foundation-project-data/wheat_forecaster/models/all_states_best_model_latest.pkl ~/wheat-app/backend/model/all_states_best_model_latest.pkl
Usage
Starting the Services

Start the backend API:

bashcd ~/wheat-app/backend
source venv/bin/activate
uvicorn wheat_price_prediction_api:app --host 0.0.0.0 --port 8000

Start the Streamlit frontend:

bashcd ~/wheat-app/frontend
source venv/bin/activate
streamlit run wheat_price_prediction_ui.py --server.port=8501 --server.address=0.0.0.0

(Optional) Start MLflow server:

bashcd ~/mlflow
pipenv run mlflow server -h 0.0.0.0 --default-artifact-root s3://foundation-project-data
Accessing the Application

Frontend UI: http://<your-server-ip>:8501
Backend API: http://<your-server-ip>:8000
MLflow server: http://<your-server-ip>:5000

API Documentation
The API documentation is available at http://<your-server-ip>:8000/docs, which provides a Swagger UI interface for interacting with the API endpoints.
Key Endpoints

/initialize: Initialize the API with the model file
/status: Get API status
/states: Get list of available states
/state_info/{state}: Get detailed information about a state
/predict: Generate price forecasts for a state
