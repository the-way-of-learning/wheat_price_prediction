#!/bin/bash
# Comprehensive deployment script for Wheat Price Prediction system

# Exit on error
set -e

# Print timestamp and heading
echo "===== Wheat Price Prediction System Deployment ====="
echo "Starting deployment at $(date)"

# Update system and install dependencies
echo "===== Installing system dependencies ====="
#sudo apt update
sudo apt install -y python3-pip pipenv virtualenv git
#sudo apt install pipenv virtualenv git
sudo apt install -y git unzip curl

# Check if AWS CLI is already installed
if ! command -v aws &> /dev/null; then
    echo "===== Installing AWS CLI ====="
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
else
    echo "===== AWS CLI already installed, skipping installation ====="
    aws --version
fi

# Create project structure
echo "===== Creating project structure ====="
mkdir -p ~/wheat-app/frontend
mkdir -p ~/wheat-app/backend/model
mkdir -p ~/mlflow

# Clone the repository
echo "===== Cloning code repository ====="
git clone https://github.com/the-way-of-learning/wheat_price_prediction.git ~/wheat_repo

# Setup MLflow environment
echo "===== Setting up MLflow environment ====="
cd ~/mlflow
pipenv install mlflow awscli boto3

# Configure AWS credentials
echo "===== Configuring AWS credentials ====="
echo "Please enter your AWS credentials:"
aws configure

# Extract AWS credentials for later use in the script
AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
AWS_REGION=$(aws configure get region)

# Download model file from S3
echo "===== Downloading model file from S3 ====="
aws s3 cp s3://foundation-project-data/wheat_forecaster/models/all_states_best_model_latest.pkl ~/wheat-app/backend/model/all_states_best_model_latest.pkl

# Move code files to appropriate locations
echo "===== Moving code files to appropriate locations ====="
# API backend
cp ~/wheat_repo/code_files/wheat_price_prediction_api.py ~/wheat-app/backend/
# Streamlit frontend
cp ~/wheat_repo/code_files/wheat_price_prediction_ui.py ~/wheat-app/frontend/

# Install backend requirements
echo "===== Installing backend requirements ====="
cd ~/wheat-app/backend
python3 -m virtualenv venv
source venv/bin/activate
pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib statsmodels matplotlib seaborn plotly mlflow boto3 awscli
deactivate

# Install frontend requirements
echo "===== Installing frontend requirements ====="
cd ~/wheat-app/frontend
python3 -m virtualenv venv
source venv/bin/activate
pip install streamlit pandas numpy plotly requests
deactivate


# Create start scripts for both services
echo "===== Creating service start scripts ====="

# Backend start script
cat > ~/wheat-app/start_backend.sh << 'EOF'
#!/bin/bash
cd ~/wheat-app/backend
source venv/bin/activate
nohup uvicorn wheat_price_prediction_api:app --host 0.0.0.0 --port 8000 > output.log 2>&1 &
echo $! > backend.pid
echo "Backend started with PID $(cat backend.pid)"
deactivate
EOF
chmod +x ~/wheat-app/start_backend.sh

# Frontend start script
cat > ~/wheat-app/start_frontend.sh << 'EOF'
#!/bin/bash
cd ~/wheat-app/frontend
source venv/bin/activate
nohup streamlit run wheat_price_prediction_ui.py --server.port=8501 --server.address=0.0.0.0 > output_frontend.log 2>&1 &
echo $! > frontend.pid
echo "Frontend started with PID $(cat frontend.pid)"
deactivate
EOF
chmod +x ~/wheat-app/start_frontend.sh

# MLflow start script
cat > ~/mlflow/start_mlflow.sh << 'EOF'
#!/bin/bash
cd ~/mlflow
pipenv run mlflow server -h 0.0.0.0 --default-artifact-root s3://foundation-project-data &
echo $! > mlflow.pid
echo "MLflow server started with PID $(cat mlflow.pid)"
EOF
chmod +x ~/mlflow/start_mlflow.sh

# Create stop scripts for all services
cat > ~/wheat-app/stop_services.sh << 'EOF'
#!/bin/bash
echo "Stopping all services..."

# Stop backend
if [ -f ~/wheat-app/backend/backend.pid ]; then
    BACKEND_PID=$(cat ~/wheat-app/backend/backend.pid)
    if ps -p $BACKEND_PID > /dev/null; then
        kill $BACKEND_PID
        echo "Backend service stopped (PID: $BACKEND_PID)"
    else
        echo "Backend service not running"
    fi
    rm ~/wheat-app/backend/backend.pid
fi

# Stop frontend
if [ -f ~/wheat-app/frontend/frontend.pid ]; then
    FRONTEND_PID=$(cat ~/wheat-app/frontend/frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null; then
        kill $FRONTEND_PID
        echo "Frontend service stopped (PID: $FRONTEND_PID)"
    else
        echo "Frontend service not running"
    fi
    rm ~/wheat-app/frontend/frontend.pid
fi

# Stop MLflow
if [ -f ~/mlflow/mlflow.pid ]; then
    MLFLOW_PID=$(cat ~/mlflow/mlflow.pid)
    if ps -p $MLFLOW_PID > /dev/null; then
        kill $MLFLOW_PID
        echo "MLflow service stopped (PID: $MLFLOW_PID)"
    else
        echo "MLflow service not running"
    fi
    rm ~/mlflow/mlflow.pid
fi

echo "All services stopped"
EOF
chmod +x ~/wheat-app/stop_services.sh

# Start the services
echo "===== Starting services ====="
~/mlflow/start_mlflow.sh
sleep 5  # Give MLflow some time to start
~/wheat-app/start_backend.sh
sleep 3  # Give backend some time to start
~/wheat-app/start_frontend.sh

# Create a convenience script to check logs
cat > ~/wheat-app/check_logs.sh << 'EOF'
#!/bin/bash
echo "===== Backend logs ====="
tail -n 50 ~/wheat-app/backend/output.log
echo ""
echo "===== Frontend logs ====="
tail -n 50 ~/wheat-app/frontend/output_frontend.log
EOF
chmod +x ~/wheat-app/check_logs.sh

# Get public IP address for final message
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com || wget -qO- http://checkip.amazonaws.com)

echo "===== Deployment complete! ====="
echo "MLflow server running at: http://$PUBLIC_IP:5000"
echo "Backend API running at: http://$PUBLIC_IP:8000"
echo "Frontend UI running at: http://$PUBLIC_IP:8501"
echo ""
echo "Use the following scripts to manage your services:"
echo "  - Check logs: ~/wheat-app/check_logs.sh"
echo "  - Stop all services: ~/wheat-app/stop_services.sh"
echo "  - Start backend: ~/wheat-app/start_backend.sh"
echo "  - Start frontend: ~/wheat-app/start_frontend.sh"
echo "  - Start MLflow: ~/mlflow/start_mlflow.sh"