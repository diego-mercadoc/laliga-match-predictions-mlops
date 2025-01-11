import os
from dotenv import load_dotenv
import mlflow

def init_mlflow():
    """Initialize MLflow with DagsHub configuration."""
    load_dotenv()
    
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_USER_TOKEN')

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
if __name__ == "__main__":
    init_mlflow() 