import os
import mlflow
import dagshub
from dotenv import load_dotenv

def setup_mlflow():
    load_dotenv()
    
    dagshub.init(
        repo_owner=os.getenv('DAGSHUB_USERNAME'),
        repo_name="ProyectoFinalCD",
        mlflow=True
    )
    
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="final-prefect-experiment")
    
    current_experiment = mlflow.get_experiment_by_name("final-prefect-experiment")
    print(f"Current experiment: {current_experiment.experiment_id}")

if __name__ == "__main__":
    setup_mlflow() 