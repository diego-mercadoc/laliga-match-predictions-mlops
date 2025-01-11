import mlflow
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from pandera import DataFrameSchema, Column, Check
import requests
from pathlib import Path
import json
import os
from prefect import flow, task
from prefect.schedules import IntervalSchedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data validation schema
training_data_schema = DataFrameSchema({
    'home_team': Column(str),
    'away_team': Column(str),
    'goals_home': Column(float, Check.greater_equal_than(0)),
    'goals_away': Column(float, Check.greater_equal_than(0)),
    'corners_home': Column(float, Check.greater_equal_than(0)),
    'corners_away': Column(float, Check.greater_equal_than(0)),
    'yellow_cards_home': Column(float, Check.greater_equal_than(0)),
    'yellow_cards_away': Column(float, Check.greater_equal_than(0))
})

@task
def validate_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate the input data using pandera schema."""
    try:
        training_data_schema.validate(df)
        return True, "Data validation successful"
    except Exception as e:
        return False, str(e)

@task
def fetch_new_data() -> pd.DataFrame:
    """Fetch new training data from the source."""
    try:
        # Replace with your actual data fetching logic
        df = pd.read_csv("data/laliga.csv")
        return df
    except Exception as e:
        logger.error(f"Error fetching new data: {str(e)}")
        raise

@task
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess the data and split into features and targets."""
    # Add your feature engineering logic here
    feature_cols = [
        'home_team_rank', 'away_team_rank',
        'home_team_form', 'away_team_form',
        'head_to_head_wins', 'head_to_head_draws',
        'home_team_goals_scored_avg', 'away_team_goals_scored_avg',
        'home_team_goals_conceded_avg', 'away_team_goals_conceded_avg'
    ]
    
    target_cols = [
        'goals_home', 'goals_away',
        'corners_home', 'corners_away',
        'yellow_cards_home', 'yellow_cards_away'
    ]
    
    X = df[feature_cols]
    y = df[target_cols]
    
    return X, y

@task
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    experiment_name: str
) -> str:
    """Train the model and log metrics to MLflow."""
    with mlflow.start_run(experiment_name=experiment_name) as run:
        # Log parameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror'
        }
        mlflow.log_params(params)
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Log metrics
        predictions = model.predict(X_train)
        metrics = {
            'mse': mean_squared_error(y_train, predictions),
            'mae': mean_absolute_error(y_train, predictions),
            'r2': r2_score(y_train, predictions)
        }
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="LaLigaPredictionsModel"
        )
        
        return run.info.run_id

@task
def evaluate_model(
    run_id: str,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Dict[str, float]:
    """Evaluate the model on test data."""
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    predictions = model.predict(X_test)
    
    metrics = {
        'test_mse': mean_squared_error(y_test, predictions),
        'test_mae': mean_absolute_error(y_test, predictions),
        'test_r2': r2_score(y_test, predictions)
    }
    
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
    
    return metrics

@task
def promote_model(run_id: str, metrics: Dict[str, float]) -> None:
    """Promote model to production if it performs better than the current champion."""
    client = mlflow.tracking.MlflowClient()
    
    # Get current champion model
    try:
        current_champion = client.get_model_version_by_alias(
            "LaLigaPredictionsModel",
            "champion"
        )
        current_metrics = json.loads(
            current_champion.description
        )['metrics']
    except:
        current_metrics = {'test_r2': -float('inf')}
    
    # Compare performance
    if metrics['test_r2'] > current_metrics['test_r2']:
        # Promote new model to champion
        new_version = client.search_model_versions(
            f"run_id = '{run_id}'"
        )[0].version
        
        client.set_model_version_alias(
            "LaLigaPredictionsModel",
            new_version,
            "champion"
        )
        
        # Update description with metrics
        client.update_model_version(
            "LaLigaPredictionsModel",
            new_version,
            description=json.dumps({'metrics': metrics})
        )
        
        logger.info(f"Promoted model version {new_version} to champion")
    else:
        logger.info("Current champion model performs better, no promotion needed")

@flow
def retraining_pipeline(
    experiment_name: str = "laliga_predictions_retraining"
) -> None:
    """Main retraining pipeline flow."""
    try:
        # Fetch and validate new data
        df = fetch_new_data()
        is_valid, validation_message = validate_data(df)
        
        if not is_valid:
            logger.error(f"Data validation failed: {validation_message}")
            return
        
        # Preprocess data
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        run_id = train_model(X_train, y_train, experiment_name)
        
        # Evaluate model
        metrics = evaluate_model(run_id, X_test, y_test)
        
        # Promote model if better
        promote_model(run_id, metrics)
        
        logger.info("Retraining pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in retraining pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up scheduled retraining
    schedule = IntervalSchedule(
        interval=timedelta(days=7)  # Retrain weekly
    )
    
    retraining_pipeline() 