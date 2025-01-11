import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import xgboost as xgb
import mlflow
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training and evaluating prediction models."""
    
    def __init__(
        self,
        experiment_name: str,
        target_variables: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """Initialize ModelTrainer.
        
        Args:
            experiment_name: Name of the MLflow experiment
            target_variables: List of target variables to predict
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.experiment_name = experiment_name
        self.target_variables = target_variables
        self.test_size = test_size
        self.random_state = random_state
        
        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)
        
        # Initialize models and scalers
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Default model parameters
        self.default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': random_state
        }
    
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Prepare data for training.
        
        Args:
            features_df: DataFrame with features
            target_df: DataFrame with target variables
            
        Returns:
            Dict containing train and test data for each target
        """
        prepared_data = {}
        
        for target in self.target_variables:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_df,
                target_df[target],
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers[target] = scaler
            
            # Store data
            prepared_data[target] = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train.values,
                'y_test': y_test.values
            }
        
        return prepared_data
    
    def train_models(
        self,
        prepared_data: Dict[str, Dict[str, np.ndarray]],
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Train models for each target variable.
        
        Args:
            prepared_data: Dictionary containing prepared data
            params: Optional model parameters
        """
        if params is None:
            params = self.default_params
        
        for target in self.target_variables:
            with mlflow.start_run(run_name=f"{target}_training"):
                logger.info(f"\nTraining model for {target}")
                
                # Log parameters
                mlflow.log_params(params)
                
                # Get data
                data = prepared_data[target]
                X_train = data['X_train']
                y_train = data['y_train']
                
                # Create and train model
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train)],
                    verbose=False
                )
                
                # Store model
                self.models[target] = model
                
                # Log model
                mlflow.xgboost.log_model(model, f"{target}_model")
                
                # Calculate and log training metrics
                train_pred = model.predict(X_train)
                train_metrics = self._calculate_metrics(y_train, train_pred)
                
                for metric, value in train_metrics.items():
                    mlflow.log_metric(f"train_{metric}", value)
                
                logger.info("Training metrics:")
                for metric, value in train_metrics.items():
                    logger.info(f"- {metric}: {value:.4f}")
    
    def evaluate_models(
        self,
        prepared_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate trained models.
        
        Args:
            prepared_data: Dictionary containing prepared data
            
        Returns:
            Dictionary containing evaluation metrics for each target
        """
        evaluation_results = {}
        
        for target in self.target_variables:
            with mlflow.start_run(run_name=f"{target}_evaluation"):
                logger.info(f"\nEvaluating model for {target}")
                
                # Get data and model
                data = prepared_data[target]
                X_test = data['X_test']
                y_test = data['y_test']
                model = self.models[target]
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred)
                evaluation_results[target] = metrics
                
                # Log metrics
                for metric, value in metrics.items():
                    mlflow.log_metric(f"test_{metric}", value)
                
                logger.info("Test metrics:")
                for metric, value in metrics.items():
                    logger.info(f"- {metric}: {value:.4f}")
                
                # Feature importance
                importance = model.feature_importances_
                feature_imp = pd.DataFrame({
                    'feature': model.get_booster().feature_names,
                    'importance': importance
                })
                feature_imp = feature_imp.sort_values('importance', ascending=False)
                
                # Log feature importance
                mlflow.log_dict(
                    feature_imp.to_dict(),
                    f"{target}_feature_importance.json"
                )
        
        return evaluation_results
    
    def cross_validate(
        self,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame,
        cv: int = 5
    ) -> Dict[str, Dict[str, List[float]]]:
        """Perform cross-validation.
        
        Args:
            features_df: DataFrame with features
            target_df: DataFrame with target variables
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation scores
        """
        cv_results = {}
        
        for target in self.target_variables:
            with mlflow.start_run(run_name=f"{target}_cross_validation"):
                logger.info(f"\nPerforming cross-validation for {target}")
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(features_df)
                y = target_df[target].values
                
                # Create model
                model = xgb.XGBRegressor(**self.default_params)
                
                # Calculate scores
                mse_scores = cross_val_score(
                    model, X_scaled, y,
                    cv=cv,
                    scoring='neg_mean_squared_error'
                )
                mae_scores = cross_val_score(
                    model, X_scaled, y,
                    cv=cv,
                    scoring='neg_mean_absolute_error'
                )
                r2_scores = cross_val_score(
                    model, X_scaled, y,
                    cv=cv,
                    scoring='r2'
                )
                
                # Store results
                cv_results[target] = {
                    'mse': -mse_scores,
                    'mae': -mae_scores,
                    'r2': r2_scores
                }
                
                # Log metrics
                for metric, scores in cv_results[target].items():
                    mlflow.log_metric(f"cv_mean_{metric}", np.mean(scores))
                    mlflow.log_metric(f"cv_std_{metric}", np.std(scores))
                
                logger.info("Cross-validation results:")
                for metric, scores in cv_results[target].items():
                    logger.info(f"- {metric}:")
                    logger.info(f"  Mean: {np.mean(scores):.4f}")
                    logger.info(f"  Std: {np.std(scores):.4f}")
        
        return cv_results
    
    def save_models(self, output_dir: str) -> None:
        """Save trained models and scalers.
        
        Args:
            output_dir: Directory to save models and scalers
        """
        import joblib
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for target in self.target_variables:
            # Save model
            model_path = os.path.join(output_dir, f"{target}_model.json")
            self.models[target].save_model(model_path)
            
            # Save scaler
            scaler_path = os.path.join(output_dir, f"{target}_scaler.pkl")
            joblib.dump(self.scalers[target], scaler_path)
            
            logger.info(f"Saved model and scaler for {target}")
    
    def load_models(self, input_dir: str) -> None:
        """Load trained models and scalers.
        
        Args:
            input_dir: Directory containing saved models and scalers
        """
        import joblib
        import os
        
        for target in self.target_variables:
            # Load model
            model_path = os.path.join(input_dir, f"{target}_model.json")
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            self.models[target] = model
            
            # Load scaler
            scaler_path = os.path.join(input_dir, f"{target}_scaler.pkl")
            self.scalers[target] = joblib.load(scaler_path)
            
            logger.info(f"Loaded model and scaler for {target}")
    
    def predict(
        self,
        features_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Make predictions using trained models.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary containing predictions for each target
        """
        predictions = {}
        
        for target in self.target_variables:
            # Scale features
            X_scaled = self.scalers[target].transform(features_df)
            
            # Make predictions
            predictions[target] = self.models[target].predict(X_scaled)
        
        return predictions
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing evaluation metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        } 