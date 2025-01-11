import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Class for monitoring model performance and data drift."""
    
    def __init__(
        self,
        experiment_name: str,
        target_variables: List[str],
        monitoring_window: int = 10,
        drift_threshold: float = 0.1,
        performance_threshold: float = 0.2
    ):
        """Initialize ModelMonitor.
        
        Args:
            experiment_name: Name of the MLflow experiment
            target_variables: List of target variables to monitor
            monitoring_window: Number of predictions to use for monitoring
            drift_threshold: Threshold for detecting data drift
            performance_threshold: Threshold for performance degradation
        """
        self.experiment_name = experiment_name
        self.target_variables = target_variables
        self.monitoring_window = monitoring_window
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        
        # Initialize monitoring history
        self.prediction_history: Dict[str, List[float]] = {
            target: [] for target in target_variables
        }
        self.actual_history: Dict[str, List[float]] = {
            target: [] for target in target_variables
        }
        self.error_history: Dict[str, List[float]] = {
            target: [] for target in target_variables
        }
        self.drift_history: Dict[str, List[Tuple[datetime, float]]] = {
            target: [] for target in target_variables
        }
        
        # Set up MLflow
        mlflow.set_experiment(f"{experiment_name}_monitoring")
    
    def update_history(
        self,
        predictions: Dict[str, np.ndarray],
        actual_values: Dict[str, np.ndarray],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update monitoring history with new predictions.
        
        Args:
            predictions: Dictionary of predictions for each target
            actual_values: Dictionary of actual values for each target
            timestamp: Optional timestamp for the predictions
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        for target in self.target_variables:
            pred = predictions[target]
            actual = actual_values[target]
            
            # Update histories
            self.prediction_history[target].extend(pred.tolist())
            self.actual_history[target].extend(actual.tolist())
            
            # Calculate and store errors
            errors = np.abs(pred - actual)
            self.error_history[target].extend(errors.tolist())
            
            # Calculate drift
            drift_score = self._calculate_drift(
                pred,
                self.prediction_history[target][-self.monitoring_window:]
            )
            self.drift_history[target].append((timestamp, drift_score))
            
            # Trim histories to monitoring window
            self._trim_histories(target)
            
            # Log to MLflow
            with mlflow.start_run(run_name=f"{target}_monitoring_{timestamp}"):
                metrics = self._calculate_metrics(target)
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
                mlflow.log_metric("drift_score", drift_score)
    
    def check_performance(self) -> Dict[str, Dict[str, float]]:
        """Check model performance metrics.
        
        Returns:
            Dictionary containing performance metrics for each target
        """
        performance_metrics = {}
        
        for target in self.target_variables:
            metrics = self._calculate_metrics(target)
            performance_metrics[target] = metrics
            
            # Log performance issues
            if metrics['rmse'] > self.performance_threshold:
                logger.warning(
                    f"High RMSE ({metrics['rmse']:.4f}) "
                    f"detected for {target}"
                )
            if metrics['mae'] > self.performance_threshold:
                logger.warning(
                    f"High MAE ({metrics['mae']:.4f}) "
                    f"detected for {target}"
                )
        
        return performance_metrics
    
    def check_drift(self) -> Dict[str, List[Dict[str, any]]]:
        """Check for data drift in predictions.
        
        Returns:
            Dictionary containing drift analysis for each target
        """
        drift_analysis = {}
        
        for target in self.target_variables:
            recent_drifts = [
                {
                    'timestamp': timestamp,
                    'drift_score': score
                }
                for timestamp, score in self.drift_history[target][-self.monitoring_window:]
            ]
            
            drift_analysis[target] = recent_drifts
            
            # Log drift issues
            recent_scores = [d['drift_score'] for d in recent_drifts]
            if any(score > self.drift_threshold for score in recent_scores):
                logger.warning(
                    f"Data drift detected for {target} "
                    f"(max score: {max(recent_scores):.4f})"
                )
        
        return drift_analysis
    
    def generate_monitoring_report(self) -> Dict[str, any]:
        """Generate comprehensive monitoring report.
        
        Returns:
            Dictionary containing monitoring report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self.check_performance(),
            'drift_analysis': self.check_drift(),
            'alerts': self._generate_alerts(),
            'statistics': self._calculate_statistics()
        }
        
        return report
    
    def save_report(self, report: Dict[str, any], file_path: str) -> None:
        """Save monitoring report to file.
        
        Args:
            report: Monitoring report to save
            file_path: Path to save the report to
        """
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved monitoring report to {file_path}")
    
    def _calculate_metrics(self, target: str) -> Dict[str, float]:
        """Calculate performance metrics for a target.
        
        Args:
            target: Target variable name
            
        Returns:
            Dictionary containing performance metrics
        """
        predictions = np.array(self.prediction_history[target][-self.monitoring_window:])
        actuals = np.array(self.actual_history[target][-self.monitoring_window:])
        
        return {
            'mse': mean_squared_error(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mae': mean_absolute_error(actuals, predictions),
            'r2': r2_score(actuals, predictions)
        }
    
    def _calculate_drift(
        self,
        new_predictions: np.ndarray,
        historical_predictions: List[float]
    ) -> float:
        """Calculate drift score between new and historical predictions.
        
        Args:
            new_predictions: Array of new predictions
            historical_predictions: List of historical predictions
            
        Returns:
            float: Drift score
        """
        if len(historical_predictions) == 0:
            return 0.0
        
        # Calculate distribution statistics
        hist_mean = np.mean(historical_predictions)
        hist_std = np.std(historical_predictions)
        new_mean = np.mean(new_predictions)
        new_std = np.std(new_predictions)
        
        # Calculate normalized difference in mean and standard deviation
        mean_diff = abs(new_mean - hist_mean) / (hist_std + 1e-6)
        std_diff = abs(new_std - hist_std) / (hist_std + 1e-6)
        
        # Combine into single drift score
        drift_score = (mean_diff + std_diff) / 2
        
        return drift_score
    
    def _trim_histories(self, target: str) -> None:
        """Trim histories to monitoring window.
        
        Args:
            target: Target variable name
        """
        window = self.monitoring_window
        self.prediction_history[target] = self.prediction_history[target][-window:]
        self.actual_history[target] = self.actual_history[target][-window:]
        self.error_history[target] = self.error_history[target][-window:]
        self.drift_history[target] = self.drift_history[target][-window:]
    
    def _generate_alerts(self) -> List[Dict[str, any]]:
        """Generate alerts based on monitoring results.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        for target in self.target_variables:
            # Check performance degradation
            metrics = self._calculate_metrics(target)
            if metrics['rmse'] > self.performance_threshold:
                alerts.append({
                    'type': 'performance',
                    'target': target,
                    'metric': 'rmse',
                    'value': metrics['rmse'],
                    'threshold': self.performance_threshold,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Check data drift
            recent_drifts = [
                score for _, score in self.drift_history[target][-self.monitoring_window:]
            ]
            if any(score > self.drift_threshold for score in recent_drifts):
                alerts.append({
                    'type': 'drift',
                    'target': target,
                    'value': max(recent_drifts),
                    'threshold': self.drift_threshold,
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for monitoring data.
        
        Returns:
            Dictionary containing statistics for each target
        """
        statistics = {}
        
        for target in self.target_variables:
            predictions = np.array(self.prediction_history[target])
            actuals = np.array(self.actual_history[target])
            errors = np.array(self.error_history[target])
            
            statistics[target] = {
                'prediction_mean': float(np.mean(predictions)),
                'prediction_std': float(np.std(predictions)),
                'actual_mean': float(np.mean(actuals)),
                'actual_std': float(np.std(actuals)),
                'error_mean': float(np.mean(errors)),
                'error_std': float(np.std(errors)),
                'sample_size': len(predictions)
            }
        
        return statistics 