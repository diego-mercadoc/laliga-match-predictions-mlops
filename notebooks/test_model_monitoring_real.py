import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from utils.model_monitoring import ModelMonitor
from utils.model_training import ModelTrainer
from utils.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load match data from file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        pd.DataFrame: Loaded match data
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepare data for monitoring.
    
    Args:
        df: DataFrame containing match data
        
    Returns:
        tuple: (features_df, target_df, feature_names)
    """
    try:
        # Engineer features
        engineer = FeatureEngineer(
            rolling_window=5,
            min_matches=3,
            home_advantage_weight=1.2
        )
        features_df, feature_names = engineer.engineer_features(df)
        logger.info(f"Engineered {len(feature_names)} features")
        
        # Prepare target variables
        target_df = pd.DataFrame({
            'goals_home': df['Goles_Local'],
            'goals_away': df['Goles_Visitante'],
            'corners_home': df['Corners_Local'],
            'corners_away': df['Corners_Visitante'],
            'yellow_cards_home': df['Amarillas_Local'],
            'yellow_cards_away': df['Amarillas_Visitante']
        })
        
        return features_df, target_df, feature_names
    
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def simulate_predictions(
    trainer: ModelTrainer,
    features_df: pd.DataFrame,
    target_df: pd.DataFrame,
    batch_size: int = 5
) -> tuple:
    """Simulate batch predictions for monitoring.
    
    Args:
        trainer: Trained ModelTrainer instance
        features_df: DataFrame with features
        target_df: DataFrame with target variables
        batch_size: Number of predictions per batch
        
    Returns:
        tuple: (predictions, actual_values)
    """
    try:
        n_samples = len(features_df)
        predictions = {}
        actual_values = {}
        
        # Get predictions in batches
        for i in range(0, n_samples, batch_size):
            batch_features = features_df.iloc[i:i + batch_size]
            batch_targets = target_df.iloc[i:i + batch_size]
            
            # Make predictions
            batch_predictions = trainer.predict(batch_features)
            
            # Store predictions and actual values
            for target in trainer.target_variables:
                if target not in predictions:
                    predictions[target] = []
                    actual_values[target] = []
                
                predictions[target].extend(batch_predictions[target])
                actual_values[target].extend(batch_targets[target].values)
        
        # Convert to numpy arrays
        for target in trainer.target_variables:
            predictions[target] = np.array(predictions[target])
            actual_values[target] = np.array(actual_values[target])
        
        return predictions, actual_values
    
    except Exception as e:
        logger.error(f"Error simulating predictions: {str(e)}")
        raise

def analyze_monitoring_results(monitor: ModelMonitor) -> None:
    """Analyze monitoring results.
    
    Args:
        monitor: ModelMonitor instance with monitoring history
    """
    try:
        logger.info("\nAnalyzing Monitoring Results:")
        
        # Check performance
        performance_metrics = monitor.check_performance()
        logger.info("\nPerformance Metrics:")
        for target, metrics in performance_metrics.items():
            logger.info(f"\n{target}:")
            for metric, value in metrics.items():
                logger.info(f"- {metric}: {value:.4f}")
        
        # Check drift
        drift_analysis = monitor.check_drift()
        logger.info("\nDrift Analysis:")
        for target, drifts in drift_analysis.items():
            if drifts:
                recent_scores = [d['drift_score'] for d in drifts]
                logger.info(f"\n{target}:")
                logger.info(f"- Latest drift score: {recent_scores[-1]:.4f}")
                logger.info(f"- Max drift score: {max(recent_scores):.4f}")
                logger.info(f"- Mean drift score: {np.mean(recent_scores):.4f}")
        
        # Generate and analyze report
        report = monitor.generate_monitoring_report()
        
        logger.info("\nMonitoring Statistics:")
        for target, stats in report['statistics'].items():
            logger.info(f"\n{target}:")
            for metric, value in stats.items():
                logger.info(f"- {metric}: {value:.4f}")
        
        logger.info("\nAlerts:")
        if report['alerts']:
            for alert in report['alerts']:
                logger.warning(
                    f"- {alert['type'].upper()} alert for {alert['target']}: "
                    f"value = {alert['value']:.4f}, "
                    f"threshold = {alert['threshold']:.4f}"
                )
        else:
            logger.info("No alerts generated")
        
        # Save report
        output_dir = '../monitoring'
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(
            output_dir,
            f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        monitor.save_report(report, report_path)
        logger.info(f"\nSaved monitoring report to {report_path}")
    
    except Exception as e:
        logger.error(f"Error analyzing monitoring results: {str(e)}")
        raise

def main():
    """Main function to test model monitoring with real data."""
    try:
        # Load data
        df = load_data('../data/laliga.csv')
        logger.info(f"Loaded {len(df)} matches")
        
        # Prepare data
        features_df, target_df, feature_names = prepare_data(df)
        
        # Initialize and train models
        trainer = ModelTrainer(
            experiment_name="laliga_predictions",
            target_variables=target_df.columns.tolist(),
            test_size=0.2,
            random_state=42
        )
        
        # Prepare data for training
        prepared_data = trainer.prepare_data(features_df, target_df)
        
        # Train models
        trainer.train_models(prepared_data)
        logger.info("Trained models for all targets")
        
        # Initialize monitor
        monitor = ModelMonitor(
            experiment_name="laliga_predictions_monitoring",
            target_variables=target_df.columns.tolist(),
            monitoring_window=10,
            drift_threshold=0.1,
            performance_threshold=0.2
        )
        
        # Simulate predictions and update monitoring
        predictions, actual_values = simulate_predictions(
            trainer,
            features_df,
            target_df,
            batch_size=5
        )
        
        # Update monitoring history
        monitor.update_history(predictions, actual_values)
        logger.info("Updated monitoring history")
        
        # Analyze monitoring results
        analyze_monitoring_results(monitor)
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 