import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
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

def prepare_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare target variables for training.
    
    Args:
        df: DataFrame containing match data
        
    Returns:
        pd.DataFrame: DataFrame with target variables
    """
    try:
        targets = pd.DataFrame({
            'goals_home': df['Goles_Local'],
            'goals_away': df['Goles_Visitante'],
            'corners_home': df['Corners_Local'],
            'corners_away': df['Corners_Visitante'],
            'yellow_cards_home': df['Amarillas_Local'],
            'yellow_cards_away': df['Amarillas_Visitante']
        })
        
        logger.info("Prepared target variables:")
        for column in targets.columns:
            stats = targets[column].describe()
            logger.info(f"\n{column}:")
            logger.info(f"- Mean: {stats['mean']:.2f}")
            logger.info(f"- Std: {stats['std']:.2f}")
            logger.info(f"- Min: {stats['min']:.0f}")
            logger.info(f"- Max: {stats['max']:.0f}")
        
        return targets
    except Exception as e:
        logger.error(f"Error preparing target variables: {str(e)}")
        raise

def analyze_predictions(
    predictions: dict,
    actual_values: pd.DataFrame,
    feature_names: list
) -> None:
    """Analyze model predictions.
    
    Args:
        predictions: Dictionary containing predictions
        actual_values: DataFrame with actual values
        feature_names: List of feature names used
    """
    try:
        logger.info("\nPrediction Analysis:")
        
        # Analyze each target variable
        for target, pred in predictions.items():
            logger.info(f"\n{target}:")
            
            # Get actual values
            actual = actual_values[target].values
            
            # Calculate metrics
            mse = np.mean((actual - pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual - pred))
            r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
            
            logger.info("Metrics:")
            logger.info(f"- MSE: {mse:.4f}")
            logger.info(f"- RMSE: {rmse:.4f}")
            logger.info(f"- MAE: {mae:.4f}")
            logger.info(f"- R²: {r2:.4f}")
            
            # Analyze prediction distribution
            pred_stats = pd.Series(pred).describe()
            logger.info("\nPrediction Distribution:")
            logger.info(f"- Mean: {pred_stats['mean']:.2f}")
            logger.info(f"- Std: {pred_stats['std']:.2f}")
            logger.info(f"- Min: {pred_stats['min']:.2f}")
            logger.info(f"- Max: {pred_stats['max']:.2f}")
            
            # Check for extreme predictions
            threshold = 3  # Standard deviations
            z_scores = np.abs((pred - np.mean(pred)) / np.std(pred))
            extreme_preds = np.where(z_scores > threshold)[0]
            
            if len(extreme_preds) > 0:
                logger.warning(f"\nFound {len(extreme_preds)} extreme predictions:")
                for idx in extreme_preds:
                    logger.warning(
                        f"- Index {idx}: "
                        f"Predicted {pred[idx]:.2f}, "
                        f"Actual {actual[idx]:.2f}, "
                        f"Z-score {z_scores[idx]:.2f}"
                    )
    
    except Exception as e:
        logger.error(f"Error analyzing predictions: {str(e)}")
        raise

def main():
    """Main function to test model training with real data."""
    try:
        # Load match data
        df = load_data('../data/laliga.csv')
        logger.info(f"Loaded {len(df)} matches")
        
        # Engineer features
        engineer = FeatureEngineer(
            rolling_window=5,
            min_matches=3,
            home_advantage_weight=1.2
        )
        features_df, feature_names = engineer.engineer_features(df)
        logger.info(f"Engineered {len(feature_names)} features")
        
        # Prepare target variables
        target_df = prepare_target_variables(df)
        
        # Initialize model trainer
        trainer = ModelTrainer(
            experiment_name="laliga_predictions",
            target_variables=target_df.columns.tolist(),
            test_size=0.2,
            random_state=42
        )
        
        # Prepare data for training
        prepared_data = trainer.prepare_data(features_df, target_df)
        logger.info("Prepared data for training")
        
        # Perform cross-validation
        cv_results = trainer.cross_validate(features_df, target_df, cv=5)
        logger.info("\nCross-validation results:")
        for target, metrics in cv_results.items():
            logger.info(f"\n{target}:")
            for metric, scores in metrics.items():
                logger.info(f"- {metric}:")
                logger.info(f"  Mean: {np.mean(scores):.4f}")
                logger.info(f"  Std: {np.std(scores):.4f}")
        
        # Train models
        trainer.train_models(prepared_data)
        logger.info("\nTrained models for all targets")
        
        # Evaluate models
        evaluation_results = trainer.evaluate_models(prepared_data)
        logger.info("\nEvaluation results:")
        for target, metrics in evaluation_results.items():
            logger.info(f"\n{target}:")
            for metric, value in metrics.items():
                logger.info(f"- {metric}: {value:.4f}")
        
        # Make predictions
        predictions = trainer.predict(features_df)
        analyze_predictions(predictions, target_df, feature_names)
        
        # Save models
        output_dir = '../models'
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_models(output_dir)
        logger.info(f"\nSaved models to {output_dir}")
        
        # Export predictions
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv('../data/predictions.csv', index=True)
        logger.info("\nExported predictions to predictions.csv")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 