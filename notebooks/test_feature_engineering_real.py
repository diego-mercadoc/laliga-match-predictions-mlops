import pandas as pd
import numpy as np
import logging
from datetime import datetime
from utils.feature_engineering import FeatureEngineer
from utils.match_exceptions import MatchStatus
from typing import Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_match_data(file_path: str) -> pd.DataFrame:
    """Load match data from CSV file."""
    try:
        df = pd.read_csv('data/laliga.csv')
        logger.info(f"Successfully loaded {len(df)} matches from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Engineer features for match prediction."""
    try:
        # Convert date to datetime
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Sort by date
        df = df.sort_values('Fecha')
        
        # Initialize feature lists
        form_features = []
        match_features = []
        
        # Calculate rolling averages for goals, corners, and cards
        for window in [3, 5]:
            # Goals
            df[f'Goals_Scored_Last{window}'] = df.groupby('Anfitrion')['GF'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'Goals_Conceded_Last{window}'] = df.groupby('Anfitrion')['GC'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            form_features.extend([f'Goals_Scored_Last{window}', f'Goals_Conceded_Last{window}'])
            
            # Expected Goals
            df[f'xG_Last{window}'] = df.groupby('Anfitrion')['xG(tm)'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'xGA_Last{window}'] = df.groupby('Anfitrion')['xGA(tm)'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            form_features.extend([f'xG_Last{window}', f'xGA_Last{window}'])
        
        # Calculate team form
        df['Points'] = df['Resultado'].map({1: 0, 2: 1, 3: 3})
        df['Form_Score'] = df.groupby('Anfitrion')['Points'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        form_features.append('Form_Score')
        
        # Calculate home/away performance
        home_stats = df[df['Sedes'] == 1].groupby('Anfitrion').agg({
            'GF': 'mean',
            'GC': 'mean',
            'xG(tm)': 'mean',
            'xGA(tm)': 'mean'
        }).add_prefix('Home_')
        
        away_stats = df[df['Sedes'] == 0].groupby('Anfitrion').agg({
            'GF': 'mean',
            'GC': 'mean',
            'xG(tm)': 'mean',
            'xGA(tm)': 'mean'
        }).add_prefix('Away_')
        
        df = df.merge(home_stats, left_on='Anfitrion', right_index=True, how='left')
        df = df.merge(away_stats, left_on='Anfitrion', right_index=True, how='left')
        
        match_features.extend(home_stats.columns.tolist())
        match_features.extend(away_stats.columns.tolist())
        
        # Calculate opponent strength
        df['Opp_Strength'] = df.groupby('Adversario')['Points'].transform('mean')
        match_features.append('Opp_Strength')
        
        # Combine all features
        feature_names = form_features + match_features
        features_df = df[feature_names].copy()
        
        # Handle missing values
        features_df = features_df.fillna(features_df.mean())
        
        return features_df, feature_names
        
    except Exception as e:
        logger.error(f"Error engineering features: {str(e)}")
        raise

def analyze_features(features_df: pd.DataFrame, feature_names: list) -> None:
    """Analyze engineered features."""
    try:
        logger.info(f"\nFeature Analysis:")
        logger.info(f"Number of features: {len(feature_names)}")
        
        # Basic statistics
        stats = features_df.describe()
        logger.info("\nFeature Statistics:")
        logger.info(stats)
        
        # Missing values
        missing = features_df.isnull().sum()
        if missing.any():
            logger.warning("\nFeatures with missing values:")
            for feature, count in missing[missing > 0].items():
                logger.warning(f"- {feature}: {count} missing values")
        
        # Correlations
        correlations = features_df.corr()
        high_correlations = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                corr = correlations.iloc[i, j]
                if abs(corr) > 0.8:
                    high_correlations.append((
                        feature_names[i],
                        feature_names[j],
                        corr
                    ))
        
        if high_correlations:
            logger.warning("\nHighly correlated features (|correlation| > 0.8):")
            for f1, f2, corr in sorted(high_correlations, key=lambda x: abs(x[2]), reverse=True):
                logger.warning(f"- {f1} vs {f2}: {corr:.3f}")
        
    except Exception as e:
        logger.error(f"Error analyzing features: {str(e)}")
        raise

def main():
    """Main function to run feature engineering and analysis."""
    try:
        # Load data
        df = load_match_data('data/laliga.csv')
        logger.info(f"Loaded {len(df)} matches")
        
        # Engineer features
        features_df, feature_names = engineer_features(df)
        
        # Analyze features
        analyze_features(features_df, feature_names)
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 