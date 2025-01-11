import pandas as pd
import numpy as np
from importance_analysis import analyze_all_targets
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def engineer_features(df):
    """Engineer features for importance analysis."""
    try:
        # Convert date to datetime
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Sort by date
        df = df.sort_values('Fecha')
        
        # Calculate form features
        df['Form_Score'] = df.groupby('Anfitrion')['GF'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        
        # Calculate rolling averages
        for window in [3, 5]:
            df[f'Goals_Tm_Last{window}'] = df.groupby('Anfitrion')['GF'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'Goals_Opp_Last{window}'] = df.groupby('Anfitrion')['GC'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        
        # Calculate weighted form
        df['Weighted_Form'] = df.groupby('Anfitrion')['GF'].transform(
            lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
        )
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        return df
        
    except Exception as e:
        logger.error(f"Error engineering features: {str(e)}")
        raise

def main():
    try:
        # Load data
        df = pd.read_csv('../data/laliga.csv')
        logger.info(f"Loaded {len(df)} matches")
        
        # Engineer features
        df = engineer_features(df)
        logger.info("Features engineered successfully")
        
        # Run analysis
        results = analyze_all_targets(df)
        
        # Print results
        print("\nFeature Importance Analysis Results:")
        for target, (importance_df, interaction_df) in results.items():
            print(f"\n{target.upper()} PREDICTION:")
            
            print("\nTop 5 Most Important Features:")
            top_features = importance_df.sort_values('avg_importance', ascending=False).head()
            print(top_features.to_string())
            
            print("\nTop 3 Feature Interactions:")
            if not interaction_df.empty:
                top_interactions = interaction_df.sort_values('interaction_score', ascending=False).head(3)
                print(top_interactions.to_string())
            else:
                print("No significant interactions found")
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 