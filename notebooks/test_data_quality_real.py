"""Test data quality on real LaLiga match data."""

import pandas as pd
import numpy as np
import logging
from utils.data_quality import DataQualityChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load match data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path} with {len(df)} matches")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def analyze_column_stats(df: pd.DataFrame) -> None:
    """Log basic statistics for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        stats = df[col].describe()
        logger.info(f"\nColumn {col} statistics:")
        logger.info(f"Mean: {stats['mean']:.2f}")
        logger.info(f"Min: {stats['min']}")
        logger.info(f"Max: {stats['max']}")
        logger.info(f"Missing values: {df[col].isnull().sum()}")

def main():
    """Run data quality checks on real data."""
    try:
        # Load data
        df = load_data('data/laliga.csv')
        
        # Initialize checker
        checker = DataQualityChecker()
        
        # Check if we have minimum required data
        if not checker.validate_minimum_data(df):
            logger.warning("Dataset does not meet minimum data requirement")
            return
        
        # Run quality checks
        metrics = checker.check_quality(df)
        
        # Log results
        logger.info("\nData Quality Metrics:")
        logger.info(f"Completeness: {metrics.completeness:.2f}")
        logger.info(f"Consistency: {metrics.consistency:.2f}")
        logger.info(f"Timeliness: {metrics.timeliness:.2f}")
        logger.info(f"Validity: {metrics.validity:.2f}")
        logger.info(f"Overall Score: {metrics.overall_score:.2f}")
        
        if metrics.issues:
            logger.warning("\nData Quality Issues:")
            for issue in metrics.issues:
                logger.warning(f"- {issue}")
        else:
            logger.info("\nNo data quality issues found")
        
        # Analyze column statistics
        logger.info("\nColumn Statistics:")
        analyze_column_stats(df)
        
    except Exception as e:
        logger.error(f"Error in data quality analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main() 