from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@dataclass
class DataQualityMetrics:
    """Class to store data quality metrics."""
    completeness: float
    consistency: float
    timeliness: float
    validity: float
    overall_score: float
    issues: List[str]

class DataQualityChecker:
    """Class to check data quality metrics for LaLiga match data."""
    
    def __init__(self, min_required_matches: int = 3):
        """Initialize DataQualityChecker.
        
        Args:
            min_required_matches: Minimum number of matches required for valid analysis
        """
        self.min_required_matches = min_required_matches
        self.required_columns = [
            'Fecha', 'Día', 'Sedes', 'Anfitrion', 'Adversario',
            'GF', 'GC', 'GF(tm)', 'GC(tm)', 'xG(tm)', 'xGA(tm)',
            'PG(tm)', 'PE(tm)', 'PP(tm)', 'Últimos 5(tm)'
        ]
        self.column_types = {
            'Fecha': datetime,
            'Día': np.number,
            'Sedes': np.number,
            'Anfitrion': str,
            'Adversario': str,
            'GF': np.number,
            'GC': np.number,
            'GF(tm)': np.number,
            'GC(tm)': np.number,
            'xG(tm)': np.number,
            'xGA(tm)': np.number,
            'PG(tm)': np.number,
            'PE(tm)': np.number,
            'PP(tm)': np.number,
            'Últimos 5(tm)': np.number
        }
        
        # Define non-negative check for numeric columns
        self.non_negative_columns = [
            'GF', 'GC', 'GF(tm)', 'GC(tm)', 'xG(tm)', 'xGA(tm)',
            'PG(tm)', 'PE(tm)', 'PP(tm)', 'Últimos 5(tm)'
        ]
    
    def calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score.
        
        Args:
            df: Input DataFrame
            
        Returns:
            float: Completeness score between 0 and 1
        """
        if df.empty:
            return 0.0
        return 1 - df[self.required_columns].isnull().sum().sum() / (df.shape[0] * len(self.required_columns))
    
    def check_consistency(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check data consistency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple containing:
                - float: Consistency score between 0 and 1
                - List[str]: List of consistency issues found
        """
        issues = []
        score = 1.0
        
        # Check for duplicate matches
        duplicates = df.duplicated(subset=['Fecha', 'Anfitrion', 'Adversario'], keep=False)
        if duplicates.any():
            score -= 0.3
            issues.append(f"Found {duplicates.sum()} duplicate matches")
        
        # Check for self-matches (team playing against itself)
        self_matches = df['Anfitrion'] == df['Adversario']
        if self_matches.any():
            score -= 0.3
            issues.append(f"Found {self_matches.sum()} self-matches")
        
        # Check for teams playing multiple matches on same day
        for fecha in df['Fecha'].unique():
            day_matches = df[df['Fecha'] == fecha]
            for team in pd.concat([day_matches['Anfitrion'], day_matches['Adversario']]).unique():
                team_matches = (day_matches['Anfitrion'] == team).sum() + (day_matches['Adversario'] == team).sum()
                if team_matches > 1:
                    score -= 0.2
                    issues.append(f"Team {team} plays {team_matches} matches on {fecha}")
        
        return max(0.0, score), issues
    
    def check_timeliness(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check data timeliness.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple containing:
                - float: Timeliness score between 0 and 1
                - List[str]: List of timeliness issues found
        """
        issues = []
        score = 1.0
        
        # Convert Fecha to datetime if it's not already
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        current_date = datetime.now()
        
        # Check for future dates (more than a week ahead)
        future_dates = df['Fecha'] > current_date + timedelta(days=7)
        if future_dates.any():
            score -= 0.3
            issues.append(f"Found {future_dates.sum()} dates more than a week in the future")
        
        # Check for very old dates (more than 2 years old)
        old_dates = df['Fecha'] < current_date - timedelta(days=730)
        if old_dates.any():
            score -= 0.3
            issues.append(f"Found {old_dates.sum()} dates more than 2 years old")
        
        # Check for missing dates
        if df['Fecha'].isnull().any():
            score -= 0.4
            issues.append(f"Found {df['Fecha'].isnull().sum()} missing dates")
        
        return max(0.0, score), issues
    
    def check_validity(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check data validity.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple containing:
                - float: Validity score between 0 and 1
                - List[str]: List of validity issues found
        """
        issues = []
        score = 1.0
        
        # Check for missing required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            score -= 0.4
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check column types
        for col, expected_type in self.column_types.items():
            if col in df.columns:
                if expected_type == np.number:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        score -= 0.2
                        issues.append(f"Column {col} should be numeric")
                elif expected_type == datetime:
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        score -= 0.2
                        issues.append(f"Column {col} should be datetime")
                elif expected_type == str:
                    if not pd.api.types.is_string_dtype(df[col]):
                        score -= 0.2
                        issues.append(f"Column {col} should be string")
        
        # Check for negative values in numeric columns
        for col in self.non_negative_columns:
            if col in df.columns:
                negative_values = df[df[col].notna()][df[col] < 0]
                if not negative_values.empty:
                    score -= 0.1
                    issues.append(
                        f"Column {col} has {len(negative_values)} negative values"
                    )
        
        return max(0.0, score), issues
    
    def validate_minimum_data(self, df: pd.DataFrame) -> bool:
        """Validate if DataFrame has minimum required matches.
        
        Args:
            df: Input DataFrame
            
        Returns:
            bool: True if DataFrame has minimum required matches
        """
        return len(df) >= self.min_required_matches
    
    def check_quality(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Check overall data quality.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataQualityMetrics: Object containing quality metrics and issues
        """
        if df.empty:
            return DataQualityMetrics(
                completeness=0.0,
                consistency=0.0,
                timeliness=0.0,
                validity=0.0,
                overall_score=0.0,
                issues=["Empty DataFrame"]
            )
        
        # Calculate individual metrics
        completeness = self.calculate_completeness(df)
        consistency_score, consistency_issues = self.check_consistency(df)
        timeliness_score, timeliness_issues = self.check_timeliness(df)
        validity_score, validity_issues = self.check_validity(df)
        
        # Combine all issues
        all_issues = []
        if consistency_issues:
            all_issues.extend(consistency_issues)
        if timeliness_issues:
            all_issues.extend(timeliness_issues)
        if validity_issues:
            all_issues.extend(validity_issues)
        
        # Calculate overall score (weighted average)
        weights = {
            'completeness': 0.3,
            'consistency': 0.3,
            'timeliness': 0.2,
            'validity': 0.2
        }
        
        overall_score = (
            completeness * weights['completeness'] +
            consistency_score * weights['consistency'] +
            timeliness_score * weights['timeliness'] +
            validity_score * weights['validity']
        )
        
        return DataQualityMetrics(
            completeness=completeness,
            consistency=consistency_score,
            timeliness=timeliness_score,
            validity=validity_score,
            overall_score=overall_score,
            issues=all_issues
        ) 