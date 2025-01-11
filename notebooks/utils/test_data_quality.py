import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .data_quality import DataQualityChecker, DataQualityMetrics

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Fecha': pd.date_range(start='2024-01-01', periods=4),
        'Día': [1, 1, 2, 2],
        'Sedes': [1, 2, 1, 2],
        'Anfitrion': ['Team A', 'Team B', 'Team C', 'Team D'],
        'Adversario': ['Team B', 'Team C', 'Team D', 'Team A'],
        'GF(tm)': [2, 1, 3, 0],
        'GC(tm)': [1, 2, 2, 1],
        'Corners_Local': [5, 6, 4, 7],
        'Corners_Visitante': [4, 5, 6, 3],
        'Amarillas_Local': [2, 3, 1, 4],
        'Amarillas_Visitante': [3, 2, 4, 2]
    })

@pytest.fixture
def quality_checker():
    """Create DataQualityChecker instance."""
    return DataQualityChecker(min_required_matches=3)

def test_completeness(quality_checker, sample_data):
    """Test completeness calculation."""
    # Test with complete data
    score = quality_checker.calculate_completeness(sample_data)
    assert score == 1.0
    
    # Test with missing data
    sample_data.iloc[0, 0] = np.nan
    score = quality_checker.calculate_completeness(sample_data)
    assert score < 1.0

def test_consistency(quality_checker, sample_data):
    """Test consistency checks."""
    # Test with consistent data
    score, issues = quality_checker.check_consistency(sample_data)
    assert score == 1.0
    assert not issues
    
    # Test with duplicate matches
    duplicate_data = pd.concat([sample_data, sample_data.iloc[[0]]])
    score, issues = quality_checker.check_consistency(duplicate_data)
    assert score < 1.0
    assert any("duplicate matches" in issue for issue in issues)
    
    # Test with self-matches
    sample_data.loc[0, 'Visitante'] = sample_data.loc[0, 'Local']
    score, issues = quality_checker.check_consistency(sample_data)
    assert score < 1.0
    assert any("self-matches" in issue for issue in issues)

def test_timeliness(quality_checker, sample_data):
    """Test timeliness checks."""
    # Test with current dates
    score, issues = quality_checker.check_timeliness(sample_data)
    assert score == 1.0
    assert not issues
    
    # Test with future dates
    sample_data.loc[0, 'Fecha'] = datetime.now() + timedelta(days=365)
    score, issues = quality_checker.check_timeliness(sample_data)
    assert score < 1.0
    assert any("future" in issue for issue in issues)
    
    # Test with old dates
    sample_data.loc[1, 'Fecha'] = datetime.now() - timedelta(days=365)
    score, issues = quality_checker.check_timeliness(sample_data)
    assert score < 1.0
    assert any("old dates" in issue for issue in issues)

def test_validity(quality_checker, sample_data):
    """Test validity checks."""
    # Test with valid data
    score, issues = quality_checker.check_validity(sample_data)
    assert score == 1.0
    assert not issues
    
    # Test with wrong column type
    sample_data['Sem.'] = sample_data['Sem.'].astype(str)
    score, issues = quality_checker.check_validity(sample_data)
    assert score < 1.0
    assert any("should be" in issue for issue in issues)
    
    # Test with missing column
    invalid_data = sample_data.drop('Estado', axis=1)
    score, issues = quality_checker.check_validity(invalid_data)
    assert score < 1.0
    assert any("Missing required column" in issue for issue in issues)

def test_validity_ranges(quality_checker, sample_data):
    """Test validity range checks."""
    # Test with valid data
    score, issues = quality_checker.check_validity(sample_data)
    assert score == 1.0
    assert not issues
    
    # Test with negative goals
    sample_data.loc[0, 'GF(tm)'] = -1
    score, issues = quality_checker.check_validity(sample_data)
    assert score < 1.0
    assert any("GF(tm)" in issue and "negative values" in issue for issue in issues)
    
    # Test with negative corners
    sample_data.loc[1, 'Corners_Local'] = -5
    score, issues = quality_checker.check_validity(sample_data)
    assert score < 1.0
    assert any("Corners_Local" in issue and "negative values" in issue for issue in issues)
    
    # Test with negative yellow cards
    sample_data.loc[2, 'Amarillas_Local'] = -2
    score, issues = quality_checker.check_validity(sample_data)
    assert score < 1.0
    assert any("Amarillas_Local" in issue and "negative values" in issue for issue in issues)
    
    # Test with high but valid values
    sample_data.loc[3, 'GF(tm)'] = 15  # Unusually high but valid
    sample_data.loc[3, 'Corners_Local'] = 40  # Unusually high but valid
    sample_data.loc[3, 'Amarillas_Local'] = 20  # Unusually high but valid
    score, issues = quality_checker.check_validity(sample_data)
    # These high values should not affect the score or create issues
    assert not any("outside valid range" in issue for issue in issues)

def test_overall_quality(quality_checker, sample_data):
    """Test overall quality assessment."""
    # Test with good quality data
    metrics = quality_checker.check_quality(sample_data)
    assert isinstance(metrics, DataQualityMetrics)
    assert metrics.overall_score > 0.9
    assert not metrics.issues
    
    # Test with poor quality data
    sample_data.iloc[0, 0] = np.nan  # Add missing value
    sample_data.loc[1, 'Fecha'] = datetime.now() + timedelta(days=365)  # Add future date
    sample_data['Sem.'] = sample_data['Sem.'].astype(str)  # Wrong type
    
    metrics = quality_checker.check_quality(sample_data)
    assert metrics.overall_score < 0.9
    assert len(metrics.issues) > 0

def test_minimum_data_validation(quality_checker, sample_data):
    """Test minimum data requirement validation."""
    # Test with sufficient data
    assert quality_checker.validate_minimum_data(sample_data)
    
    # Test with insufficient data
    insufficient_data = sample_data.iloc[:2]
    assert not quality_checker.validate_minimum_data(insufficient_data)

def test_edge_cases(quality_checker):
    """Test edge cases and error handling."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    metrics = quality_checker.check_quality(empty_df)
    assert metrics.overall_score == 0
    assert len(metrics.issues) > 0
    
    # Test with single row
    single_row = pd.DataFrame({
        'Sem.': [1],
        'Local': ['Team A'],
        'Visitante': ['Team B'],
        'Estado': ['Fin'],
        'Fecha': [datetime.now()]
    })
    metrics = quality_checker.check_quality(single_row)
    assert metrics.overall_score > 0
    
    # Test with all null values
    null_df = pd.DataFrame({
        'Sem.': [np.nan] * 3,
        'Local': [np.nan] * 3,
        'Visitante': [np.nan] * 3,
        'Estado': [np.nan] * 3,
        'Fecha': [np.nan] * 3
    })
    metrics = quality_checker.check_quality(null_df)
    assert metrics.completeness == 0
    assert len(metrics.issues) > 0 

def test_overall_quality_with_new_features(quality_checker, sample_data):
    """Test overall quality assessment with new features."""
    # Test with good quality data
    metrics = quality_checker.check_quality(sample_data)
    assert isinstance(metrics, DataQualityMetrics)
    assert metrics.overall_score > 0.9
    assert not metrics.issues
    
    # Test with poor quality data
    sample_data.iloc[0, 0] = np.nan  # Add missing value
    sample_data.loc[1, 'Corners_Local'] = -1  # Negative corners
    sample_data.loc[2, 'Amarillas_Local'] = -2  # Negative yellow cards
    sample_data['GF(tm)'] = sample_data['GF(tm)'].astype(str)  # Wrong type
    
    metrics = quality_checker.check_quality(sample_data)
    assert metrics.overall_score < 0.9
    assert len(metrics.issues) > 0
    assert any("Corners_Local" in issue and "negative values" in issue for issue in metrics.issues)
    assert any("Amarillas_Local" in issue and "negative values" in issue for issue in metrics.issues)
    assert any("GF(tm)" in issue for issue in metrics.issues) 