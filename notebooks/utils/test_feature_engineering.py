import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .feature_engineering import FeatureEngineer
from .match_exceptions import MatchStatus

@pytest.fixture
def sample_data():
    """Create sample match data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10)
    
    data = {
        'Sem.': range(1, 11),
        'Local': ['Team A', 'Team B', 'Team C', 'Team A', 'Team B',
                 'Team C', 'Team A', 'Team B', 'Team C', 'Team A'],
        'Visitante': ['Team B', 'Team C', 'Team A', 'Team C', 'Team A',
                     'Team B', 'Team B', 'Team C', 'Team A', 'Team C'],
        'Estado': [MatchStatus.COMPLETED.value] * 8 + [MatchStatus.SCHEDULED.value] * 2,
        'Fecha': dates,
        'Goles_Local': [2, 1, 0, 3, 1, 2, 1, 0, np.nan, np.nan],
        'Goles_Visitante': [1, 1, 2, 0, 2, 2, 1, 1, np.nan, np.nan],
        'Corners_Local': [5, 4, 3, 6, 4, 5, 4, 3, np.nan, np.nan],
        'Corners_Visitante': [4, 3, 5, 2, 5, 4, 3, 4, np.nan, np.nan],
        'Amarillas_Local': [2, 1, 3, 1, 2, 2, 1, 2, np.nan, np.nan],
        'Amarillas_Visitante': [3, 2, 1, 2, 1, 3, 2, 1, np.nan, np.nan]
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer(
        rolling_window=3,
        min_matches=2,
        home_advantage_weight=1.2
    )

def test_initialization(feature_engineer):
    """Test FeatureEngineer initialization."""
    assert feature_engineer.rolling_window == 3
    assert feature_engineer.min_matches == 2
    assert feature_engineer.home_advantage_weight == 1.2
    
    # Check feature groups
    assert all(f in feature_engineer.form_features for f in [
        'wins', 'draws', 'losses', 'goals_scored', 'goals_conceded',
        'corners_for', 'corners_against', 'yellow_cards'
    ])
    
    assert all(f in feature_engineer.head2head_features for f in [
        'h2h_wins', 'h2h_draws', 'h2h_losses',
        'h2h_goals_scored', 'h2h_goals_conceded',
        'h2h_corners_for', 'h2h_corners_against',
        'h2h_yellow_cards'
    ])
    
    assert all(f in feature_engineer.time_features for f in [
        'days_since_last_match',
        'is_weekend',
        'is_evening_match'
    ])

def test_calculate_team_form(feature_engineer, sample_data):
    """Test team form calculation."""
    date = datetime(2024, 1, 8)
    
    # Test home team form
    home_form = feature_engineer._calculate_team_form(
        sample_data, 'Team A', date, True
    )
    
    assert isinstance(home_form, dict)
    assert all(key in home_form for key in feature_engineer.form_features)
    assert all(isinstance(value, (float, np.float64)) for value in home_form.values())
    
    # Test away team form
    away_form = feature_engineer._calculate_team_form(
        sample_data, 'Team B', date, False
    )
    
    assert isinstance(away_form, dict)
    assert all(key in away_form for key in feature_engineer.form_features)
    assert all(isinstance(value, (float, np.float64)) for value in away_form.values())
    
    # Test home advantage weight
    for key in home_form:
        if not np.isnan(home_form[key]):
            assert home_form[key] > away_form[key]

def test_calculate_head2head(feature_engineer, sample_data):
    """Test head-to-head statistics calculation."""
    date = datetime(2024, 1, 8)
    
    h2h_stats = feature_engineer._calculate_head2head(
        sample_data, 'Team A', 'Team B', date
    )
    
    assert isinstance(h2h_stats, dict)
    assert all(key in h2h_stats for key in feature_engineer.head2head_features)
    assert all(isinstance(value, (float, np.float64)) for value in h2h_stats.values())
    
    # Test with teams that haven't played each other
    no_h2h_stats = feature_engineer._calculate_head2head(
        sample_data, 'Team D', 'Team E', date
    )
    
    assert all(np.isnan(value) for value in no_h2h_stats.values())

def test_calculate_time_features(feature_engineer, sample_data):
    """Test time-based features calculation."""
    date = datetime(2024, 1, 8, 19, 0)  # Evening match on a Monday
    
    time_features = feature_engineer._calculate_time_features(
        sample_data, 'Team A', date
    )
    
    assert isinstance(time_features, dict)
    assert all(key in time_features for key in feature_engineer.time_features)
    
    assert isinstance(time_features['days_since_last_match'], (int, float))
    assert isinstance(time_features['is_weekend'], int)
    assert isinstance(time_features['is_evening_match'], int)
    
    assert time_features['is_weekend'] == 0  # Monday
    assert time_features['is_evening_match'] == 1  # 19:00

def test_engineer_features(feature_engineer, sample_data):
    """Test full feature engineering process."""
    features_df, feature_names = feature_engineer.engineer_features(
        sample_data,
        target_date=datetime(2024, 1, 8)
    )
    
    assert isinstance(features_df, pd.DataFrame)
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    
    # Check that all expected feature groups are present
    expected_prefixes = ['home_', 'away_', 'h2h_']
    for prefix in expected_prefixes:
        assert any(f.startswith(prefix) for f in feature_names)
    
    # Check that only completed matches are included
    assert len(features_df) == sum(sample_data['Estado'] == MatchStatus.COMPLETED.value)
    
    # Check for missing values
    assert not features_df.isnull().all().any()  # No completely null columns
    
    # Check value ranges
    assert features_df.filter(like='_wins').between(0, 1).all().all()
    assert features_df.filter(like='_draws').between(0, 1).all().all()
    assert features_df.filter(like='_losses').between(0, 1).all().all()
    assert features_df.filter(like='is_weekend').isin([0, 1]).all().all()
    assert features_df.filter(like='is_evening_match').isin([0, 1]).all().all()

def test_edge_cases(feature_engineer):
    """Test edge cases and error handling."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame(columns=[
        'Sem.', 'Local', 'Visitante', 'Estado', 'Fecha',
        'Goles_Local', 'Goles_Visitante', 'Corners_Local', 'Corners_Visitante',
        'Amarillas_Local', 'Amarillas_Visitante'
    ])
    
    features_df, feature_names = feature_engineer.engineer_features(empty_df)
    assert len(features_df) == 0
    assert len(feature_names) == 0
    
    # Test with single match
    single_match_df = pd.DataFrame({
        'Sem.': [1],
        'Local': ['Team A'],
        'Visitante': ['Team B'],
        'Estado': [MatchStatus.COMPLETED.value],
        'Fecha': [datetime.now()],
        'Goles_Local': [1],
        'Goles_Visitante': [0],
        'Corners_Local': [5],
        'Corners_Visitante': [4],
        'Amarillas_Local': [2],
        'Amarillas_Visitante': [1]
    })
    
    features_df, feature_names = feature_engineer.engineer_features(single_match_df)
    assert len(features_df) == 1
    
    # Test with all matches in future
    future_df = pd.DataFrame({
        'Sem.': [1],
        'Local': ['Team A'],
        'Visitante': ['Team B'],
        'Estado': [MatchStatus.SCHEDULED.value],
        'Fecha': [datetime.now() + timedelta(days=7)],
        'Goles_Local': [np.nan],
        'Goles_Visitante': [np.nan],
        'Corners_Local': [np.nan],
        'Corners_Visitante': [np.nan],
        'Amarillas_Local': [np.nan],
        'Amarillas_Visitante': [np.nan]
    })
    
    features_df, feature_names = feature_engineer.engineer_features(future_df)
    assert len(features_df) == 0 