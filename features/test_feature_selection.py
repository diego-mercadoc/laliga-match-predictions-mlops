import pytest
import pandas as pd
import numpy as np
from features.feature_selection import (
    analyze_correlations,
    lasso_feature_selection,
    ridge_feature_selection,
    rfe_feature_selection,
    select_features_for_target,
    select_features_all_targets,
    select_stable_features
)
import os

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create features with known correlations
    x1 = np.random.normal(0, 1, n_samples)
    x2 = x1 * 0.9 + np.random.normal(0, 0.1, n_samples)  # Highly correlated with x1
    x3 = np.random.normal(0, 1, n_samples)  # Independent
    x4 = np.random.normal(0, 1, n_samples)  # Independent
    y = 2*x1 + 0.5*x3 + 0.1*x4 + np.random.normal(0, 0.1, n_samples)
    
    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'target': y
    })

@pytest.fixture
def feature_sets():
    """Create sample feature sets."""
    return {
        'goals': ['x1', 'x2', 'x3', 'x4'],
        'corners': ['x1', 'x2', 'x3'],
        'cards': ['x2', 'x3', 'x4']
    }

@pytest.fixture
def targets():
    """Create sample targets."""
    return {
        'goals': ['target', 'target'],  # Using same target for simplicity
        'corners': ['target', 'target'],
        'cards': ['target', 'target']
    }

def test_analyze_correlations(sample_data):
    """Test correlation analysis."""
    features = ['x1', 'x2', 'x3', 'x4']
    corr_matrix, to_drop = analyze_correlations(sample_data, features, threshold=0.85)
    
    assert isinstance(corr_matrix, pd.DataFrame)
    assert isinstance(to_drop, list)
    assert len(to_drop) > 0  # Should detect x1-x2 correlation
    assert os.path.exists('plots/correlation_matrix.png')

def test_lasso_feature_selection(sample_data):
    """Test LASSO feature selection."""
    X = sample_data[['x1', 'x2', 'x3', 'x4']]
    y = sample_data['target']
    
    selected = lasso_feature_selection(X, y)
    assert isinstance(selected, list)
    assert len(selected) > 0
    assert all(f in X.columns for f in selected)

def test_ridge_feature_selection(sample_data):
    """Test Ridge feature selection."""
    X = sample_data[['x1', 'x2', 'x3', 'x4']]
    y = sample_data['target']
    
    selected = ridge_feature_selection(X, y)
    assert isinstance(selected, list)
    assert len(selected) > 0
    assert all(f in X.columns for f in selected)

def test_rfe_feature_selection(sample_data):
    """Test RFE feature selection."""
    X = sample_data[['x1', 'x2', 'x3', 'x4']]
    y = sample_data['target']
    
    selected = rfe_feature_selection(X, y)
    assert isinstance(selected, list)
    assert len(selected) > 0
    assert all(f in X.columns for f in selected)
    assert os.path.exists('plots/rfe_scores.png')

def test_select_features_for_target(sample_data):
    """Test complete feature selection for a single target."""
    features = ['x1', 'x2', 'x3', 'x4']
    results = select_features_for_target(sample_data, features, 'target')
    
    assert isinstance(results, dict)
    assert all(method in results for method in ['correlation', 'lasso', 'ridge', 'rfe', 'consensus'])
    assert all(isinstance(features, list) for features in results.values())
    assert all(all(f in sample_data.columns for f in features) for features in results.values())

def test_select_features_all_targets(sample_data, feature_sets, targets):
    """Test feature selection for all targets."""
    results = select_features_all_targets(sample_data, feature_sets, targets)
    
    assert isinstance(results, dict)
    assert all(target in results for target in targets.keys())
    assert all(isinstance(target_results, dict) for target_results in results.values())
    assert all(
        all(method in target_results for method in ['correlation', 'lasso', 'ridge', 'rfe', 'consensus'])
        for target_results in results.values()
    )

def test_select_stable_features(sample_data):
    """Test selection of stable and important features."""
    # Create mock importance DataFrame
    importance_df = pd.DataFrame({
        'feature': ['x1', 'x2', 'x3', 'x4'],
        'rf_home_stability': [0.9, 0.8, 0.3, 0.7],
        'rf_away_stability': [0.85, 0.75, 0.4, 0.65],
        'xgb_home_stability': [0.95, 0.85, 0.35, 0.75],
        'xgb_away_stability': [0.9, 0.8, 0.45, 0.7],
        'avg_importance': [0.3, 0.02, 0.01, 0.005]
    })
    
    # Test with default thresholds
    selected = select_stable_features(importance_df)
    assert isinstance(selected, list)
    assert 'x1' in selected  # High stability and importance
    assert 'x2' in selected  # High stability, borderline importance
    assert 'x3' not in selected  # Low stability
    assert 'x4' not in selected  # Low importance
    
    # Test with custom thresholds
    selected = select_stable_features(importance_df, stability_threshold=0.9, importance_threshold=0.1)
    assert 'x1' in selected  # Only x1 meets both criteria
    assert len(selected) == 1
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame(columns=importance_df.columns)
    selected = select_stable_features(empty_df)
    assert len(selected) == 0

def test_enhanced_feature_selection(sample_data):
    """Test the enhanced feature selection process with stability."""
    features = ['x1', 'x2', 'x3', 'x4']
    results = select_features_for_target(sample_data, features, 'target')
    
    # Check that all selection methods are present
    assert all(method in results for method in 
              ['correlation', 'lasso', 'ridge', 'rfe', 'stable_important', 'consensus'])
    
    # Check that stable_important features are included
    assert isinstance(results['stable_important'], list)
    assert all(f in features for f in results['stable_important'])
    
    # Check that consensus requires more agreement
    consensus_features = results['consensus']
    for feature in consensus_features:
        count = sum(1 for method_results in results.values() if feature in method_results)
        assert count >= 3, f"Feature {feature} selected by fewer than 3 methods"

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up generated files after tests."""
    yield
    # Remove generated plots after tests
    if os.path.exists('plots'):
        for file in os.listdir('plots'):
            os.remove(os.path.join('plots', file))
        os.rmdir('plots') 