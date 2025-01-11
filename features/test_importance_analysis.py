import pytest
import pandas as pd
import numpy as np
from importance_analysis import analyze_target_importance, analyze_all_targets, TARGETS, FEATURE_SETS
import os
from sklearn.ensemble import RandomForestRegressor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create base DataFrame with common features
    df = pd.DataFrame({
        'Sedes': np.random.randint(0, 2, n_samples),
        'Día': np.random.randint(1, 8, n_samples),
    })
    
    # Add features for goals
    for feature in FEATURE_SETS['goals']:
        if feature not in df:
            df[feature] = np.random.uniform(0, 4, n_samples)
    
    # Add features for corners
    for feature in FEATURE_SETS['corners']:
        if feature not in df:
            df[feature] = np.random.uniform(0, 12, n_samples)
    
    # Add features for cards
    for feature in FEATURE_SETS['cards']:
        if feature not in df:
            df[feature] = np.random.uniform(0, 5, n_samples)
    
    # Add target variables
    for target_list in TARGETS.values():
        for target in target_list:
            if target not in df:
                df[target] = np.random.randint(0, 5, n_samples)
    
    return df

def test_analyze_target_importance(sample_data):
    """Test the analyze_target_importance function for each target type."""
    for target_type in TARGETS.keys():
        result = analyze_target_importance(sample_data, target_type)
        
        # Check basic structure
        assert isinstance(result, pd.DataFrame)
        assert 'feature' in result.columns
        assert 'avg_importance_home' in result.columns
        assert 'avg_importance_away' in result.columns
        assert 'avg_importance' in result.columns
        
        # Check that we have results for all models
        expected_columns = [
            'feature',
            'rf_home', 'rf_away',
            'xgb_home', 'xgb_away',
            'lr_home', 'lr_away',
            'ridge_home', 'ridge_away',
            'lasso_home', 'lasso_away',
            'svr_home', 'svr_away',
            'avg_importance_home',
            'avg_importance_away',
            'avg_importance'
        ]
        for col in expected_columns:
            assert col in result.columns
        
        # Check that importance values are valid
        assert (result['avg_importance'] >= 0).all()
        assert (result['avg_importance_home'] >= 0).all()
        assert (result['avg_importance_away'] >= 0).all()
        
        # Check that we have the correct number of features
        assert len(result) == len(FEATURE_SETS[target_type])
        
        # Check that features are sorted by importance
        assert (result['avg_importance'].diff().fillna(0) <= 0).all()  # Descending order
        
        # Check that plot was generated
        assert os.path.exists(f'plots/importance_{target_type}.png')

def test_analyze_all_targets(sample_data):
    """Test the analyze_all_targets function."""
    results = analyze_all_targets(sample_data)
    
    # Check that we have results for all targets
    assert set(results.keys()) == set(TARGETS.keys())
    
    # Check each result
    for target_type, result in results.items():
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(FEATURE_SETS[target_type])
        assert 'avg_importance' in result.columns
        assert (result['avg_importance'] >= 0).all()

def test_stability_score(sample_data):
    """Test feature importance stability calculation."""
    # Prepare test data
    features = FEATURE_SETS['goals']
    X = sample_data[features]
    y = sample_data[TARGETS['goals'][0]]
    
    # Test with RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    stability_results = calculate_stability_score(model, X, y, features, n_iterations=10)
    
    # Check results structure
    assert isinstance(stability_results, pd.DataFrame)
    assert all(col in stability_results.columns 
              for col in ['feature', 'mean_importance', 'std_importance', 'stability_score'])
    
    # Check values are in valid ranges
    assert all(0 <= score <= 1 for score in stability_results['stability_score'])
    assert all(score >= 0 for score in stability_results['mean_importance'])
    assert all(score >= 0 for score in stability_results['std_importance'])
    
    # Test with different models
    for model_name, model in MODELS.items():
        try:
            stability_results = calculate_stability_score(model, X, y, features, n_iterations=5)
            assert isinstance(stability_results, pd.DataFrame)
            assert len(stability_results) == len(features)
        except Exception as e:
            # Some models might not support feature importance
            assert str(e) in ["Model doesn't support feature importance analysis"]

def test_weighted_importance(sample_data):
    """Test weighted importance calculation with stability scores."""
    # Run importance analysis
    importance_df = analyze_target_importance(sample_data, 'goals')
    
    # Check that weighted averages are calculated
    assert 'avg_importance_home' in importance_df.columns
    assert 'avg_importance_away' in importance_df.columns
    assert 'avg_importance' in importance_df.columns
    
    # Check that stability scores are used in weighting
    stability_cols = [col for col in importance_df.columns if col.endswith('_stability')]
    assert len(stability_cols) > 0
    
    # Verify weights are properly applied
    assert all(0 <= score <= 1 for score in importance_df['avg_importance'])
    assert all(importance_df['avg_importance'] <= 
              importance_df[['avg_importance_home', 'avg_importance_away']].max(axis=1))

def test_interaction_importance(sample_data):
    """Test feature interaction importance calculation."""
    # Prepare test data
    features = FEATURE_SETS['goals']
    X = sample_data[features]
    y = sample_data[TARGETS['goals'][0]]
    
    # Test with RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    
    # Test with specific feature pairs
    feature_pairs = [
        (features[0], features[1]),
        (features[1], features[2])
    ]
    interaction_df = calculate_interaction_importance(
        model, X, y, feature_pairs=feature_pairs
    )
    
    # Check results structure
    assert isinstance(interaction_df, pd.DataFrame)
    assert all(col in interaction_df.columns for col in [
        'feature1', 'feature2', 'interaction_score',
        'individual_score1', 'individual_score2', 'joint_score'
    ])
    assert len(interaction_df) == len(feature_pairs)
    
    # Check value ranges
    assert all(isinstance(score, (int, float)) for score in interaction_df['interaction_score'])
    assert all(isinstance(score, (int, float)) for score in interaction_df['individual_score1'])
    assert all(isinstance(score, (int, float)) for score in interaction_df['individual_score2'])
    assert all(isinstance(score, (int, float)) for score in interaction_df['joint_score'])

def test_permutation_importance(sample_data):
    """Test permutation importance calculation."""
    # Prepare test data
    features = FEATURE_SETS['goals']
    X = sample_data[features]
    y = sample_data[TARGETS['goals'][0]]
    
    # Test with RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    
    # Test single feature
    score_single = _permutation_importance(model, X, y, features[0], n_iterations=5)
    assert isinstance(score_single, float)
    
    # Test multiple features
    score_multiple = _permutation_importance(model, X, y, features[:2], n_iterations=5)
    assert isinstance(score_multiple, float)

def test_interaction_analysis_integration(sample_data):
    """Test integration of interaction analysis with main importance analysis."""
    # Run importance analysis for goals
    importance_df, interaction_df = analyze_target_importance(sample_data, 'goals')
    
    # Check importance results
    assert isinstance(importance_df, pd.DataFrame)
    assert 'avg_importance' in importance_df.columns
    assert len(importance_df) == len(FEATURE_SETS['goals'])
    
    # Check interaction results
    assert isinstance(interaction_df, pd.DataFrame)
    if not interaction_df.empty:
        assert all(col in interaction_df.columns for col in [
            'feature1', 'feature2', 'interaction_score',
            'individual_score1', 'individual_score2', 'joint_score'
        ])
        
        # Check that interaction plot was generated
        assert os.path.exists(f'plots/interaction_goals.png')

def test_plot_interaction_importance(sample_data):
    """Test interaction importance plotting."""
    # Create sample interaction results
    interaction_df = pd.DataFrame({
        'feature1': ['f1', 'f1', 'f2'],
        'feature2': ['f2', 'f3', 'f3'],
        'interaction_score': [0.5, 0.3, 0.2],
        'individual_score1': [0.4, 0.3, 0.2],
        'individual_score2': [0.3, 0.2, 0.1],
        'joint_score': [1.2, 0.8, 0.5]
    })
    
    # Generate plot
    plot_interaction_importance(interaction_df, 'test')
    
    # Check that plot was generated
    assert os.path.exists('plots/interaction_test.png')

def test_interaction_importance():
    """Test feature interaction importance calculation."""
    # Create synthetic dataset with known interactions
    n_samples = 1000
    X = pd.DataFrame({
        'f1': np.random.normal(0, 1, n_samples),
        'f2': np.random.normal(0, 1, n_samples),
        'f3': np.random.normal(0, 1, n_samples)
    })
    
    # Create target with interaction between f1 and f2
    X['f1*f2'] = X['f1'] * X['f2']
    y = 2*X['f1'] + 3*X['f2'] + 5*X['f1*f2'] + X['f3'] + np.random.normal(0, 0.1, n_samples)
    X = X.drop('f1*f2', axis=1)
    
    # Fit random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Calculate interaction importance
    interaction_df = calculate_interaction_importance(model, X, y, n_iterations=10)
    
    # Assertions
    assert isinstance(interaction_df, pd.DataFrame)
    assert len(interaction_df) == 3  # Number of possible pairs for 3 features
    assert all(col in interaction_df.columns for col in ['feature1', 'feature2', 'interaction_score'])
    
    # Check that f1-f2 interaction is strongest
    f1f2_score = interaction_df[
        ((interaction_df['feature1'] == 'f1') & (interaction_df['feature2'] == 'f2')) |
        ((interaction_df['feature1'] == 'f2') & (interaction_df['feature2'] == 'f1'))
    ]['interaction_score'].iloc[0]
    
    assert f1f2_score == interaction_df['interaction_score'].max()

def test_permutation_importance():
    """Test permutation importance calculation."""
    # Create synthetic dataset
    n_samples = 1000
    X = pd.DataFrame({
        'f1': np.random.normal(0, 1, n_samples),
        'f2': np.random.normal(0, 1, n_samples)
    })
    y = 2*X['f1'] + 0.5*X['f2'] + np.random.normal(0, 0.1, n_samples)
    
    # Fit random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Calculate permutation importance
    imp_f1 = _permutation_importance(model, X, y, 'f1', n_iterations=10)
    imp_f2 = _permutation_importance(model, X, y, 'f2', n_iterations=10)
    
    # Assertions
    assert isinstance(imp_f1, float)
    assert isinstance(imp_f2, float)
    assert imp_f1 > 0  # Important feature should have positive importance
    assert imp_f1 > imp_f2  # f1 should be more important than f2

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up generated files after tests."""
    yield
    # Remove generated plots after tests
    if os.path.exists('plots'):
        for file in os.listdir('plots'):
            os.remove(os.path.join('plots', file))
        os.rmdir('plots') 