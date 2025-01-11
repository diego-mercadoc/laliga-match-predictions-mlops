import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import mlflow
import os
import tempfile
from .model_training import ModelTrainer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create features
    features = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    })
    
    # Create targets
    targets = pd.DataFrame({
        'goals': np.random.poisson(2, n_samples),
        'corners': np.random.poisson(5, n_samples),
        'yellow_cards': np.random.poisson(3, n_samples)
    })
    
    return features, targets

@pytest.fixture
def model_trainer():
    """Create ModelTrainer instance."""
    return ModelTrainer(
        experiment_name="test_experiment",
        target_variables=['goals', 'corners', 'yellow_cards'],
        test_size=0.2,
        random_state=42
    )

def test_initialization(model_trainer):
    """Test ModelTrainer initialization."""
    assert model_trainer.experiment_name == "test_experiment"
    assert model_trainer.target_variables == ['goals', 'corners', 'yellow_cards']
    assert model_trainer.test_size == 0.2
    assert model_trainer.random_state == 42
    
    # Check default parameters
    assert isinstance(model_trainer.default_params, dict)
    assert 'max_depth' in model_trainer.default_params
    assert 'learning_rate' in model_trainer.default_params
    assert 'n_estimators' in model_trainer.default_params

def test_prepare_data(model_trainer, sample_data):
    """Test data preparation."""
    features_df, target_df = sample_data
    prepared_data = model_trainer.prepare_data(features_df, target_df)
    
    # Check structure
    assert isinstance(prepared_data, dict)
    assert all(target in prepared_data for target in model_trainer.target_variables)
    
    # Check data splits
    for target in model_trainer.target_variables:
        data = prepared_data[target]
        assert 'X_train' in data
        assert 'X_test' in data
        assert 'y_train' in data
        assert 'y_test' in data
        
        # Check shapes
        n_samples = len(features_df)
        expected_test_samples = int(n_samples * model_trainer.test_size)
        expected_train_samples = n_samples - expected_test_samples
        
        assert data['X_train'].shape[0] == expected_train_samples
        assert data['X_test'].shape[0] == expected_test_samples
        assert data['y_train'].shape[0] == expected_train_samples
        assert data['y_test'].shape[0] == expected_test_samples
        
        # Check scaling
        assert isinstance(model_trainer.scalers[target], StandardScaler)

def test_train_models(model_trainer, sample_data):
    """Test model training."""
    features_df, target_df = sample_data
    prepared_data = model_trainer.prepare_data(features_df, target_df)
    
    # Train models
    model_trainer.train_models(prepared_data)
    
    # Check models
    assert len(model_trainer.models) == len(model_trainer.target_variables)
    for target in model_trainer.target_variables:
        assert isinstance(model_trainer.models[target], xgb.XGBRegressor)

def test_evaluate_models(model_trainer, sample_data):
    """Test model evaluation."""
    features_df, target_df = sample_data
    prepared_data = model_trainer.prepare_data(features_df, target_df)
    model_trainer.train_models(prepared_data)
    
    # Evaluate models
    evaluation_results = model_trainer.evaluate_models(prepared_data)
    
    # Check results
    assert isinstance(evaluation_results, dict)
    assert all(target in evaluation_results for target in model_trainer.target_variables)
    
    for target, metrics in evaluation_results.items():
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Check metric values
        assert all(isinstance(value, float) for value in metrics.values())
        assert all(value >= 0 for value in metrics.values())  # Non-negative metrics
        assert metrics['r2'] <= 1.0  # R² should be <= 1

def test_cross_validate(model_trainer, sample_data):
    """Test cross-validation."""
    features_df, target_df = sample_data
    
    # Perform cross-validation
    cv_results = model_trainer.cross_validate(features_df, target_df, cv=3)
    
    # Check results
    assert isinstance(cv_results, dict)
    assert all(target in cv_results for target in model_trainer.target_variables)
    
    for target, metrics in cv_results.items():
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Check scores
        assert all(isinstance(scores, np.ndarray) for scores in metrics.values())
        assert all(len(scores) == 3 for scores in metrics.values())  # 3-fold CV

def test_save_load_models(model_trainer, sample_data):
    """Test saving and loading models."""
    features_df, target_df = sample_data
    prepared_data = model_trainer.prepare_data(features_df, target_df)
    model_trainer.train_models(prepared_data)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models
        model_trainer.save_models(temp_dir)
        
        # Check saved files
        for target in model_trainer.target_variables:
            assert os.path.exists(os.path.join(temp_dir, f"{target}_model.json"))
            assert os.path.exists(os.path.join(temp_dir, f"{target}_scaler.pkl"))
        
        # Create new trainer and load models
        new_trainer = ModelTrainer(
            experiment_name="test_experiment",
            target_variables=model_trainer.target_variables
        )
        new_trainer.load_models(temp_dir)
        
        # Check loaded models
        assert len(new_trainer.models) == len(model_trainer.models)
        assert len(new_trainer.scalers) == len(model_trainer.scalers)

def test_predict(model_trainer, sample_data):
    """Test making predictions."""
    features_df, target_df = sample_data
    prepared_data = model_trainer.prepare_data(features_df, target_df)
    model_trainer.train_models(prepared_data)
    
    # Make predictions
    predictions = model_trainer.predict(features_df)
    
    # Check predictions
    assert isinstance(predictions, dict)
    assert all(target in predictions for target in model_trainer.target_variables)
    
    for target, pred in predictions.items():
        assert isinstance(pred, np.ndarray)
        assert len(pred) == len(features_df)
        assert all(p >= 0 for p in pred)  # Non-negative predictions for count data

def test_calculate_metrics():
    """Test metric calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    trainer = ModelTrainer("test", ["test"])
    metrics = trainer._calculate_metrics(y_true, y_pred)
    
    # Check metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    
    # Check values
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0
    assert metrics['r2'] <= 1.0

def test_edge_cases(model_trainer):
    """Test edge cases and error handling."""
    # Empty data
    empty_features = pd.DataFrame(columns=['feature1', 'feature2', 'feature3'])
    empty_targets = pd.DataFrame(columns=model_trainer.target_variables)
    
    with pytest.raises(ValueError):
        model_trainer.prepare_data(empty_features, empty_targets)
    
    # Single sample
    single_features = pd.DataFrame({
        'feature1': [1],
        'feature2': [2],
        'feature3': [3]
    })
    single_targets = pd.DataFrame({
        'goals': [2],
        'corners': [5],
        'yellow_cards': [3]
    })
    
    with pytest.raises(ValueError):
        model_trainer.cross_validate(single_features, single_targets)
    
    # Missing features
    missing_features = pd.DataFrame({
        'feature1': [1, 2, np.nan],
        'feature2': [np.nan, 5, 6],
        'feature3': [7, 8, 9]
    })
    valid_targets = pd.DataFrame({
        'goals': [2, 3, 1],
        'corners': [5, 4, 6],
        'yellow_cards': [3, 2, 1]
    })
    
    prepared_data = model_trainer.prepare_data(missing_features, valid_targets)
    model_trainer.train_models(prepared_data)
    predictions = model_trainer.predict(missing_features)
    
    assert all(not np.isnan(pred).any() for pred in predictions.values()) 