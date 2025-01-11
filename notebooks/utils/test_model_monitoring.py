import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import json
from .model_monitoring import ModelMonitor

@pytest.fixture
def sample_monitor():
    """Create ModelMonitor instance for testing."""
    return ModelMonitor(
        experiment_name="test_monitoring",
        target_variables=['goals', 'corners', 'yellow_cards'],
        monitoring_window=5,
        drift_threshold=0.1,
        performance_threshold=0.2
    )

@pytest.fixture
def sample_data():
    """Create sample predictions and actual values."""
    np.random.seed(42)
    n_samples = 10
    
    predictions = {
        'goals': np.random.poisson(2, n_samples),
        'corners': np.random.poisson(5, n_samples),
        'yellow_cards': np.random.poisson(3, n_samples)
    }
    
    actual_values = {
        'goals': predictions['goals'] + np.random.normal(0, 0.5, n_samples),
        'corners': predictions['corners'] + np.random.normal(0, 1, n_samples),
        'yellow_cards': predictions['yellow_cards'] + np.random.normal(0, 0.5, n_samples)
    }
    
    return predictions, actual_values

def test_initialization(sample_monitor):
    """Test ModelMonitor initialization."""
    assert sample_monitor.experiment_name == "test_monitoring"
    assert sample_monitor.target_variables == ['goals', 'corners', 'yellow_cards']
    assert sample_monitor.monitoring_window == 5
    assert sample_monitor.drift_threshold == 0.1
    assert sample_monitor.performance_threshold == 0.2
    
    # Check history initialization
    for target in sample_monitor.target_variables:
        assert isinstance(sample_monitor.prediction_history[target], list)
        assert isinstance(sample_monitor.actual_history[target], list)
        assert isinstance(sample_monitor.error_history[target], list)
        assert isinstance(sample_monitor.drift_history[target], list)

def test_update_history(sample_monitor, sample_data):
    """Test updating monitoring history."""
    predictions, actual_values = sample_data
    timestamp = datetime.now()
    
    # Update history
    sample_monitor.update_history(predictions, actual_values, timestamp)
    
    # Check history updates
    for target in sample_monitor.target_variables:
        assert len(sample_monitor.prediction_history[target]) == len(predictions[target])
        assert len(sample_monitor.actual_history[target]) == len(actual_values[target])
        assert len(sample_monitor.error_history[target]) == len(predictions[target])
        assert len(sample_monitor.drift_history[target]) == 1
        
        # Check drift history format
        drift_entry = sample_monitor.drift_history[target][0]
        assert isinstance(drift_entry, tuple)
        assert isinstance(drift_entry[0], datetime)
        assert isinstance(drift_entry[1], float)

def test_check_performance(sample_monitor, sample_data):
    """Test performance checking."""
    predictions, actual_values = sample_data
    sample_monitor.update_history(predictions, actual_values)
    
    # Check performance
    performance_metrics = sample_monitor.check_performance()
    
    assert isinstance(performance_metrics, dict)
    for target in sample_monitor.target_variables:
        assert target in performance_metrics
        metrics = performance_metrics[target]
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Check metric values
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['r2'] <= 1.0

def test_check_drift(sample_monitor, sample_data):
    """Test drift checking."""
    predictions, actual_values = sample_data
    sample_monitor.update_history(predictions, actual_values)
    
    # Check drift
    drift_analysis = sample_monitor.check_drift()
    
    assert isinstance(drift_analysis, dict)
    for target in sample_monitor.target_variables:
        assert target in drift_analysis
        drifts = drift_analysis[target]
        assert isinstance(drifts, list)
        
        for drift in drifts:
            assert 'timestamp' in drift
            assert 'drift_score' in drift
            assert isinstance(drift['drift_score'], float)
            assert drift['drift_score'] >= 0

def test_generate_monitoring_report(sample_monitor, sample_data):
    """Test monitoring report generation."""
    predictions, actual_values = sample_data
    sample_monitor.update_history(predictions, actual_values)
    
    # Generate report
    report = sample_monitor.generate_monitoring_report()
    
    assert isinstance(report, dict)
    assert 'timestamp' in report
    assert 'performance_metrics' in report
    assert 'drift_analysis' in report
    assert 'alerts' in report
    assert 'statistics' in report
    
    # Check statistics
    stats = report['statistics']
    for target in sample_monitor.target_variables:
        assert target in stats
        target_stats = stats[target]
        assert 'prediction_mean' in target_stats
        assert 'prediction_std' in target_stats
        assert 'actual_mean' in target_stats
        assert 'actual_std' in target_stats
        assert 'error_mean' in target_stats
        assert 'error_std' in target_stats
        assert 'sample_size' in target_stats

def test_save_report(sample_monitor, sample_data):
    """Test saving monitoring report."""
    predictions, actual_values = sample_data
    sample_monitor.update_history(predictions, actual_values)
    report = sample_monitor.generate_monitoring_report()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "monitoring_report.json")
        
        # Save report
        sample_monitor.save_report(report, file_path)
        
        # Check file exists
        assert os.path.exists(file_path)
        
        # Load and verify content
        with open(file_path, 'r') as f:
            loaded_report = json.load(f)
            assert loaded_report == report

def test_calculate_drift():
    """Test drift calculation."""
    monitor = ModelMonitor("test", ["test"])
    
    # Test with no historical data
    new_pred = np.array([1, 2, 3])
    hist_pred = []
    drift_score = monitor._calculate_drift(new_pred, hist_pred)
    assert drift_score == 0.0
    
    # Test with similar distributions
    new_pred = np.array([1, 2, 3, 4, 5])
    hist_pred = [1, 2, 3, 4, 5]
    drift_score = monitor._calculate_drift(new_pred, hist_pred)
    assert drift_score < 0.1  # Small drift
    
    # Test with different distributions
    new_pred = np.array([10, 11, 12, 13, 14])
    hist_pred = [1, 2, 3, 4, 5]
    drift_score = monitor._calculate_drift(new_pred, hist_pred)
    assert drift_score > 0.1  # Large drift

def test_trim_histories(sample_monitor, sample_data):
    """Test history trimming."""
    predictions, actual_values = sample_data
    
    # Update history multiple times
    for _ in range(3):
        sample_monitor.update_history(predictions, actual_values)
    
    # Check histories are trimmed to window size
    for target in sample_monitor.target_variables:
        assert len(sample_monitor.prediction_history[target]) <= sample_monitor.monitoring_window
        assert len(sample_monitor.actual_history[target]) <= sample_monitor.monitoring_window
        assert len(sample_monitor.error_history[target]) <= sample_monitor.monitoring_window
        assert len(sample_monitor.drift_history[target]) <= sample_monitor.monitoring_window

def test_generate_alerts(sample_monitor, sample_data):
    """Test alert generation."""
    predictions, actual_values = sample_data
    
    # Create poor predictions to trigger alerts
    poor_predictions = {
        target: values + np.random.normal(2, 1, len(values))
        for target, values in predictions.items()
    }
    
    # Update history with poor predictions
    sample_monitor.update_history(poor_predictions, actual_values)
    
    # Generate alerts
    alerts = sample_monitor._generate_alerts()
    
    assert isinstance(alerts, list)
    for alert in alerts:
        assert 'type' in alert
        assert alert['type'] in ['performance', 'drift']
        assert 'target' in alert
        assert alert['target'] in sample_monitor.target_variables
        assert 'timestamp' in alert
        assert 'value' in alert
        assert 'threshold' in alert

def test_edge_cases(sample_monitor):
    """Test edge cases and error handling."""
    # Empty predictions
    empty_predictions = {
        target: np.array([])
        for target in sample_monitor.target_variables
    }
    empty_actuals = {
        target: np.array([])
        for target in sample_monitor.target_variables
    }
    
    # Update with empty data
    sample_monitor.update_history(empty_predictions, empty_actuals)
    
    # Check performance metrics with no data
    metrics = sample_monitor.check_performance()
    for target in sample_monitor.target_variables:
        assert all(np.isnan(value) for value in metrics[target].values())
    
    # Single value
    single_predictions = {
        target: np.array([1.0])
        for target in sample_monitor.target_variables
    }
    single_actuals = {
        target: np.array([1.0])
        for target in sample_monitor.target_variables
    }
    
    # Update with single value
    sample_monitor.update_history(single_predictions, single_actuals)
    
    # Check drift calculation with single value
    drift_analysis = sample_monitor.check_drift()
    for target in sample_monitor.target_variables:
        assert len(drift_analysis[target]) > 0 