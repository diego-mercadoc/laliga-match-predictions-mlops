# MLOps Documentation

## Overview

This document outlines the MLOps practices and workflows implemented in the LaLiga Match Prediction System. It covers model lifecycle management, training pipelines, monitoring, and deployment strategies.

## Model Management

### Model Registry
- Registry Name: "LaLigaPredictionsModel"
- Aliases:
  - "champion": Best performing model
  - "challenger": Model being evaluated
  - "LaLigaBestModel": Production model

### Model Versioning
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "LaLigaPredictionsModel"
model_version = client.create_model_version(
    name=model_name,
    source="runs:/run_id/model",
    run_id="run_id"
)
```

## Training Pipeline

### Data Processing
1. Data Collection
   - Historical data from `laliga.csv`
   - Current season data from Excel
   - Live data from fbref.com API

2. Feature Engineering
   - Team performance metrics
   - Form indicators
   - Head-to-head statistics
   - Match context features

3. Feature Selection
   - Correlation analysis
   - Lasso/Ridge selection
   - Recursive feature elimination
   - Target-specific selection

### Model Training
```python
@task
def train_model(X_train, y_train, params):
    with mlflow.start_run():
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        mlflow.log_params(params)
        return model

@flow
def training_pipeline():
    data = load_data()
    features = engineer_features(data)
    model = train_model(features)
    evaluate_model(model)
```

## Monitoring

### Metrics Collection
- Model performance metrics
- Prediction latency
- Feature drift
- Data quality metrics

### Alerts
```yaml
alerts:
  - name: model_accuracy_drop
    condition: accuracy < 0.7
    duration: 5m
    
  - name: prediction_latency
    condition: latency > 200ms
    duration: 1m
    
  - name: data_drift
    condition: drift_score > 0.3
    duration: 1h
```

## Deployment

### Model Deployment
1. Champion/Challenger Pattern
   ```python
   def promote_challenger():
       client = MlflowClient()
       challenger = client.get_model_version_by_alias(
           name="LaLigaPredictionsModel",
           alias="challenger"
       )
       if challenger.metrics["accuracy"] > current_champion.metrics["accuracy"]:
           client.set_registered_model_alias(
               name="LaLigaPredictionsModel",
               alias="champion",
               version=challenger.version
           )
   ```

2. Rollback Procedure
   ```python
   def rollback_model():
       client = MlflowClient()
       previous_version = client.get_model_version(
           name="LaLigaPredictionsModel",
           version=current_version - 1
       )
       client.transition_model_version_stage(
           name="LaLigaPredictionsModel",
           version=previous_version.version,
           stage="Production"
       )
   ```

## Experiment Tracking

### MLflow Configuration
```python
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("laliga_predictions")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(artifacts_dir)
```

### Experiment Organization
- Separate experiments for each prediction type
- Tagged runs for feature selection studies
- Nested runs for hyperparameter tuning

## Quality Assurance

### Data Validation
```python
def validate_data(df: pd.DataFrame) -> bool:
    checks = [
        check_missing_values(df),
        check_data_types(df),
        check_value_ranges(df),
        check_feature_correlation(df)
    ]
    return all(checks)
```

### Model Validation
- Cross-validation metrics
- Feature importance analysis
- Prediction confidence scores
- Error analysis

## Automation

### Prefect Flows
```python
@flow
def end_to_end_pipeline():
    with mlflow.start_run():
        # Data processing
        data = extract_data()
        features = transform_data(data)
        
        # Model training
        model = train_model(features)
        metrics = evaluate_model(model)
        
        # Model registration
        if metrics["accuracy"] > threshold:
            register_model(model)
```

### Scheduled Jobs
```python
from prefect.schedules import IntervalSchedule

schedule = IntervalSchedule(
    interval=timedelta(days=1),
    start_date=datetime.utcnow()
)

@flow(schedule=schedule)
def daily_retraining():
    run_training_pipeline()
    update_model_registry()
```

## Best Practices

1. Version Control
   - Code versioning with Git
   - Data versioning with DVC
   - Model versioning with MLflow

2. Testing
   - Unit tests for components
   - Integration tests for pipelines
   - Model validation tests

3. Documentation
   - Code documentation
   - Model cards
   - Experiment tracking
   - Pipeline documentation

4. Monitoring
   - Performance metrics
   - Resource utilization
   - Error tracking
   - Data drift detection 