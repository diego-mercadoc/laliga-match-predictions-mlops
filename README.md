# LaLiga Match Outcome Prediction System

A comprehensive machine learning system for predicting various outcomes in Spanish La Liga football matches, including match results, goals scored, corners awarded, and yellow cards received.

## Features

- **Match Predictions**
  - Win/Draw/Loss probabilities
  - Expected goals for each team
  - Expected corners for each team
  - Expected yellow cards for each team
  - Confidence scores for predictions

- **MLOps Features**
  - Automated model retraining pipeline
  - A/B testing with champion/challenger models
  - Model performance monitoring
  - Data validation pipeline
  - Feature importance analysis
  - Model drift detection

- **Monitoring & Visualization**
  - Real-time performance metrics
  - Custom Grafana dashboards
  - Prometheus metrics integration
  - Automated alerts for anomalies
  - Various visualization types for analysis

- **API Features**
  - RESTful endpoints
  - JWT authentication
  - Rate limiting
  - Comprehensive error handling
  - Swagger/OpenAPI documentation

## Data Sources

The system integrates data from three complementary sources to ensure robust predictions:

### Historical Data (`laliga.csv`)
- Primary source for win/draw/loss predictions
- Contains historical match statistics and outcomes
- Used for feature engineering and model training
- Maintained for historical analysis and model validation

### Current Season Reference (`LaLiga Dataset 2023-2024.xlsx`)
- Reference data for current season matches
- Used to validate live data fetches
- Provides backup when API is unavailable
- Contains additional features for goals/corners/cards predictions

### Live Data (fbref.com API)
- Real-time match data and statistics
- Updates current season information
- Provides latest team performance metrics
- Source for new prediction features

### Data Flow
1. Historical data → Feature engineering → Model training
2. Current season data → Validation → Feature updates
3. Live data → Real-time predictions → Model serving

### Data Quality Measures
- Automated validation between sources
- Completeness checks for required features
- Consistency validation across data sources
- Timeliness checks for live data updates

## Architecture

The system follows a microservices architecture with the following components:

- **Backend (FastAPI)**
  - Model serving
  - API endpoints
  - Data validation
  - Performance monitoring

- **MLflow**
  - Experiment tracking
  - Model registry
  - Model versioning
  - Artifact storage

- **MinIO**
  - Object storage for models
  - Backup storage
  - Data versioning

- **Prometheus & Grafana**
  - Metrics collection
  - Performance visualization
  - Alert management
  - Custom dashboards

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- 8GB+ RAM
- 20GB+ disk space

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/laliga-predictions.git
   cd laliga-predictions
   ```

2. Create a `.env` file:
   ```bash
   cp src/.env.example src/.env
   ```

3. Update the environment variables in `.env` with your settings.

4. Build and start the services:
   ```bash
   cd src
   docker-compose up -d
   ```

5. Access the services:
   - API: http://localhost:8000
   - MLflow: http://localhost:5000
   - MinIO Console: http://localhost:9001
   - Grafana: http://localhost:3000
   - Prometheus: http://localhost:9090

## Usage

### Making Predictions

```python
import requests

def predict_match(home_team, away_team, date):
    url = "http://localhost:8000/predict"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "home_team": home_team,
        "away_team": away_team,
        "date": date,
        "home_form": [1, 1, 0, 1, 0],
        "away_form": [1, 1, 1, 0, 1],
        "head_to_head": [1, 0, 1, 1, 0]
    }
    
    response = requests.post(url, json=data, headers=headers)
    return response.json()

# Example usage
prediction = predict_match("Real Madrid", "Barcelona", "2024-01-20")
print(prediction)
```

### Monitoring Performance

```python
import requests
from datetime import datetime, timedelta

def monitor_model_performance():
    url = "http://localhost:8000/model/performance"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(url, headers=headers)
    metrics = response.json()
    
    # Alert if accuracy drops below threshold
    if metrics["accuracy"] < 0.7:
        send_alert("Low model accuracy detected")
    
    # Check for model drift
    if metrics["drift_score"] > 0.3:
        schedule_retraining()
    
    return metrics

# Run monitoring hourly
while True:
    metrics = monitor_model_performance()
    print(f"Current model accuracy: {metrics['accuracy']}")
    time.sleep(3600)
```

## Development

### Project Structure

```
.
├── data/
│   ├── LaLiga Dataset 2023-2024.xlsx
│   └── laliga.csv
├── notebooks/
│   ├── EDA_Preprocesado.ipynb
│   ├── EDA_sin_Preprocesar.ipynb
│   ├── Entrega2_Experiments.ipynb
│   └── Preprocesamiento.ipynb
├── src/
│   ├── backend/
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── utils/
│   ├── grafana/
│   │   ├── dashboards/
│   │   └── provisioning/
│   └── prometheus/
│       ├── alert.rules.yml
│       └── prometheus.yml
└── README.md
```

### Adding New Features

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test thoroughly

3. Submit a pull request with:
   - Clear description of changes
   - Any new dependencies
   - Test results
   - Documentation updates

## Testing

Run the test suite:
```bash
cd src/backend
pytest
```

## Documentation

- [API Documentation](src/backend/API_DOCUMENTATION.md)
- [Model Documentation](docs/MODEL.md)
- [MLOps Documentation](docs/MLOPS.md)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data source: [Spanish Football Federation](https://www.rfef.es/)
- MLOps best practices: [MLOps.org](https://ml-ops.org/)
- FastAPI framework: [FastAPI](https://fastapi.tiangolo.com/)
- MLflow: [MLflow](https://mlflow.org/)

## System Architecture Documentation

Detailed system architecture documentation can be found in the following files:

- [Data Dependencies](docs/diagrams/data_dependencies.md) - Overview of data, service, code, and documentation dependencies
- [Data Flow Diagrams](docs/diagrams/data_flow.md) - Detailed data ingestion, model pipeline, and service integration flows
- [Validation Measures](docs/diagrams/validation_measures.md) - Comprehensive validation and quality measures for data and models
- [Failure Recovery](docs/diagrams/failure_recovery.md) - Analysis of failure points and recovery procedures