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
  - Optional `X-API-Key` authentication through the `API_KEY` environment variable
  - PySpark-based FBref feature pipeline
  - Data-quality and feature-preview endpoints
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
   git clone https://github.com/JuanPab2009/ProyectoFinalCD.git
   cd ProyectoFinalCD
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

### Data Quality And Predictions

The production backend runs the PySpark API in `src/backend/api.py`. It prepares the same model feature contract as the original notebook, validates nulls, duplicate/self matches, and missing feature columns before serving predictions.

```bash
cd src
docker-compose up -d backend
```

Check data quality for a jornada:

```bash
curl "http://localhost:8000/data/quality?jornada=12"
```

Preview the model-ready feature rows:

```bash
curl "http://localhost:8000/features/preview?jornada=12&limit=5"
```

Call predictions. Numeric NULL/NaN feature values are imputed with per-column medians by default and reported in the response:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"jornada": 12}'
```

For live MLflow predictions configure `MLFLOW_TRACKING_URI`, `MLFLOW_MODEL_URI`, and the DagsHub/MLflow credentials accepted by MLflow. Without those credentials, `/health`, `/data/quality`, and `/features/preview` still work, while `/predict` returns a clear `503`.

### Automated Data Refresh

Raw La Liga CSVs from Football-Data are cached under `data/raw/football_data`. The refresh runner can update the cache, rebuild temporal features, rerun the model comparison grid, and regenerate API artifacts:

```bash
py -3.11 src\backend\services\refresh_pipeline.py --force-download
```

For a lightweight raw-cache check without retraining:

```bash
py -3.11 src\backend\services\refresh_pipeline.py --skip-experiments
```

The API exposes the latest refresh state at:

```bash
curl http://localhost:8000/refresh/status
```

A scheduled Codex automation named `Refresh LaLiga multi-season data` runs Tuesdays and Fridays at 06:30.

The multi-season pipeline now validates every source file and writes a source manifest to `artifacts/multi_season/data_source_report.json`. Feature engineering includes leakage-safe Elo, season rank before kickoff, 3/5/10-match rolling form, head-to-head history, rest days, shot efficiency, and normalized market-implied probabilities from Football-Data odds. See `docs/data_sources.md` for the source rules and leakage policy.

### GitHub And DagsHub Publishing

The MLOps-ready GitHub remote is:

```bash
git remote add mlops https://github.com/diego-mercadoc/laliga-match-predictions-mlops.git
git push -u mlops HEAD:main
```

DagsHub requires an authenticated DagsHub session before a repository can be created. After logging in, create the matching DagsHub project and push/mirror the same code and artifacts:

```bash
dagshub login --token <your-dagshub-token>
dagshub repo create laliga-match-predictions-mlops
git remote add dagshub https://dagshub.com/<dagshub-user>/laliga-match-predictions-mlops.git
git push dagshub main
```

### Making Predictions From Python

```python
import requests

def predict_jornada(jornada):
    url = "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}
    data = {"jornada": jornada}
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()

prediction = predict_jornada(12)
print(prediction["data_quality"])
```

### Monitoring Data Quality

```python
import requests

def monitor_data_quality(jornada=12):
    url = f"http://localhost:8000/data/quality?jornada={jornada}"
    headers = {}  # Add {"X-API-Key": "..."} if API_KEY is configured.
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    report = response.json()
    
    if not report["is_valid"]:
        send_alert({"jornada": jornada, "issues": report["issues"]})
    
    return report

quality = monitor_data_quality(12)
print(quality["issue_count"])
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
│   │   ├── api.py
│   │   ├── services/
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
