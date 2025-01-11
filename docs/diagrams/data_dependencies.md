# Data Dependencies

## 1. Data Flow
```ascii
Historical Data (laliga.csv)
↓
Feature Engineering
↓
Model Training
↓
Model Registry (MLflow)
↓
Prediction Service

Current Season Data (Excel)
↓
Data Validation
↓
Feature Updates
↓
Live Predictions

Live Data (fbref.com)
↓
Data Processing
↓
Feature Updates
↓
Real-time Predictions
```

## 2. Service Dependencies
```ascii
MinIO (Object Storage)
↑
MLflow (Model Registry)
↑
Backend (FastAPI)
↑
Frontend (Streamlit)
↑
Prometheus
↑
Grafana
↑
Monitoring Dashboard
```

## 3. Code Dependencies
```ascii
features/
  feature_selection.py
  importance_analysis.py
  ↓
notebooks/
  Training_pipeline.py
  ↓
src/
  backend/main.py
  frontend/main.py
  ↓
Docker Services
```

## 4. Documentation Dependencies
```ascii
.cursorrules
↔
README.md
↔
DOCKER_DOCUMENTATION.md
↔
docs/MLOPS.md
↔
src/backend/API_DOCUMENTATION.md
↔
src/frontend/FRONTEND_DOCUMENTATION.md
``` 