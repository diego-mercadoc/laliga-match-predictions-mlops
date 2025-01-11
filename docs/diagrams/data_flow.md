# Data Flow Diagrams

## 1. Data Ingestion Flow

```ascii
Historical Data (laliga.csv)          Current Season (Excel)          Live Data (fbref.com)
↓                                     ↓                               ↓
Data Validation                       Data Validation                 Data Validation
- Schema check                        - Schema check                  - API response check
- Type validation                     - Type validation               - Data completeness
- Range checks                        - Range checks                  - Rate limiting
↓                                     ↓                               ↓
Data Cleaning                         Data Cleaning                   Data Processing
- Missing values                      - Missing values                - HTML parsing
- Outlier detection                   - Date formatting               - JSON extraction
- Format standardization              - Team name mapping             - Error handling
↓                                     ↓                               ↓
Feature Engineering <----------------> Feature Merging <-------------> Feature Updates
↓
Quality Checks
- Feature completeness
- Correlation analysis
- Distribution checks
```

## 2. Model Pipeline Flow

```ascii
Feature Selection                     Model Training                  Model Evaluation
↓                                     ↓                               ↓
Correlation Analysis                  Cross Validation                Performance Metrics
- Threshold: 0.85                     - 5-fold CV                     - Accuracy
- Feature dropping                    - Time series split             - Precision/Recall
↓                                     ↓                               ↓
Lasso/Ridge Selection                 Hyperparameter Tuning          Model Comparison
- L1/L2 regularization               - Hyperopt                      - Champion vs Challenger
- Feature importance                 - Bayesian optimization         - Drift detection
↓                                     ↓                               ↓
RFE Selection                         Model Registry                  Model Deployment
- Recursive elimination              - MLflow tracking               - Version control
- Cross-validation                   - Artifact storage              - Rollback capability
```

## 3. Service Integration Flow

```ascii
MinIO                                MLflow                          FastAPI
↓                                     ↓                               ↓
Object Storage                        Experiment Tracking             API Endpoints
- Model artifacts                     - Run history                   - Predictions
- Training data                       - Metrics                       - Monitoring
- Metrics                            - Parameters                     - Health checks
↓                                     ↓                               ↓
Backup Management                     Model Registry                  Load Balancing
- Version control                     - Model versions                - Rate limiting
- Data versioning                     - Stage transitions             - Authentication
↓                                     ↓                               ↓
Access Control                        Deployment Control              Error Handling
- Authentication                      - Champion/Challenger           - Validation
- Authorization                       - A/B testing                   - Logging
``` 