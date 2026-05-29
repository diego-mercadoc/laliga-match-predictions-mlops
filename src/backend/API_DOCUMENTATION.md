# LaLiga Predictions API Documentation

## Overview

The backend is served from `api.py` and uses a PySpark pipeline to build the same 48-column model feature contract used by the original notebook. FBref data is validated before any prediction is attempted.

## Base URL

```text
http://localhost:8000
```

## Authentication

Local development is open by default. If `API_KEY` is set, send it with each protected request:

```http
X-API-Key: <your-api-key>
```

## Endpoints

### Health

```http
GET /health
```

Returns API health, pipeline type, feature count, and lazy MLflow model metadata.

### Data Quality

```http
GET /data/quality?jornada=12
X-API-Key: <your-api-key>
```

Validates FBref schedule and statistics after PySpark joins. The report includes missing feature columns, null counts, duplicate rows, and self-match rows.

### Feature Preview

```http
GET /features/preview?jornada=12&limit=5
X-API-Key: <your-api-key>
```

Returns model-ready rows, the expected feature columns, and the data-quality report. This endpoint is useful for debugging the ETL without loading MLflow.

### Predict

```http
POST /predict
Content-Type: application/json
X-API-Key: <your-api-key>

{
  "jornada": 12,
  "impute_missing": true
}
```

`/predict` validates the PySpark feature frame, optionally imputes numeric NULL/NaN feature values with per-column medians, loads MLflow lazily, and returns one prediction row per team view.

```json
{
  "jornada": 12,
  "predictions": [
    {
      "Anfitrion": "Barcelona",
      "Adversario": "Real Madrid",
      "Sedes": 1,
      "Probabilidad_Victoria": 0.45,
      "Probabilidad_Empate": 0.25,
      "Probabilidad_Derrota": 0.3,
      "Goles_Predichos_Local": 1.8,
      "Goles_Predichos_Visitante": 1.2
    }
  ],
  "data_quality": {
    "is_valid": true,
    "issue_count": 0,
    "issues": []
  },
  "imputation": {
    "strategy": "median",
    "filled_counts": {},
    "fill_values": {},
    "total_filled": 0
  },
  "model_version": "runs:/e8e41ab35bd34545a81ccb039080a64c/model",
  "feature_count": 48,
  "generated_at": "2026-05-28T00:00:00Z"
}
```

If MLflow or DagsHub credentials are missing, `/predict` returns `503` with a clear configuration message. `/health`, `/data/quality`, and `/features/preview` do not require model access.

### Refresh Status

```http
GET /refresh/status
```

Returns the latest data-refresh manifest from `artifacts/multi_season/refresh_status.json`.

### Start Refresh

```http
POST /refresh/run
Content-Type: application/json
X-API-Key: <your-api-key>

{
  "force_download": true,
  "skip_experiments": false
}
```

Starts the refresh pipeline in a background Python process. It updates raw Football-Data CSVs, rebuilds features, reruns experiments, and regenerates prediction artifacts.

## Runtime Configuration

- `MLFLOW_TRACKING_URI`: defaults to `https://dagshub.com/JuanPab2009/ProyectoFinalCD.mlflow`
- `MLFLOW_MODEL_URI`: defaults to `runs:/e8e41ab35bd34545a81ccb039080a64c/model`
- `API_KEY`: optional local API key

For private DagsHub/MLflow access, configure the username/token environment variables accepted by MLflow before calling `/predict`.

## Error Handling

- `403`: `API_KEY` is configured and the request did not include the right `X-API-Key`
- `422`: FBref data could not produce a valid feature frame
- `503`: MLflow model could not be loaded
- `500`: unexpected server error
