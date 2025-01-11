# LaLiga Predictions API Documentation

## Overview

The LaLiga Predictions API provides endpoints for predicting various outcomes in Spanish La Liga football matches, including match results, goals scored, corners awarded, and yellow cards received. The API is built with FastAPI and follows RESTful principles.

## Base URL

```
http://localhost:8000
```

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_token>
```

## Endpoints

### 1. Predictions

#### Make a Prediction
```http
POST /predict
Content-Type: application/json
Authorization: Bearer <your_token>

{
  "home_team": "Real Madrid",
  "away_team": "Barcelona",
  "date": "2024-01-20",
  "home_form": [1, 1, 0, 1, 0],
  "away_form": [1, 1, 1, 0, 1],
  "head_to_head": [1, 0, 1, 1, 0]
}
```

Response:
```json
{
  "match_outcome": {
    "home_win_probability": 0.45,
    "draw_probability": 0.25,
    "away_win_probability": 0.30
  },
  "goals_prediction": {
    "home_goals": 2,
    "away_goals": 1
  },
  "corners_prediction": {
    "home_corners": 5,
    "away_corners": 4
  },
  "cards_prediction": {
    "home_yellow_cards": 2,
    "away_yellow_cards": 3
  },
  "confidence_score": 0.85,
  "model_version": "production"
}
```

### 2. Model Performance

#### Get Model Performance Metrics
```http
GET /model/performance
Authorization: Bearer <your_token>
```

Response:
```json
{
  "accuracy": 0.82,
  "precision": 0.79,
  "recall": 0.81,
  "f1_score": 0.80,
  "predictions_count": 1000,
  "last_retrained": "2024-01-10T00:00:00Z",
  "drift_score": 0.05
}
```

#### Get Feature Importance
```http
GET /model/feature-importance
Authorization: Bearer <your_token>
```

Response:
```json
{
  "features": [
    {
      "name": "home_form",
      "importance": 0.25
    },
    {
      "name": "away_form",
      "importance": 0.23
    },
    {
      "name": "head_to_head",
      "importance": 0.20
    }
  ]
}
```

### 3. Model Management

#### Get Model Status
```http
GET /model/status
Authorization: Bearer <your_token>
```

Response:
```json
{
  "status": "healthy",
  "version": "production",
  "last_updated": "2024-01-10T00:00:00Z",
  "total_predictions": 1000,
  "average_latency": 0.15
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

### Common Error Codes

- `400 Bad Request`: Invalid input data
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

Example Error Response:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid team name provided",
    "details": {
      "field": "home_team",
      "reason": "Team not found in database"
    }
  }
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- 100 requests per minute per IP address
- Rate limit headers are included in responses:
  ```
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 95
  X-RateLimit-Reset: 1704870000
  ```

## Troubleshooting Guide

### Common Issues

1. Authentication Failures
   - Check if the token is valid and not expired
   - Ensure the token is included in the Authorization header
   - Verify the token format: `Bearer <token>`

2. Invalid Input Data
   - Verify team names match exactly with the database
   - Ensure date format is YYYY-MM-DD
   - Check that arrays (form, head_to_head) have correct length

3. Rate Limiting
   - Implement exponential backoff
   - Cache frequently requested data
   - Consider upgrading to a higher tier if needed

4. High Latency
   - Check network connectivity
   - Verify server load
   - Consider using batch predictions for multiple matches

### Monitoring

The API provides monitoring endpoints:

```http
GET /metrics
```

Key metrics to monitor:
- Request latency
- Error rates
- Prediction accuracy
- Model drift
- Resource usage

### Logging

Logs are in JSON format and include:
- Request ID
- Timestamp
- User ID
- Endpoint
- Response time
- Error details (if any)

## Best Practices

1. Prediction Requests
   - Batch predictions when possible
   - Include all available features for better accuracy
   - Handle prediction confidence scores appropriately

2. Error Handling
   - Implement retry logic with exponential backoff
   - Log all errors for debugging
   - Handle edge cases gracefully

3. Performance
   - Cache frequently used data
   - Use compression for large requests
   - Monitor response times and error rates

## Example Usage Scenarios

### 1. Match Outcome Prediction

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

### 2. Model Performance Monitoring

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

## Support

For additional support:
- Check the logs for detailed error messages
- Monitor the `/metrics` endpoint for system health
- Contact the development team for persistent issues

## Updates and Maintenance

The API is regularly updated with:
- Model retraining (weekly)
- Performance improvements
- Bug fixes
- New features

Stay updated with the latest changes by monitoring the version endpoint:
```http
GET /version
``` 