import requests
import json
from datetime import datetime

def get_auth_token():
    """Get authentication token from the API."""
    url = "http://localhost:8000/auth/token"
    data = {
        "username": "test_user",
        "password": "test_password"
    }
    response = requests.post(url, data=data)
    return response.json()["access_token"]

def predict_match(home_team, away_team, date, token):
    """Make a prediction for a match."""
    url = "http://localhost:8000/predict"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "home_team": home_team,
        "away_team": away_team,
        "date": date,
        "home_form": [1, 1, 0, 1, 0],  # Example form data
        "away_form": [1, 1, 1, 0, 1],
        "head_to_head": [1, 0, 1, 1, 0]
    }
    
    response = requests.post(url, json=data, headers=headers)
    return response.json()

def get_model_performance(token):
    """Get model performance metrics."""
    url = "http://localhost:8000/model/performance"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    return response.json()

def main():
    # Get authentication token
    token = get_auth_token()
    
    # Test matches to predict
    matches = [
        ("Real Madrid", "Barcelona", "2024-01-20"),
        ("Atletico Madrid", "Sevilla", "2024-01-21"),
        ("Valencia", "Athletic Bilbao", "2024-01-22")
    ]
    
    # Make predictions
    print("\nMatch Predictions:")
    print("-----------------")
    for home_team, away_team, date in matches:
        prediction = predict_match(home_team, away_team, date, token)
        print(f"\n{home_team} vs {away_team} ({date}):")
        print(json.dumps(prediction, indent=2))
    
    # Get model performance
    print("\nModel Performance:")
    print("-----------------")
    performance = get_model_performance(token)
    print(json.dumps(performance, indent=2))

if __name__ == "__main__":
    main() 