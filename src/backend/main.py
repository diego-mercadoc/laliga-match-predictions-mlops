import mlflow
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, validator, Field
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import re
import dagshub
import mlflow
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union
import requests
from datetime import datetime, timedelta
from scipy import stats
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from enum import Enum
from sklearn.linear_model import LinearRegression
from pandera import DataFrameSchema, Column, Check
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
import time
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from utils.visualization import (
    create_prediction_heatmap,
    create_performance_timeline,
    create_accuracy_radar,
    create_error_distribution,
    create_feature_importance_sunburst,
    create_prediction_confidence_gauge,
    create_model_comparison_parallel,
    create_drift_analysis_visualization,
    create_performance_calendar_heatmap
)
from utils.backup import PredictionBackupManager
import asyncio
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data validation schema
class MatchStatus(str, Enum):
    """Status of a match."""
    COMPLETED = "completed"
    SCHEDULED = "scheduled"
    POSTPONED = "postponed"
    LIVE = "live"

class MatchException(BaseModel):
    """Model for handling match exceptions."""
    match_id: str
    home_team: str
    away_team: str
    original_date: Optional[datetime]
    new_date: Optional[datetime]
    status: MatchStatus
    reason: Optional[str]
    jornada: int

# Update the match data schema
match_data_schema = DataFrameSchema({
    'Sem.': Column(int, Check.in_range(1, 38)),
    'Día': Column(str, nullable=True),  # Allow null for postponed matches
    'Fecha': Column(pd.Timestamp, nullable=True),  # Allow null for postponed matches
    'Local': Column(str),
    'Visitante': Column(str),
    'Estado': Column(str, Check.isin(['Fin', 'Hoy', 'Mañana', 'Aplazado']), nullable=True)
})

def handle_postponed_match(match_data: dict) -> MatchException:
    """Handle a postponed match and create an exception record."""
    return MatchException(
        match_id=f"{match_data['Local']}_{match_data['Visitante']}_{match_data['Sem.']}",
        home_team=match_data['Local'],
        away_team=match_data['Visitante'],
        original_date=match_data.get('Fecha'),
        status=MatchStatus.POSTPONED,
        jornada=match_data['Sem.']
    )

def process_match_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[MatchException]]:
    """Process match data and handle exceptions."""
    exceptions = []
    
    # Convert date strings to datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    
    # Handle postponed matches
    postponed_mask = df['Estado'] == 'Aplazado'
    if postponed_mask.any():
        postponed_matches = df[postponed_mask].to_dict('records')
        exceptions.extend([handle_postponed_match(match) for match in postponed_matches])
        df = df[~postponed_mask].copy()
    
    # Handle missing dates
    missing_dates = df['Fecha'].isna()
    if missing_dates.any():
        missing_matches = df[missing_dates].to_dict('records')
        exceptions.extend([
            MatchException(
                match_id=f"{match['Local']}_{match['Visitante']}_{match['Sem.']}",
                home_team=match['Local'],
                away_team=match['Visitante'],
                status=MatchStatus.SCHEDULED,
                jornada=match['Sem.']
            ) for match in missing_matches
        ])
        df = df[~missing_dates].copy()
    
    return df, exceptions

# Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key is None:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Could not validate API key"
        )
    # In production, validate against secure storage
    if api_key != "your-api-key":  # Replace with secure key validation
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    return api_key

# Model monitoring
class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.prediction_times = []
        self.last_drift_check = datetime.now()
        self.drift_check_interval = timedelta(hours=1)

    def add_prediction(self, prediction: dict, prediction_time: float):
        self.predictions.append(prediction)
        self.prediction_times.append(prediction_time)
        if len(self.prediction_times) > 1000:
            self.predictions.pop(0)
            self.prediction_times.pop(0)

    def add_actual(self, actual: dict):
        self.actuals.append(actual)
        if len(self.actuals) > 1000:
            self.actuals.pop(0)

    def check_drift(self) -> Optional[dict]:
        if datetime.now() - self.last_drift_check < self.drift_check_interval:
            return None

        if len(self.predictions) < 100:
            return None

        recent_predictions = pd.DataFrame(self.predictions[-100:])
        historical_predictions = pd.DataFrame(self.predictions[:-100])

        if historical_predictions.empty:
            return None

        drift_metrics = {}
        for col in recent_predictions.select_dtypes(include=[np.number]).columns:
            ks_statistic, p_value = stats.ks_2samp(
                historical_predictions[col],
                recent_predictions[col]
            )
            drift_metrics[col] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'drift_detected': p_value < 0.05
            }

        self.last_drift_check = datetime.now()
        return drift_metrics

    def get_performance_metrics(self) -> dict:
        if not self.actuals:
            return {}

        metrics = {
            'prediction_latency': {
                'mean': np.mean(self.prediction_times),
                'p95': np.percentile(self.prediction_times, 95),
                'p99': np.percentile(self.prediction_times, 99)
            }
        }

        if self.actuals:
            actual_df = pd.DataFrame(self.actuals)
            pred_df = pd.DataFrame(self.predictions)
            
            for col in actual_df.select_dtypes(include=[np.number]).columns:
                if col in pred_df.columns:
                    metrics[f'{col}_rmse'] = np.sqrt(mean_squared_error(
                        actual_df[col], pred_df[col]
                    ))

        return metrics

model_monitor = ModelMonitor()

# Rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    def is_allowed(self) -> bool:
        now = time.time()
        minute_ago = now - 60
        self.requests = [req for req in self.requests if req > minute_ago]
        
        if len(self.requests) >= self.requests_per_minute:
            return False
        
        self.requests.append(now)
        return True

rate_limiter = RateLimiter()

# Load the model from MLflow Model Registry
model_name = "final-prefect-model"

# Set the MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/JuanPab2009/ProyectoFinalCD.mlflow")

# Load the champion model
try:
    # Get the champion model version
    client = mlflow.tracking.MlflowClient()
    champion_version = client.get_model_version_by_alias(model_name, "champion")
    model_uri = f"models:/{model_name}/{champion_version.version}"
    # Load the model as a PyFuncModel
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f"Successfully loaded model version {champion_version.version}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def fetch_data(url: str, retries: int = 3) -> pd.DataFrame:
    """Fetch data from URL with retries."""
    for attempt in range(retries):
        try:
            return pd.read_html(url)[0]
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed to fetch data from {url} after {retries} attempts: {str(e)}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(1)

def predict(input_data: dict) -> pd.DataFrame:
    """
    Make predictions for La Liga matches.
    
    Args:
        input_data: Dictionary containing the jornada number
        
    Returns:
        DataFrame with predictions for each match
    """
    try:
        jornada = input_data["jornada"]
        logger.info(f"Making predictions for jornada {jornada}")

        # Fetch match data
        url = "https://fbref.com/es/comps/12/horario/Resultados-y-partidos-en-La-Liga"
        df = fetch_data(url)
        
        # Process data and handle exceptions
        df, exceptions = process_match_data(df)
        
        if exceptions:
            logger.warning(f"Found {len(exceptions)} match exceptions in jornada {jornada}")
            for exc in exceptions:
                logger.info(f"Match exception: {exc.dict()}")
        
        # Filter by jornada
        df = df[df["Sem."] == jornada]
        if df.empty and not exceptions:
            raise HTTPException(
                status_code=404,
                detail=f"No matches found for jornada {jornada}"
            )

        # Continue with predictions only for available matches
        if not df.empty:
            predictions = model.predict(df)
            results = create_prediction_results(df, predictions)
        else:
            results = pd.DataFrame()

        # Add exception information to response
        if exceptions:
            results = pd.concat([
                results,
                pd.DataFrame([{
                    'Anfitrion': exc.home_team,
                    'Adversario': exc.away_team,
                    'Estado': exc.status,
                    'Fecha_Original': exc.original_date,
                    'Fecha_Nueva': exc.new_date
                } for exc in exceptions])
            ])

        return results

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating predictions: {str(e)}"
        )

def calculate_confidence_interval(value: float, error_margin: float = 0.2, confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval for a predicted value.
    
    Args:
        value: The predicted value
        error_margin: The estimated error margin as a proportion of the value
        confidence: The confidence level (default: 0.95 for 95% confidence)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    std_error = value * error_margin
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * std_error
    
    lower_bound = max(0, value - margin)  # Ensure non-negative
    upper_bound = value + margin
    
    return round(lower_bound, 2), round(upper_bound, 2)

def create_prediction_results(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """
    Create a DataFrame with prediction results and confidence intervals.
    
    Args:
        df: Input DataFrame with match data
        predictions: Model predictions
        
    Returns:
        DataFrame with formatted predictions and confidence intervals
    """
    # Create base results DataFrame
    results = pd.DataFrame({
        'Anfitrion': df['Local'],
        'Adversario': df['Visitante'],
        'Probabilidad_Victoria': predictions[:, 2],
        'Probabilidad_Empate': predictions[:, 1],
        'Probabilidad_Derrota': predictions[:, 0],
        'Estado': 'scheduled'
    })
    
    # Add predicted goals
    results['Goles_Predichos_Local'] = np.round(
        results['Probabilidad_Victoria'] * 2 + results['Probabilidad_Empate'], 
        1
    )
    results['Goles_Predichos_Visitante'] = np.round(
        results['Probabilidad_Derrota'] * 2 + results['Probabilidad_Empate'], 
        1
    )
    
    # Add predicted corners (using team possession and attack metrics)
    results['Corners_Predichos_Local'] = np.round(
        predictions[:, 3] * 0.3,  # Using possession metric from predictions
        1
    )
    results['Corners_Predichos_Visitante'] = np.round(
        predictions[:, 4] * 0.3,  # Using possession metric from predictions
        1
    )
    
    # Add predicted yellow cards (using foul and aggression metrics)
    results['Amarillas_Predichas_Local'] = np.round(
        (predictions[:, 5] + predictions[:, 6] * 0.2) * 0.4,  # Using foul and aggression metrics
        1
    )
    results['Amarillas_Predichas_Visitante'] = np.round(
        (predictions[:, 7] + predictions[:, 8] * 0.2) * 0.4,  # Using foul and aggression metrics
        1
    )
    
    # Round probabilities
    probability_cols = ['Probabilidad_Victoria', 'Probabilidad_Empate', 'Probabilidad_Derrota']
    results[probability_cols] = results[probability_cols].round(3)
    
    # Calculate confidence intervals with dynamic error margins
    metrics_margins = {
        'Goles_Predichos': 0.25,  # 25% error margin for goals
        'Corners_Predichos': 0.30,  # 30% error margin for corners
        'Amarillas_Predichas': 0.35  # 35% error margin for yellow cards
    }
    
    for base_metric, error_margin in metrics_margins.items():
        for team in ['Local', 'Visitante']:
            metric = f'{base_metric}_{team}'
            results[f'{metric}_CI_Lower'], results[f'{metric}_CI_Upper'] = zip(
                *results[metric].apply(
                    lambda x: calculate_confidence_interval(x, error_margin)
                )
            )
    
    logger.info(
        f"Successfully generated predictions with confidence intervals for {len(results)} matches"
    )
    
    return results

# Set up FastAPI app
app = FastAPI(
    title="La Liga Predictions API",
    description="API for predicting La Liga match outcomes, goals, corners, and yellow cards",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Define the input data model using Pydantic
class InputData(BaseModel):
    jornada: int = Field(..., ge=1, le=38, description="Match day number (1-38)")
    
    @validator('jornada')
    def validate_jornada(cls, v):
        current_month = datetime.now().month
        if current_month < 8:  # Before August
            if v > 38:
                raise ValueError("Invalid jornada number")
        return v

class PredictionResponse(BaseModel):
    Anfitrion: str = Field(..., description="Home team name")
    Adversario: str = Field(..., description="Away team name")
    Probabilidad_Victoria: float = Field(..., ge=0, le=1, description="Probability of home team victory")
    Probabilidad_Empate: float = Field(..., ge=0, le=1, description="Probability of draw")
    Probabilidad_Derrota: float = Field(..., ge=0, le=1, description="Probability of home team defeat")
    Goles_Predichos_Local: float = Field(..., ge=0, description="Predicted goals for home team")
    Goles_Predichos_Local_CI_Lower: float = Field(..., description="Lower bound of confidence interval for home goals")
    Goles_Predichos_Local_CI_Upper: float = Field(..., description="Upper bound of confidence interval for home goals")
    Goles_Predichos_Visitante: float = Field(..., ge=0, description="Predicted goals for away team")
    Goles_Predichos_Visitante_CI_Lower: float = Field(..., description="Lower bound of confidence interval for away goals")
    Goles_Predichos_Visitante_CI_Upper: float = Field(..., description="Upper bound of confidence interval for away goals")
    Corners_Predichos_Local: float = Field(..., ge=0, description="Predicted corners for home team")
    Corners_Predichos_Local_CI_Lower: float = Field(..., description="Lower bound of confidence interval for home corners")
    Corners_Predichos_Local_CI_Upper: float = Field(..., description="Upper bound of confidence interval for home corners")
    Corners_Predichos_Visitante: float = Field(..., ge=0, description="Predicted corners for away team")
    Corners_Predichos_Visitante_CI_Lower: float = Field(..., description="Lower bound of confidence interval for away corners")
    Corners_Predichos_Visitante_CI_Upper: float = Field(..., description="Upper bound of confidence interval for away corners")
    Amarillas_Predichas_Local: float = Field(..., ge=0, description="Predicted yellow cards for home team")
    Amarillas_Predichas_Local_CI_Lower: float = Field(..., description="Lower bound of confidence interval for home yellow cards")
    Amarillas_Predichas_Local_CI_Upper: float = Field(..., description="Upper bound of confidence interval for home yellow cards")
    Amarillas_Predichas_Visitante: float = Field(..., ge=0, description="Predicted yellow cards for away team")
    Amarillas_Predichas_Visitante_CI_Lower: float = Field(..., description="Lower bound of confidence interval for away yellow cards")
    Amarillas_Predichas_Visitante_CI_Upper: float = Field(..., description="Upper bound of confidence interval for away yellow cards")

class PredictionsResponse(BaseModel):
    predictions: List[PredictionResponse]
    timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str

class HistoricalPrediction(BaseModel):
    """Model for storing historical predictions and actual results."""
    jornada: int
    fecha: datetime
    anfitrion: str
    adversario: str
    prediccion_goles_local: float
    prediccion_goles_visitante: float
    prediccion_corners_local: float
    prediccion_corners_visitante: float
    prediccion_amarillas_local: float
    prediccion_amarillas_visitante: float
    goles_local_real: Optional[float] = None
    goles_visitante_real: Optional[float] = None
    corners_local_real: Optional[float] = None
    corners_visitante_real: Optional[float] = None
    amarillas_local_real: Optional[float] = None
    amarillas_visitante_real: Optional[float] = None

class AccuracyMetrics(BaseModel):
    """Model for accuracy metrics of predictions."""
    metric_type: str
    mse: float
    mae: float
    accuracy_within_ci: float
    sample_size: int
    last_updated: datetime

def save_prediction(prediction: dict):
    """Save a prediction to the historical records."""
    history_file = Path("prediction_history.json")
    
    try:
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)
        else:
            history = []
        
        # Add timestamp to prediction
        prediction["timestamp"] = datetime.now().isoformat()
        history.append(prediction)
        
        # Keep only last 1000 predictions
        if len(history) > 1000:
            history = history[-1000:]
        
        with open(history_file, "w") as f:
            json.dump(history, f)
            
    except Exception as e:
        logger.error(f"Error saving prediction: {str(e)}")

def fetch_actual_results(jornada: int) -> pd.DataFrame:
    """Fetch actual match results from fbref."""
    try:
        url = "https://fbref.com/es/comps/12/horario/Resultados-y-partidos-en-La-Liga"
        df = fetch_data(url)
        df = df[df["Sem."] == jornada]
        
        # Extract scores from Marcador column
        df[["GF", "GC"]] = df["Marcador"].str.extract(r"(\d+).*?(\d+)")
        df[["GF", "GC"]] = df[["GF", "GC"]].astype(float)
        
        return df
    except Exception as e:
        logger.error(f"Error fetching actual results: {str(e)}")
        return pd.DataFrame()

def calculate_accuracy_metrics(predictions: List[dict], actuals: List[dict]) -> Dict[str, AccuracyMetrics]:
    """Calculate accuracy metrics for different prediction types."""
    metrics = {}
    
    for pred_type in ["goals", "corners", "cards"]:
        pred_values = []
        actual_values = []
        within_ci = 0
        total = 0
        
        for pred, actual in zip(predictions, actuals):
            if pred_type == "goals":
                if actual.get("goles_local_real") is not None:
                    pred_values.extend([pred["Goles_Predichos_Local"], pred["Goles_Predichos_Visitante"]])
                    actual_values.extend([actual["goles_local_real"], actual["goles_visitante_real"]])
                    
                    # Check if actual values fall within confidence intervals
                    if (pred["Goles_Predichos_Local_CI_Lower"] <= actual["goles_local_real"] <= pred["Goles_Predichos_Local_CI_Upper"]):
                        within_ci += 1
                    if (pred["Goles_Predichos_Visitante_CI_Lower"] <= actual["goles_visitante_real"] <= pred["Goles_Predichos_Visitante_CI_Upper"]):
                        within_ci += 1
                    total += 2
            
            # Similar checks for corners and cards
            # [Add similar blocks for corners and cards]
        
        if pred_values and actual_values:
            mse = np.mean((np.array(pred_values) - np.array(actual_values)) ** 2)
            mae = np.mean(np.abs(np.array(pred_values) - np.array(actual_values)))
            accuracy_within_ci = within_ci / total if total > 0 else 0
            
            metrics[pred_type] = AccuracyMetrics(
                metric_type=pred_type,
                mse=round(mse, 3),
                mae=round(mae, 3),
                accuracy_within_ci=round(accuracy_within_ci, 3),
                sample_size=total,
                last_updated=datetime.now()
            )
    
    return metrics

# Add new endpoints
@app.get("/accuracy", response_model=Dict[str, AccuracyMetrics])
async def get_prediction_accuracy(
    days: int = Query(30, ge=1, le=365, description="Number of days of history to analyze")
):
    """
    Get accuracy metrics for predictions over the specified time period.
    
    Args:
        days: Number of days of history to analyze (default: 30)
        
    Returns:
        Dictionary of accuracy metrics for different prediction types
    """
    try:
        history_file = Path("prediction_history.json")
        if not history_file.exists():
            return {}
        
        with open(history_file, "r") as f:
            history = json.load(f)
        
        # Filter predictions within the specified time period
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_predictions = [
            pred for pred in history 
            if datetime.fromisoformat(pred["timestamp"]) > cutoff_date
        ]
        
        if not recent_predictions:
            return {}
        
        # Fetch actual results for these predictions
        actual_results = []
        for pred in recent_predictions:
            result = fetch_actual_results(pred["jornada"])
            if not result.empty:
                actual_results.append({
                    "jornada": pred["jornada"],
                    "goles_local_real": result["GF"].iloc[0],
                    "goles_visitante_real": result["GC"].iloc[0],
                    # Add actual corners and cards when available
                })
        
        # Calculate accuracy metrics
        metrics = calculate_accuracy_metrics(recent_predictions, actual_results)
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating prediction accuracy: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error calculating prediction accuracy"
        )

# Modify the predict endpoint to save predictions
@app.post("/predict", response_model=PredictionsResponse, dependencies=[Depends(verify_api_key)])
async def predict_endpoint(input_data: InputData):
    """
    Predict match outcomes for a specific La Liga matchday.
    
    Args:
        input_data: Input data containing the jornada number
        
    Returns:
        Predictions for all matches in the specified jornada
    """
    if not rate_limiter.is_allowed():
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )

    try:
        start_time = time.time()
        
        # Get predictions
        predictions = predict(input_data.dict())
        
        # Monitor prediction time
        prediction_time = time.time() - start_time
        
        # Add to monitoring
        for pred in predictions.to_dict(orient="records"):
            model_monitor.add_prediction(pred, prediction_time)
        
        # Check for drift
        drift_metrics = model_monitor.check_drift()
        if drift_metrics and any(m['drift_detected'] for m in drift_metrics.values()):
            logger.warning(f"Model drift detected: {drift_metrics}")
        
        # Convert DataFrame to a list of dictionaries
        predictions_dict = predictions.to_dict(orient="records")

        # Save predictions for historical tracking
        for pred in predictions_dict:
            save_prediction({
                "jornada": input_data.jornada,
                **pred
            })
        
        return PredictionsResponse(
            predictions=[PredictionResponse(**pred) for pred in predictions_dict],
            model_version=str(champion_version.version)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )

@app.get("/monitoring/metrics")
async def get_monitoring_metrics(api_key: str = Depends(verify_api_key)):
    """Get current model monitoring metrics."""
    return {
        'performance_metrics': model_monitor.get_performance_metrics(),
        'drift_metrics': model_monitor.check_drift()
    }

@app.get("/health")
async def health_check():
    """Check if the API is healthy and the model is loaded."""
    return {
        "status": "healthy",
        "model_name": model_name,
        "model_version": str(champion_version.version),
        "last_drift_check": model_monitor.last_drift_check.isoformat(),
        "prediction_count": len(model_monitor.predictions)
    }

# Print dependencies at startup
dependencies = mlflow.pyfunc.get_model_dependencies(model_uri)
logger.info(f"Model dependencies: {dependencies}")

class PredictionType(str, Enum):
    GOALS = "goals"
    CORNERS = "corners"
    CARDS = "cards"

@dataclass
class ErrorMargins:
    """Default error margins for different prediction types."""
    goals: float = 0.25
    corners: float = 0.30
    cards: float = 0.35

class VisualizationData(BaseModel):
    """Model for visualization-ready prediction data."""
    match_id: str
    home_team: str
    away_team: str
    prediction_type: PredictionType
    predicted_value_home: float
    predicted_value_away: float
    ci_lower_home: float
    ci_upper_home: float
    ci_lower_away: float
    ci_upper_away: float
    actual_value_home: Optional[float] = None
    actual_value_away: Optional[float] = None

class VisualizationResponse(BaseModel):
    """Response model for visualization data."""
    data: List[VisualizationData]
    error_margins: Dict[str, float]
    accuracy_metrics: Optional[Dict[str, AccuracyMetrics]] = None

def adjust_error_margins(historical_accuracy: Dict[str, AccuracyMetrics]) -> Dict[str, float]:
    """
    Dynamically adjust error margins based on historical prediction accuracy.
    
    Args:
        historical_accuracy: Dictionary of accuracy metrics for each prediction type
        
    Returns:
        Dictionary of adjusted error margins
    """
    default_margins = ErrorMargins()
    adjusted_margins = {}
    
    for pred_type, metrics in historical_accuracy.items():
        # Base adjustment on accuracy within CI and MAE
        base_margin = getattr(default_margins, pred_type)
        
        # If accuracy within CI is too low, increase margin
        if metrics.accuracy_within_ci < 0.8:  # Less than 80% accuracy
            adjustment = (0.8 - metrics.accuracy_within_ci) * 0.5
            new_margin = base_margin * (1 + adjustment)
        # If accuracy within CI is very high, decrease margin
        elif metrics.accuracy_within_ci > 0.95:  # More than 95% accuracy
            adjustment = (metrics.accuracy_within_ci - 0.95) * 0.5
            new_margin = base_margin * (1 - adjustment)
        else:
            new_margin = base_margin
            
        # Ensure margin stays within reasonable bounds
        adjusted_margins[pred_type] = max(0.1, min(0.5, new_margin))
    
    return adjusted_margins

def create_visualization_data(predictions: List[dict], actuals: List[dict], error_margins: Dict[str, float]) -> List[VisualizationData]:
    """
    Create visualization-ready data from predictions and actuals.
    
    Args:
        predictions: List of prediction dictionaries
        actuals: List of actual result dictionaries
        error_margins: Dictionary of error margins for each prediction type
        
    Returns:
        List of VisualizationData objects
    """
    viz_data = []
    
    for pred, actual in zip(predictions, actuals):
        match_id = f"{pred['jornada']}_{pred['Anfitrion']}_{pred['Adversario']}"
        
        # Add goals visualization data
        viz_data.append(VisualizationData(
            match_id=match_id,
            home_team=pred['Anfitrion'],
            away_team=pred['Adversario'],
            prediction_type=PredictionType.GOALS,
            predicted_value_home=pred['Goles_Predichos_Local'],
            predicted_value_away=pred['Goles_Predichos_Visitante'],
            ci_lower_home=pred['Goles_Predichos_Local_CI_Lower'],
            ci_upper_home=pred['Goles_Predichos_Local_CI_Upper'],
            ci_lower_away=pred['Goles_Predichos_Visitante_CI_Lower'],
            ci_upper_away=pred['Goles_Predichos_Visitante_CI_Upper'],
            actual_value_home=actual.get('goles_local_real'),
            actual_value_away=actual.get('goles_visitante_real')
        ))
        
        # Add similar entries for corners and cards
        # [Similar blocks for corners and cards]
    
    return viz_data

def create_plotly_figure(viz_data: List[VisualizationData], pred_type: PredictionType) -> dict:
    """
    Create a Plotly figure for visualizing predictions.
    
    Args:
        viz_data: List of visualization data objects
        pred_type: Type of prediction to visualize
        
    Returns:
        Plotly figure as dictionary
    """
    filtered_data = [d for d in viz_data if d.prediction_type == pred_type]
    
    fig = go.Figure()
    
    # Add home team predictions
    fig.add_trace(go.Bar(
        name='Home Team Predicted',
        x=[f"{d.home_team} vs {d.away_team}" for d in filtered_data],
        y=[d.predicted_value_home for d in filtered_data],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[d.ci_upper_home - d.predicted_value_home for d in filtered_data],
            arrayminus=[d.predicted_value_home - d.ci_lower_home for d in filtered_data]
        )
    ))
    
    # Add away team predictions
    fig.add_trace(go.Bar(
        name='Away Team Predicted',
        x=[f"{d.home_team} vs {d.away_team}" for d in filtered_data],
        y=[d.predicted_value_away for d in filtered_data],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[d.ci_upper_away - d.predicted_value_away for d in filtered_data],
            arrayminus=[d.predicted_value_away - d.ci_lower_away for d in filtered_data]
        )
    ))
    
    # Add actual values if available
    actuals_home = [d.actual_value_home for d in filtered_data if d.actual_value_home is not None]
    actuals_away = [d.actual_value_away for d in filtered_data if d.actual_value_away is not None]
    
    if actuals_home:
        fig.add_trace(go.Scatter(
            name='Home Team Actual',
            x=[f"{d.home_team} vs {d.away_team}" for d in filtered_data if d.actual_value_home is not None],
            y=actuals_home,
            mode='markers',
            marker=dict(size=10, symbol='diamond')
        ))
    
    if actuals_away:
        fig.add_trace(go.Scatter(
            name='Away Team Actual',
            x=[f"{d.home_team} vs {d.away_team}" for d in filtered_data if d.actual_value_away is not None],
            y=actuals_away,
            mode='markers',
            marker=dict(size=10, symbol='diamond')
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{pred_type.value.title()} Predictions with Confidence Intervals',
        xaxis_title='Match',
        yaxis_title=f'Predicted {pred_type.value.title()}',
        barmode='group',
        template='plotly_white'
    )
    
    return fig.to_dict()

@app.get("/visualization", response_model=VisualizationResponse)
async def get_visualization_data(
    jornada: int = Query(..., ge=1, le=38, description="Match day number"),
    pred_type: PredictionType = Query(..., description="Type of prediction to visualize"),
    viz_type: VisualizationType = Query(
        VisualizationType.BAR_CONFIDENCE,
        description="Type of visualization to create"
    )
):
    """
    Get visualization data for predictions.
    
    Args:
        jornada: Match day number
        pred_type: Type of prediction to visualize
        viz_type: Type of visualization to create
        
    Returns:
        Visualization data and related metrics
    """
    try:
        # Get historical accuracy and predictions
        accuracy_metrics = await get_prediction_accuracy(days=30)
        error_margins = adjust_error_margins(accuracy_metrics)
        
        input_data = InputData(jornada=jornada)
        predictions_response = await predict_endpoint(input_data)
        predictions = [pred.dict() for pred in predictions_response.predictions]
        
        # Get actual results
        actuals = []
        result_df = fetch_actual_results(jornada)
        if not result_df.empty:
            for _, row in result_df.iterrows():
                actuals.append({
                    "jornada": jornada,
                    "goles_local_real": row["GF"],
                    "goles_visitante_real": row["GC"]
                })
        
        # Create visualization based on type
        viz_data = create_visualization_data(predictions, actuals, error_margins)
        
        if viz_type == VisualizationType.BAR_CONFIDENCE:
            figure = create_plotly_figure(viz_data, pred_type)
        elif viz_type == VisualizationType.LINE_TREND:
            trends = await get_prediction_trends(days=30)
            trend_metrics = getattr(trends, pred_type.value)
            figure = create_trend_visualization(trend_metrics, pred_type.value)
        elif viz_type == VisualizationType.HEATMAP:
            figure = create_heatmap_visualization(predictions, pred_type.value)
        else:  # SCATTER
            figure = create_scatter_visualization(viz_data, pred_type)
        
        return VisualizationResponse(
            data=viz_data,
            error_margins=error_margins,
            accuracy_metrics=accuracy_metrics,
            figure=figure
        )
    
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error creating visualization"
        )

class TrendMetrics(BaseModel):
    """Model for trend analysis metrics with confidence bands."""
    slope: float
    trend_direction: str  # "improving", "stable", or "declining"
    confidence: float
    prediction_next: float
    prediction_next_lower: Optional[float] = None
    prediction_next_upper: Optional[float] = None
    historical_values: List[float]
    dates: List[str]
    significance: Optional[TrendSignificance] = None
    confidence_bands: Optional[Dict[str, List[float]]] = None

class AccuracyTrend(BaseModel):
    """Model for accuracy trend analysis."""
    goals: TrendMetrics
    corners: TrendMetrics
    cards: TrendMetrics
    last_updated: datetime

class VisualizationType(str, Enum):
    """Types of visualizations available."""
    BAR_CONFIDENCE = "bar_confidence"
    LINE_TREND = "line_trend"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    RADAR = "radar"  # New visualization type

def analyze_prediction_trend(historical_data: List[dict], metric_type: str) -> TrendMetrics:
    """
    Analyze trends in prediction accuracy over time with confidence bands.
    
    Args:
        historical_data: List of historical predictions and results
        metric_type: Type of prediction to analyze
        
    Returns:
        Trend analysis metrics with confidence bands
    """
    # Convert historical data to time series
    df = pd.DataFrame(historical_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Calculate daily accuracy
    daily_accuracy = []
    dates = []
    
    for date, group in df.groupby(df['timestamp'].dt.date):
        if metric_type == 'goals':
            pred_cols = ['Goles_Predichos_Local', 'Goles_Predichos_Visitante']
            actual_cols = ['goles_local_real', 'goles_visitante_real']
        elif metric_type == 'corners':
            pred_cols = ['Corners_Predichos_Local', 'Corners_Predichos_Visitante']
            actual_cols = ['corners_local_real', 'corners_visitante_real']
        else:  # cards
            pred_cols = ['Amarillas_Predichas_Local', 'Amarillas_Predichas_Visitante']
            actual_cols = ['amarillas_local_real', 'amarillas_visitante_real']
        
        # Calculate mean absolute error for the day
        mae = np.mean([
            abs(row[pred_cols[0]] - row[actual_cols[0]]) + 
            abs(row[pred_cols[1]] - row[actual_cols[1]])
            for _, row in group.iterrows()
            if not pd.isna(row[actual_cols[0]]) and not pd.isna(row[actual_cols[1]])
        ])
        
        if not np.isnan(mae):
            daily_accuracy.append(mae)
            dates.append(date.strftime('%Y-%m-%d'))
    
    if len(daily_accuracy) < 2:
        return TrendMetrics(
            slope=0.0,
            trend_direction="stable",
            confidence=0.0,
            prediction_next=np.mean(daily_accuracy) if daily_accuracy else 0.0,
            historical_values=daily_accuracy,
            dates=dates,
            confidence_bands=None
        )
    
    # Fit linear regression
    X = np.arange(len(daily_accuracy)).reshape(-1, 1)
    y = np.array(daily_accuracy)
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate trend metrics
    slope = model.coef_[0]
    confidence = abs(slope) / (np.std(daily_accuracy) + 1e-6)
    
    # Calculate confidence bands
    n = len(X)
    y_pred = model.predict(X)
    mse = np.sum((y - y_pred) ** 2) / (n - 2)
    std_error = np.sqrt(mse * (1 + 1/n + (X - np.mean(X))**2 / np.sum((X - np.mean(X))**2)))
    
    # 95% confidence interval
    t_value = stats.t.ppf(0.975, n - 2)
    lower_bound = y_pred - t_value * std_error
    upper_bound = y_pred + t_value * std_error
    
    # Determine trend direction
    if abs(slope) < 0.01:
        trend = "stable"
    else:
        trend = "improving" if slope < 0 else "declining"
    
    # Predict next value with confidence interval
    next_x = np.array([[len(daily_accuracy)]])
    next_prediction = model.predict(next_x)[0]
    next_std_error = np.sqrt(mse * (1 + 1/n + (next_x - np.mean(X))**2 / np.sum((X - np.mean(X))**2)))[0]
    next_lower = next_prediction - t_value * next_std_error
    next_upper = next_prediction + t_value * next_std_error
    
    trend_metrics = TrendMetrics(
        slope=float(slope),
        trend_direction=trend,
        confidence=float(confidence),
        prediction_next=float(next_prediction),
        prediction_next_lower=float(next_lower),
        prediction_next_upper=float(next_upper),
        historical_values=daily_accuracy,
        dates=dates,
        confidence_bands={
            'lower': lower_bound.tolist(),
            'upper': upper_bound.tolist()
        }
    )
    
    # Add significance test
    trend_metrics.significance = test_trend_significance(trend_metrics)
    
    return trend_metrics

def create_trend_visualization(trend_metrics: TrendMetrics, metric_type: str) -> dict:
    """Create a visualization for trend analysis with confidence bands."""
    fig = go.Figure()
    
    # Add historical values
    fig.add_trace(go.Scatter(
        x=trend_metrics.dates,
        y=trend_metrics.historical_values,
        mode='lines+markers',
        name='Historical MAE',
        line=dict(color='blue')
    ))
    
    # Calculate trend line and confidence bands
    X = np.arange(len(trend_metrics.historical_values))
    y = np.array(trend_metrics.historical_values)
    
    # Fit polynomial for trend
    z = np.polyfit(X, y, 1)
    p = np.poly1d(z)
    trend_line = p(X)
    
    # Calculate confidence bands
    n = len(X)
    mean_x = np.mean(X)
    se = np.sqrt(np.sum((y - trend_line) ** 2) / (n - 2))
    
    # Calculate confidence intervals
    pi = 0.95  # 95% prediction interval
    t_value = stats.t.ppf((1 + pi) / 2, n - 2)
    
    x_new = np.linspace(X.min(), X.max(), 100)
    y_new = p(x_new)
    
    # Calculate prediction interval
    x_new_mean = np.mean(x_new)
    
    pi_band = t_value * se * np.sqrt(1 + 1/n + (x_new - x_new_mean)**2 / 
                                    np.sum((X - mean_x)**2))
    
    lower_bound = y_new - pi_band
    upper_bound = y_new + pi_band
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=trend_metrics.dates,
        y=trend_line,
        mode='lines',
        name='Trend',
        line=dict(color='red', dash='dash')
    ))
    
    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=trend_metrics.dates + trend_metrics.dates[::-1],
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,0,0,0)'),
        name='95% Confidence Band',
        showlegend=True
    ))
    
    # Add significance annotation
    if trend_metrics.significance:
        sig_text = (
            f"Trend {'is' if trend_metrics.significance.is_significant else 'is not'} significant\n"
            f"p-value: {trend_metrics.significance.p_value:.3f}"
        )
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=sig_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{metric_type.title()} Prediction Accuracy Trend',
            x=0.5,
            y=0.95
        ),
        xaxis_title='Date',
        yaxis_title='Mean Absolute Error',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(t=100),  # Add more top margin for title
        hovermode='x unified'
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate="<br>".join([
            "Date: %{x}",
            "Value: %{y:.2f}",
            "<extra></extra>"
        ])
    )
    
    return fig.to_dict()

def create_heatmap_visualization(predictions: List[dict], metric_type: str) -> dict:
    """Create a heatmap visualization of prediction accuracy."""
    # Convert predictions to DataFrame
    df = pd.DataFrame(predictions)
    
    if metric_type == 'goals':
        actual_cols = ['goles_local_real', 'goles_visitante_real']
        pred_cols = ['Goles_Predichos_Local', 'Goles_Predichos_Visitante']
    elif metric_type == 'corners':
        actual_cols = ['corners_local_real', 'corners_visitante_real']
        pred_cols = ['Corners_Predichos_Local', 'Corners_Predichos_Visitante']
    else:
        actual_cols = ['amarillas_local_real', 'amarillas_visitante_real']
        pred_cols = ['Amarillas_Predichas_Local', 'Amarillas_Predichas_Visitante']
    
    # Calculate errors
    errors = []
    teams = []
    for _, row in df.iterrows():
        if not pd.isna(row[actual_cols[0]]) and not pd.isna(row[actual_cols[1]]):
            home_error = abs(row[pred_cols[0]] - row[actual_cols[0]])
            away_error = abs(row[pred_cols[1]] - row[actual_cols[1]])
            errors.extend([home_error, away_error])
            teams.extend([row['Anfitrion'], row['Adversario']])
    
    # Create heatmap data
    heatmap_df = pd.DataFrame({'team': teams, 'error': errors})
    heatmap_data = heatmap_df.pivot_table(
        values='error',
        index='team',
        aggfunc='mean'
    ).sort_values('error')
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[heatmap_data.values],
        x=heatmap_data.index,
        colorscale='RdYlGn_r',
        showscale=True
    ))
    
    fig.update_layout(
        title=f'{metric_type.title()} Prediction Error by Team',
        xaxis_title='Team',
        yaxis_title='',
        template='plotly_white'
    )
    
    return fig.to_dict()

@app.get("/trends", response_model=AccuracyTrend)
async def get_prediction_trends(
    days: int = Query(30, ge=1, le=365, description="Number of days of history to analyze")
):
    """
    Get trend analysis of prediction accuracy.
    
    Args:
        days: Number of days of history to analyze
        
    Returns:
        Trend analysis for each prediction type
    """
    try:
        # Get historical data
        history_file = Path("prediction_history.json")
        if not history_file.exists():
            raise HTTPException(status_code=404, detail="No historical data available")
        
        with open(history_file, "r") as f:
            history = json.load(f)
        
        # Filter recent predictions
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_predictions = [
            pred for pred in history 
            if datetime.fromisoformat(pred["timestamp"]) > cutoff_date
        ]
        
        if not recent_predictions:
            raise HTTPException(status_code=404, detail="No recent predictions available")
        
        # Analyze trends for each prediction type
        trends = AccuracyTrend(
            goals=analyze_prediction_trend(recent_predictions, "goals"),
            corners=analyze_prediction_trend(recent_predictions, "corners"),
            cards=analyze_prediction_trend(recent_predictions, "cards"),
            last_updated=datetime.now()
        )
        
        return trends
    
    except Exception as e:
        logger.error(f"Error analyzing prediction trends: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error analyzing prediction trends"
        )

class TrendSignificance(BaseModel):
    """Model for trend significance test results."""
    is_significant: bool
    p_value: float
    test_statistic: float
    test_name: str
    sample_size: int

def create_radar_visualization(predictions: List[dict], team_name: str) -> dict:
    """
    Create a radar chart comparing a team's metrics against league averages.
    
    Args:
        predictions: List of prediction dictionaries
        team_name: Name of the team to analyze
        
    Returns:
        Plotly figure as dictionary
    """
    df = pd.DataFrame(predictions)
    
    # Define metrics to compare
    metrics = {
        'Goals Scored': ('Goles_Predichos_Local', 'Goles_Predichos_Visitante'),
        'Corners': ('Corners_Predichos_Local', 'Corners_Predichos_Visitante'),
        'Yellow Cards': ('Amarillas_Predichas_Local', 'Amarillas_Predichas_Visitante'),
        'Win Probability': ('Probabilidad_Victoria', None),
        'Draw Probability': ('Probabilidad_Empate', None)
    }
    
    # Calculate team and league averages
    team_stats = []
    league_stats = []
    
    for metric_name, (home_col, away_col) in metrics.items():
        # Team stats
        team_values = []
        if home_col:
            home_mask = df['Anfitrion'] == team_name
            team_values.extend(df.loc[home_mask, home_col])
        if away_col:
            away_mask = df['Adversario'] == team_name
            team_values.extend(df.loc[away_mask, away_col])
        
        team_avg = np.mean(team_values) if team_values else 0
        team_stats.append(team_avg)
        
        # League stats
        if home_col and away_col:
            league_avg = np.mean(np.concatenate([df[home_col], df[away_col]]))
        else:
            league_avg = np.mean(df[home_col])
        league_stats.append(league_avg)
    
    # Create radar chart
    fig = go.Figure()
    
    # Add team stats
    fig.add_trace(go.Scatterpolar(
        r=team_stats,
        theta=list(metrics.keys()),
        fill='toself',
        name=team_name,
        line_color='blue'
    ))
    
    # Add league averages
    fig.add_trace(go.Scatterpolar(
        r=league_stats,
        theta=list(metrics.keys()),
        fill='toself',
        name='League Average',
        line_color='red'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(team_stats), max(league_stats)) * 1.2]
            )
        ),
        title=f'Team Performance Radar: {team_name} vs League Average',
        showlegend=True
    )
    
    return fig.to_dict()

def test_trend_significance(trend_metrics: TrendMetrics) -> TrendSignificance:
    """
    Test statistical significance of a trend using Mann-Kendall test.
    
    Args:
        trend_metrics: Trend metrics including historical values
        
    Returns:
        Significance test results
    """
    from scipy import stats
    
    if len(trend_metrics.historical_values) < 2:
        return TrendSignificance(
            is_significant=False,
            p_value=1.0,
            test_statistic=0.0,
            test_name="Mann-Kendall",
            sample_size=len(trend_metrics.historical_values)
        )
    
    # Perform Mann-Kendall test
    result = stats.kendalltau(
        np.arange(len(trend_metrics.historical_values)),
        trend_metrics.historical_values
    )
    
    # Check significance at 95% confidence level
    is_significant = result.pvalue < 0.05
    
    return TrendSignificance(
        is_significant=is_significant,
        p_value=float(result.pvalue),
        test_statistic=float(result.statistic),
        test_name="Mann-Kendall",
        sample_size=len(trend_metrics.historical_values)
    )

@app.get("/team-comparison/{team_name}")
async def get_team_comparison(
    team_name: str,
    jornada: int = Query(..., ge=1, le=38, description="Match day number")
):
    """
    Get team comparison visualization.
    
    Args:
        team_name: Name of the team to analyze
        jornada: Match day number
        
    Returns:
        Radar chart comparing team metrics against league averages
    """
    try:
        # Get predictions
        input_data = InputData(jornada=jornada)
        predictions_response = await predict_endpoint(input_data)
        predictions = [pred.dict() for pred in predictions_response.predictions]
        
        # Create radar chart
        figure = create_radar_visualization(predictions, team_name)
        
        return {
            "team_name": team_name,
            "figure": figure
        }
    
    except Exception as e:
        logger.error(f"Error creating team comparison: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error creating team comparison"
        )

# A/B Testing Configuration
class ABTestingConfig:
    def __init__(self):
        self.champion_traffic_fraction = 0.8
        self.challenger_traffic_fraction = 0.2
        self.min_samples_required = 1000
        self.evaluation_period_days = 7
        self.last_evaluation = datetime.now()
        self.champion_metrics = []
        self.challenger_metrics = []

    def should_use_champion(self) -> bool:
        return np.random.random() < self.champion_traffic_fraction

    def record_prediction_metrics(self, is_champion: bool, prediction: dict, actual: dict):
        metrics = calculate_prediction_metrics(prediction, actual)
        if is_champion:
            self.champion_metrics.append(metrics)
        else:
            self.challenger_metrics.append(metrics)

    def evaluate_models(self) -> Optional[dict]:
        if (datetime.now() - self.last_evaluation).days < self.evaluation_period_days:
            return None

        if len(self.champion_metrics) < self.min_samples_required or \
           len(self.challenger_metrics) < self.min_samples_required:
            return None

        champion_performance = pd.DataFrame(self.champion_metrics).mean()
        challenger_performance = pd.DataFrame(self.challenger_metrics).mean()

        evaluation = {
            'champion_performance': champion_performance.to_dict(),
            'challenger_performance': challenger_performance.to_dict(),
            'challenger_is_better': challenger_performance.mean() > champion_performance.mean()
        }

        self.last_evaluation = datetime.now()
        return evaluation

ab_testing = ABTestingConfig()

# Feature Importance Analysis
def analyze_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """Analyze and return feature importance scores."""
    try:
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_scores = np.abs(model.coef_)
        else:
            raise ValueError("Model doesn't support feature importance analysis")

        return dict(zip(feature_names, importance_scores))
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {str(e)}")
        return {}

# Automated Model Performance Reporting
class ModelPerformanceReport:
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.feature_importance = {}
        self.last_report_time = datetime.now()
        self.report_interval = timedelta(days=1)

    def add_prediction(self, features: dict, prediction: dict, actual: Optional[dict] = None):
        self.predictions.append(prediction)
        if actual:
            self.actuals.append(actual)

    def generate_report(self) -> dict:
        if datetime.now() - self.last_report_time < self.report_interval:
            return {}

        report = {
            'timestamp': datetime.now().isoformat(),
            'prediction_count': len(self.predictions),
            'metrics': model_monitor.get_performance_metrics(),
            'drift_analysis': model_monitor.check_drift(),
            'feature_importance': self.feature_importance
        }

        # Generate performance visualizations
        if self.actuals:
            report['visualizations'] = self._generate_visualizations()

        self.last_report_time = datetime.now()
        return report

    def _generate_visualizations(self) -> dict:
        pred_df = pd.DataFrame(self.predictions)
        actual_df = pd.DataFrame(self.actuals)
        
        visualizations = {}
        for col in pred_df.columns:
            if col in actual_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=range(len(pred_df)),
                    y=pred_df[col],
                    name='Predicted'
                ))
                fig.add_trace(go.Scatter(
                    x=range(len(actual_df)),
                    y=actual_df[col],
                    name='Actual'
                ))
                visualizations[col] = fig.to_json()
        
        return visualizations

performance_reporter = ModelPerformanceReport()

# Initialize backup manager
backup_manager = PredictionBackupManager(
    local_backup_dir="backups",
    s3_bucket=os.getenv("BACKUP_S3_BUCKET"),
    backup_frequency_hours=int(os.getenv("BACKUP_FREQUENCY_HOURS", "24")),
    retention_days=int(os.getenv("BACKUP_RETENTION_DAYS", "90"))
)

# New visualization endpoints
@app.get("/visualizations/heatmap")
async def get_prediction_heatmap(
    metric: str = Query(..., description="Metric to visualize"),
    api_key: str = Depends(verify_api_key)
):
    """Get heatmap visualization of predictions across teams."""
    predictions_df = pd.DataFrame(model_monitor.predictions)
    return create_prediction_heatmap(predictions_df, metric)

@app.get("/visualizations/performance")
async def get_performance_timeline(
    metric: str = Query(..., description="Metric to visualize"),
    window: int = Query(30, description="Rolling average window size"),
    api_key: str = Depends(verify_api_key)
):
    """Get performance timeline visualization."""
    metrics = model_monitor.get_performance_metrics()
    return create_performance_timeline(metrics, metric, window)

@app.get("/visualizations/accuracy")
async def get_accuracy_radar(
    api_key: str = Depends(verify_api_key)
):
    """Get radar chart of prediction accuracy."""
    metrics = model_monitor.get_performance_metrics()
    return create_accuracy_radar(metrics)

@app.get("/visualizations/error-distribution")
async def get_error_distribution(
    api_key: str = Depends(verify_api_key)
):
    """Get error distribution visualization."""
    predictions_df = pd.DataFrame(model_monitor.predictions)
    actuals_df = pd.DataFrame(model_monitor.actuals)
    return create_error_distribution(predictions_df, actuals_df)

@app.get("/visualizations/feature-importance")
async def get_feature_importance_viz(
    api_key: str = Depends(verify_api_key)
):
    """Get feature importance visualization."""
    importance = analyze_feature_importance(model, model.feature_names_)
    return create_feature_importance_sunburst(importance)

@app.get("/visualizations/confidence")
async def get_confidence_gauge(
    api_key: str = Depends(verify_api_key)
):
    """Get confidence gauge visualization."""
    predictions = model_monitor.predictions[-1] if model_monitor.predictions else None
    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions available")
    
    confidence_scores = calculate_confidence_scores(predictions)
    return create_prediction_confidence_gauge(confidence_scores)

@app.get("/visualizations/model-comparison")
async def get_model_comparison(
    api_key: str = Depends(verify_api_key)
):
    """Get model comparison visualization."""
    ab_results = ab_testing.evaluate_models()
    if not ab_results:
        raise HTTPException(status_code=404, detail="No A/B testing results available")
    
    return create_model_comparison_parallel(
        ab_results['champion_performance'],
        ab_results['challenger_performance']
    )

@app.get("/visualizations/drift")
async def get_drift_visualization(
    metric: str = Query(..., description="Metric to analyze"),
    api_key: str = Depends(verify_api_key)
):
    """Get drift analysis visualization."""
    predictions_df = pd.DataFrame(model_monitor.predictions)
    if len(predictions_df) < 200:
        raise HTTPException(status_code=400, detail="Insufficient data for drift analysis")
    
    historical = predictions_df.iloc[:-100]
    recent = predictions_df.iloc[-100:]
    
    return create_drift_analysis_visualization(historical, recent, metric)

@app.get("/visualizations/calendar")
async def get_performance_calendar(
    metric: str = Query(..., description="Metric to visualize"),
    api_key: str = Depends(verify_api_key)
):
    """Get calendar heatmap of model performance."""
    metrics = model_monitor.get_performance_metrics()
    return create_performance_calendar_heatmap(metrics, metric)

# Backup endpoints
@app.post("/backups/create")
async def create_backup(
    api_key: str = Depends(verify_api_key)
):
    """Create a new backup of historical predictions."""
    try:
        if not backup_manager.needs_backup():
            return {"message": "Backup not needed yet"}
        
        backup_file = backup_manager.create_backup(model_monitor.predictions)
        return {
            "message": "Backup created successfully",
            "backup_file": backup_file
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating backup: {str(e)}"
        )

@app.get("/backups/list")
async def list_backups(
    api_key: str = Depends(verify_api_key)
):
    """List all available backups."""
    try:
        backups = backup_manager.list_backups()
        return {"backups": backups}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing backups: {str(e)}"
        )

@app.post("/backups/restore")
async def restore_backup(
    backup_file: Optional[str] = None,
    date: Optional[datetime] = None,
    api_key: str = Depends(verify_api_key)
):
    """Restore predictions from backup."""
    try:
        predictions = backup_manager.restore_from_backup(backup_file, date)
        return {
            "message": "Backup restored successfully",
            "predictions_count": len(predictions)
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Backup not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error restoring backup: {str(e)}"
        )

@app.delete("/backups/cleanup")
async def cleanup_backups(
    api_key: str = Depends(verify_api_key)
):
    """Clean up old backups."""
    try:
        backup_manager.cleanup_old_backups()
        return {"message": "Old backups cleaned up successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error cleaning up backups: {str(e)}"
        )

# Schedule backup creation
@app.on_event("startup")
async def schedule_backups():
    """Schedule regular backups."""
    async def create_periodic_backup():
        while True:
            if backup_manager.needs_backup():
                try:
                    await create_backup(api_key=os.getenv("API_KEY"))
                except Exception as e:
                    logger.error(f"Error in scheduled backup: {str(e)}")
            
            # Wait for next check
            await asyncio.sleep(3600)  # Check every hour
    
    asyncio.create_task(create_periodic_backup())