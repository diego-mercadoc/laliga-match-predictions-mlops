import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

def create_prediction_heatmap(predictions: pd.DataFrame, metric: str) -> dict:
    """Create a heatmap showing prediction patterns across teams."""
    pivot_table = pd.pivot_table(
        predictions,
        values=metric,
        index='home_team',
        columns='away_team',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlBu',
        text=np.round(pivot_table.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'{metric} Distribution Across Teams',
        xaxis_title='Away Team',
        yaxis_title='Home Team'
    )
    
    return fig.to_dict()

def create_performance_timeline(
    metrics: List[Dict],
    metric_name: str,
    window: int = 30
) -> dict:
    """Create a timeline visualization of model performance."""
    df = pd.DataFrame(metrics)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Calculate rolling average
    df[f'{metric_name}_rolling'] = df[metric_name].rolling(window=window).mean()
    
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df[metric_name],
        mode='markers',
        name='Actual',
        marker=dict(size=6)
    ))
    
    # Add rolling average
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df[f'{metric_name}_rolling'],
        mode='lines',
        name=f'{window}-day Rolling Average',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title=f'{metric_name} Performance Timeline',
        xaxis_title='Date',
        yaxis_title=metric_name,
        showlegend=True
    )
    
    return fig.to_dict()

def create_accuracy_radar(metrics: Dict[str, float]) -> dict:
    """Create a radar chart showing accuracy across different prediction types."""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title='Prediction Accuracy by Type'
    )
    
    return fig.to_dict()

def create_error_distribution(predictions: pd.DataFrame, actuals: pd.DataFrame) -> dict:
    """Create error distribution visualization."""
    errors = predictions - actuals
    
    fig = go.Figure()
    
    for column in errors.columns:
        fig.add_trace(go.Histogram(
            x=errors[column],
            name=column,
            opacity=0.7,
            nbinsx=30
        ))
    
    fig.update_layout(
        title='Error Distribution by Prediction Type',
        xaxis_title='Prediction Error',
        yaxis_title='Count',
        barmode='overlay'
    )
    
    return fig.to_dict()

def create_feature_importance_sunburst(feature_importance: Dict[str, float]) -> dict:
    """Create a sunburst chart of feature importance."""
    # Group features by category
    categories = {
        'team': ['home_team_rank', 'away_team_rank', 'home_team_form', 'away_team_form'],
        'historical': ['head_to_head_wins', 'head_to_head_draws'],
        'performance': ['goals_scored_avg', 'goals_conceded_avg']
    }
    
    # Prepare data for sunburst
    data = []
    for category, features in categories.items():
        for feature in features:
            if feature in feature_importance:
                data.append({
                    'category': category,
                    'feature': feature,
                    'importance': feature_importance[feature]
                })
    
    df = pd.DataFrame(data)
    
    fig = px.sunburst(
        df,
        path=['category', 'feature'],
        values='importance',
        title='Feature Importance Distribution'
    )
    
    return fig.to_dict()

def create_prediction_confidence_gauge(confidence_scores: Dict[str, float]) -> dict:
    """Create gauge charts for prediction confidence."""
    fig = go.Figure()
    
    for i, (metric, score) in enumerate(confidence_scores.items()):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={'row': 0, 'column': i},
            title={'text': metric},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 60], 'color': "red"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
    
    fig.update_layout(
        grid={'rows': 1, 'columns': len(confidence_scores)},
        title='Prediction Confidence Scores'
    )
    
    return fig.to_dict()

def create_model_comparison_parallel(
    champion_metrics: Dict[str, float],
    challenger_metrics: Dict[str, float]
) -> dict:
    """Create parallel coordinates plot comparing model versions."""
    metrics = list(champion_metrics.keys())
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=[0, 1],
                colorscale=[[0, 'blue'], [1, 'red']]
            ),
            dimensions=[
                dict(
                    range=[
                        min(champion_metrics[m], challenger_metrics[m]),
                        max(champion_metrics[m], challenger_metrics[m])
                    ],
                    label=m,
                    values=[champion_metrics[m], challenger_metrics[m]]
                ) for m in metrics
            ]
        )
    )
    
    fig.update_layout(
        title='Champion vs Challenger Model Comparison',
        showlegend=True
    )
    
    return fig.to_dict()

def create_drift_analysis_visualization(
    historical_predictions: pd.DataFrame,
    recent_predictions: pd.DataFrame,
    metric: str
) -> dict:
    """Create visualization for drift analysis."""
    fig = go.Figure()
    
    # Add historical distribution
    fig.add_trace(go.Histogram(
        x=historical_predictions[metric],
        name='Historical',
        opacity=0.7,
        nbinsx=30
    ))
    
    # Add recent distribution
    fig.add_trace(go.Histogram(
        x=recent_predictions[metric],
        name='Recent',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.update_layout(
        title=f'Distribution Comparison for {metric}',
        xaxis_title=metric,
        yaxis_title='Count',
        barmode='overlay'
    )
    
    return fig.to_dict()

def create_performance_calendar_heatmap(
    metrics: List[Dict],
    metric_name: str
) -> dict:
    """Create a calendar heatmap of model performance."""
    df = pd.DataFrame(metrics)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    daily_avg = df.groupby('date')[metric_name].mean().reset_index()
    
    fig = go.Figure(data=go.Heatmap(
        x=daily_avg['date'],
        y=['Performance'],
        z=[daily_avg[metric_name]],
        colorscale='RdYlBu_r'
    ))
    
    fig.update_layout(
        title=f'Daily {metric_name} Performance',
        xaxis_title='Date',
        yaxis_title='',
        height=200
    )
    
    return fig.to_dict() 