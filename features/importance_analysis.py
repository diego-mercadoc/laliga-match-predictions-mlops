"""
Feature Importance Analysis Module for LaLiga Match Prediction System

This module implements comprehensive feature importance analysis for the LaLiga match
prediction system. It provides tools for analyzing and visualizing feature importance
across different prediction targets and model types.

Key Components:
    - Cross-validated Importance: Calculates robust feature importance scores
    - Multi-target Analysis: Separate analysis for goals, corners, and cards
    - Visualization: Generates importance plots and analysis reports
    - Model-specific Analysis: Supports multiple model types (RF, XGBoost, etc.)

Target Categories:
    - Goals: Home/Away goals prediction features
    - Corners: Corner kick prediction features
    - Cards: Yellow card prediction features

Feature Sets:
    Each target category has a predefined set of relevant features, including:
    - Direct statistics (goals, corners, cards)
    - Form indicators (last 3/5 matches)
    - Expected values (xG, xGA)
    - Team performance metrics
    - Match context (venue, day)
    - Referee statistics (for cards)

Quality Measures:
    - Cross-validation for robust importance scores
    - Multiple model types for validation
    - Standardized feature scaling
    - Comprehensive visualization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define targets
TARGETS = {
    'goals': ['GF', 'GC'],  # Home/Away goals
    'performance': ['Form_Score', 'Weighted_Form']  # Team performance
}

# Define feature sets for each target
FEATURE_SETS = {
    'goals': [
        'GF', 'GC', 'xG(tm)', 'xGA(tm)',
        'Form_Score', 'Weighted_Form',
        'Goals_Tm_Last3', 'Goals_Tm_Last5',
        'Goals_Opp_Last3', 'Goals_Opp_Last5',
        'Sedes', 'Día'
    ],
    'performance': [
        'Form_Score', 'Weighted_Form',
        'Goals_Tm_Last3', 'Goals_Tm_Last5',
        'Goals_Opp_Last3', 'Goals_Opp_Last5',
        'GF', 'GC', 'xG(tm)', 'xGA(tm)',
        'Sedes', 'Día'
    ]
}

# Define the models to use for feature importance
MODELS = {
    'rf': RandomForestRegressor(random_state=42),
    'xgb': XGBRegressor(random_state=42),
    'lr': LinearRegression(),
    'ridge': Ridge(random_state=42),
    'lasso': Lasso(random_state=42),
    'svr': SVR(kernel='rbf')
}

def calculate_cv_importance(model, X: pd.DataFrame, y: pd.Series, feature_names: list, n_splits=5) -> np.ndarray:
    """Calculate feature importance using cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    importances = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            imp = np.abs(model.coef_)
        else:
            imp = np.zeros(len(feature_names))
            for i in range(len(feature_names)):
                X_temp = X_val.copy()
                X_temp.iloc[:, i] = np.random.permutation(X_temp.iloc[:, i])
                imp[i] = -np.mean(cross_val_score(model, X_temp, y_val, cv=3))
        
        importances.append(imp)
    
    return np.mean(importances, axis=0)

def plot_feature_importance(importance_df: pd.DataFrame, target_type: str, output_dir: str = 'plots'):
    """Generate feature importance plots."""
    plt.figure(figsize=(12, 8))
    
    # Plot average importance
    sns.barplot(data=importance_df.head(10), 
                x='avg_importance', 
                y='feature',
                palette='viridis')
    
    plt.title(f'Top 10 Most Important Features for {target_type} Prediction')
    plt.xlabel('Average Importance')
    plt.ylabel('Feature')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/importance_{target_type}.png')
    plt.close()

def calculate_stability_score(model, X: pd.DataFrame, y: pd.Series, feature_names: list, n_iterations: int = 100) -> pd.DataFrame:
    """Calculate feature importance stability using bootstrap sampling.
    
    Args:
        model: Fitted model with feature_importances_ or coef_ attribute
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names
        n_iterations: Number of bootstrap iterations
        
    Returns:
        DataFrame with mean importance, std deviation, and stability scores
    """
    logger.info(f"Calculating stability scores with {n_iterations} iterations")
    
    n_samples = X.shape[0]
    importances = []
    
    for _ in range(n_iterations):
        # Bootstrap sampling
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]
        
        # Fit model and get importance scores
        model.fit(X_boot, y_boot)
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            imp = np.abs(model.coef_)
        else:
            raise ValueError("Model doesn't support feature importance analysis")
        
        importances.append(imp)
    
    # Calculate statistics
    importances = np.array(importances)
    mean_importance = np.mean(importances, axis=0)
    std_importance = np.std(importances, axis=0)
    
    # Calculate stability score (1 - coefficient of variation, bounded to [0,1])
    stability_score = 1 - np.minimum(std_importance / (mean_importance + 1e-10), 1)
    
    return pd.DataFrame({
        'feature': feature_names,
        'mean_importance': mean_importance,
        'std_importance': std_importance,
        'stability_score': stability_score
    })

def calculate_interaction_importance(model, X: pd.DataFrame, y: pd.Series, feature_pairs: list = None, n_iterations: int = 50) -> pd.DataFrame:
    """Calculate feature interaction importance using permutation importance.
    
    Args:
        model: Fitted model
        X: Feature matrix
        y: Target variable
        feature_pairs: List of feature pairs to analyze. If None, analyze all pairs
        n_iterations: Number of permutation iterations
        
    Returns:
        DataFrame with interaction scores for feature pairs
    """
    logger.info("Calculating feature interaction importance...")
    
    # If no pairs specified, create all possible pairs
    if feature_pairs is None:
        features = X.columns.tolist()
        feature_pairs = [(f1, f2) for i, f1 in enumerate(features) 
                        for f2 in features[i+1:]]
    
    # Fit model once
    model.fit(X, y)
    
    interaction_scores = []
    for f1, f2 in feature_pairs:
        # Calculate individual importance
        imp1 = _permutation_importance(model, X, y, f1, n_iterations)
        imp2 = _permutation_importance(model, X, y, f2, n_iterations)
        
        # Calculate joint importance
        joint_imp = _permutation_importance(model, X, y, [f1, f2], n_iterations)
        
        # Calculate interaction score (synergy/redundancy)
        interaction_score = joint_imp - (imp1 + imp2)
        
        interaction_scores.append({
            'feature1': f1,
            'feature2': f2,
            'interaction_score': interaction_score,
            'individual_score1': imp1,
            'individual_score2': imp2,
            'joint_score': joint_imp
        })
    
    return pd.DataFrame(interaction_scores)

def _permutation_importance(model, X: pd.DataFrame, y: pd.Series, features: Union[str, list], n_iterations: int) -> float:
    """Calculate permutation importance for single feature or feature group."""
    base_score = model.score(X, y)
    importance_scores = []
    
    features = [features] if isinstance(features, str) else features
    
    for _ in range(n_iterations):
        X_permuted = X.copy()
        # Permute specified features
        for feature in features:
            X_permuted[feature] = np.random.permutation(X_permuted[feature])
        
        permuted_score = model.score(X_permuted, y)
        importance_scores.append(base_score - permuted_score)
    
    return np.mean(importance_scores)

def plot_interaction_importance(interaction_df: pd.DataFrame, target_type: str, output_dir: str = 'plots'):
    """Generate feature interaction importance plots."""
    # Plot interaction heatmap
    plt.figure(figsize=(12, 10))
    
    # Create interaction matrix
    features = list(set(interaction_df['feature1'].unique()) | set(interaction_df['feature2'].unique()))
    n_features = len(features)
    interaction_matrix = np.zeros((n_features, n_features))
    
    for _, row in interaction_df.iterrows():
        i = features.index(row['feature1'])
        j = features.index(row['feature2'])
        interaction_matrix[i, j] = row['interaction_score']
        interaction_matrix[j, i] = row['interaction_score']
    
    sns.heatmap(
        interaction_matrix,
        xticklabels=features,
        yticklabels=features,
        cmap='coolwarm',
        center=0,
        annot=True
    )
    
    plt.title(f'Feature Interaction Importance for {target_type} Prediction')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/interaction_{target_type}.png')
    plt.close()

def analyze_target_importance(data: pd.DataFrame, target_type: str, output_dir: str = 'plots') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze feature importance for a specific target type.
    
    Args:
        data: Input DataFrame
        target_type: Type of target ('goals', 'corners', 'cards')
        output_dir: Directory to save plots
        
    Returns:
        Tuple of (importance_df, interaction_df)
    """
    logger.info(f"Analyzing feature importance for {target_type}")
    
    # Get features and targets for this type
    features = FEATURE_SETS[target_type]
    targets = TARGETS[target_type]
    
    X = data[features]
    importance_dfs = []
    interaction_dfs = []
    
    for target in targets:
        y = data[target]
        
        # Calculate stability scores
        stability_df = calculate_stability_score(
            RandomForestRegressor(n_estimators=100, random_state=42),
            X, y, features
        )
        
        # Calculate interaction importance
        interaction_df = calculate_interaction_importance(
            RandomForestRegressor(n_estimators=100, random_state=42),
            X, y, n_iterations=50
        )
        
        # Generate interaction plots automatically
        plot_interaction_importance(interaction_df, f"{target_type}_{target}", output_dir)
        
        # Weight importance by stability
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': stability_df['mean_importance'] * stability_df['stability_score']
        })
        
        importance_dfs.append(importance_df.set_index('feature'))
        interaction_dfs.append(interaction_df)
    
    # Average importance across targets
    final_importance = pd.concat(importance_dfs, axis=1)
    final_importance['avg_importance'] = final_importance.mean(axis=1)
    
    # Combine interaction results
    final_interactions = pd.concat(interaction_dfs).groupby(['feature1', 'feature2']).mean().reset_index()
    
    return final_importance.reset_index(), final_interactions

def analyze_all_targets(df: pd.DataFrame) -> dict:
    """
    Analyze feature importance for all targets.
    
    Args:
        df: Preprocessed DataFrame with all features
        
    Returns:
        Dictionary containing importance DataFrames for each target
    """
    results = {}
    for target_type in TARGETS.keys():
        results[target_type] = analyze_target_importance(df, target_type)
    return results 