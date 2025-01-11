"""
Feature Selection Module for LaLiga Match Prediction System

This module implements various feature selection techniques for the LaLiga match prediction system.
It provides tools for analyzing feature correlations, performing feature selection using different
methods (Lasso, Ridge, RFE), and managing feature sets for different prediction targets.

Key Components:
    - Correlation Analysis: Identifies and handles highly correlated features
    - Lasso Selection: Uses L1 regularization for sparse feature selection
    - Ridge Selection: Uses L2 regularization for feature importance
    - RFE Selection: Recursive feature elimination with cross-validation
    - Target-specific Selection: Manages feature sets for different prediction targets

Usage:
    The module supports both individual target feature selection and batch processing
    for all prediction targets (goals, corners, cards). It integrates with the main
    training pipeline and supports the automated feature selection process.

Quality Measures:
    - Cross-validation for feature importance
    - Correlation thresholding
    - Multiple selection methods for robustness
    - Visualization of feature relationships
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import os
import logging

logger = logging.getLogger(__name__)

def analyze_correlations(df: pd.DataFrame, features: list, threshold: float = 0.85) -> tuple[pd.DataFrame, list]:
    """
    Analyze correlations between features and identify highly correlated pairs.
    
    Args:
        df: Input DataFrame
        features: List of feature names to analyze
        threshold: Correlation threshold for feature removal
        
    Returns:
        Tuple of (correlation matrix, list of features to drop)
    """
    logger.info(f"Analyzing correlations for {len(features)} features")
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    # Find highly correlated feature pairs
    to_drop = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # Drop the feature with higher mean correlation with other features
                f1, f2 = features[i], features[j]
                c1 = corr_matrix[f1].abs().mean()
                c2 = corr_matrix[f2].abs().mean()
                to_drop.append(f1 if c1 > c2 else f2)
                logger.debug(f"High correlation ({corr_matrix.iloc[i, j]:.3f}) between {f1} and {f2}")
    
    to_drop = list(set(to_drop))  # Remove duplicates
    logger.info(f"Found {len(to_drop)} features to drop due to high correlation")
    
    return corr_matrix, to_drop

def lasso_feature_selection(X: pd.DataFrame, y: pd.Series, alpha: float = 0.01) -> list:
    """
    Select features using LASSO regression.
    
    Args:
        X: Feature matrix
        y: Target variable
        alpha: L1 regularization parameter
        
    Returns:
        List of selected feature names
    """
    logger.info("Running LASSO feature selection")
    
    # Fit LASSO model
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X, y)
    
    # Select features with non-zero coefficients
    selected = X.columns[abs(lasso.coef_) > 1e-5].tolist()
    logger.info(f"LASSO selected {len(selected)} features")
    
    return selected

def ridge_feature_selection(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> list:
    """
    Select features using Ridge regression.
    
    Args:
        X: Feature matrix
        y: Target variable
        alpha: L2 regularization parameter
        
    Returns:
        List of selected feature names
    """
    logger.info("Running Ridge feature selection")
    
    # Fit Ridge model
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X, y)
    
    # Select features with significant coefficients
    selected = X.columns[abs(ridge.coef_) > np.std(ridge.coef_) * 0.1].tolist()
    logger.info(f"Ridge selected {len(selected)} features")
    
    return selected

def rfe_feature_selection(X: pd.DataFrame, y: pd.Series, n_features_to_select: int = None) -> list:
    """
    Select features using Recursive Feature Elimination with cross-validation.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_features_to_select: Number of features to select (default: half of features)
        
    Returns:
        List of selected feature names
    """
    logger.info("Running RFE feature selection")
    
    if n_features_to_select is None:
        n_features_to_select = X.shape[1] // 2
    
    # Initialize estimator
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Perform RFE
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)
    
    # Get selected features
    selected = X.columns[selector.support_].tolist()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot RFE scores
    plt.figure(figsize=(10, 6))
    # Calculate feature importance scores for each feature subset size
    scores = []
    for i in range(1, len(X.columns) + 1):
        selector_i = RFE(estimator=estimator, n_features_to_select=i, step=1)
        selector_i.fit(X, y)
        score = cross_val_score(
            estimator=estimator,
            X=X.iloc[:, selector_i.support_],
            y=y,
            cv=5,
            scoring='neg_mean_squared_error'
        ).mean()
        scores.append(-score)  # Convert to positive MSE
    
    plt.plot(range(1, len(X.columns) + 1), scores)
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-validation MSE')
    plt.title('RFE Feature Selection Scores')
    plt.savefig('plots/rfe_scores.png')
    plt.close()
    
    logger.info(f"RFE selected {len(selected)} features")
    return selected

def select_stable_features(importance_df: pd.DataFrame, stability_threshold: float = 0.5, importance_threshold: float = 0.01) -> list:
    """
    Select features based on both importance and stability scores.
    
    Args:
        importance_df: DataFrame with importance and stability scores
        stability_threshold: Minimum stability score required
        importance_threshold: Minimum importance score required
        
    Returns:
        List of selected stable and important features
    """
    logger.info("Selecting features based on stability and importance...")
    
    # Get stability columns
    stability_cols = [col for col in importance_df.columns if col.endswith('_stability')]
    
    # Calculate mean stability across models
    importance_df['mean_stability'] = importance_df[stability_cols].mean(axis=1)
    
    # Select features that meet both criteria
    selected = importance_df[
        (importance_df['mean_stability'] >= stability_threshold) &
        (importance_df['avg_importance'] >= importance_threshold)
    ]['feature'].tolist()
    
    logger.info(f"Selected {len(selected)} stable and important features")
    return selected

def select_features_for_target(df: pd.DataFrame, features: list, target: str) -> dict:
    """
    Perform complete feature selection for a single target variable.
    
    Args:
        df: Input DataFrame
        features: List of feature names
        target: Target variable name
        
    Returns:
        Dictionary with selected features for each method and consensus
    """
    logger.info(f"Performing feature selection for target: {target}")
    
    results = {}
    
    # Correlation analysis
    _, to_drop = analyze_correlations(df, features)
    results['correlation'] = [f for f in features if f not in to_drop]
    
    # Prepare data for model-based selection
    X = df[features]
    y = df[target]
    
    # LASSO selection
    results['lasso'] = lasso_feature_selection(X, y)
    
    # Ridge selection
    results['ridge'] = ridge_feature_selection(X, y)
    
    # RFE selection
    results['rfe'] = rfe_feature_selection(X, y)
    
    # Importance-based selection with stability
    importance_df = analyze_target_importance(df, target)
    results['stable_important'] = select_stable_features(importance_df)
    
    # Create consensus features (selected by at least 3 methods)
    all_selected = set()
    for selected in results.values():
        all_selected.update(selected)
    
    consensus = []
    for feature in all_selected:
        count = sum(1 for selected in results.values() if feature in selected)
        if count >= 3:  # Increased threshold to include stable_important
            consensus.append(feature)
    
    results['consensus'] = consensus
    logger.info(f"Consensus selection: {len(consensus)} features")
    
    return results

def select_features_all_targets(df: pd.DataFrame, feature_sets: dict, targets: dict) -> dict:
    """
    Perform feature selection for all targets.
    
    Args:
        df: Input DataFrame
        feature_sets: Dictionary mapping target types to feature lists
        targets: Dictionary mapping target types to target variable names
        
    Returns:
        Dictionary with feature selection results for each target
    """
    logger.info("Starting feature selection for all targets")
    
    results = {}
    for target_type, features in feature_sets.items():
        target = targets[target_type][0]  # Use first target for now
        results[target_type] = select_features_for_target(df, features, target)
    
    return results 