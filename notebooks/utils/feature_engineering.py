import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from .match_exceptions import MatchStatus

class FeatureEngineer:
    """Class for engineering features from LaLiga match data."""
    
    def __init__(
        self,
        rolling_window: int = 5,
        min_matches: int = 3,
        home_advantage_weight: float = 1.2
    ):
        """Initialize FeatureEngineer.
        
        Args:
            rolling_window: Number of matches to use for rolling averages
            min_matches: Minimum matches required for valid statistics
            home_advantage_weight: Weight to apply to home team statistics
        """
        self.rolling_window = rolling_window
        self.min_matches = min_matches
        self.home_advantage_weight = home_advantage_weight
        
        # Initialize feature groups
        self.form_features = [
            'wins', 'draws', 'losses',
            'goals_scored', 'goals_conceded',
            'corners_for', 'corners_against',
            'yellow_cards'
        ]
        
        self.head2head_features = [
            'h2h_wins', 'h2h_draws', 'h2h_losses',
            'h2h_goals_scored', 'h2h_goals_conceded',
            'h2h_corners_for', 'h2h_corners_against',
            'h2h_yellow_cards'
        ]
        
        self.time_features = [
            'days_since_last_match',
            'is_weekend',
            'is_evening_match'
        ]
    
    def _calculate_team_form(
        self,
        df: pd.DataFrame,
        team: str,
        date: datetime,
        is_home: bool
    ) -> Dict[str, float]:
        """Calculate form statistics for a team.
        
        Args:
            df: Match data DataFrame
            team: Team name
            date: Current match date
            is_home: Whether team is home team
            
        Returns:
            Dict[str, float]: Form statistics
        """
        # Get previous matches
        past_matches = df[
            ((df['Local'] == team) | (df['Visitante'] == team)) &
            (df['Fecha'] < date) &
            (df['Estado'] == MatchStatus.COMPLETED.value)
        ].sort_values('Fecha', ascending=False).head(self.rolling_window)
        
        if len(past_matches) < self.min_matches:
            return {feature: np.nan for feature in self.form_features}
        
        stats = {
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'corners_for': 0,
            'corners_against': 0,
            'yellow_cards': 0
        }
        
        for _, match in past_matches.iterrows():
            is_home_match = match['Local'] == team
            
            # Get match outcome
            if match['Goles_Local'] > match['Goles_Visitante']:
                stats['wins'] += 1 if is_home_match else 0
                stats['losses'] += 0 if is_home_match else 1
            elif match['Goles_Local'] < match['Goles_Visitante']:
                stats['wins'] += 0 if is_home_match else 1
                stats['losses'] += 1 if is_home_match else 0
            else:
                stats['draws'] += 1
            
            # Get goals
            if is_home_match:
                stats['goals_scored'] += match['Goles_Local']
                stats['goals_conceded'] += match['Goles_Visitante']
                stats['corners_for'] += match['Corners_Local']
                stats['corners_against'] += match['Corners_Visitante']
                stats['yellow_cards'] += match['Amarillas_Local']
            else:
                stats['goals_scored'] += match['Goles_Visitante']
                stats['goals_conceded'] += match['Goles_Local']
                stats['corners_for'] += match['Corners_Visitante']
                stats['corners_against'] += match['Corners_Local']
                stats['yellow_cards'] += match['Amarillas_Visitante']
        
        # Calculate averages
        num_matches = len(past_matches)
        for key in stats:
            stats[key] = stats[key] / num_matches
        
        # Apply home advantage weight if applicable
        if is_home:
            for key in stats:
                stats[key] *= self.home_advantage_weight
        
        return stats
    
    def _calculate_head2head(
        self,
        df: pd.DataFrame,
        home_team: str,
        away_team: str,
        date: datetime
    ) -> Dict[str, float]:
        """Calculate head-to-head statistics.
        
        Args:
            df: Match data DataFrame
            home_team: Home team name
            away_team: Away team name
            date: Current match date
            
        Returns:
            Dict[str, float]: Head-to-head statistics
        """
        # Get previous meetings
        h2h_matches = df[
            (
                ((df['Local'] == home_team) & (df['Visitante'] == away_team)) |
                ((df['Local'] == away_team) & (df['Visitante'] == home_team))
            ) &
            (df['Fecha'] < date) &
            (df['Estado'] == MatchStatus.COMPLETED.value)
        ].sort_values('Fecha', ascending=False)
        
        if len(h2h_matches) == 0:
            return {feature: np.nan for feature in self.head2head_features}
        
        stats = {
            'h2h_wins': 0,
            'h2h_draws': 0,
            'h2h_losses': 0,
            'h2h_goals_scored': 0,
            'h2h_goals_conceded': 0,
            'h2h_corners_for': 0,
            'h2h_corners_against': 0,
            'h2h_yellow_cards': 0
        }
        
        for _, match in h2h_matches.iterrows():
            home_is_first = match['Local'] == home_team
            
            # Get match outcome
            if match['Goles_Local'] > match['Goles_Visitante']:
                stats['h2h_wins'] += 1 if home_is_first else 0
                stats['h2h_losses'] += 0 if home_is_first else 1
            elif match['Goles_Local'] < match['Goles_Visitante']:
                stats['h2h_wins'] += 0 if home_is_first else 1
                stats['h2h_losses'] += 1 if home_is_first else 0
            else:
                stats['h2h_draws'] += 1
            
            # Get other statistics
            if home_is_first:
                stats['h2h_goals_scored'] += match['Goles_Local']
                stats['h2h_goals_conceded'] += match['Goles_Visitante']
                stats['h2h_corners_for'] += match['Corners_Local']
                stats['h2h_corners_against'] += match['Corners_Visitante']
                stats['h2h_yellow_cards'] += match['Amarillas_Local']
            else:
                stats['h2h_goals_scored'] += match['Goles_Visitante']
                stats['h2h_goals_conceded'] += match['Goles_Local']
                stats['h2h_corners_for'] += match['Corners_Visitante']
                stats['h2h_corners_against'] += match['Corners_Local']
                stats['h2h_yellow_cards'] += match['Amarillas_Visitante']
        
        # Calculate averages
        num_matches = len(h2h_matches)
        for key in stats:
            stats[key] = stats[key] / num_matches
        
        return stats
    
    def _calculate_time_features(
        self,
        df: pd.DataFrame,
        team: str,
        date: datetime
    ) -> Dict[str, float]:
        """Calculate time-based features.
        
        Args:
            df: Match data DataFrame
            team: Team name
            date: Current match date
            
        Returns:
            Dict[str, float]: Time-based features
        """
        # Find last match date
        last_match = df[
            ((df['Local'] == team) | (df['Visitante'] == team)) &
            (df['Fecha'] < date) &
            (df['Estado'] == MatchStatus.COMPLETED.value)
        ].sort_values('Fecha', ascending=False).head(1)
        
        features = {}
        
        # Calculate days since last match
        if len(last_match) > 0:
            features['days_since_last_match'] = (date - last_match.iloc[0]['Fecha']).days
        else:
            features['days_since_last_match'] = np.nan
        
        # Weekend indicator (Saturday or Sunday)
        features['is_weekend'] = 1 if date.weekday() >= 5 else 0
        
        # Evening match indicator (after 18:00)
        features['is_evening_match'] = 1 if date.hour >= 18 else 0
        
        return features
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        target_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Engineer features for match prediction.
        
        Args:
            df: Match data DataFrame
            target_date: Optional date to filter matches up to
            
        Returns:
            Tuple containing:
                - pd.DataFrame: DataFrame with engineered features
                - List[str]: List of feature names
        """
        if target_date is None:
            target_date = datetime.now()
        
        # Create copy of DataFrame
        df = df.copy()
        
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
            df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Filter matches up to target date
        df = df[df['Fecha'] <= target_date].copy()
        
        # Initialize feature DataFrame
        features_df = pd.DataFrame()
        
        # Calculate features for each match
        for idx, match in df.iterrows():
            if match['Estado'] != MatchStatus.COMPLETED.value:
                continue
            
            match_features = {}
            
            # Team form features
            home_form = self._calculate_team_form(
                df, match['Local'], match['Fecha'], True
            )
            away_form = self._calculate_team_form(
                df, match['Visitante'], match['Fecha'], False
            )
            
            for key, value in home_form.items():
                match_features[f'home_{key}'] = value
            for key, value in away_form.items():
                match_features[f'away_{key}'] = value
            
            # Head-to-head features
            h2h_stats = self._calculate_head2head(
                df, match['Local'], match['Visitante'], match['Fecha']
            )
            match_features.update(h2h_stats)
            
            # Time features
            home_time = self._calculate_time_features(
                df, match['Local'], match['Fecha']
            )
            away_time = self._calculate_time_features(
                df, match['Visitante'], match['Fecha']
            )
            
            for key, value in home_time.items():
                match_features[f'home_{key}'] = value
            for key, value in away_time.items():
                match_features[f'away_{key}'] = value
            
            # Add match features to DataFrame
            features_df = pd.concat([
                features_df,
                pd.DataFrame([match_features], index=[idx])
            ])
        
        # Get list of feature names
        feature_names = list(features_df.columns)
        
        return features_df, feature_names 