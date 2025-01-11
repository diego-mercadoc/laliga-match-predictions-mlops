import pandas as pd
import logging
from datetime import datetime
from utils.match_exceptions import MatchStatus, MatchException, MatchExceptionHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_match_data(file_path: str) -> pd.DataFrame:
    """Load match data from file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        pd.DataFrame: Loaded match data
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def identify_exceptions(df: pd.DataFrame) -> list[MatchException]:
    """Identify match exceptions in the data.
    
    Args:
        df: DataFrame containing match data
        
    Returns:
        list[MatchException]: List of identified exceptions
    """
    exceptions = []
    
    try:
        # Convert date column to datetime
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Check for postponed matches
        postponed_matches = df[df['Estado'] == MatchStatus.POSTPONED.value]
        for _, match in postponed_matches.iterrows():
            exceptions.append(MatchException(
                match_id=f"{match['Sem.']}_{match['Local']}_{match['Visitante']}",
                home_team=match['Local'],
                away_team=match['Visitante'],
                original_date=match['Fecha'],
                new_date=None,
                status=MatchStatus.POSTPONED,
                reason="Match postponed",
                created_at=datetime.now()
            ))
        
        # Check for cancelled matches
        cancelled_matches = df[df['Estado'] == MatchStatus.CANCELLED.value]
        for _, match in cancelled_matches.iterrows():
            exceptions.append(MatchException(
                match_id=f"{match['Sem.']}_{match['Local']}_{match['Visitante']}",
                home_team=match['Local'],
                away_team=match['Visitante'],
                original_date=match['Fecha'],
                new_date=None,
                status=MatchStatus.CANCELLED,
                reason="Match cancelled",
                created_at=datetime.now()
            ))
        
        # Check for suspended matches
        suspended_matches = df[df['Estado'] == MatchStatus.SUSPENDED.value]
        for _, match in suspended_matches.iterrows():
            exceptions.append(MatchException(
                match_id=f"{match['Sem.']}_{match['Local']}_{match['Visitante']}",
                home_team=match['Local'],
                away_team=match['Visitante'],
                original_date=match['Fecha'],
                new_date=None,
                status=MatchStatus.SUSPENDED,
                reason="Match suspended",
                created_at=datetime.now()
            ))
        
        logger.info(f"Identified {len(exceptions)} exceptions in the data")
        return exceptions
    
    except Exception as e:
        logger.error(f"Error identifying exceptions: {str(e)}")
        raise

def analyze_exceptions(handler: MatchExceptionHandler) -> None:
    """Analyze and log exception patterns.
    
    Args:
        handler: MatchExceptionHandler instance with exceptions
    """
    try:
        # Get summary statistics
        summary = handler.get_exception_summary()
        
        logger.info("\nException Summary:")
        logger.info(f"Total Exceptions: {summary['total_exceptions']}")
        logger.info(f"Unresolved Exceptions: {summary['unresolved_exceptions']}")
        
        logger.info("\nExceptions by Status:")
        for status, count in summary['exceptions_by_status'].items():
            if count > 0:
                logger.info(f"- {status.name}: {count}")
        
        # Analyze team patterns
        all_teams = set()
        for exc in handler.exceptions:
            all_teams.add(exc.home_team)
            all_teams.add(exc.away_team)
        
        logger.info("\nTeam Analysis:")
        for team in sorted(all_teams):
            team_exceptions = handler.get_exceptions_by_team(team)
            if team_exceptions:
                logger.info(f"\n{team}:")
                logger.info(f"- Total Exceptions: {len(team_exceptions)}")
                status_counts = {}
                for exc in team_exceptions:
                    status_counts[exc.status] = status_counts.get(exc.status, 0) + 1
                for status, count in status_counts.items():
                    logger.info(f"- {status.name}: {count}")
        
    except Exception as e:
        logger.error(f"Error analyzing exceptions: {str(e)}")
        raise

def main():
    """Main function to test match exceptions with real data."""
    try:
        # Load match data
        df = load_match_data('../data/laliga.csv')
        logger.info(f"Loaded {len(df)} matches")
        
        # Create exception handler
        handler = MatchExceptionHandler()
        
        # Identify and add exceptions
        exceptions = identify_exceptions(df)
        for exc in exceptions:
            handler.add_exception(exc)
        
        # Analyze exceptions
        analyze_exceptions(handler)
        
        # Export exceptions for future reference
        handler.export_exceptions('../data/match_exceptions.json')
        logger.info("\nExported exceptions to match_exceptions.json")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 