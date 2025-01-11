from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum

class MatchStatus(Enum):
    """Enum for match status."""
    SCHEDULED = "Hoy"
    POSTPONED = "Aplazado"
    COMPLETED = "Fin"
    CANCELLED = "Cancelado"
    IN_PROGRESS = "En curso"
    SUSPENDED = "Suspendido"

@dataclass
class MatchException:
    """Class to represent a match exception."""
    match_id: str
    home_team: str
    away_team: str
    original_date: datetime
    new_date: Optional[datetime]
    status: MatchStatus
    reason: str
    resolution: Optional[str] = None
    created_at: datetime = datetime.now()

class MatchExceptionHandler:
    """Class to handle match exceptions."""
    
    def __init__(self):
        """Initialize MatchExceptionHandler."""
        self.exceptions: List[MatchException] = []
    
    def add_exception(self, exception: MatchException) -> None:
        """Add a new match exception.
        
        Args:
            exception: MatchException object to add
        """
        self.exceptions.append(exception)
    
    def get_exceptions_by_team(self, team_name: str) -> List[MatchException]:
        """Get all exceptions involving a specific team.
        
        Args:
            team_name: Name of the team
            
        Returns:
            List[MatchException]: List of exceptions involving the team
        """
        return [
            exc for exc in self.exceptions
            if exc.home_team == team_name or exc.away_team == team_name
        ]
    
    def get_exceptions_by_status(self, status: MatchStatus) -> List[MatchException]:
        """Get all exceptions with a specific status.
        
        Args:
            status: MatchStatus to filter by
            
        Returns:
            List[MatchException]: List of exceptions with the specified status
        """
        return [exc for exc in self.exceptions if exc.status == status]
    
    def get_exceptions_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[MatchException]:
        """Get all exceptions within a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List[MatchException]: List of exceptions within the date range
        """
        return [
            exc for exc in self.exceptions
            if start_date <= exc.original_date <= end_date
        ]
    
    def resolve_exception(
        self,
        match_id: str,
        new_date: Optional[datetime],
        resolution: str
    ) -> None:
        """Resolve a match exception.
        
        Args:
            match_id: ID of the match to resolve
            new_date: New date for the match (if rescheduled)
            resolution: Description of how the exception was resolved
        """
        for exc in self.exceptions:
            if exc.match_id == match_id:
                exc.new_date = new_date
                exc.resolution = resolution
                if new_date:
                    exc.status = MatchStatus.SCHEDULED
                break
    
    def get_unresolved_exceptions(self) -> List[MatchException]:
        """Get all unresolved exceptions.
        
        Returns:
            List[MatchException]: List of unresolved exceptions
        """
        return [exc for exc in self.exceptions if not exc.resolution]
    
    def get_exception_summary(self) -> dict:
        """Get a summary of all exceptions.
        
        Returns:
            dict: Summary statistics of exceptions
        """
        total = len(self.exceptions)
        unresolved = len(self.get_unresolved_exceptions())
        by_status = {
            status: len(self.get_exceptions_by_status(status))
            for status in MatchStatus
        }
        
        return {
            "total_exceptions": total,
            "unresolved_exceptions": unresolved,
            "exceptions_by_status": by_status
        }
    
    def clear_resolved_exceptions(self) -> None:
        """Remove all resolved exceptions from the list."""
        self.exceptions = [exc for exc in self.exceptions if not exc.resolution]
    
    def export_exceptions(self, file_path: str) -> None:
        """Export exceptions to a file.
        
        Args:
            file_path: Path to export the exceptions to
        """
        import json
        from datetime import datetime
        
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, MatchStatus):
                return obj.value
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(file_path, 'w') as f:
            json.dump(
                [vars(exc) for exc in self.exceptions],
                f,
                default=datetime_handler,
                indent=2
            )
    
    @classmethod
    def import_exceptions(cls, file_path: str) -> 'MatchExceptionHandler':
        """Import exceptions from a file.
        
        Args:
            file_path: Path to import the exceptions from
            
        Returns:
            MatchExceptionHandler: New handler with imported exceptions
        """
        import json
        from datetime import datetime
        
        handler = cls()
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        for exc_data in data:
            # Convert string dates back to datetime
            exc_data['original_date'] = datetime.fromisoformat(exc_data['original_date'])
            if exc_data['new_date']:
                exc_data['new_date'] = datetime.fromisoformat(exc_data['new_date'])
            exc_data['created_at'] = datetime.fromisoformat(exc_data['created_at'])
            
            # Convert status string to enum
            exc_data['status'] = MatchStatus(exc_data['status'])
            
            # Create and add exception
            handler.add_exception(MatchException(**exc_data))
        
        return handler 