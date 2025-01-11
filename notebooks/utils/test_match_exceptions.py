import pytest
from datetime import datetime, timedelta
import os
import json
from .match_exceptions import MatchStatus, MatchException, MatchExceptionHandler

@pytest.fixture
def sample_exception():
    """Create a sample match exception."""
    return MatchException(
        match_id="2024_01",
        home_team="Barcelona",
        away_team="Real Madrid",
        original_date=datetime(2024, 1, 1),
        new_date=None,
        status=MatchStatus.POSTPONED,
        reason="Weather conditions",
        created_at=datetime(2024, 1, 1, 12, 0)
    )

@pytest.fixture
def handler_with_exceptions(sample_exception):
    """Create a handler with some sample exceptions."""
    handler = MatchExceptionHandler()
    
    # Add the sample exception
    handler.add_exception(sample_exception)
    
    # Add a few more exceptions
    handler.add_exception(MatchException(
        match_id="2024_02",
        home_team="Atletico Madrid",
        away_team="Sevilla",
        original_date=datetime(2024, 1, 2),
        new_date=datetime(2024, 1, 15),
        status=MatchStatus.SCHEDULED,
        reason="Stadium maintenance",
        resolution="Rescheduled",
        created_at=datetime(2024, 1, 2, 12, 0)
    ))
    
    handler.add_exception(MatchException(
        match_id="2024_03",
        home_team="Valencia",
        away_team="Barcelona",
        original_date=datetime(2024, 1, 3),
        new_date=None,
        status=MatchStatus.CANCELLED,
        reason="Security concerns",
        created_at=datetime(2024, 1, 3, 12, 0)
    ))
    
    return handler

def test_match_status_enum():
    """Test MatchStatus enum values."""
    assert MatchStatus.SCHEDULED.value == "Hoy"
    assert MatchStatus.POSTPONED.value == "Aplazado"
    assert MatchStatus.COMPLETED.value == "Fin"
    assert MatchStatus.CANCELLED.value == "Cancelado"
    assert MatchStatus.IN_PROGRESS.value == "En curso"
    assert MatchStatus.SUSPENDED.value == "Suspendido"

def test_match_exception_creation(sample_exception):
    """Test MatchException creation and attributes."""
    assert sample_exception.match_id == "2024_01"
    assert sample_exception.home_team == "Barcelona"
    assert sample_exception.away_team == "Real Madrid"
    assert sample_exception.status == MatchStatus.POSTPONED
    assert sample_exception.resolution is None

def test_add_exception(handler_with_exceptions):
    """Test adding exceptions to handler."""
    assert len(handler_with_exceptions.exceptions) == 3

def test_get_exceptions_by_team(handler_with_exceptions):
    """Test getting exceptions by team."""
    barcelona_exceptions = handler_with_exceptions.get_exceptions_by_team("Barcelona")
    assert len(barcelona_exceptions) == 2
    
    sevilla_exceptions = handler_with_exceptions.get_exceptions_by_team("Sevilla")
    assert len(sevilla_exceptions) == 1

def test_get_exceptions_by_status(handler_with_exceptions):
    """Test getting exceptions by status."""
    postponed = handler_with_exceptions.get_exceptions_by_status(MatchStatus.POSTPONED)
    assert len(postponed) == 1
    
    scheduled = handler_with_exceptions.get_exceptions_by_status(MatchStatus.SCHEDULED)
    assert len(scheduled) == 1
    
    cancelled = handler_with_exceptions.get_exceptions_by_status(MatchStatus.CANCELLED)
    assert len(cancelled) == 1

def test_get_exceptions_by_date_range(handler_with_exceptions):
    """Test getting exceptions by date range."""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 2)
    
    exceptions = handler_with_exceptions.get_exceptions_by_date_range(start_date, end_date)
    assert len(exceptions) == 2

def test_resolve_exception(handler_with_exceptions):
    """Test resolving an exception."""
    match_id = "2024_01"
    new_date = datetime(2024, 1, 15)
    resolution = "Weather improved, match rescheduled"
    
    handler_with_exceptions.resolve_exception(match_id, new_date, resolution)
    
    resolved_exception = next(exc for exc in handler_with_exceptions.exceptions if exc.match_id == match_id)
    assert resolved_exception.new_date == new_date
    assert resolved_exception.resolution == resolution
    assert resolved_exception.status == MatchStatus.SCHEDULED

def test_get_unresolved_exceptions(handler_with_exceptions):
    """Test getting unresolved exceptions."""
    unresolved = handler_with_exceptions.get_unresolved_exceptions()
    assert len(unresolved) == 2  # Two exceptions without resolution

def test_get_exception_summary(handler_with_exceptions):
    """Test getting exception summary."""
    summary = handler_with_exceptions.get_exception_summary()
    
    assert summary["total_exceptions"] == 3
    assert summary["unresolved_exceptions"] == 2
    assert summary["exceptions_by_status"][MatchStatus.POSTPONED] == 1
    assert summary["exceptions_by_status"][MatchStatus.SCHEDULED] == 1
    assert summary["exceptions_by_status"][MatchStatus.CANCELLED] == 1

def test_clear_resolved_exceptions(handler_with_exceptions):
    """Test clearing resolved exceptions."""
    handler_with_exceptions.clear_resolved_exceptions()
    assert len(handler_with_exceptions.exceptions) == 2  # One exception was resolved

def test_export_import_exceptions(handler_with_exceptions, tmp_path):
    """Test exporting and importing exceptions."""
    # Export exceptions
    export_path = os.path.join(tmp_path, "exceptions.json")
    handler_with_exceptions.export_exceptions(export_path)
    
    # Verify file exists and contains valid JSON
    assert os.path.exists(export_path)
    with open(export_path, 'r') as f:
        data = json.load(f)
        assert len(data) == 3
    
    # Import exceptions
    new_handler = MatchExceptionHandler.import_exceptions(export_path)
    assert len(new_handler.exceptions) == 3
    
    # Verify imported data matches original
    original_summary = handler_with_exceptions.get_exception_summary()
    imported_summary = new_handler.get_exception_summary()
    assert original_summary == imported_summary

def test_edge_cases():
    """Test edge cases and error handling."""
    handler = MatchExceptionHandler()
    
    # Empty handler
    assert len(handler.get_unresolved_exceptions()) == 0
    assert handler.get_exception_summary()["total_exceptions"] == 0
    
    # Invalid date range
    end_date = datetime(2024, 1, 1)
    start_date = end_date + timedelta(days=1)
    assert len(handler.get_exceptions_by_date_range(start_date, end_date)) == 0
    
    # Non-existent team
    assert len(handler.get_exceptions_by_team("Non-existent Team")) == 0
    
    # Resolve non-existent match
    handler.resolve_exception("non_existent_id", None, "No such match")
    assert len(handler.get_unresolved_exceptions()) == 0 