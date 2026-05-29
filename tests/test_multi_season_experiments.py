import pandas as pd

from services.multi_season_experiments import build_temporal_features, choose_current_season_split
from services.refresh_pipeline import read_refresh_status


def test_temporal_features_use_only_prior_matches():
    matches = pd.DataFrame(
        [
            {
                "Season": "2526",
                "Date": pd.Timestamp("2025-08-01"),
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTHG": 2,
                "FTAG": 1,
                "FTR": "H",
                "HS": 10,
                "AS": 7,
                "HST": 4,
                "AST": 2,
                "HC": 5,
                "AC": 3,
                "HY": 1,
                "AY": 2,
            },
            {
                "Season": "2526",
                "Date": pd.Timestamp("2025-08-08"),
                "HomeTeam": "Team B",
                "AwayTeam": "Team A",
                "FTHG": 0,
                "FTAG": 0,
                "FTR": "D",
                "HS": 8,
                "AS": 9,
                "HST": 1,
                "AST": 3,
                "HC": 4,
                "AC": 4,
                "HY": 3,
                "AY": 1,
            },
        ]
    )

    features = build_temporal_features(matches)

    first = features.iloc[0]
    second = features.iloc[1]
    assert first["home_matches_played"] == 0
    assert first["away_matches_played"] == 0
    assert first["home_elo"] == 1500
    assert first["away_elo"] == 1500
    assert first["elo_diff"] == 60
    assert first["home_rest_days"] == 7
    assert first["away_rest_days"] == 7
    assert second["home_matches_played"] == 1
    assert second["away_matches_played"] == 1
    assert second["home_points_per_match"] == 0
    assert second["away_points_per_match"] == 3
    assert second["home_elo"] < 1500
    assert second["away_elo"] > 1500
    assert second["home_rest_days"] == 7
    assert second["away_rest_days"] == 7


def test_current_season_split_is_inside_current_season():
    rows = []
    for index in range(130):
        rows.append(
            {
                "Season": "2526",
                "Date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=index),
                "FTR": "H",
            }
        )
    features = pd.DataFrame(rows)

    split_date = choose_current_season_split(features)

    assert split_date == "2025-11-29"


def test_read_refresh_status_handles_never_run(tmp_path):
    status = read_refresh_status(tmp_path)

    assert status["state"] == "never_run"
    assert status["status_path"].endswith("refresh_status.json")
