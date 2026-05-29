import pandas as pd

from services.multi_season_experiments import build_temporal_features, choose_current_season_split, validate_source_files
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
                "AvgCH": 1.8,
                "AvgCD": 3.5,
                "AvgCA": 4.4,
                "AvgC>2.5": 1.9,
                "AvgC<2.5": 1.95,
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
                "AvgCH": 2.2,
                "AvgCD": 3.1,
                "AvgCA": 3.0,
                "AvgC>2.5": 2.0,
                "AvgC<2.5": 1.85,
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
    assert first["home_recent_3_points_per_match"] == 1
    assert first["home_season_points_per_match"] == 0
    assert first["h2h_matches"] == 0
    assert first["market_sources_count"] == 1
    assert first["market_home_prob"] > first["market_away_prob"]
    assert second["home_matches_played"] == 1
    assert second["away_matches_played"] == 1
    assert second["home_points_per_match"] == 0
    assert second["away_points_per_match"] == 3
    assert second["home_recent_3_points_per_match"] == 0
    assert second["away_recent_3_points_per_match"] == 3
    assert second["home_season_points_per_match"] == 0
    assert second["away_season_points_per_match"] == 3
    assert second["h2h_matches"] == 1
    assert second["h2h_home_points_per_match"] == 0
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


def test_validate_source_files_reports_market_coverage(tmp_path):
    source = tmp_path / "SP1_2526.csv"
    pd.DataFrame(
        [
            {
                "Date": "01/08/2025",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTR": "H",
                "FTHG": 2,
                "FTAG": 1,
                "AvgCH": 1.8,
                "AvgCD": 3.5,
                "AvgCA": 4.4,
            }
        ]
    ).to_csv(source, index=False)

    report = validate_source_files([source])[0]

    assert report["missing_required_columns"] == []
    assert report["result_rows"] == 1
    assert report["avg_closing_odds_coverage"] == 1.0
