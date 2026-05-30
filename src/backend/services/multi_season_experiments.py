from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from services.laliga_spark_pipeline import create_spark_session


CURRENT_SEASON = "2526"
DEFAULT_SEASONS = [
    "1011",
    "1112",
    "1213",
    "1314",
    "1415",
    "1516",
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
]
FOOTBALL_DATA_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"

MARKET_ODDS_GROUPS = [
    ("AvgCH", "AvgCD", "AvgCA"),
    ("MaxCH", "MaxCD", "MaxCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("PSCH", "PSCD", "PSCA"),
    ("AvgH", "AvgD", "AvgA"),
    ("MaxH", "MaxD", "MaxA"),
    ("B365H", "B365D", "B365A"),
    ("BbAvH", "BbAvD", "BbAvA"),
    ("BbMxH", "BbMxD", "BbMxA"),
    ("BWH", "BWD", "BWA"),
    ("IWH", "IWD", "IWA"),
    ("LBH", "LBD", "LBA"),
    ("SBH", "SBD", "SBA"),
    ("WHH", "WHD", "WHA"),
    ("SJH", "SJD", "SJA"),
    ("VCH", "VCD", "VCA"),
    ("GBH", "GBD", "GBA"),
    ("BSH", "BSD", "BSA"),
]

TOTAL_GOALS_ODDS_GROUPS = [
    ("AvgC>2.5", "AvgC<2.5"),
    ("MaxC>2.5", "MaxC<2.5"),
    ("B365C>2.5", "B365C<2.5"),
    ("Avg>2.5", "Avg<2.5"),
    ("Max>2.5", "Max<2.5"),
    ("B365>2.5", "B365<2.5"),
]

MARKET_COLUMNS = sorted(
    {
        column
        for group in [*MARKET_ODDS_GROUPS, *TOTAL_GOALS_ODDS_GROUPS]
        for column in group
    }
)

RAW_COLUMNS = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
    *MARKET_COLUMNS,
]

FEATURE_COLUMNS = [
    "home_points_per_match",
    "away_points_per_match",
    "home_goals_for_per_match",
    "away_goals_for_per_match",
    "home_goals_against_per_match",
    "away_goals_against_per_match",
    "home_shots_per_match",
    "away_shots_per_match",
    "home_shots_on_target_per_match",
    "away_shots_on_target_per_match",
    "home_corners_per_match",
    "away_corners_per_match",
    "home_cards_per_match",
    "away_cards_per_match",
    "home_shot_accuracy",
    "away_shot_accuracy",
    "home_conversion_rate",
    "away_conversion_rate",
    "home_win_rate",
    "away_win_rate",
    "home_draw_rate",
    "away_draw_rate",
    "home_loss_rate",
    "away_loss_rate",
    "home_recent_points_per_match",
    "away_recent_points_per_match",
    "home_recent_goal_diff",
    "away_recent_goal_diff",
    "home_recent_3_points_per_match",
    "away_recent_3_points_per_match",
    "home_recent_3_goal_diff",
    "away_recent_3_goal_diff",
    "home_recent_3_shots_on_target",
    "away_recent_3_shots_on_target",
    "home_recent_3_corners",
    "away_recent_3_corners",
    "home_recent_5_points_per_match",
    "away_recent_5_points_per_match",
    "home_recent_5_goal_diff",
    "away_recent_5_goal_diff",
    "home_recent_5_shots_on_target",
    "away_recent_5_shots_on_target",
    "home_recent_5_corners",
    "away_recent_5_corners",
    "home_recent_10_points_per_match",
    "away_recent_10_points_per_match",
    "home_recent_10_goal_diff",
    "away_recent_10_goal_diff",
    "home_recent_10_shots_on_target",
    "away_recent_10_shots_on_target",
    "home_recent_10_corners",
    "away_recent_10_corners",
    "home_home_points_per_match",
    "away_away_points_per_match",
    "home_home_win_rate",
    "home_home_goal_diff_per_match",
    "home_home_goals_for_per_match",
    "home_home_goals_against_per_match",
    "home_home_shots_on_target_per_match",
    "away_away_loss_rate",
    "away_away_goal_diff_per_match",
    "away_away_goals_for_per_match",
    "away_away_goals_against_per_match",
    "away_away_shots_on_target_per_match",
    "home_season_points_per_match",
    "away_season_points_per_match",
    "home_season_goal_diff_per_match",
    "away_season_goal_diff_per_match",
    "home_season_shot_accuracy",
    "away_season_shot_accuracy",
    "home_season_conversion_rate",
    "away_season_conversion_rate",
    "home_season_rank_prior",
    "away_season_rank_prior",
    "season_rank_diff",
    "h2h_home_points_per_match",
    "h2h_home_goal_diff_per_match",
    "h2h_matches",
    "home_matches_played",
    "away_matches_played",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "short_rest_home",
    "short_rest_away",
    "matchday",
    "points_ppm_diff",
    "win_rate_diff",
    "goal_diff_ppm_diff",
    "shots_on_target_ppm_diff",
    "corners_ppm_diff",
    "conversion_rate_diff",
    "season_points_ppm_diff",
    "season_goal_diff_ppm_diff",
    "recent_5_points_diff",
    "recent_5_goal_diff_diff",
    "recent_10_points_diff",
    "recent_10_goal_diff_diff",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "market_home_advantage",
    "market_home_vs_not_home_prob",
    "market_home_favorite_margin",
    "market_is_home_favorite",
    "market_home_logit",
    "market_entropy",
    "market_overround",
    "market_sources_count",
    "market_over25_prob",
    "market_under25_prob",
    "market_goals_sources_count",
    "is_home_promoted_proxy",
    "is_away_promoted_proxy",
    "season_age",
]

LABEL_TO_RESULT = {0.0: "D", 1.0: "A", 2.0: "H"}
RESULT_TO_SPANISH = {"H": "Local", "D": "Empate", "A": "Visitante"}


@dataclass
class ExperimentConfig:
    name: str
    seasons: List[str]
    model_family: str
    recency_half_life: Optional[float] = None
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    name: str
    model_family: str
    train_rows: int
    test_rows: int
    seasons: List[str]
    accuracy: float
    f1_weighted: float
    log_loss_proxy: float
    recency_half_life: Optional[float]
    params: Dict[str, float]


@dataclass
class MultiSeasonRunResult:
    split_date: str
    train_current_rows: int
    test_current_rows: int
    best_experiment: ExperimentResult
    experiments: List[ExperimentResult]
    output_dir: str
    data_rows: int
    seasons: List[str]


def validate_source_files(paths: Iterable[Path]) -> List[Dict[str, object]]:
    reports: List[Dict[str, object]] = []
    required_columns = {"Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"}
    for path in paths:
        pdf = pd.read_csv(path, encoding="utf-8-sig")
        date_series = _parse_match_dates(pdf.get("Date"))
        market_columns = [column for column in MARKET_COLUMNS if column in pdf.columns]
        rows = len(pdf)
        reports.append(
            {
                "path": str(path),
                "season": path.stem.split("_")[-1],
                "rows": rows,
                "date_min": None if date_series.dropna().empty else str(date_series.min().date()),
                "date_max": None if date_series.dropna().empty else str(date_series.max().date()),
                "missing_required_columns": sorted(required_columns - set(pdf.columns)),
                "duplicate_fixture_rows": int(pdf.duplicated(subset=["Date", "HomeTeam", "AwayTeam"]).sum())
                if required_columns.issubset(pdf.columns)
                else None,
                "result_rows": int(pdf["FTR"].isin(["H", "D", "A"]).sum()) if "FTR" in pdf.columns else 0,
                "market_columns_present": market_columns,
                "market_column_count": len(market_columns),
                "avg_closing_odds_coverage": float(
                    pdf[[column for column in ["AvgCH", "AvgCD", "AvgCA"] if column in pdf.columns]]
                    .dropna()
                    .shape[0]
                    / rows
                )
                if rows and {"AvgCH", "AvgCD", "AvgCA"}.issubset(pdf.columns)
                else 0.0,
            }
        )
    return reports


def _parse_match_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], format="%d/%m/%y", errors="coerce")
    return parsed


def download_football_data(
    raw_dir: Path,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    force: bool = False,
) -> List[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for season in seasons:
        path = raw_dir / f"SP1_{season}.csv"
        if force or not path.exists():
            url = FOOTBALL_DATA_TEMPLATE.format(season=season)
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            path.write_bytes(response.content)
        paths.append(path)
    return paths


def load_match_data(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        season = path.stem.split("_")[-1]
        pdf = pd.read_csv(path, encoding="utf-8-sig")
        pdf["Season"] = season
        frames.append(pdf)

    df = pd.concat(frames, ignore_index=True)
    available = [col for col in RAW_COLUMNS if col in df.columns]
    df = df[available].copy()
    for missing in set(RAW_COLUMNS) - set(df.columns):
        df[missing] = pd.NA

    df["Date"] = _parse_match_dates(df["Date"])
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"])
    df = df[df["FTR"].isin(["H", "D", "A"])].copy()
    numeric_cols = [col for col in RAW_COLUMNS if col not in {"Season", "Date", "HomeTeam", "AwayTeam", "FTR"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return df


def _empty_stats() -> Dict[str, float]:
    return {
        "matches": 0,
        "points": 0.0,
        "gf": 0.0,
        "ga": 0.0,
        "shots": 0.0,
        "shots_target": 0.0,
        "corners": 0.0,
        "cards": 0.0,
        "wins": 0.0,
        "draws": 0.0,
        "losses": 0.0,
    }


def _averages(stats: Dict[str, float], prefix: str, default_matches: float = 1.0) -> Dict[str, float]:
    matches = max(stats["matches"], default_matches)
    shots = max(stats["shots"], 1.0)
    shots_target = max(stats["shots_target"], 1.0)
    return {
        f"{prefix}_points_per_match": stats["points"] / matches,
        f"{prefix}_goals_for_per_match": stats["gf"] / matches,
        f"{prefix}_goals_against_per_match": stats["ga"] / matches,
        f"{prefix}_shots_per_match": stats["shots"] / matches,
        f"{prefix}_shots_on_target_per_match": stats["shots_target"] / matches,
        f"{prefix}_corners_per_match": stats["corners"] / matches,
        f"{prefix}_cards_per_match": stats["cards"] / matches,
        f"{prefix}_shot_accuracy": stats["shots_target"] / shots,
        f"{prefix}_conversion_rate": stats["gf"] / shots,
        f"{prefix}_goals_per_shot_on_target": stats["gf"] / shots_target,
        f"{prefix}_win_rate": stats["wins"] / matches,
        f"{prefix}_draw_rate": stats["draws"] / matches,
        f"{prefix}_loss_rate": stats["losses"] / matches,
        f"{prefix}_matches_played": stats["matches"],
    }


def _update_stats(stats: Dict[str, float], gf: float, ga: float, shots: float, shots_target: float, corners: float, cards: float) -> None:
    stats["matches"] += 1
    stats["gf"] += gf
    stats["ga"] += ga
    stats["shots"] += 0.0 if pd.isna(shots) else float(shots)
    stats["shots_target"] += 0.0 if pd.isna(shots_target) else float(shots_target)
    stats["corners"] += 0.0 if pd.isna(corners) else float(corners)
    stats["cards"] += 0.0 if pd.isna(cards) else float(cards)
    if gf > ga:
        stats["points"] += 3
        stats["wins"] += 1
    elif gf == ga:
        stats["points"] += 1
        stats["draws"] += 1
    else:
        stats["losses"] += 1


def _recent_window_features(matches: List[Dict[str, float]], prefix: str, window: int) -> Dict[str, float]:
    recent_matches = matches[-window:]
    if not recent_matches:
        return {
            f"{prefix}_recent_{window}_points_per_match": 1.0,
            f"{prefix}_recent_{window}_goal_diff": 0.0,
            f"{prefix}_recent_{window}_shots_on_target": 0.0,
            f"{prefix}_recent_{window}_corners": 0.0,
        }
    count = len(recent_matches)
    return {
        f"{prefix}_recent_{window}_points_per_match": sum(item["points"] for item in recent_matches) / count,
        f"{prefix}_recent_{window}_goal_diff": sum(item["gf"] - item["ga"] for item in recent_matches) / count,
        f"{prefix}_recent_{window}_shots_on_target": sum(item["shots_target"] for item in recent_matches) / count,
        f"{prefix}_recent_{window}_corners": sum(item["corners"] for item in recent_matches) / count,
    }


def _market_features(match: pd.Series) -> Dict[str, float]:
    implied_rows = []
    for home_col, draw_col, away_col in MARKET_ODDS_GROUPS:
        odds = [match.get(home_col), match.get(draw_col), match.get(away_col)]
        if any(pd.isna(value) or float(value) <= 1.0 for value in odds):
            continue
        raw = [1.0 / float(value) for value in odds]
        overround = sum(raw)
        if overround <= 0:
            continue
        implied_rows.append(
            {
                "home": raw[0] / overround,
                "draw": raw[1] / overround,
                "away": raw[2] / overround,
                "overround": overround,
            }
        )

    if implied_rows:
        home_prob = sum(item["home"] for item in implied_rows) / len(implied_rows)
        draw_prob = sum(item["draw"] for item in implied_rows) / len(implied_rows)
        away_prob = sum(item["away"] for item in implied_rows) / len(implied_rows)
        overround = sum(item["overround"] for item in implied_rows) / len(implied_rows)
    else:
        home_prob, draw_prob, away_prob, overround = 0.45, 0.27, 0.28, 1.0

    entropy = -sum(prob * math.log(max(prob, 1e-15)) for prob in [home_prob, draw_prob, away_prob])

    goal_rows = []
    for over_col, under_col in TOTAL_GOALS_ODDS_GROUPS:
        odds = [match.get(over_col), match.get(under_col)]
        if any(pd.isna(value) or float(value) <= 1.0 for value in odds):
            continue
        raw = [1.0 / float(value) for value in odds]
        overround_goals = sum(raw)
        if overround_goals <= 0:
            continue
        goal_rows.append((raw[0] / overround_goals, raw[1] / overround_goals))

    if goal_rows:
        over25_prob = sum(item[0] for item in goal_rows) / len(goal_rows)
        under25_prob = sum(item[1] for item in goal_rows) / len(goal_rows)
    else:
        over25_prob, under25_prob = 0.50, 0.50

    return {
        "market_home_prob": home_prob,
        "market_draw_prob": draw_prob,
        "market_away_prob": away_prob,
        "market_home_advantage": home_prob - away_prob,
        "market_home_vs_not_home_prob": home_prob - (draw_prob + away_prob),
        "market_home_favorite_margin": home_prob - max(draw_prob, away_prob),
        "market_is_home_favorite": 1.0 if home_prob >= max(draw_prob, away_prob) else 0.0,
        "market_home_logit": math.log(max(home_prob, 1e-6) / max(1.0 - home_prob, 1e-6)),
        "market_entropy": entropy,
        "market_overround": overround,
        "market_sources_count": float(len(implied_rows)),
        "market_over25_prob": over25_prob,
        "market_under25_prob": under25_prob,
        "market_goals_sources_count": float(len(goal_rows)),
    }


def _prior_rank(season_stats: Dict[tuple[str, str], Dict[str, float]], season: str, team: str) -> int:
    teams = sorted({key_team for key_season, key_team in season_stats if key_season == season})
    if team not in teams:
        teams.append(team)

    ranked = sorted(
        teams,
        key=lambda item: (
            -season_stats.get((season, item), _empty_stats())["points"],
            -(season_stats.get((season, item), _empty_stats())["gf"] - season_stats.get((season, item), _empty_stats())["ga"]),
            -season_stats.get((season, item), _empty_stats())["gf"],
            item,
        ),
    )
    return ranked.index(team) + 1


def _h2h_features(matches: List[Dict[str, float]], home: str) -> Dict[str, float]:
    recent_matches = matches[-6:]
    if not recent_matches:
        return {
            "h2h_home_points_per_match": 1.0,
            "h2h_home_goal_diff_per_match": 0.0,
            "h2h_matches": 0.0,
        }
    count = len(recent_matches)
    points = sum(item["points_by_team"].get(home, 0.0) for item in recent_matches)
    goal_diff = sum(item["goal_diff_by_team"].get(home, 0.0) for item in recent_matches)
    return {
        "h2h_home_points_per_match": points / count,
        "h2h_home_goal_diff_per_match": goal_diff / count,
        "h2h_matches": float(count),
    }


def _home_win_matchup_features(row: Dict[str, object]) -> Dict[str, float]:
    home_goal_diff = float(row["home_goals_for_per_match"]) - float(row["home_goals_against_per_match"])
    away_goal_diff = float(row["away_goals_for_per_match"]) - float(row["away_goals_against_per_match"])
    return {
        "rest_days_diff": float(row["home_rest_days"]) - float(row["away_rest_days"]),
        "short_rest_home": 1.0 if float(row["home_rest_days"]) <= 3 else 0.0,
        "short_rest_away": 1.0 if float(row["away_rest_days"]) <= 3 else 0.0,
        "points_ppm_diff": float(row["home_points_per_match"]) - float(row["away_points_per_match"]),
        "win_rate_diff": float(row["home_win_rate"]) - float(row["away_win_rate"]),
        "goal_diff_ppm_diff": home_goal_diff - away_goal_diff,
        "shots_on_target_ppm_diff": float(row["home_shots_on_target_per_match"]) - float(row["away_shots_on_target_per_match"]),
        "corners_ppm_diff": float(row["home_corners_per_match"]) - float(row["away_corners_per_match"]),
        "conversion_rate_diff": float(row["home_conversion_rate"]) - float(row["away_conversion_rate"]),
        "season_points_ppm_diff": float(row["home_season_points_per_match"]) - float(row["away_season_points_per_match"]),
        "season_goal_diff_ppm_diff": float(row["home_season_goal_diff_per_match"]) - float(row["away_season_goal_diff_per_match"]),
        "recent_5_points_diff": float(row["home_recent_5_points_per_match"]) - float(row["away_recent_5_points_per_match"]),
        "recent_5_goal_diff_diff": float(row["home_recent_5_goal_diff"]) - float(row["away_recent_5_goal_diff"]),
        "recent_10_points_diff": float(row["home_recent_10_points_per_match"]) - float(row["away_recent_10_points_per_match"]),
        "recent_10_goal_diff_diff": float(row["home_recent_10_goal_diff"]) - float(row["away_recent_10_goal_diff"]),
    }


def build_temporal_features(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    first_season = str(matches["Season"].min())
    all_time: Dict[str, Dict[str, float]] = {}
    home_only: Dict[str, Dict[str, float]] = {}
    away_only: Dict[str, Dict[str, float]] = {}
    season_only: Dict[tuple[str, str], Dict[str, float]] = {}
    h2h: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    recent: Dict[str, List[Dict[str, float]]] = {}
    season_team_matches: Dict[tuple[str, str], int] = {}
    elo_ratings: Dict[str, float] = {}
    last_played: Dict[str, pd.Timestamp] = {}
    elo_k = 20.0
    home_advantage = 60.0

    for _, match in matches.sort_values("Date").iterrows():
        home = str(match["HomeTeam"])
        away = str(match["AwayTeam"])
        season = str(match["Season"])
        match_date = pd.Timestamp(match["Date"])
        for team in [home, away]:
            all_time.setdefault(team, _empty_stats())
            home_only.setdefault(team, _empty_stats())
            away_only.setdefault(team, _empty_stats())
            season_only.setdefault((season, team), _empty_stats())
            recent.setdefault(team, [])
            elo_ratings.setdefault(team, 1500.0)

        home_stats = _averages(all_time[home], "home")
        away_stats = _averages(all_time[away], "away")
        home_season_stats = _averages(season_only[(season, home)], "home_season")
        away_season_stats = _averages(season_only[(season, away)], "away_season")
        home_elo = elo_ratings[home]
        away_elo = elo_ratings[away]
        home_rest = (match_date - last_played[home]).days if home in last_played else 7
        away_rest = (match_date - last_played[away]).days if away in last_played else 7
        home_rank = _prior_rank(season_only, season, home)
        away_rank = _prior_rank(season_only, season, away)
        h2h_key = tuple(sorted([home, away]))
        row = {
            "Season": season,
            "Date": match_date,
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": int(match["FTHG"]),
            "FTAG": int(match["FTAG"]),
            "FTR": str(match["FTR"]),
            "HC": 0 if pd.isna(match.get("HC")) else int(match.get("HC")),
            "AC": 0 if pd.isna(match.get("AC")) else int(match.get("AC")),
            "HY": 0 if pd.isna(match.get("HY")) else int(match.get("HY")),
            "AY": 0 if pd.isna(match.get("AY")) else int(match.get("AY")),
            "HR": 0 if pd.isna(match.get("HR")) else int(match.get("HR")),
            "AR": 0 if pd.isna(match.get("AR")) else int(match.get("AR")),
            "matchday": max(
                season_team_matches.get((season, home), 0),
                season_team_matches.get((season, away), 0),
            )
            + 1,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo + home_advantage - away_elo,
            "home_rest_days": min(home_rest, 21),
            "away_rest_days": min(away_rest, 21),
            "home_season_points_per_match": home_season_stats["home_season_points_per_match"],
            "away_season_points_per_match": away_season_stats["away_season_points_per_match"],
            "home_season_goal_diff_per_match": home_season_stats["home_season_goals_for_per_match"]
            - home_season_stats["home_season_goals_against_per_match"],
            "away_season_goal_diff_per_match": away_season_stats["away_season_goals_for_per_match"]
            - away_season_stats["away_season_goals_against_per_match"],
            "home_season_shot_accuracy": home_season_stats["home_season_shot_accuracy"],
            "away_season_shot_accuracy": away_season_stats["away_season_shot_accuracy"],
            "home_season_conversion_rate": home_season_stats["home_season_conversion_rate"],
            "away_season_conversion_rate": away_season_stats["away_season_conversion_rate"],
            "home_season_rank_prior": home_rank,
            "away_season_rank_prior": away_rank,
            "season_rank_diff": away_rank - home_rank,
            **_h2h_features(h2h.get(h2h_key, []), home),
            **_market_features(match),
            **home_stats,
            **away_stats,
        }

        home_home = _averages(home_only[home], "home_home")
        away_away = _averages(away_only[away], "away_away")
        row["home_home_points_per_match"] = home_home["home_home_points_per_match"]
        row["away_away_points_per_match"] = away_away["away_away_points_per_match"]
        row["home_home_win_rate"] = home_home["home_home_win_rate"]
        row["home_home_goal_diff_per_match"] = (
            home_home["home_home_goals_for_per_match"] - home_home["home_home_goals_against_per_match"]
        )
        row["home_home_goals_for_per_match"] = home_home["home_home_goals_for_per_match"]
        row["home_home_goals_against_per_match"] = home_home["home_home_goals_against_per_match"]
        row["home_home_shots_on_target_per_match"] = home_home["home_home_shots_on_target_per_match"]
        row["away_away_loss_rate"] = away_away["away_away_loss_rate"]
        row["away_away_goal_diff_per_match"] = (
            away_away["away_away_goals_for_per_match"] - away_away["away_away_goals_against_per_match"]
        )
        row["away_away_goals_for_per_match"] = away_away["away_away_goals_for_per_match"]
        row["away_away_goals_against_per_match"] = away_away["away_away_goals_against_per_match"]
        row["away_away_shots_on_target_per_match"] = away_away["away_away_shots_on_target_per_match"]

        for team, prefix in [(home, "home"), (away, "away")]:
            recent_matches = recent[team][-5:]
            if recent_matches:
                row[f"{prefix}_recent_points_per_match"] = sum(item["points"] for item in recent_matches) / len(recent_matches)
                row[f"{prefix}_recent_goal_diff"] = sum(item["gf"] - item["ga"] for item in recent_matches) / len(recent_matches)
            else:
                row[f"{prefix}_recent_points_per_match"] = 1.0
                row[f"{prefix}_recent_goal_diff"] = 0.0
            for window in [3, 5, 10]:
                row.update(_recent_window_features(recent[team], prefix, window))

        row.update(_home_win_matchup_features(row))
        row["is_home_promoted_proxy"] = 1 if all_time[home]["matches"] < 15 and season != first_season else 0
        row["is_away_promoted_proxy"] = 1 if all_time[away]["matches"] < 15 and season != first_season else 0
        row["season_age"] = DEFAULT_SEASONS.index(season) if season in DEFAULT_SEASONS else 0
        rows.append(row)

        home_goals = float(match["FTHG"])
        away_goals = float(match["FTAG"])
        _update_stats(all_time[home], home_goals, away_goals, match.get("HS"), match.get("HST"), match.get("HC"), match.get("HY"))
        _update_stats(all_time[away], away_goals, home_goals, match.get("AS"), match.get("AST"), match.get("AC"), match.get("AY"))
        _update_stats(home_only[home], home_goals, away_goals, match.get("HS"), match.get("HST"), match.get("HC"), match.get("HY"))
        _update_stats(away_only[away], away_goals, home_goals, match.get("AS"), match.get("AST"), match.get("AC"), match.get("AY"))
        _update_stats(season_only[(season, home)], home_goals, away_goals, match.get("HS"), match.get("HST"), match.get("HC"), match.get("HY"))
        _update_stats(season_only[(season, away)], away_goals, home_goals, match.get("AS"), match.get("AST"), match.get("AC"), match.get("AY"))

        home_points = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
        away_points = 3 if away_goals > home_goals else 1 if away_goals == home_goals else 0
        recent[home].append(
            {
                "points": home_points,
                "gf": home_goals,
                "ga": away_goals,
                "shots_target": 0.0 if pd.isna(match.get("HST")) else float(match.get("HST")),
                "corners": 0.0 if pd.isna(match.get("HC")) else float(match.get("HC")),
            }
        )
        recent[away].append(
            {
                "points": away_points,
                "gf": away_goals,
                "ga": home_goals,
                "shots_target": 0.0 if pd.isna(match.get("AST")) else float(match.get("AST")),
                "corners": 0.0 if pd.isna(match.get("AC")) else float(match.get("AC")),
            }
        )
        h2h.setdefault(h2h_key, []).append(
            {
                "points_by_team": {home: float(home_points), away: float(away_points)},
                "goal_diff_by_team": {home: home_goals - away_goals, away: away_goals - home_goals},
            }
        )
        season_team_matches[(season, home)] = season_team_matches.get((season, home), 0) + 1
        season_team_matches[(season, away)] = season_team_matches.get((season, away), 0) + 1
        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400.0))
        actual_home = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        elo_delta = elo_k * (actual_home - expected_home)
        elo_ratings[home] = home_elo + elo_delta
        elo_ratings[away] = away_elo - elo_delta
        last_played[home] = match_date
        last_played[away] = match_date

    return pd.DataFrame(rows)


def choose_current_season_split(features: pd.DataFrame, current_season: str = CURRENT_SEASON) -> str:
    current = features[features["Season"] == current_season].sort_values("Date").reset_index(drop=True)
    if current.empty:
        raise ValueError(f"No rows found for current season {current_season}")
    index = min(max(120, int(len(current) * 0.40)), len(current) - 1)
    return str(current.iloc[index]["Date"].date())


def _with_sample_weights(frame: DataFrame, half_life: Optional[float]) -> DataFrame:
    if half_life is None:
        return frame.withColumn("sample_weight", F.lit(1.0))

    max_age = frame.agg(F.max("season_age")).first()[0]
    return frame.withColumn(
        "sample_weight",
        F.pow(F.lit(0.5), (F.lit(float(max_age)) - F.col("season_age")) / F.lit(float(half_life))),
    )


def _pipeline(model_family: str, params: Optional[Dict[str, float]] = None) -> Pipeline:
    params = params or {}
    label_indexer = StringIndexer(inputCol="FTR", outputCol="label", handleInvalid="error")
    imputer = Imputer(inputCols=FEATURE_COLUMNS, outputCols=[f"{col}_imputed" for col in FEATURE_COLUMNS], strategy="median")
    assembler = VectorAssembler(inputCols=[f"{col}_imputed" for col in FEATURE_COLUMNS], outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    if model_family == "random_forest":
        classifier = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            weightCol="sample_weight",
            predictionCol="prediction",
            probabilityCol="probability",
            numTrees=int(params.get("numTrees", 220)),
            maxDepth=int(params.get("maxDepth", 7)),
            minInstancesPerNode=int(params.get("minInstancesPerNode", 4)),
            seed=42,
        )
    elif model_family == "logistic_regression":
        classifier = LogisticRegression(
            labelCol="label",
            featuresCol="features",
            weightCol="sample_weight",
            predictionCol="prediction",
            probabilityCol="probability",
            maxIter=int(params.get("maxIter", 120)),
            regParam=float(params.get("regParam", 0.08)),
            elasticNetParam=float(params.get("elasticNetParam", 0.0)),
        )
    elif model_family == "decision_tree":
        classifier = DecisionTreeClassifier(
            labelCol="label",
            featuresCol="features",
            weightCol="sample_weight",
            predictionCol="prediction",
            probabilityCol="probability",
            maxDepth=int(params.get("maxDepth", 5)),
            minInstancesPerNode=int(params.get("minInstancesPerNode", 6)),
            seed=42,
        )
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
    return Pipeline(stages=[label_indexer, imputer, assembler, scaler, classifier])


def _log_loss_proxy(predictions: DataFrame) -> float:
    return float(
        predictions.withColumn("probability_array", vector_to_array("probability"))
        .withColumn(
            "picked_probability",
            F.greatest(
                F.element_at(F.col("probability_array"), F.col("label").cast("int") + F.lit(1)),
                F.lit(1e-15),
            ),
        )
        .select((-F.avg(F.log("picked_probability"))).alias("log_loss"))
        .first()["log_loss"]
    )


def run_experiment(
    spark: SparkSession,
    features: pd.DataFrame,
    split_date: str,
    config: ExperimentConfig,
    current_season: str = CURRENT_SEASON,
    cached_features: Optional[DataFrame] = None,
) -> tuple[ExperimentResult, DataFrame, object]:
    spark_features = cached_features if cached_features is not None else spark.createDataFrame(features)
    spark_features = spark_features.filter(F.col("Season").isin(config.seasons))
    spark_features = _with_sample_weights(spark_features, config.recency_half_life)
    train = spark_features.filter(
        (F.col("Season") != current_season) | (F.to_date("Date") < F.lit(split_date))
    )
    test = spark_features.filter(
        (F.col("Season") == current_season) & (F.to_date("Date") >= F.lit(split_date))
    )
    train_count = train.count()
    test_count = test.count()
    model = _pipeline(config.model_family, config.params).fit(train)
    predictions = model.transform(test)

    accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
    f1_weighted = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(predictions)
    log_loss = _log_loss_proxy(predictions)
    result = ExperimentResult(
        name=config.name,
        model_family=config.model_family,
        train_rows=train_count,
        test_rows=test_count,
        seasons=config.seasons,
        accuracy=float(accuracy),
        f1_weighted=float(f1_weighted),
        log_loss_proxy=log_loss,
        recency_half_life=config.recency_half_life,
        params=config.params,
    )
    return result, predictions, model


def experiment_grid() -> List[ExperimentConfig]:
    def trailing(count: int) -> List[str]:
        return DEFAULT_SEASONS[-min(count, len(DEFAULT_SEASONS)) :]

    windows = {
        "current_only": [CURRENT_SEASON],
        "last_2": trailing(2),
        "last_4": trailing(4),
        "last_6": trailing(6),
        "last_8": trailing(8),
        "last_10": trailing(10),
        "last_12": trailing(12),
        f"all_{len(DEFAULT_SEASONS)}": DEFAULT_SEASONS,
    }
    configs: List[ExperimentConfig] = []
    model_variants = [
        ("logistic_regression_l2", "logistic_regression", {"regParam": 0.08, "elasticNetParam": 0.0}),
        ("random_forest_regularized", "random_forest", {"numTrees": 80, "maxDepth": 5, "minInstancesPerNode": 8}),
    ]
    for window_name, seasons in windows.items():
        for variant_name, model_family, params in model_variants:
            configs.append(ExperimentConfig(f"{window_name}_{variant_name}", seasons, model_family, None, params))
            if len(seasons) >= 4:
                configs.append(ExperimentConfig(f"{window_name}_{variant_name}_recency_1", seasons, model_family, 1.0, params))
                configs.append(ExperimentConfig(f"{window_name}_{variant_name}_recency_2", seasons, model_family, 2.0, params))
                configs.append(ExperimentConfig(f"{window_name}_{variant_name}_recency_4", seasons, model_family, 4.0, params))
                if len(seasons) >= 8:
                    configs.append(ExperimentConfig(f"{window_name}_{variant_name}_recency_8", seasons, model_family, 8.0, params))
    return configs


def _prediction_output(predictions: DataFrame, model) -> pd.DataFrame:
    labels = model.stages[0].labels
    label_map = {float(index): label for index, label in enumerate(labels)}
    map_expr = F.create_map(*[item for pair in label_map.items() for item in (F.lit(pair[0]), F.lit(pair[1]))])
    pred = predictions.withColumn("PredictedFTR", map_expr[F.col("prediction")])
    pdf = pred.select("Date", "HomeTeam", "AwayTeam", "FTR", "PredictedFTR", "probability").orderBy("Date").toPandas()
    pdf["Resultado"] = pdf["FTR"].map(RESULT_TO_SPANISH)
    pdf["Prediccion"] = pdf["PredictedFTR"].map(RESULT_TO_SPANISH)
    pdf["correct"] = pdf["FTR"] == pdf["PredictedFTR"]
    return pdf.drop(columns=["probability"])


def run_multi_season_experiments(
    repo_root: Path,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    current_season: str = CURRENT_SEASON,
    spark: Optional[SparkSession] = None,
    force_download: bool = False,
) -> MultiSeasonRunResult:
    owns_spark = spark is None
    spark = spark or create_spark_session("laliga-multi-season-experiments", master="local[2]")
    output_dir = repo_root / "artifacts" / "multi_season"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_paths = download_football_data(repo_root / "data" / "raw" / "football_data", seasons, force=force_download)
        source_report = validate_source_files(raw_paths)
        source_report_path = output_dir / "data_source_report.json"
        source_report_path.write_text(json.dumps(source_report, indent=2, ensure_ascii=False), encoding="utf-8")
        matches = load_match_data(raw_paths)
        features = build_temporal_features(matches)
        features_path = output_dir / "laliga_multi_season_features.csv"
        features.to_csv(features_path, index=False, encoding="utf-8")
        split_date = choose_current_season_split(features, current_season)

        experiment_results: List[ExperimentResult] = []
        best_tuple = None
        spark_features = spark.createDataFrame(features).cache()
        spark_features.count()
        try:
            for config in experiment_grid():
                result, predictions, model = run_experiment(
                    spark,
                    features,
                    split_date,
                    config,
                    current_season,
                    cached_features=spark_features,
                )
                experiment_results.append(result)
                if best_tuple is None or (result.log_loss_proxy, -result.accuracy) < (best_tuple[0].log_loss_proxy, -best_tuple[0].accuracy):
                    best_tuple = (result, predictions, model)
        finally:
            spark_features.unpersist()

        assert best_tuple is not None
        best_result, best_predictions, best_model = best_tuple
        predictions_pdf = _prediction_output(best_predictions, best_model)
        predictions_path = output_dir / "current_season_predictions.csv"
        predictions_pdf.to_csv(predictions_path, index=False, encoding="utf-8")

        experiments_path = output_dir / "experiment_results.csv"
        pd.DataFrame([asdict(item) for item in experiment_results]).to_csv(experiments_path, index=False, encoding="utf-8")

        run_result = MultiSeasonRunResult(
            split_date=split_date,
            train_current_rows=len(features[(features["Season"] == current_season) & (features["Date"] < pd.to_datetime(split_date))]),
            test_current_rows=len(features[(features["Season"] == current_season) & (features["Date"] >= pd.to_datetime(split_date))]),
            best_experiment=best_result,
            experiments=experiment_results,
            output_dir=str(output_dir),
            data_rows=len(features),
            seasons=list(seasons),
        )
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "split_date": run_result.split_date,
                    "train_current_rows": run_result.train_current_rows,
                    "test_current_rows": run_result.test_current_rows,
                    "data_rows": run_result.data_rows,
                    "seasons": run_result.seasons,
                    "best_experiment": asdict(run_result.best_experiment),
                    "experiments": [asdict(item) for item in run_result.experiments],
                    "predictions_csv": str(predictions_path),
                    "features_csv": str(features_path),
                    "experiments_csv": str(experiments_path),
                    "data_source_report_json": str(source_report_path),
                    "feature_count": len(FEATURE_COLUMNS),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return run_result
    finally:
        if owns_spark:
            spark.stop()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    result = run_multi_season_experiments(repo_root)
    print(json.dumps(
        {
            "split_date": result.split_date,
            "data_rows": result.data_rows,
            "best_experiment": asdict(result.best_experiment),
            "output_dir": result.output_dir,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
