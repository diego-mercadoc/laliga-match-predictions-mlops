from __future__ import annotations

import json
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
DEFAULT_SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
FOOTBALL_DATA_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"

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
    "home_home_points_per_match",
    "away_away_points_per_match",
    "home_matches_played",
    "away_matches_played",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_rest_days",
    "away_rest_days",
    "matchday",
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

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
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
    return {
        f"{prefix}_points_per_match": stats["points"] / matches,
        f"{prefix}_goals_for_per_match": stats["gf"] / matches,
        f"{prefix}_goals_against_per_match": stats["ga"] / matches,
        f"{prefix}_shots_per_match": stats["shots"] / matches,
        f"{prefix}_shots_on_target_per_match": stats["shots_target"] / matches,
        f"{prefix}_corners_per_match": stats["corners"] / matches,
        f"{prefix}_cards_per_match": stats["cards"] / matches,
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


def build_temporal_features(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    first_season = str(matches["Season"].min())
    all_time: Dict[str, Dict[str, float]] = {}
    home_only: Dict[str, Dict[str, float]] = {}
    away_only: Dict[str, Dict[str, float]] = {}
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
            recent.setdefault(team, [])
            elo_ratings.setdefault(team, 1500.0)

        home_stats = _averages(all_time[home], "home")
        away_stats = _averages(all_time[away], "away")
        home_elo = elo_ratings[home]
        away_elo = elo_ratings[away]
        home_rest = (match_date - last_played[home]).days if home in last_played else 7
        away_rest = (match_date - last_played[away]).days if away in last_played else 7
        row = {
            "Season": season,
            "Date": match_date,
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": int(match["FTHG"]),
            "FTAG": int(match["FTAG"]),
            "FTR": str(match["FTR"]),
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
            **home_stats,
            **away_stats,
        }

        home_home = _averages(home_only[home], "home_home")
        away_away = _averages(away_only[away], "away_away")
        row["home_home_points_per_match"] = home_home["home_home_points_per_match"]
        row["away_away_points_per_match"] = away_away["away_away_points_per_match"]

        for team, prefix in [(home, "home"), (away, "away")]:
            recent_matches = recent[team][-5:]
            if recent_matches:
                row[f"{prefix}_recent_points_per_match"] = sum(item["points"] for item in recent_matches) / len(recent_matches)
                row[f"{prefix}_recent_goal_diff"] = sum(item["gf"] - item["ga"] for item in recent_matches) / len(recent_matches)
            else:
                row[f"{prefix}_recent_points_per_match"] = 1.0
                row[f"{prefix}_recent_goal_diff"] = 0.0

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

        recent[home].append({"points": 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0, "gf": home_goals, "ga": away_goals})
        recent[away].append({"points": 3 if away_goals > home_goals else 1 if away_goals == home_goals else 0, "gf": away_goals, "ga": home_goals})
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
) -> tuple[ExperimentResult, DataFrame, object]:
    spark_features = spark.createDataFrame(features[features["Season"].isin(config.seasons)])
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
    windows = {
        "current_only": ["2526"],
        "last_2": ["2425", "2526"],
        "last_4": ["2223", "2324", "2425", "2526"],
        "last_8": DEFAULT_SEASONS,
    }
    configs: List[ExperimentConfig] = []
    model_variants = [
        ("logistic_regression_l2", "logistic_regression", {"regParam": 0.08, "elasticNetParam": 0.0}),
        ("random_forest_regularized", "random_forest", {"numTrees": 80, "maxDepth": 5, "minInstancesPerNode": 8}),
    ]
    for window_name, seasons in windows.items():
        for variant_name, model_family, params in model_variants:
            configs.append(ExperimentConfig(f"{window_name}_{variant_name}", seasons, model_family, None, params))
            if window_name in {"last_4", "last_8"}:
                configs.append(ExperimentConfig(f"{window_name}_{variant_name}_recency_2", seasons, model_family, 2.0, params))
                configs.append(ExperimentConfig(f"{window_name}_{variant_name}_recency_4", seasons, model_family, 4.0, params))
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
        matches = load_match_data(raw_paths)
        features = build_temporal_features(matches)
        features_path = output_dir / "laliga_multi_season_features.csv"
        features.to_csv(features_path, index=False, encoding="utf-8")
        split_date = choose_current_season_split(features, current_season)

        experiment_results: List[ExperimentResult] = []
        best_tuple = None
        for config in experiment_grid():
            result, predictions, model = run_experiment(spark, features, split_date, config, current_season)
            experiment_results.append(result)
            if best_tuple is None or (result.log_loss_proxy, -result.accuracy) < (best_tuple[0].log_loss_proxy, -best_tuple[0].accuracy):
                best_tuple = (result, predictions, model)

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
