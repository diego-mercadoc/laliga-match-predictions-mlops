from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from services.multi_season_experiments import CURRENT_SEASON, DEFAULT_SEASONS, FEATURE_COLUMNS, choose_current_season_split
from services.multi_target_experiments import _best_binary_threshold


TARGET_NAME = "home_win"


@dataclass
class HomeWinExperimentResult:
    name: str
    model_name: str
    feature_set: str
    train_window: str
    train_seasons: List[str]
    recency_half_life: Optional[float]
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_count: int
    decision_threshold: float
    validation_accuracy: float
    validation_balanced_accuracy: float
    validation_f1_weighted: float
    validation_roc_auc: Optional[float]
    validation_brier_score: Optional[float]
    validation_selection_metric: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_f1_weighted: float
    test_roc_auc: Optional[float]
    test_log_loss: Optional[float]
    test_brier_score: Optional[float]
    test_selection_metric: float
    majority_class_rate: float
    baseline_accuracy_delta: float
    notes: str


def _feature_sets() -> Dict[str, List[str]]:
    market = [column for column in FEATURE_COLUMNS if column.startswith("market_")]
    non_market = [column for column in FEATURE_COLUMNS if not column.startswith("market_")]
    form_market_elo = [
        column
        for column in FEATURE_COLUMNS
        if column.startswith("market_")
        or column.endswith("_elo")
        or column in {"elo_diff", "season_rank_diff", "home_rest_days", "away_rest_days", "matchday"}
        or "_recent_" in column
        or "_season_" in column
        or column.startswith("h2h_")
    ]
    compact = [
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
        "home_elo",
        "away_elo",
        "elo_diff",
        "season_rank_diff",
        "home_recent_5_points_per_match",
        "away_recent_5_points_per_match",
        "home_recent_5_goal_diff",
        "away_recent_5_goal_diff",
        "home_season_points_per_match",
        "away_season_points_per_match",
        "home_home_points_per_match",
        "away_away_points_per_match",
        "home_rest_days",
        "away_rest_days",
        "matchday",
        "points_ppm_diff",
        "win_rate_diff",
        "goal_diff_ppm_diff",
        "season_points_ppm_diff",
        "season_goal_diff_ppm_diff",
        "recent_5_points_diff",
        "recent_5_goal_diff_diff",
        "home_home_win_rate",
        "home_home_goal_diff_per_match",
        "away_away_loss_rate",
        "away_away_goal_diff_per_match",
        "rest_days_diff",
    ]
    return {
        "all_features": list(FEATURE_COLUMNS),
        "no_market": non_market,
        "market_only": market,
        "form_market_elo": form_market_elo,
        "compact": [column for column in compact if column in FEATURE_COLUMNS],
    }


def _feature_pipeline(model, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _model_factories() -> Dict[str, Callable[[], Pipeline]]:
    return {
        "logistic_l2_c1": lambda: _feature_pipeline(LogisticRegression(max_iter=800, C=1.0), scale=True),
        "logistic_l2_c03": lambda: _feature_pipeline(LogisticRegression(max_iter=800, C=0.3), scale=True),
        "gaussian_nb_smoothing_1e9": lambda: _feature_pipeline(GaussianNB(var_smoothing=1e-9), scale=True),
        "gaussian_nb_smoothing_1e7": lambda: _feature_pipeline(GaussianNB(var_smoothing=1e-7), scale=True),
        "gaussian_nb_smoothing_1e5": lambda: _feature_pipeline(GaussianNB(var_smoothing=1e-5), scale=True),
        "histgb_l2_002": lambda: _feature_pipeline(
            HistGradientBoostingClassifier(max_iter=220, learning_rate=0.035, l2_regularization=0.02, random_state=42)
        ),
        "histgb_l2_010": lambda: _feature_pipeline(
            HistGradientBoostingClassifier(max_iter=260, learning_rate=0.025, l2_regularization=0.10, random_state=42)
        ),
        "extra_trees_depth8": lambda: _feature_pipeline(
            ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=1)
        ),
        "extra_trees_depth12": lambda: _feature_pipeline(
            ExtraTreesClassifier(n_estimators=500, max_depth=12, min_samples_leaf=12, random_state=42, n_jobs=1)
        ),
        "random_forest_depth8": lambda: _feature_pipeline(
            RandomForestClassifier(n_estimators=420, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=1)
        ),
        "gradient_boosting": lambda: _feature_pipeline(
            GradientBoostingClassifier(n_estimators=180, learning_rate=0.035, max_depth=2, random_state=42)
        ),
    }


def _training_windows(seasons: Sequence[str]) -> Dict[str, List[str]]:
    ordered = [season for season in DEFAULT_SEASONS if season in set(seasons)]

    def trailing(count: int) -> List[str]:
        return ordered[-min(count, len(ordered)) :]

    windows = {
        "current_only": [CURRENT_SEASON],
        "last_2": trailing(2),
        "last_4": trailing(4),
        "last_6": trailing(6),
        "last_8": trailing(8),
        "last_10": trailing(10),
        "last_12": trailing(12),
        f"all_{len(ordered)}": ordered,
    }
    return {name: value for name, value in windows.items() if value and CURRENT_SEASON in value}


def _split_train_validation(train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ordered = train.assign(_target=y_train.values).sort_values("Date").reset_index(drop=True)
    cut_index = min(max(60, int(len(ordered) * 0.82)), len(ordered) - 20)
    if cut_index <= 0:
        cut_index = max(1, int(len(ordered) * 0.75))
    inner_train = ordered.iloc[:cut_index].copy()
    validation = ordered.iloc[cut_index:].copy()
    return inner_train, validation, inner_train["_target"].astype(int), validation["_target"].astype(int)


def _safe_roc_auc(y_true: pd.Series, probability: np.ndarray) -> Optional[float]:
    try:
        return float(roc_auc_score(y_true, probability))
    except ValueError:
        return None


def _safe_log_loss(y_true: pd.Series, probabilities: np.ndarray) -> Optional[float]:
    try:
        return float(log_loss(y_true, probabilities, labels=[0, 1]))
    except ValueError:
        return None


def _score(y_true: pd.Series, probability: np.ndarray, threshold: float) -> Dict[str, Optional[float]]:
    predicted = (probability >= threshold).astype(int)
    auc = _safe_roc_auc(y_true, probability)
    balanced = float(balanced_accuracy_score(y_true, predicted))
    f1 = float(f1_score(y_true, predicted, average="weighted", zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": balanced,
        "f1_weighted": f1,
        "roc_auc": auc,
        "brier_score": float(brier_score_loss(y_true, probability)),
        "selection_metric": (balanced * 0.55) + ((auc if auc is not None else balanced) * 0.30) + (f1 * 0.15),
    }


def _recency_weight(frame: pd.DataFrame, half_life: Optional[float]) -> np.ndarray:
    if half_life is None:
        return np.ones(len(frame), dtype=float)
    max_age = float(frame["season_age"].max())
    ages = frame["season_age"].astype(float).to_numpy()
    return np.power(0.5, (max_age - ages) / float(half_life))


def _fit(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, train_frame: pd.DataFrame, half_life: Optional[float]) -> Pipeline:
    class_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    sample_weight = class_weight * _recency_weight(train_frame, half_life)
    model.fit(x_train, y_train, model__sample_weight=sample_weight)
    return model


def _positive_probability(model: Pipeline, x_frame: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["model"].classes_)
    probabilities = model.predict_proba(x_frame)
    return probabilities[:, classes.index(1)]


def _run_config(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set_name: str,
    feature_columns: List[str],
    train_window_name: str,
    train_seasons: List[str],
    model_name: str,
    model_factory: Callable[[], Pipeline],
    recency_half_life: Optional[float],
) -> HomeWinExperimentResult:
    candidate_train = train[train["Season"].isin(train_seasons)].copy()
    y_candidate = (candidate_train["FTR"] == "H").astype(int)
    y_test = (test["FTR"] == "H").astype(int)
    inner_train, validation, y_inner, y_validation = _split_train_validation(candidate_train, y_candidate)

    model = model_factory()
    _fit(model, inner_train[feature_columns], y_inner, inner_train, recency_half_life)
    validation_probability = _positive_probability(model, validation[feature_columns])
    threshold = _best_binary_threshold(y_validation, validation_probability)
    validation_scores = _score(y_validation, validation_probability, threshold)

    final_model = model_factory()
    _fit(final_model, candidate_train[feature_columns], y_candidate, candidate_train, recency_half_life)
    test_probability = _positive_probability(final_model, test[feature_columns])
    test_scores = _score(y_test, test_probability, threshold)
    test_probabilities = np.column_stack([1.0 - test_probability, test_probability])
    majority_rate = float(y_test.value_counts(normalize=True).max())
    baseline_delta = float(test_scores["accuracy"] or 0.0) - majority_rate
    notes = ""
    if baseline_delta <= 0:
        notes = "Does not beat home-win majority baseline on temporal test split."

    recency_name = "none" if recency_half_life is None else str(recency_half_life).replace(".", "p")
    return HomeWinExperimentResult(
        name=f"{TARGET_NAME}_{feature_set_name}_{train_window_name}_{model_name}_recency_{recency_name}",
        model_name=model_name,
        feature_set=feature_set_name,
        train_window=train_window_name,
        train_seasons=train_seasons,
        recency_half_life=recency_half_life,
        train_rows=len(candidate_train),
        validation_rows=len(validation),
        test_rows=len(test),
        feature_count=len(feature_columns),
        decision_threshold=float(threshold),
        validation_accuracy=float(validation_scores["accuracy"] or 0.0),
        validation_balanced_accuracy=float(validation_scores["balanced_accuracy"] or 0.0),
        validation_f1_weighted=float(validation_scores["f1_weighted"] or 0.0),
        validation_roc_auc=validation_scores["roc_auc"],
        validation_brier_score=validation_scores["brier_score"],
        validation_selection_metric=float(validation_scores["selection_metric"] or 0.0),
        test_accuracy=float(test_scores["accuracy"] or 0.0),
        test_balanced_accuracy=float(test_scores["balanced_accuracy"] or 0.0),
        test_f1_weighted=float(test_scores["f1_weighted"] or 0.0),
        test_roc_auc=test_scores["roc_auc"],
        test_log_loss=_safe_log_loss(y_test, test_probabilities),
        test_brier_score=test_scores["brier_score"],
        test_selection_metric=float(test_scores["selection_metric"] or 0.0),
        majority_class_rate=majority_rate,
        baseline_accuracy_delta=baseline_delta,
        notes=notes,
    )


def _run_config_batch(train: pd.DataFrame, test: pd.DataFrame, configs: List[tuple]) -> List[HomeWinExperimentResult]:
    with threadpool_limits(limits=1):
        return [_run_config(train, test, *config) for config in configs]


def _chunk_configs(configs: List[tuple], chunk_count: int) -> List[List[tuple]]:
    if chunk_count <= 1:
        return [configs]
    chunk_size = max(1, math.ceil(len(configs) / chunk_count))
    return [configs[index : index + chunk_size] for index in range(0, len(configs), chunk_size)]


def run_home_win_experiments(
    repo_root: Path,
    current_season: str = CURRENT_SEASON,
    n_jobs: Optional[int] = None,
    backend: Optional[str] = None,
    max_configs: Optional[int] = None,
    search_mode: str = "balanced",
) -> Dict[str, object]:
    started = time.perf_counter()
    output_dir = repo_root / "artifacts" / "home_win"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = repo_root / "artifacts" / "multi_season" / "laliga_multi_season_features.csv"
    features = pd.read_csv(feature_path)
    features["Season"] = features["Season"].astype(str)
    features["Date"] = pd.to_datetime(features["Date"])
    split_date = choose_current_season_split(features, current_season)
    train = features[(features["Season"] != current_season) | (features["Date"] < pd.to_datetime(split_date))].copy()
    test = features[(features["Season"] == current_season) & (features["Date"] >= pd.to_datetime(split_date))].copy()

    feature_sets = _feature_sets()
    windows = _training_windows(features["Season"].unique())
    models = _model_factories()
    recency_options: List[Optional[float]] = [None, 1.0, 2.0, 4.0, 8.0]
    if search_mode == "balanced":
        feature_sets = {name: feature_sets[name] for name in ["all_features", "compact", "form_market_elo", "market_only", "no_market"]}
        windows = {name: windows[name] for name in ["last_2", "last_4", "last_6", "last_8"]}
        models = {
            name: models[name]
            for name in [
                "logistic_l2_c03",
                "gaussian_nb_smoothing_1e9",
                "gaussian_nb_smoothing_1e7",
                "gaussian_nb_smoothing_1e5",
                "histgb_l2_002",
                "histgb_l2_010",
                "extra_trees_depth8",
            ]
        }
        recency_options = [None, 1.0, 2.0, 4.0]
    elif search_mode != "exhaustive":
        raise ValueError(f"Unsupported search mode: {search_mode}")

    configs = []
    for feature_set_name, feature_columns in feature_sets.items():
        for train_window_name, train_seasons in windows.items():
            for model_name, model_factory in models.items():
                for recency_half_life in recency_options:
                    if train_window_name in {"current_only", "last_2"} and recency_half_life is not None:
                        continue
                    configs.append(
                        (
                            feature_set_name,
                            feature_columns,
                            train_window_name,
                            train_seasons,
                            model_name,
                            model_factory,
                            recency_half_life,
                        )
                    )
    if max_configs is not None:
        configs = configs[: max(1, max_configs)]

    cpu_count = os.cpu_count() or 2
    workers = n_jobs if n_jobs is not None else max(1, cpu_count)
    if workers == -1:
        workers = cpu_count
    workers = max(1, min(workers, cpu_count))
    joblib_backend = backend or "loky"
    chunks = _chunk_configs(configs, workers * 4)
    batch_results = Parallel(n_jobs=workers, backend=joblib_backend, batch_size=1, pre_dispatch="all", verbose=0)(
        delayed(_run_config_batch)(train, test, chunk) for chunk in chunks
    )
    results = [result for batch in batch_results for result in batch]
    result_dicts = [asdict(result) for result in results]
    results_df = pd.DataFrame(result_dicts).sort_values(
        ["validation_selection_metric", "test_selection_metric", "test_accuracy"],
        ascending=False,
    )
    robust_results = results_df[(results_df["train_rows"] >= 500) & (results_df["validation_rows"] >= 100)].copy()
    best_by_validation_unrestricted = results_df.iloc[0].to_dict()
    best_by_validation = (robust_results if not robust_results.empty else results_df).iloc[0].to_dict()
    best_by_test = results_df.sort_values(
        ["test_selection_metric", "test_accuracy", "validation_selection_metric"],
        ascending=False,
    ).iloc[0].to_dict()

    results_path = output_dir / "experiment_results.csv"
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "best_validation_predictions.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8")

    best_columns = _feature_sets()[best_by_validation["feature_set"]]
    best_train_seasons = list(best_by_validation["train_seasons"])
    if isinstance(best_train_seasons, str):
        best_train_seasons = [item.strip().strip("'") for item in best_train_seasons.strip("[]").split(",") if item.strip()]
    best_recency = best_by_validation["recency_half_life"]
    best_recency = None if pd.isna(best_recency) else float(best_recency)
    best_train = train[train["Season"].isin(best_train_seasons)].copy()
    y_best_train = (best_train["FTR"] == "H").astype(int)
    y_test = (test["FTR"] == "H").astype(int)
    best_model = _model_factories()[best_by_validation["model_name"]]()
    _fit(best_model, best_train[best_columns], y_best_train, best_train, best_recency)
    best_probability = _positive_probability(best_model, test[best_columns])
    best_threshold = float(best_by_validation["decision_threshold"])
    predictions = pd.DataFrame(
        {
            "Date": test["Date"].values,
            "HomeTeam": test["HomeTeam"].values,
            "AwayTeam": test["AwayTeam"].values,
            "actual_home_win": y_test.values,
            "prob_home_win": best_probability,
            "predicted_home_win": (best_probability >= best_threshold).astype(int),
            "decision_threshold": best_threshold,
        }
    )
    predictions["correct"] = predictions["actual_home_win"] == predictions["predicted_home_win"]
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")

    payload = _json_ready(
        {
        "target": TARGET_NAME,
        "search_mode": search_mode,
        "split_date": split_date,
        "train_rows": len(train),
        "test_rows": len(test),
        "experiment_count": len(results),
        "parallel_jobs": workers,
        "parallel_backend": joblib_backend,
        "config_chunk_count": len(chunks),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "feature_set_count": len(feature_sets),
        "model_count": len(models),
        "best_by_validation": best_by_validation,
        "best_by_validation_unrestricted": best_by_validation_unrestricted,
        "best_by_test_audit": best_by_test,
        "results_csv": str(results_path),
        "predictions_csv": str(predictions_path),
        }
    )
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run high-throughput home-win experiments.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs. Defaults to all CPU cores; use -1 for all cores.")
    parser.add_argument("--backend", choices=["threading", "loky"], default=None, help="joblib backend. Defaults to loky processes.")
    parser.add_argument("--max-configs", type=int, default=None, help="Optional deterministic cap for smoke tests.")
    parser.add_argument("--search-mode", choices=["balanced", "exhaustive"], default="balanced", help="Balanced for recurring refresh; exhaustive for research runs.")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    result = run_home_win_experiments(
        repo_root,
        n_jobs=args.n_jobs,
        backend=args.backend,
        max_configs=args.max_configs,
        search_mode=args.search_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
