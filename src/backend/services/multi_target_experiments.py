from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from services.multi_season_experiments import CURRENT_SEASON, FEATURE_COLUMNS, choose_current_season_split


CLASSIFICATION_TARGETS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "match_result_1x2": lambda df: df["FTR"],
    "home_win": lambda df: (df["FTR"] == "H").astype(int),
    "draw": lambda df: (df["FTR"] == "D").astype(int),
    "away_win": lambda df: (df["FTR"] == "A").astype(int),
    "home_or_draw_double_chance": lambda df: df["FTR"].isin(["H", "D"]).astype(int),
    "away_or_draw_double_chance": lambda df: df["FTR"].isin(["A", "D"]).astype(int),
    "over_0_5_goals": lambda df: ((df["FTHG"] + df["FTAG"]) > 0.5).astype(int),
    "over_1_5_goals": lambda df: ((df["FTHG"] + df["FTAG"]) > 1.5).astype(int),
    "over_2_5_goals": lambda df: ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int),
    "over_3_5_goals": lambda df: ((df["FTHG"] + df["FTAG"]) > 3.5).astype(int),
    "under_5_5_goals": lambda df: ((df["FTHG"] + df["FTAG"]) < 5.5).astype(int),
    "under_6_5_goals": lambda df: ((df["FTHG"] + df["FTAG"]) < 6.5).astype(int),
    "both_teams_score": lambda df: ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int),
    "home_scores": lambda df: (df["FTHG"] > 0).astype(int),
    "away_scores": lambda df: (df["FTAG"] > 0).astype(int),
    "total_corners_over_8_5": lambda df: ((df["HC"] + df["AC"]) > 8.5).astype(int),
    "total_corners_over_9_5": lambda df: ((df["HC"] + df["AC"]) > 9.5).astype(int),
    "total_yellow_cards_over_4_5": lambda df: ((df["HY"] + df["AY"]) > 4.5).astype(int),
    "any_red_card": lambda df: ((df["HR"] + df["AR"]) > 0).astype(int),
}

REGRESSION_TARGETS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "home_goals": lambda df: df["FTHG"],
    "away_goals": lambda df: df["FTAG"],
    "total_goals": lambda df: df["FTHG"] + df["FTAG"],
    "total_corners": lambda df: df["HC"] + df["AC"],
    "total_yellow_cards": lambda df: df["HY"] + df["AY"],
}


@dataclass
class TargetResult:
    target: str
    task_type: str
    model_name: str
    train_rows: int
    test_rows: int
    target_rate: Optional[float]
    majority_class_rate: Optional[float]
    accuracy: Optional[float]
    balanced_accuracy: Optional[float]
    f1_weighted: Optional[float]
    roc_auc: Optional[float]
    log_loss: Optional[float]
    brier_score: Optional[float]
    mae: Optional[float]
    rmse: Optional[float]
    r2: Optional[float]
    baseline_metric: float
    selection_metric: float
    notes: str


def _feature_pipeline(model, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _classification_models(is_multiclass: bool) -> Dict[str, Pipeline]:
    return {
        "dummy_most_frequent": _feature_pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic_l2_balanced": _feature_pipeline(
            LogisticRegression(max_iter=500, class_weight="balanced" if not is_multiclass else None, n_jobs=None),
            scale=True,
        ),
        "hist_gradient_boosting": _feature_pipeline(
            HistGradientBoostingClassifier(max_iter=160, learning_rate=0.045, l2_regularization=0.02, random_state=42)
        ),
        "extra_trees": _feature_pipeline(
            ExtraTreesClassifier(
                n_estimators=320,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced" if not is_multiclass else None,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }


def _regression_models() -> Dict[str, Pipeline]:
    return {
        "dummy_median": _feature_pipeline(DummyRegressor(strategy="median")),
        "ridge": _feature_pipeline(Ridge(alpha=8.0), scale=True),
        "hist_gradient_boosting": _feature_pipeline(
            HistGradientBoostingRegressor(max_iter=180, learning_rate=0.045, l2_regularization=0.02, random_state=42)
        ),
        "extra_trees": _feature_pipeline(
            ExtraTreesRegressor(n_estimators=320, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=-1)
        ),
    }


def _safe_log_loss(y_true, probabilities, labels) -> Optional[float]:
    try:
        return float(log_loss(y_true, probabilities, labels=labels))
    except ValueError:
        return None


def _safe_roc_auc(y_true, probabilities, classes) -> Optional[float]:
    try:
        if len(classes) == 2:
            positive_index = list(classes).index(1) if 1 in classes else 1
            return float(roc_auc_score(y_true, probabilities[:, positive_index]))
        return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted"))
    except ValueError:
        return None


def _classification_result(name: str, model_name: str, model: Pipeline, x_test: pd.DataFrame, y_train, y_test) -> TargetResult:
    predicted = model.predict(x_test)
    classes = getattr(model.named_steps["model"], "classes_", np.unique(y_train))
    probabilities = model.predict_proba(x_test) if hasattr(model.named_steps["model"], "predict_proba") else None
    counts = pd.Series(y_test).value_counts(normalize=True)
    majority_rate = float(counts.max())
    target_rate = float(pd.Series(y_test).mean()) if set(pd.Series(y_test).dropna().unique()).issubset({0, 1}) else None
    auc = _safe_roc_auc(y_test, probabilities, classes) if probabilities is not None else None
    loss = _safe_log_loss(y_test, probabilities, labels=list(classes)) if probabilities is not None else None
    brier = None
    if probabilities is not None and len(classes) == 2 and set(classes).issubset({0, 1}):
        positive_index = list(classes).index(1)
        brier = float(brier_score_loss(y_test, probabilities[:, positive_index]))

    balanced = float(balanced_accuracy_score(y_test, predicted))
    accuracy = float(accuracy_score(y_test, predicted))
    f1 = float(f1_score(y_test, predicted, average="weighted", zero_division=0))
    selection = (balanced + f1) / 2.0
    notes = ""
    if target_rate is not None and (target_rate < 0.15 or target_rate > 0.85):
        notes = "Class is highly imbalanced; accuracy is not enough by itself."
    return TargetResult(
        target=name,
        task_type="classification",
        model_name=model_name,
        train_rows=len(y_train),
        test_rows=len(y_test),
        target_rate=target_rate,
        majority_class_rate=majority_rate,
        accuracy=accuracy,
        balanced_accuracy=balanced,
        f1_weighted=f1,
        roc_auc=auc,
        log_loss=loss,
        brier_score=brier,
        mae=None,
        rmse=None,
        r2=None,
        baseline_metric=majority_rate,
        selection_metric=selection,
        notes=notes,
    )


def _regression_result(name: str, model_name: str, model: Pipeline, x_test: pd.DataFrame, y_train, y_test) -> TargetResult:
    predicted = model.predict(x_test)
    mae = float(mean_absolute_error(y_test, predicted))
    rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
    r2 = float(r2_score(y_test, predicted))
    baseline_mae = float(mean_absolute_error(y_test, np.repeat(np.median(y_train), len(y_test))))
    selection = baseline_mae - mae
    return TargetResult(
        target=name,
        task_type="regression",
        model_name=model_name,
        train_rows=len(y_train),
        test_rows=len(y_test),
        target_rate=None,
        majority_class_rate=None,
        accuracy=None,
        balanced_accuracy=None,
        f1_weighted=None,
        roc_auc=None,
        log_loss=None,
        brier_score=None,
        mae=mae,
        rmse=rmse,
        r2=r2,
        baseline_metric=baseline_mae,
        selection_metric=selection,
        notes="Selection metric is baseline MAE improvement.",
    )


def _split_features(features: pd.DataFrame, split_date: str, current_season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dated = features.copy()
    dated["Date"] = pd.to_datetime(dated["Date"])
    train = dated[(dated["Season"] != current_season) | (dated["Date"] < pd.to_datetime(split_date))].copy()
    test = dated[(dated["Season"] == current_season) & (dated["Date"] >= pd.to_datetime(split_date))].copy()
    return train, test


def _select_best(results: List[TargetResult]) -> TargetResult:
    if results[0].task_type == "classification":
        return max(results, key=lambda item: (item.selection_metric, item.accuracy or 0.0))
    return max(results, key=lambda item: (item.selection_metric, -(item.mae or 999.0)))


def run_multi_target_experiments(
    repo_root: Path,
    current_season: str = CURRENT_SEASON,
) -> Dict[str, object]:
    output_dir = repo_root / "artifacts" / "multi_target"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = repo_root / "artifacts" / "multi_season" / "laliga_multi_season_features.csv"
    features = pd.read_csv(feature_path)
    features["Season"] = features["Season"].astype(str)
    features["Date"] = pd.to_datetime(features["Date"])
    split_date = choose_current_season_split(features, current_season)
    train, test = _split_features(features, split_date, current_season)
    x_train = train[FEATURE_COLUMNS]
    x_test = test[FEATURE_COLUMNS]

    target_results: List[TargetResult] = []
    predictions_frames: List[pd.DataFrame] = []

    for target_name, target_fn in CLASSIFICATION_TARGETS.items():
        y_train = target_fn(train)
        y_test = target_fn(test)
        is_multiclass = y_train.nunique() > 2
        candidates: List[TargetResult] = []
        for model_name, model in _classification_models(is_multiclass).items():
            model.fit(x_train, y_train)
            candidates.append(_classification_result(target_name, model_name, model, x_test, y_train, y_test))
        best = _select_best(candidates)
        target_results.append(best)
        best_model = _classification_models(is_multiclass)[best.model_name]
        best_model.fit(x_train, y_train)
        predictions_frames.append(
            pd.DataFrame(
                {
                    "Date": test["Date"].values,
                    "HomeTeam": test["HomeTeam"].values,
                    "AwayTeam": test["AwayTeam"].values,
                    "target": target_name,
                    "actual": y_test.values,
                    "predicted": best_model.predict(x_test),
                }
            )
        )

    for target_name, target_fn in REGRESSION_TARGETS.items():
        y_train = target_fn(train)
        y_test = target_fn(test)
        candidates = []
        for model_name, model in _regression_models().items():
            model.fit(x_train, y_train)
            candidates.append(_regression_result(target_name, model_name, model, x_test, y_train, y_test))
        best = _select_best(candidates)
        target_results.append(best)
        best_model = _regression_models()[best.model_name]
        best_model.fit(x_train, y_train)
        predictions_frames.append(
            pd.DataFrame(
                {
                    "Date": test["Date"].values,
                    "HomeTeam": test["HomeTeam"].values,
                    "AwayTeam": test["AwayTeam"].values,
                    "target": target_name,
                    "actual": y_test.values,
                    "predicted": best_model.predict(x_test),
                }
            )
        )

    result_dicts = [asdict(item) for item in target_results]
    results_path = output_dir / "target_results.csv"
    predictions_path = output_dir / "target_predictions.csv"
    summary_path = output_dir / "summary.json"
    pd.DataFrame(result_dicts).to_csv(results_path, index=False, encoding="utf-8")
    pd.concat(predictions_frames, ignore_index=True).to_csv(predictions_path, index=False, encoding="utf-8")

    classification = [item for item in target_results if item.task_type == "classification"]
    regression = [item for item in target_results if item.task_type == "regression"]
    best_accuracy = max(classification, key=lambda item: item.accuracy or 0.0)
    best_balanced = max(classification, key=lambda item: item.selection_metric)
    best_regression = max(regression, key=lambda item: item.selection_metric)

    payload = {
        "split_date": split_date,
        "feature_count": len(FEATURE_COLUMNS),
        "train_rows": len(train),
        "test_rows": len(test),
        "target_count": len(target_results),
        "best_accuracy_target": asdict(best_accuracy),
        "best_balanced_target": asdict(best_balanced),
        "best_regression_target": asdict(best_regression),
        "targets": result_dicts,
        "results_csv": str(results_path),
        "predictions_csv": str(predictions_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    result = run_multi_target_experiments(repo_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
