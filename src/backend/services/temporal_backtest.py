from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from services.laliga_spark_pipeline import (
    PREDICTION_FEATURE_COLUMNS,
    _q,
    create_spark_session,
    impute_missing_feature_values,
    validate_prediction_frame,
)

SAFE_FEATURE_COLUMNS = [f"feature_{index:03d}" for index, _ in enumerate(PREDICTION_FEATURE_COLUMNS)]


@dataclass
class SplitDecision:
    train_rows: int
    test_rows: int
    cutoff_index: int
    cutoff_date: str
    reason: str


@dataclass
class BacktestResult:
    split: SplitDecision
    metrics: Dict[str, float]
    confusion_matrix: List[Dict[str, int]]
    imputation: Dict[str, object]
    output_csv: str


def choose_early_season_split(pdf: pd.DataFrame) -> SplitDecision:
    """Choose a temporal split before the table is fully settled.

    The dataset has one row per Barcelona match, so this picks the first cut
    between 35% and 45% of the season that has every observed target class in
    training. That keeps the split early while avoiding a classifier that never
    sees losses/draws/wins.
    """
    sorted_pdf = pdf.sort_values("Fecha").reset_index(drop=True)
    n_rows = len(sorted_pdf)
    observed_classes = set(sorted_pdf["Resultado"].dropna().astype(int))
    min_index = max(8, int(n_rows * 0.35))
    max_index = max(min_index + 1, int(n_rows * 0.45))

    selected_index: Optional[int] = None
    for cutoff_index in range(min_index, max_index + 1):
        train_classes = set(sorted_pdf.iloc[:cutoff_index]["Resultado"].dropna().astype(int))
        if observed_classes.issubset(train_classes):
            selected_index = cutoff_index
            break

    if selected_index is None:
        selected_index = min(max_index, n_rows - 1)

    cutoff_date = str(sorted_pdf.iloc[selected_index - 1]["Fecha"].date())
    return SplitDecision(
        train_rows=selected_index,
        test_rows=n_rows - selected_index,
        cutoff_index=selected_index,
        cutoff_date=cutoff_date,
        reason=(
            "Temporal split around the first 40% of the season, after all target "
            "classes are represented but before the final table is clear."
        ),
    )


def _prepare_frame(spark: SparkSession, csv_path: Path) -> tuple[DataFrame, Dict[str, object]]:
    pdf = pd.read_csv(csv_path)
    pdf["Fecha"] = pd.to_datetime(pdf["Fecha"], errors="coerce")

    frame = spark.createDataFrame(pdf)
    imputed_frame, imputation = impute_missing_feature_values(frame)
    quality = validate_prediction_frame(imputed_frame)
    if not quality.is_valid:
        raise ValueError(f"Feature dataset is still invalid after imputation: {quality.issues}")

    return imputed_frame, imputation.to_dict()


def _with_safe_feature_columns(frame: DataFrame) -> DataFrame:
    safe_columns = [
        _q(original).cast("double").alias(safe)
        for original, safe in zip(PREDICTION_FEATURE_COLUMNS, SAFE_FEATURE_COLUMNS)
    ]
    return frame.select("*", *safe_columns)


def _build_pipeline() -> Pipeline:
    assembler = VectorAssembler(inputCols=SAFE_FEATURE_COLUMNS, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    label_indexer = StringIndexer(inputCol="Resultado", outputCol="label", handleInvalid="error")
    classifier = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        predictionCol="prediction",
        probabilityCol="probability",
        seed=42,
        numTrees=120,
        maxDepth=4,
        minInstancesPerNode=2,
    )
    return Pipeline(stages=[label_indexer, assembler, scaler, classifier])


def _fallback_pipeline() -> Pipeline:
    assembler = VectorAssembler(inputCols=SAFE_FEATURE_COLUMNS, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    label_indexer = StringIndexer(inputCol="Resultado", outputCol="label", handleInvalid="error")
    classifier = LogisticRegression(
        labelCol="label",
        featuresCol="features",
        predictionCol="prediction",
        probabilityCol="probability",
        maxIter=100,
        regParam=0.25,
        elasticNetParam=0.0,
    )
    return Pipeline(stages=[label_indexer, assembler, scaler, classifier])


def run_temporal_backtest(
    csv_path: Path,
    output_dir: Path,
    spark: Optional[SparkSession] = None,
) -> BacktestResult:
    owns_spark = spark is None
    spark = spark or create_spark_session("laliga-temporal-backtest", master="local[1]")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        feature_frame, imputation = _prepare_frame(spark, csv_path)
        sorted_pdf = feature_frame.orderBy("Fecha").toPandas()
        split = choose_early_season_split(sorted_pdf)

        sorted_frame = _with_safe_feature_columns(spark.createDataFrame(sorted_pdf))
        with_index = sorted_frame.withColumn("row_number", F.row_number().over(Window.orderBy("Fecha")) - 1)
        train = with_index.filter(F.col("row_number") < split.cutoff_index).drop("row_number")
        test = with_index.filter(F.col("row_number") >= split.cutoff_index).drop("row_number")

        try:
            model = _build_pipeline().fit(train)
            model_family = "spark_random_forest"
        except Exception:
            model = _fallback_pipeline().fit(train)
            model_family = "spark_logistic_regression"

        predictions = model.transform(test)
        label_model = model.stages[0]
        labels = [int(float(label)) for label in label_model.labels]
        label_map = {float(index): label for index, label in enumerate(labels)}
        map_expr = F.create_map(*[item for pair in label_map.items() for item in (F.lit(pair[0]), F.lit(pair[1]))])
        predictions = predictions.withColumn("predicted_resultado", map_expr[F.col("prediction")])

        evaluator_accuracy = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy",
        )
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1",
        )
        metrics = {
            "accuracy": float(evaluator_accuracy.evaluate(predictions)),
            "f1_weighted": float(evaluator_f1.evaluate(predictions)),
            "test_rows": float(split.test_rows),
            "train_rows": float(split.train_rows),
            "model_family": model_family,
        }

        selected_columns = [
            "Fecha",
            "Anfitrion",
            "Adversario",
            "Sedes",
            "Resultado",
            "predicted_resultado",
            "GF",
            "GC",
        ]
        output_pdf = predictions.select(*selected_columns).orderBy("Fecha").toPandas()
        output_pdf["correct"] = output_pdf["Resultado"] == output_pdf["predicted_resultado"]
        output_pdf["Resultado_label"] = output_pdf["Resultado"].map({1: "Derrota", 2: "Empate", 3: "Victoria"})
        output_pdf["Prediccion_label"] = output_pdf["predicted_resultado"].map({1: "Derrota", 2: "Empate", 3: "Victoria"})

        output_csv = output_dir / "temporal_backtest_predictions.csv"
        output_pdf.to_csv(output_csv, index=False, encoding="utf-8")

        confusion_pdf = (
            output_pdf.groupby(["Resultado_label", "Prediccion_label"])
            .size()
            .reset_index(name="count")
        )

        result = BacktestResult(
            split=split,
            metrics=metrics,
            confusion_matrix=confusion_pdf.to_dict(orient="records"),
            imputation=imputation,
            output_csv=str(output_csv),
        )

        return result
    finally:
        if owns_spark:
            spark.stop()


def main() -> None:
    import json

    repo_root = Path(__file__).resolve().parents[3]
    result = run_temporal_backtest(
        csv_path=repo_root / "data" / "laliga.csv",
        output_dir=repo_root / "artifacts" / "backtests",
    )
    output_json = repo_root / "artifacts" / "backtests" / "temporal_backtest_metrics.json"
    output_json.write_text(
        json.dumps(
            {
                "split": asdict(result.split),
                "metrics": result.metrics,
                "confusion_matrix": result.confusion_matrix,
                "imputation": result.imputation,
                "output_csv": result.output_csv,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(output_json)
    print(json.dumps(asdict(result.split), indent=2, ensure_ascii=False))
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
