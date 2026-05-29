from __future__ import annotations

from dataclasses import asdict, dataclass
from io import StringIO
import math
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


FBREF_SCHEDULE_URL = "https://fbref.com/es/comps/12/horario/Resultados-y-partidos-en-La-Liga"
FBREF_STATS_URL = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"

BASE_FEATURE_COLUMNS = [
    "Edad",
    "Pos.",
    "Ass",
    "TPint",
    "PrgC",
    "PrgP",
    "% de TT",
    "Dist",
    "% Cmp",
    "Dist. tot.",
    "TklG",
    "Int",
    "Err",
    "RL",
    "PG",
    "PE",
    "PP",
    "GF",
    "GC",
    "xG",
    "xGA",
    "Últimos 5",
    "Máximo Goleador del Equipo",
]

PREDICTION_FEATURE_COLUMNS = [
    "Día",
    "Sedes",
    "Edad(opp)",
    "Pos.(opp)",
    "Ass(opp)",
    "TPint(opp)",
    "PrgC(opp)",
    "PrgP(opp)",
    "% de TT(opp)",
    "Dist(opp)",
    "% Cmp(opp)",
    "Dist. tot.(opp)",
    "TklG(opp)",
    "Int(opp)",
    "Err(opp)",
    "RL(opp)",
    "PG(opp)",
    "PE(opp)",
    "PP(opp)",
    "GF(opp)",
    "GC(opp)",
    "xG(opp)",
    "xGA(opp)",
    "Últimos 5(opp)",
    "Máximo Goleador del Equipo(opp)",
    "Edad(tm)",
    "Pos.(tm)",
    "Ass(tm)",
    "TPint(tm)",
    "PrgC(tm)",
    "PrgP(tm)",
    "% de TT(tm)",
    "Dist(tm)",
    "% Cmp(tm)",
    "Dist. tot.(tm)",
    "TklG(tm)",
    "Int(tm)",
    "Err(tm)",
    "RL(tm)",
    "PG(tm)",
    "PE(tm)",
    "PP(tm)",
    "GF(tm)",
    "GC(tm)",
    "xG(tm)",
    "xGA(tm)",
    "Últimos 5(tm)",
    "Máximo Goleador del Equipo(tm)",
]

NUMERIC_BASE_COLUMNS = [col for col in BASE_FEATURE_COLUMNS if col != "Pos."]


class DataQualityError(ValueError):
    """Raised when source tables cannot produce a valid prediction dataset."""


@dataclass
class SparkDataQualityReport:
    rows: int
    columns: int
    is_valid: bool
    issue_count: int
    issues: List[str]
    missing_required_columns: List[str]
    null_counts: Dict[str, int]
    duplicate_match_rows: int
    self_match_rows: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class FeatureImputationReport:
    strategy: str
    filled_counts: Dict[str, int]
    fill_values: Dict[str, float]

    @property
    def total_filled(self) -> int:
        return sum(self.filled_counts.values())

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["total_filled"] = self.total_filled
        return data


@dataclass
class PreparedPredictionFrame:
    frame: DataFrame
    quality: SparkDataQualityReport

    @property
    def feature_columns(self) -> List[str]:
        return list(PREDICTION_FEATURE_COLUMNS)

    def features_to_pandas(self) -> pd.DataFrame:
        return self.frame.select(*[_q(col) for col in PREDICTION_FEATURE_COLUMNS]).toPandas()

    def rows_to_pandas(self) -> pd.DataFrame:
        return self.frame.toPandas()


def _q(column_name: str):
    return F.col(f"`{column_name}`")


def create_spark_session(
    app_name: str = "laliga-prediction-pipeline",
    master: Optional[str] = None,
) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)

    return (
        builder.config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )


def fetch_html_tables(
    url: str,
    timeout_seconds: int = 30,
    retries: int = 3,
    user_agent: str = "Mozilla/5.0 (compatible; LaLigaPredictions/1.0)",
) -> List[pd.DataFrame]:
    last_error: Optional[Exception] = None
    headers = {"User-Agent": user_agent}

    for _ in range(retries):
        try:
            response = requests.get(url, timeout=timeout_seconds, headers=headers)
            response.raise_for_status()
            return pd.read_html(StringIO(response.text))
        except Exception as exc:  # pragma: no cover - exercised through callers in integration
            last_error = exc

    raise DataQualityError(f"Could not fetch FBref tables from {url}: {last_error}")


def _flatten_pandas_columns(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.copy()
    if isinstance(pdf.columns, pd.MultiIndex):
        flattened = []
        for parts in pdf.columns:
            non_empty = [str(part).strip() for part in parts if str(part).strip() and "Unnamed:" not in str(part)]
            flattened.append(non_empty[-1] if non_empty else str(parts[-1]).strip())
        pdf.columns = flattened
    else:
        pdf.columns = [str(col).strip() for col in pdf.columns]

    seen: Dict[str, int] = {}
    deduped = []
    for col in pdf.columns:
        count = seen.get(col, 0)
        deduped.append(col if count == 0 else f"{col}_{count}")
        seen[col] = count + 1
    pdf.columns = deduped
    return pdf


def _require_columns(pdf: pd.DataFrame, columns: Sequence[str], table_name: str) -> None:
    missing = [col for col in columns if col not in pdf.columns]
    if missing:
        raise DataQualityError(f"{table_name} is missing required columns: {missing}")


def _spark_from_pandas(spark: SparkSession, pdf: pd.DataFrame) -> DataFrame:
    return spark.createDataFrame(pdf.where(pd.notnull(pdf), None))


def _to_double(column_name: str):
    return F.regexp_replace(
        F.regexp_replace(_q(column_name).cast("string"), "%", ""),
        ",",
        ".",
    ).cast("double")


def _missing_condition(column_name: str):
    return _q(column_name).isNull() | F.isnan(_q(column_name).cast("double"))


def _select_required(df: DataFrame, columns: Iterable[str]) -> DataFrame:
    return df.select(*[_q(col).alias(col) for col in columns])


def build_match_frame(spark: SparkSession, schedule_pdf: pd.DataFrame, jornada: int) -> DataFrame:
    if not 1 <= int(jornada) <= 38:
        raise DataQualityError("jornada must be between 1 and 38")

    schedule_pdf = _flatten_pandas_columns(schedule_pdf)
    required = ["Sem.", "Día", "Fecha", "Local", "Visitante"]
    _require_columns(schedule_pdf, required, "schedule")

    days = F.create_map(
        F.lit("Lun"),
        F.lit(1),
        F.lit("Mar"),
        F.lit(2),
        F.lit("Mié"),
        F.lit(3),
        F.lit("Mie"),
        F.lit(3),
        F.lit("Jue"),
        F.lit(4),
        F.lit("Vie"),
        F.lit(5),
        F.lit("Sáb"),
        F.lit(6),
        F.lit("Sab"),
        F.lit(6),
        F.lit("Dom"),
        F.lit(7),
    )

    schedule = (
        _spark_from_pandas(spark, schedule_pdf)
        .select(*[_q(col).alias(col) for col in required])
        .withColumn("Sem.", _q("Sem.").cast("int"))
        .withColumn("Fecha", F.to_date(_q("Fecha")))
        .withColumn("Día", days[_q("Día")].cast("int"))
        .filter(_q("Sem.") == int(jornada))
        .filter(_q("Local").isNotNull() & _q("Visitante").isNotNull())
    )

    home = schedule.select(
        _q("Fecha"),
        _q("Día"),
        F.lit(1).alias("Sedes"),
        _q("Local").alias("Anfitrion"),
        _q("Visitante").alias("Adversario"),
    )
    away = schedule.select(
        _q("Fecha"),
        _q("Día"),
        F.lit(0).alias("Sedes"),
        _q("Visitante").alias("Anfitrion"),
        _q("Local").alias("Adversario"),
    )

    return home.unionByName(away)


def _extract_basic_stats(spark: SparkSession, stats_pdf: pd.DataFrame) -> DataFrame:
    stats_pdf = _flatten_pandas_columns(stats_pdf)
    columns = [
        "RL",
        "Equipo",
        "PG",
        "PE",
        "PP",
        "GF",
        "GC",
        "xG",
        "xGA",
        "Últimos 5",
        "Máximo Goleador del Equipo",
    ]
    _require_columns(stats_pdf, columns, "basic stats")
    basic = _select_required(_spark_from_pandas(spark, stats_pdf), columns)

    recent_form_expr = """
        aggregate(
            split(trim(coalesce(`Últimos 5`, '')), ' +'),
            0,
            (acc, result) -> acc + CASE
                WHEN result = 'PG' THEN 3
                WHEN result = 'PE' THEN 1
                ELSE 0
            END
        )
    """

    numeric_columns = ["RL", "PG", "PE", "PP", "GF", "GC", "xG", "xGA"]
    for col in numeric_columns:
        basic = basic.withColumn(col, _to_double(col))

    return (
        basic.withColumn("Últimos 5", F.expr(recent_form_expr).cast("double"))
        .withColumn(
            "Máximo Goleador del Equipo",
            F.regexp_extract(_q("Máximo Goleador del Equipo").cast("string"), r"\b(\d+)\b", 1).cast("double"),
        )
        .withColumn("Equipo", F.trim(_q("Equipo")))
    )


def _extract_table(
    spark: SparkSession,
    stats_pdf: pd.DataFrame,
    columns: Sequence[str],
    table_name: str,
) -> DataFrame:
    stats_pdf = _flatten_pandas_columns(stats_pdf)
    _require_columns(stats_pdf, columns, table_name)
    frame = _select_required(_spark_from_pandas(spark, stats_pdf), columns).withColumn("Equipo", F.trim(_q("Equipo")))
    for col in columns:
        if col != "Equipo":
            frame = frame.withColumn(col, _to_double(col))
    return frame


def build_team_stats_frame(spark: SparkSession, stats_tables: Sequence[pd.DataFrame]) -> DataFrame:
    if len(stats_tables) <= 16:
        raise DataQualityError("FBref stats response did not include all expected tables")

    basic = _extract_basic_stats(spark, stats_tables[0])
    attack = _extract_table(
        spark,
        stats_tables[2],
        ["Equipo", "Edad", "Pos.", "Ass", "TPint", "PrgC", "PrgP"],
        "attack stats",
    )
    shots = _extract_table(spark, stats_tables[8], ["Equipo", "% de TT", "Dist"], "shot stats")
    passes = _extract_table(spark, stats_tables[10], ["Equipo", "% Cmp", "Dist. tot."], "passing stats")
    defense = _extract_table(spark, stats_tables[16], ["Equipo", "TklG", "Int", "Err"], "defense stats")

    return (
        attack.join(shots, on="Equipo", how="left")
        .join(passes, on="Equipo", how="left")
        .join(defense, on="Equipo", how="left")
        .join(basic, on="Equipo", how="left")
    )


def _stats_with_suffix(stats: DataFrame, suffix: str) -> DataFrame:
    columns = [_q("Equipo").alias(f"Equipo_{suffix}")]
    columns.extend(_q(col).alias(f"{col}({suffix})") for col in BASE_FEATURE_COLUMNS)
    return stats.select(*columns)


def build_prediction_frame(
    spark: SparkSession,
    schedule_pdf: pd.DataFrame,
    stats_tables: Sequence[pd.DataFrame],
    jornada: int,
) -> PreparedPredictionFrame:
    matches = build_match_frame(spark, schedule_pdf, jornada)
    stats = build_team_stats_frame(spark, stats_tables)

    opponent_stats = _stats_with_suffix(stats, "opp")
    team_stats = _stats_with_suffix(stats, "tm")

    prediction_frame = (
        matches.join(opponent_stats, _q("Adversario") == F.col("Equipo_opp"), how="left")
        .join(team_stats, _q("Anfitrion") == F.col("Equipo_tm"), how="left")
        .drop("Equipo_opp", "Equipo_tm")
    )

    quality = validate_prediction_frame(prediction_frame)
    return PreparedPredictionFrame(frame=prediction_frame, quality=quality)


def prepare_prediction_frame_from_fbref(
    spark: SparkSession,
    jornada: int,
    schedule_url: str = FBREF_SCHEDULE_URL,
    stats_url: str = FBREF_STATS_URL,
) -> PreparedPredictionFrame:
    schedule_tables = fetch_html_tables(schedule_url)
    stats_tables = fetch_html_tables(stats_url)
    if not schedule_tables:
        raise DataQualityError("FBref schedule response did not include any tables")
    return build_prediction_frame(spark, schedule_tables[0], stats_tables, jornada)


def validate_feature_ready_pandas(
    spark: SparkSession,
    feature_pdf: pd.DataFrame,
    required_columns: Sequence[str] = PREDICTION_FEATURE_COLUMNS,
) -> SparkDataQualityReport:
    """Validate an already materialized feature dataset such as data/laliga.csv."""
    feature_pdf = _flatten_pandas_columns(feature_pdf)
    if "Fecha" in feature_pdf.columns:
        feature_pdf = feature_pdf.copy()
        feature_pdf["Fecha"] = pd.to_datetime(feature_pdf["Fecha"], errors="coerce")

    return validate_prediction_frame(
        _spark_from_pandas(spark, feature_pdf),
        required_columns=required_columns,
    )


def impute_missing_feature_values(
    frame: DataFrame,
    feature_columns: Sequence[str] = PREDICTION_FEATURE_COLUMNS,
    strategy: str = "median",
) -> tuple[DataFrame, FeatureImputationReport]:
    """Fill NULL/NaN numeric feature values with per-column medians."""
    if strategy != "median":
        raise ValueError("Only median imputation is currently supported")

    missing_columns = [col for col in feature_columns if col not in frame.columns]
    if missing_columns:
        raise DataQualityError(f"Cannot impute missing feature columns: {missing_columns}")

    filled_counts: Dict[str, int] = {}
    fill_values: Dict[str, float] = {}
    imputed = frame

    count_exprs = [
        F.sum(F.when(_missing_condition(col), F.lit(1)).otherwise(F.lit(0))).alias(col)
        for col in feature_columns
    ]
    missing_counts = {
        key: int(value)
        for key, value in imputed.select(*count_exprs).first().asDict().items()
    }

    for col, missing_count in missing_counts.items():
        if missing_count == 0:
            continue

        median_row = imputed.select(
            F.expr(f"percentile_approx(`{col}`, 0.5)").alias("median")
        ).first()
        median = median_row["median"] if median_row else None
        fill_value = 0.0 if median is None or math.isnan(float(median)) else float(median)

        imputed = imputed.withColumn(
            col,
            F.when(_missing_condition(col), F.lit(fill_value)).otherwise(_q(col)),
        )
        filled_counts[col] = int(missing_count)
        fill_values[col] = fill_value

    return imputed, FeatureImputationReport(
        strategy=strategy,
        filled_counts=filled_counts,
        fill_values=fill_values,
    )


def validate_prediction_frame(
    frame: DataFrame,
    required_columns: Sequence[str] = PREDICTION_FEATURE_COLUMNS,
) -> SparkDataQualityReport:
    missing_required = [col for col in required_columns if col not in frame.columns]
    issues: List[str] = []
    if missing_required:
        issues.append(f"Missing required feature columns: {missing_required}")

    rows = frame.count()
    if rows == 0:
        issues.append("Prediction frame has no rows")
    if rows % 2 != 0:
        issues.append("Prediction frame should have an even number of team-view rows")

    self_match_rows = 0
    duplicate_match_rows = 0
    null_counts: Dict[str, int] = {}

    if rows and not missing_required:
        null_exprs = [
            F.sum(
                F.when(
                    _missing_condition(col),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ).alias(col)
            for col in required_columns
        ]
        null_counts = {key: int(value) for key, value in frame.select(*null_exprs).first().asDict().items()}
        null_columns = [col for col, count in null_counts.items() if count > 0]
        if null_columns:
            issues.append(f"Feature columns contain nulls: {null_columns}")

    if {"Anfitrion", "Adversario"}.issubset(set(frame.columns)):
        self_match_rows = frame.filter(_q("Anfitrion") == _q("Adversario")).count()
        if self_match_rows:
            issues.append(f"Found {self_match_rows} self-match rows")

        duplicate_match_rows = (
            frame.groupBy("Fecha", "Anfitrion", "Adversario")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
        if duplicate_match_rows:
            issues.append(f"Found {duplicate_match_rows} duplicate match rows")

    return SparkDataQualityReport(
        rows=rows,
        columns=len(frame.columns),
        is_valid=not issues,
        issue_count=len(issues),
        issues=issues,
        missing_required_columns=missing_required,
        null_counts=null_counts,
        duplicate_match_rows=duplicate_match_rows,
        self_match_rows=self_match_rows,
    )
