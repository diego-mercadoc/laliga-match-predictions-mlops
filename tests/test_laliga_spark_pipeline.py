import pandas as pd
import pytest

from services.laliga_spark_pipeline import (
    PREDICTION_FEATURE_COLUMNS,
    build_prediction_frame,
    create_spark_session,
    impute_missing_feature_values,
    validate_prediction_frame,
    validate_feature_ready_pandas,
)


@pytest.fixture(scope="session")
def spark():
    session = create_spark_session("laliga-pipeline-tests", master="local[1]")
    yield session
    session.stop()


def _stats_tables(team_names):
    basic = pd.DataFrame(
        {
            "RL": [1, 2, 3, 4],
            "Equipo": team_names,
            "PG": [10, 8, 7, 6],
            "PE": [2, 3, 4, 5],
            "PP": [1, 2, 3, 4],
            "GF": [30, 25, 22, 18],
            "GC": [10, 12, 15, 19],
            "xG": [28.5, 24.2, 20.0, 17.8],
            "xGA": [9.8, 11.1, 16.2, 18.0],
            "Últimos 5": ["PG PG PE PP PG", "PE PG PP PG PE", "PP PG PE PE PG", "PP PP PG PE PG"],
            "Máximo Goleador del Equipo": ["Jugador A 12", "Jugador B 9", "Jugador C 8", "Jugador D 7"],
        }
    )
    attack = pd.DataFrame(
        {
            "Equipo": team_names,
            "Edad": [27.1, 26.2, 25.8, 28.0],
            "Pos.": [55.0, 51.0, 49.0, 47.0],
            "Ass": [20, 18, 15, 12],
            "TPint": [3, 2, 2, 1],
            "PrgC": [180, 160, 140, 130],
            "PrgP": [420, 390, 350, 300],
        }
    )
    shots = pd.DataFrame({"Equipo": team_names, "% de TT": [35.0, 33.0, 31.0, 30.0], "Dist": [16.2, 17.1, 18.0, 18.4]})
    passes = pd.DataFrame({"Equipo": team_names, "% Cmp": [88.0, 85.0, 82.0, 80.0], "Dist. tot.": [9500, 9100, 8700, 8300]})
    defense = pd.DataFrame({"Equipo": team_names, "TklG": [120, 130, 125, 140], "Int": [90, 86, 84, 82], "Err": [3, 4, 5, 6]})

    tables = [pd.DataFrame()] * 17
    tables[0] = basic
    tables[2] = attack
    tables[8] = shots
    tables[10] = passes
    tables[16] = defense
    return tables


def test_build_prediction_frame_returns_team_view_rows(spark):
    schedule = pd.DataFrame(
        {
            "Sem.": [1, 1],
            "Día": ["Sáb", "Dom"],
            "Fecha": ["2024-08-17", "2024-08-18"],
            "Local": ["Team A", "Team C"],
            "Visitante": ["Team B", "Team D"],
        }
    )

    prepared = build_prediction_frame(spark, schedule, _stats_tables(["Team A", "Team B", "Team C", "Team D"]), jornada=1)
    rows = prepared.rows_to_pandas()

    assert prepared.quality.is_valid
    assert prepared.quality.rows == 4
    assert set(PREDICTION_FEATURE_COLUMNS).issubset(rows.columns)
    assert set(rows["Sedes"]) == {0, 1}
    assert rows[PREDICTION_FEATURE_COLUMNS].isna().sum().sum() == 0


def test_quality_report_detects_missing_team_stats(spark):
    schedule = pd.DataFrame(
        {
            "Sem.": [1],
            "Día": ["Sáb"],
            "Fecha": ["2024-08-17"],
            "Local": ["Team A"],
            "Visitante": ["Unknown Team"],
        }
    )

    prepared = build_prediction_frame(spark, schedule, _stats_tables(["Team A", "Team B", "Team C", "Team D"]), jornada=1)

    assert not prepared.quality.is_valid
    assert prepared.quality.issue_count >= 1
    assert any("nulls" in issue for issue in prepared.quality.issues)


def test_recent_form_and_top_scorer_are_parsed_in_spark(spark):
    schedule = pd.DataFrame(
        {
            "Sem.": [1],
            "Día": ["Mar"],
            "Fecha": ["2024-08-17"],
            "Local": ["Team A"],
            "Visitante": ["Team B"],
        }
    )

    prepared = build_prediction_frame(spark, schedule, _stats_tables(["Team A", "Team B", "Team C", "Team D"]), jornada=1)
    rows = prepared.rows_to_pandas()
    team_a = rows[rows["Anfitrion"] == "Team A"].iloc[0]

    assert team_a["Día"] == 2
    assert team_a["Últimos 5(tm)"] == 10
    assert team_a["Máximo Goleador del Equipo(tm)"] == 12


def test_validate_feature_ready_dataset_detects_null_features(spark):
    feature_pdf = pd.DataFrame(
        {
            "Fecha": ["2024-08-17", "2024-08-18"],
            "Anfitrion": ["Team A", "Team B"],
            "Adversario": ["Team B", "Team A"],
            **{col: [1.0, 1.0] for col in PREDICTION_FEATURE_COLUMNS},
        }
    )
    feature_pdf.loc[0, "xG(tm)"] = None

    report = validate_feature_ready_pandas(spark, feature_pdf)

    assert not report.is_valid
    assert report.null_counts["xG(tm)"] == 1

    frame = spark.createDataFrame(feature_pdf)
    imputed, imputation = impute_missing_feature_values(frame)
    imputed_report = validate_prediction_frame(imputed)

    assert imputation.total_filled == 1
    assert imputation.filled_counts["xG(tm)"] == 1
    assert imputed_report.is_valid
