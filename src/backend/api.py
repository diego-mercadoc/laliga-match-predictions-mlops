from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from pyspark.sql import SparkSession

from services.laliga_spark_pipeline import (
    DataQualityError,
    FeatureImputationReport,
    PREDICTION_FEATURE_COLUMNS,
    SparkDataQualityReport,
    create_spark_session,
    impute_missing_feature_values,
    prepare_prediction_frame_from_fbref,
    validate_prediction_frame,
)
from services.model_service import MlflowModelService, ModelUnavailableError
from services.refresh_pipeline import read_refresh_status, start_refresh_subprocess


app = FastAPI(
    title="LaLiga Predictions API",
    description="PySpark-backed API for La Liga prediction features, validation and model serving.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_spark: Optional[SparkSession] = None
_model_service = MlflowModelService()


class PredictRequest(BaseModel):
    jornada: int = Field(..., ge=1, le=38, description="La Liga matchday number")
    impute_missing: bool = Field(
        True,
        description="Fill numeric NULL/NaN feature values with medians before prediction.",
    )


class QualityResponse(BaseModel):
    rows: int
    columns: int
    is_valid: bool
    issue_count: int
    issues: List[str]
    missing_required_columns: List[str]
    null_counts: Dict[str, int]
    duplicate_match_rows: int
    self_match_rows: int


class ImputationResponse(BaseModel):
    strategy: str
    filled_counts: Dict[str, int]
    fill_values: Dict[str, float]
    total_filled: int


class PredictionRow(BaseModel):
    Anfitrion: str
    Adversario: str
    Fecha: Optional[str] = None
    Sedes: int
    Probabilidad_Victoria: float
    Probabilidad_Empate: float
    Probabilidad_Derrota: float
    Goles_Predichos_Local: float
    Goles_Predichos_Local_CI_Lower: float
    Goles_Predichos_Local_CI_Upper: float
    Goles_Predichos_Visitante: float
    Goles_Predichos_Visitante_CI_Lower: float
    Goles_Predichos_Visitante_CI_Upper: float
    Corners_Predichos_Local: float
    Corners_Predichos_Local_CI_Lower: float
    Corners_Predichos_Local_CI_Upper: float
    Corners_Predichos_Visitante: float
    Corners_Predichos_Visitante_CI_Lower: float
    Corners_Predichos_Visitante_CI_Upper: float
    Amarillas_Predichas_Local: float
    Amarillas_Predichas_Local_CI_Lower: float
    Amarillas_Predichas_Local_CI_Upper: float
    Amarillas_Predichas_Visitante: float
    Amarillas_Predichas_Visitante_CI_Lower: float
    Amarillas_Predichas_Visitante_CI_Upper: float


class PredictionsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    jornada: int
    predictions: List[PredictionRow]
    data_quality: QualityResponse
    imputation: Optional[ImputationResponse] = None
    model_version: str
    feature_count: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class RefreshRequest(BaseModel):
    force_download: bool = Field(False, description="Re-download raw football-data CSVs even when cached.")
    skip_experiments: bool = Field(False, description="Only refresh raw CSVs; do not rebuild features/models.")


def _get_spark() -> SparkSession:
    global _spark
    if _spark is None:
        _spark = create_spark_session()
    return _spark


async def verify_api_key(x_api_key: Annotated[Optional[str], Header()] = None) -> None:
    import os

    configured_key = os.getenv("API_KEY")
    if configured_key and x_api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )


def _quality_response(report: SparkDataQualityReport) -> QualityResponse:
    return QualityResponse(**report.to_dict())


def _imputation_response(report: Optional[FeatureImputationReport]) -> Optional[ImputationResponse]:
    if report is None:
        return None
    return ImputationResponse(**report.to_dict())


def _confidence_interval(value: float, error_margin: float) -> tuple[float, float]:
    lower = max(0.0, value * (1.0 - error_margin))
    upper = value * (1.0 + error_margin)
    return round(lower, 2), round(upper, 2)


def _safe_float(value, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _probabilities(prediction) -> tuple[float, float, float]:
    values = np.asarray(prediction).astype(float).ravel()
    if values.size >= 3:
        probs = np.maximum(values[:3], 0.0)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return round(float(probs[2]), 4), round(float(probs[1]), 4), round(float(probs[0]), 4)

    if values.size == 1:
        label = int(round(values[0]))
        return (
            1.0 if label == 2 else 0.0,
            1.0 if label == 1 else 0.0,
            1.0 if label == 0 else 0.0,
        )

    return 0.34, 0.32, 0.34


def _prediction_rows(source_rows: pd.DataFrame, model_predictions) -> List[PredictionRow]:
    predictions_array = np.asarray(model_predictions)
    if predictions_array.ndim == 1:
        prediction_iterable = predictions_array.reshape(-1, 1)
    else:
        prediction_iterable = predictions_array

    rows: List[PredictionRow] = []
    for (_, row), prediction in zip(source_rows.iterrows(), prediction_iterable):
        home_prob, draw_prob, away_prob = _probabilities(prediction)

        home_goals = round(max(0.0, _safe_float(row.get("xG(tm)"), 1.1)), 2)
        away_goals = round(max(0.0, _safe_float(row.get("xG(opp)"), 1.0)), 2)
        home_corners = round(max(0.0, _safe_float(row.get("PrgC(tm)")) / 8.0 + _safe_float(row.get("% de TT(tm)")) / 20.0), 2)
        away_corners = round(max(0.0, _safe_float(row.get("PrgC(opp)")) / 8.0 + _safe_float(row.get("% de TT(opp)")) / 20.0), 2)
        home_cards = round(max(0.0, _safe_float(row.get("TklG(tm)")) / 12.0 + _safe_float(row.get("Err(tm)")) * 0.15), 2)
        away_cards = round(max(0.0, _safe_float(row.get("TklG(opp)")) / 12.0 + _safe_float(row.get("Err(opp)")) * 0.15), 2)

        goals_home_ci = _confidence_interval(home_goals, 0.25)
        goals_away_ci = _confidence_interval(away_goals, 0.25)
        corners_home_ci = _confidence_interval(home_corners, 0.30)
        corners_away_ci = _confidence_interval(away_corners, 0.30)
        cards_home_ci = _confidence_interval(home_cards, 0.35)
        cards_away_ci = _confidence_interval(away_cards, 0.35)

        fecha = row.get("Fecha")
        fecha_text = None if pd.isna(fecha) else str(fecha)

        rows.append(
            PredictionRow(
                Anfitrion=str(row["Anfitrion"]),
                Adversario=str(row["Adversario"]),
                Fecha=fecha_text,
                Sedes=int(row["Sedes"]),
                Probabilidad_Victoria=home_prob,
                Probabilidad_Empate=draw_prob,
                Probabilidad_Derrota=away_prob,
                Goles_Predichos_Local=home_goals,
                Goles_Predichos_Local_CI_Lower=goals_home_ci[0],
                Goles_Predichos_Local_CI_Upper=goals_home_ci[1],
                Goles_Predichos_Visitante=away_goals,
                Goles_Predichos_Visitante_CI_Lower=goals_away_ci[0],
                Goles_Predichos_Visitante_CI_Upper=goals_away_ci[1],
                Corners_Predichos_Local=home_corners,
                Corners_Predichos_Local_CI_Lower=corners_home_ci[0],
                Corners_Predichos_Local_CI_Upper=corners_home_ci[1],
                Corners_Predichos_Visitante=away_corners,
                Corners_Predichos_Visitante_CI_Lower=corners_away_ci[0],
                Corners_Predichos_Visitante_CI_Upper=corners_away_ci[1],
                Amarillas_Predichas_Local=home_cards,
                Amarillas_Predichas_Local_CI_Lower=cards_home_ci[0],
                Amarillas_Predichas_Local_CI_Upper=cards_home_ci[1],
                Amarillas_Predichas_Visitante=away_cards,
                Amarillas_Predichas_Visitante_CI_Lower=cards_away_ci[0],
                Amarillas_Predichas_Visitante_CI_Upper=cards_away_ci[1],
            )
        )
    return rows


@app.get("/health")
async def health_check() -> Dict[str, object]:
    metadata = _model_service.metadata
    return {
        "status": "healthy",
        "pipeline": "pyspark",
        "feature_count": len(PREDICTION_FEATURE_COLUMNS),
        "model": metadata.__dict__,
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home_page() -> str:
    metadata = _model_service.metadata
    model_state = "Cargado" if metadata.loaded else "Listo para cargar"
    return f"""
    <!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>LaLiga Predictions API</title>
      <style>
        :root {{
          color-scheme: light;
          --ink: #17202a;
          --muted: #5f6b7a;
          --line: #d8dee7;
          --panel: #ffffff;
          --bg: #f5f7fa;
          --accent: #0f766e;
          --accent-soft: #d9f4ef;
          --warn: #9a3412;
          --warn-soft: #ffedd5;
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: var(--ink);
          background: var(--bg);
        }}
        header {{
          border-bottom: 1px solid var(--line);
          background: #fff;
        }}
        .wrap {{
          max-width: 1120px;
          margin: 0 auto;
          padding: 28px 24px;
        }}
        h1 {{
          margin: 0;
          font-size: clamp(28px, 4vw, 46px);
          line-height: 1.05;
          letter-spacing: 0;
        }}
        .subhead {{
          margin: 12px 0 0;
          max-width: 780px;
          color: var(--muted);
          font-size: 17px;
          line-height: 1.55;
        }}
        main .wrap {{
          display: grid;
          grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr);
          gap: 20px;
          align-items: start;
        }}
        section, aside {{
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 20px;
        }}
        h2 {{
          margin: 0 0 14px;
          font-size: 18px;
          letter-spacing: 0;
        }}
        .grid {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
        }}
        a.action {{
          display: block;
          min-height: 94px;
          padding: 16px;
          border: 1px solid var(--line);
          border-radius: 8px;
          color: inherit;
          text-decoration: none;
          background: #fff;
        }}
        a.action:hover {{
          border-color: var(--accent);
          background: #fbfffe;
        }}
        .label {{
          display: block;
          font-weight: 700;
          margin-bottom: 6px;
        }}
        .hint {{
          color: var(--muted);
          font-size: 14px;
          line-height: 1.4;
        }}
        dl {{
          display: grid;
          grid-template-columns: auto 1fr;
          gap: 12px 16px;
          margin: 0;
        }}
        dt {{
          color: var(--muted);
        }}
        dd {{
          margin: 0;
          min-width: 0;
          overflow-wrap: anywhere;
          font-weight: 600;
        }}
        .pill {{
          display: inline-flex;
          align-items: center;
          min-height: 30px;
          padding: 4px 10px;
          border-radius: 999px;
          background: var(--accent-soft);
          color: var(--accent);
          font-weight: 700;
        }}
        .note {{
          margin-top: 16px;
          padding: 12px;
          border-radius: 8px;
          background: var(--warn-soft);
          color: var(--warn);
          font-size: 14px;
          line-height: 1.45;
        }}
        code {{
          padding: 2px 5px;
          border-radius: 5px;
          background: #edf1f6;
          font-size: 0.94em;
        }}
        @media (max-width: 820px) {{
          main .wrap, .grid {{
            grid-template-columns: 1fr;
          }}
          .wrap {{
            padding: 22px 16px;
          }}
        }}
      </style>
    </head>
    <body>
      <header>
        <div class="wrap">
          <h1>LaLiga Predictions API</h1>
          <p class="subhead">Backend en FastAPI con pipeline PySpark para preparar features, validar calidad de datos y servir predicciones desde MLflow.</p>
        </div>
      </header>
      <main>
        <div class="wrap">
          <section>
            <h2>Acciones rápidas</h2>
            <div class="grid">
              <a class="action" href="/docs">
                <span class="label">Swagger UI</span>
                <span class="hint">Probar endpoints desde una interfaz interactiva.</span>
              </a>
              <a class="action" href="/data/quality?jornada=12">
                <span class="label">Calidad de datos</span>
                <span class="hint">Validar features de la jornada 12.</span>
              </a>
              <a class="action" href="/features/preview?jornada=12&limit=5">
                <span class="label">Preview de features</span>
                <span class="hint">Ver filas listas para el modelo.</span>
              </a>
              <a class="action" href="/backtest/latest">
                <span class="label">Backtest temporal</span>
                <span class="hint">Ver partición, métricas y predicciones generadas.</span>
              </a>
              <a class="action" href="/experiments/latest">
                <span class="label">Experimentos multi-temporada</span>
                <span class="hint">Comparar modelos, ventanas y filtrar predicciones por equipo.</span>
              </a>
              <a class="action" href="/refresh/status">
                <span class="label">Estado de refresh</span>
                <span class="hint">Ver cuándo se actualizaron los datos y modelos.</span>
              </a>
              <a class="action" href="/health">
                <span class="label">Health JSON</span>
                <span class="hint">Respuesta técnica para monitoreo.</span>
              </a>
            </div>
          </section>
          <aside>
            <h2>Estado</h2>
            <dl>
              <dt>API</dt>
              <dd><span class="pill">Healthy</span></dd>
              <dt>Pipeline</dt>
              <dd>PySpark</dd>
              <dt>Features</dt>
              <dd>48</dd>
              <dt>Modelo</dt>
              <dd>{model_state}</dd>
              <dt>URI</dt>
              <dd><code>{metadata.model_uri}</code></dd>
            </dl>
            <div class="note">
              <strong>Nota:</strong> <code>/health</code> está pensado para máquinas. Para ver y probar la API, usa <code>/docs</code> o esta pantalla.
            </div>
          </aside>
        </div>
      </main>
    </body>
    </html>
    """


@app.get("/refresh/status")
async def refresh_status() -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    return read_refresh_status(repo_root)


@app.post("/refresh/run", dependencies=[Depends(verify_api_key)])
async def refresh_run(request: RefreshRequest) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    return start_refresh_subprocess(
        repo_root,
        force_download=request.force_download,
        skip_experiments=request.skip_experiments,
    )


@app.get("/experiments/latest", response_class=HTMLResponse, include_in_schema=False)
async def latest_experiments_page(team: Optional[str] = Query(None)) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "artifacts" / "multi_season"
    summary_path = output_dir / "summary.json"
    experiments_path = output_dir / "experiment_results.csv"
    predictions_path = output_dir / "current_season_predictions.csv"

    if not summary_path.exists() or not experiments_path.exists() or not predictions_path.exists():
        return """
        <!doctype html>
        <html lang="es"><head><meta charset="utf-8"><title>Experimentos no disponibles</title></head>
        <body style="font-family:system-ui;margin:32px">
          <h1>Experimentos no disponibles</h1>
          <p>Ejecuta <code>py -3.11 src\\backend\\services\\multi_season_experiments.py</code> para generar el reporte.</p>
          <p><a href="/">Volver</a></p>
        </body></html>
        """

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    experiments = pd.read_csv(experiments_path).sort_values("log_loss_proxy").head(12)
    predictions = pd.read_csv(predictions_path)
    teams = sorted(set(predictions["HomeTeam"]) | set(predictions["AwayTeam"]))
    selected_team = team if team in teams else None
    filtered_predictions = predictions
    if selected_team:
        filtered_predictions = predictions[
            (predictions["HomeTeam"] == selected_team) | (predictions["AwayTeam"] == selected_team)
        ]

    experiments_table = experiments[
        ["name", "model_family", "accuracy", "f1_weighted", "log_loss_proxy", "train_rows", "test_rows", "recency_half_life"]
    ].to_html(index=False, classes="results", border=0, float_format=lambda value: f"{value:.3f}")
    predictions_table = filtered_predictions[
        ["Date", "HomeTeam", "AwayTeam", "Resultado", "Prediccion", "correct"]
    ].to_html(index=False, classes="results", border=0)

    options = "\n".join(
        f'<option value="{name}" {"selected" if name == selected_team else ""}>{name}</option>'
        for name in teams
    )
    best = summary["best_experiment"]
    best_model_name = best["model_family"].replace("_", " ")
    correct = int(filtered_predictions["correct"].sum())
    total = len(filtered_predictions)
    title_suffix = f" para {selected_team}" if selected_team else ""

    return f"""
    <!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Experimentos multi-temporada</title>
      <style>
        body {{
          margin: 0;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: #17202a;
          background: #f5f7fa;
        }}
        .wrap {{ max-width: 1220px; margin: 0 auto; padding: 28px 24px; }}
        h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 44px); letter-spacing: 0; }}
        p {{ color: #5f6b7a; line-height: 1.5; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 22px 0; }}
        .metric, .panel {{ background: #fff; border: 1px solid #d8dee7; border-radius: 8px; padding: 16px; }}
        .metric span {{ display: block; color: #5f6b7a; font-size: 13px; }}
        .metric strong {{ display: block; margin-top: 6px; font-size: 25px; overflow-wrap: anywhere; }}
        table.results {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d8dee7; border-radius: 8px; overflow: hidden; margin-bottom: 24px; }}
        .results th, .results td {{ border-bottom: 1px solid #e7ebf0; padding: 10px 12px; text-align: left; font-size: 14px; }}
        .results th {{ background: #edf1f6; }}
        select, button {{ min-height: 38px; border: 1px solid #bdc7d4; border-radius: 7px; padding: 6px 10px; font: inherit; }}
        button {{ background: #0f766e; color: #fff; font-weight: 700; cursor: pointer; }}
        code {{ background: #edf1f6; border-radius: 5px; padding: 2px 5px; }}
        a {{ color: #0f766e; font-weight: 700; }}
        @media (max-width: 820px) {{
          .wrap {{ padding: 22px 16px; }}
          .metrics {{ grid-template-columns: 1fr 1fr; }}
          table.results {{ display: block; overflow-x: auto; }}
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <p><a href="/">Volver</a></p>
        <h1>Experimentos multi-temporada</h1>
        <p>Datos de Football-Data para La Liga 2018/19-2025/26. Corte temporal en <strong>{summary["split_date"]}</strong>; el tramo posterior de 2025/26 se trata como predicción fuera de muestra.</p>
        <div class="metrics">
          <div class="metric"><span>Partidos</span><strong>{summary["data_rows"]}</strong></div>
          <div class="metric"><span>Mejor modelo</span><strong>{best_model_name}</strong></div>
          <div class="metric"><span>Mejor ventana</span><strong>{", ".join(best["seasons"])}</strong></div>
          <div class="metric"><span>Accuracy / F1</span><strong>{best["accuracy"]:.1%} / {best["f1_weighted"]:.2f}</strong></div>
        </div>
        <div class="panel">
          <form method="get" action="/experiments/latest">
            <label for="team"><strong>Filtrar equipo:</strong></label>
            <select id="team" name="team">
              <option value="">Todos</option>
              {options}
            </select>
            <button type="submit">Aplicar</button>
          </form>
          <p>Predicciones{title_suffix}: {correct} correctas de {total}. El mejor modelo se eligió por menor log-loss proxy, no sólo por accuracy.</p>
        </div>
        <h2>Top experimentos</h2>
        {experiments_table}
        <h2>Predicciones 2025/26{title_suffix}</h2>
        {predictions_table}
      </div>
    </body>
    </html>
    """


@app.get("/backtest/latest", response_class=HTMLResponse, include_in_schema=False)
async def latest_backtest_page() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    metrics_path = repo_root / "artifacts" / "backtests" / "temporal_backtest_metrics.json"
    predictions_path = repo_root / "artifacts" / "backtests" / "temporal_backtest_predictions.csv"

    if not metrics_path.exists() or not predictions_path.exists():
        return """
        <!doctype html>
        <html lang="es"><head><meta charset="utf-8"><title>Backtest no disponible</title></head>
        <body style="font-family:system-ui;margin:32px">
          <h1>Backtest no disponible</h1>
          <p>Ejecuta <code>py -3.11 src\\backend\\services\\temporal_backtest.py</code> para generar el reporte.</p>
          <p><a href="/">Volver</a></p>
        </body></html>
        """

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    table_html = predictions[
        ["Fecha", "Adversario", "Sedes", "Resultado_label", "Prediccion_label", "correct"]
    ].to_html(index=False, classes="results", border=0)

    accuracy = metrics["metrics"]["accuracy"]
    f1_weighted = metrics["metrics"]["f1_weighted"]
    split = metrics["split"]
    imputation = metrics["imputation"]
    correct = int(predictions["correct"].sum())
    total = len(predictions)

    return f"""
    <!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Backtest temporal</title>
      <style>
        body {{
          margin: 0;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: #17202a;
          background: #f5f7fa;
        }}
        .wrap {{
          max-width: 1180px;
          margin: 0 auto;
          padding: 28px 24px;
        }}
        h1 {{
          margin: 0 0 8px;
          font-size: clamp(28px, 4vw, 44px);
          letter-spacing: 0;
        }}
        p {{
          color: #5f6b7a;
          line-height: 1.5;
        }}
        .metrics {{
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 12px;
          margin: 22px 0;
        }}
        .metric, .panel {{
          background: #fff;
          border: 1px solid #d8dee7;
          border-radius: 8px;
          padding: 16px;
        }}
        .metric span {{
          display: block;
          color: #5f6b7a;
          font-size: 13px;
        }}
        .metric strong {{
          display: block;
          margin-top: 6px;
          font-size: 26px;
        }}
        table.results {{
          width: 100%;
          border-collapse: collapse;
          background: #fff;
          border: 1px solid #d8dee7;
          border-radius: 8px;
          overflow: hidden;
        }}
        .results th, .results td {{
          border-bottom: 1px solid #e7ebf0;
          padding: 10px 12px;
          text-align: left;
          font-size: 14px;
        }}
        .results th {{
          background: #edf1f6;
        }}
        code {{
          background: #edf1f6;
          border-radius: 5px;
          padding: 2px 5px;
        }}
        a {{
          color: #0f766e;
          font-weight: 700;
        }}
        @media (max-width: 820px) {{
          .wrap {{ padding: 22px 16px; }}
          .metrics {{ grid-template-columns: 1fr 1fr; }}
          table.results {{ display: block; overflow-x: auto; }}
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <p><a href="/">Volver</a></p>
        <h1>Backtest temporal</h1>
        <p>{split["reason"]}</p>
        <div class="metrics">
          <div class="metric"><span>Corte</span><strong>{split["cutoff_date"]}</strong></div>
          <div class="metric"><span>Train / Test</span><strong>{split["train_rows"]} / {split["test_rows"]}</strong></div>
          <div class="metric"><span>Accuracy</span><strong>{accuracy:.0%}</strong></div>
          <div class="metric"><span>F1 weighted</span><strong>{f1_weighted:.2f}</strong></div>
        </div>
        <div class="panel">
          <p><strong>Resultado:</strong> {correct} de {total} partidos predichos correctamente. Modelo: <code>{metrics["metrics"]["model_family"]}</code>.</p>
          <p><strong>Imputación:</strong> {imputation["total_filled"]} valores rellenados por mediana antes de entrenar.</p>
        </div>
        <h2>Predicciones</h2>
        {table_html}
      </div>
    </body>
    </html>
    """


@app.get("/data/quality", response_model=QualityResponse, dependencies=[Depends(verify_api_key)])
async def data_quality(jornada: int = Query(..., ge=1, le=38)) -> QualityResponse:
    try:
        prepared = prepare_prediction_frame_from_fbref(_get_spark(), jornada)
        return _quality_response(prepared.quality)
    except DataQualityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/features/preview", dependencies=[Depends(verify_api_key)])
async def feature_preview(
    jornada: int = Query(..., ge=1, le=38),
    limit: int = Query(5, ge=1, le=20),
) -> Dict[str, object]:
    try:
        prepared = prepare_prediction_frame_from_fbref(_get_spark(), jornada)
        rows = prepared.frame.limit(limit).toPandas().to_dict(orient="records")
        return {
            "jornada": jornada,
            "columns": prepared.frame.columns,
            "feature_columns": prepared.feature_columns,
            "rows": rows,
            "data_quality": prepared.quality.to_dict(),
        }
    except DataQualityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionsResponse, dependencies=[Depends(verify_api_key)])
async def predict(request: PredictRequest) -> PredictionsResponse:
    try:
        prepared = prepare_prediction_frame_from_fbref(_get_spark(), request.jornada)
    except DataQualityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    quality = prepared.quality
    imputation_report: Optional[FeatureImputationReport] = None
    prediction_frame = prepared.frame

    if not quality.is_valid and request.impute_missing:
        prediction_frame, imputation_report = impute_missing_feature_values(prediction_frame)
        quality = validate_prediction_frame(prediction_frame)

    if not quality.is_valid:
        raise HTTPException(status_code=422, detail=prepared.quality.to_dict())

    source_rows = prediction_frame.toPandas()
    model_features = source_rows[PREDICTION_FEATURE_COLUMNS]

    try:
        raw_predictions = _model_service.predict(model_features)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictionsResponse(
        jornada=request.jornada,
        predictions=_prediction_rows(source_rows, raw_predictions),
        data_quality=_quality_response(quality),
        imputation=_imputation_response(imputation_report),
        model_version=_model_service.metadata.model_uri,
        feature_count=len(PREDICTION_FEATURE_COLUMNS),
    )
