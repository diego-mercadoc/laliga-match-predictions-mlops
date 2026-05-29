from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


DEFAULT_TRACKING_URI = "https://dagshub.com/JuanPab2009/ProyectoFinalCD.mlflow"
DEFAULT_MODEL_URI = "runs:/e8e41ab35bd34545a81ccb039080a64c/model"


class ModelUnavailableError(RuntimeError):
    """Raised when the prediction model cannot be loaded from MLflow."""


@dataclass
class ModelMetadata:
    model_uri: str
    tracking_uri: Optional[str]
    loaded: bool = False
    error: Optional[str] = None


class MlflowModelService:
    """Lazy MLflow wrapper so the API can start without DagsHub credentials."""

    def __init__(
        self,
        model_uri: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.model_uri = model_uri or os.getenv("MLFLOW_MODEL_URI", DEFAULT_MODEL_URI)
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        self._model: Optional[Any] = None
        self._last_error: Optional[str] = None

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_uri=self.model_uri,
            tracking_uri=self.tracking_uri,
            loaded=self._model is not None,
            error=self._last_error,
        )

    def load(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            import mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            self._model = mlflow.pyfunc.load_model(self.model_uri)
            self._last_error = None
            return self._model
        except Exception as exc:  # pragma: no cover - depends on remote registry/auth
            self._last_error = str(exc)
            raise ModelUnavailableError(
                "MLflow model is unavailable. Configure MLFLOW_TRACKING_URI, "
                "MLFLOW_MODEL_URI and DagsHub/MLflow credentials before calling /predict."
            ) from exc

    def predict(self, features: pd.DataFrame):
        model = self.load()
        return model.predict(features)
