# src/ibov_pipeline/register_model.py
"""Registra o modelo treinado no MLflow Model Registry com metadata de governança.

Fluxo:
    1. Localiza o run mais recente (ou recebe run_id explícito).
    2. Registra no Model Registry com nome padronizado.
    3. Adiciona tags obrigatórias de governança (GAP 05 do Datathon).
    4. Avalia no holdout imutável e loga métricas finais.

Uso:
    python -m src.ibov_pipeline.register_model
    python -m src.ibov_pipeline.register_model --run-id <RUN_ID>

Uso programático:
    from src.ibov_pipeline.register_model import run
    model_version = run(run_id="abc123")
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from src.ibov_pipeline.config import cfg
from src.ibov_pipeline.train_lstm import compute_metrics

logger = logging.getLogger(__name__)


# Schema obrigatório de tags (GAP 05)
REQUIRED_TAGS: dict[str, type] = {
    "model_name": str,
    "model_version": str,
    "model_type": str,
    "framework": str,
    "owner": str,
    "risk_level": str,
    "fairness_checked": str,     # MLflow tags são sempre string
    "data_start_date": str,
    "data_end_date": str,
    "phase": str,
}


def _get_latest_run_id(experiment_name: str) -> str:
    """Retorna o run_id do run mais recente no experimento LSTM.

    Args:
        experiment_name: Nome do experimento MLflow.

    Returns:
        run_id do run mais recente com tag pipeline_stage='lstm'.

    Raises:
        ValueError: Se nenhum run LSTM for encontrado.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.pipeline_stage = 'lstm'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            "Nenhum run LSTM encontrado. Execute train_lstm.py primeiro."
        )

    run_id = runs[0].info.run_id
    logger.info("Run mais recente encontrado: %s", run_id)
    return run_id


def _evaluate_on_holdout(model, scaler) -> dict[str, float]:
    """Avalia o modelo no holdout imutável e retorna métricas.

    Args:
        model: Modelo Keras carregado.
        scaler: MinMaxScaler para reverter normalização.

    Returns:
        Dicionário de métricas no holdout.

    Raises:
        FileNotFoundError: Se os artefatos de holdout não existirem.
    """
    holdout_x = Path(cfg.data.holdout_x_path)
    holdout_y = Path(cfg.data.holdout_y_path)

    if not holdout_x.exists() or not holdout_y.exists():
        raise FileNotFoundError(
            "Holdout não encontrado. Execute feature_engineering.py primeiro."
        )

    X_holdout = np.load(str(holdout_x))
    y_holdout = np.load(str(holdout_y))

    y_pred = model.predict(X_holdout, verbose=0).flatten()
    metrics = compute_metrics(y_holdout, y_pred, scaler)

    logger.info(
        "Holdout: MAE=%.2f | RMSE=%.2f | MAPE=%.4f%%",
        metrics["mae"], metrics["rmse"], metrics["mape"] * 100,
    )
    return {f"holdout_{k}": v for k, v in metrics.items()}


def register(
    run_id: str,
    model_version_tag: str = "1.0.0",
) -> str:
    """Registra o modelo no MLflow Model Registry.

    Args:
        run_id: ID do run MLflow a registrar.
        model_version_tag: Versão semântica para a tag de governança.

    Returns:
        Versão registrada no Model Registry (ex: '1').
    """
    client = MlflowClient()
    model_name = cfg.registry.model_name

    # --- Registra no Model Registry ---
    model_uri = f"runs:/{run_id}/keras_model"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = registered.version
    logger.info("Modelo registrado: %s v%s (run=%s)", model_name, version, run_id)

    # --- Tags obrigatórias de governança (GAP 05) ---
    run = client.get_run(run_id)
    tags = {
        "model_name": model_name,
        "model_version": model_version_tag,
        "model_type": "regression",
        "framework": "tensorflow-keras",
        "owner": cfg.registry.owner,
        "risk_level": cfg.registry.risk_level,
        "fairness_checked": "false",          # Atualizar após auditoria
        "data_start_date": cfg.data.start_date,
        "data_end_date": cfg.data.end_date or "today",
        "phase": "datathon-fase05",
        "git_sha": run.data.tags.get("mlflow.source.git.commit", "unknown"),
        "training_run_id": run_id,
    }

    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, value)

    # Valida que todas as tags obrigatórias foram definidas
    for required_key in REQUIRED_TAGS:
        if required_key not in tags:
            logger.warning("Tag obrigatória ausente: %s", required_key)

    logger.info("Tags de governança aplicadas: %d tags", len(tags))

    # --- Avaliação no holdout + log de métricas finais ---
    scaler = joblib.load(cfg.data.scaler_path)
    model = mlflow.keras.load_model(model_uri)
    holdout_metrics = _evaluate_on_holdout(model, scaler)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(holdout_metrics)

    logger.info("Métricas do holdout logadas no run original.")

    # --- Transição para Staging (aguarda aprovação humana para Production) ---
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.info(
        "Modelo %s v%s → Staging. "
        "Aprovação manual necessária para promover para Production.",
        model_name, version,
    )

    return version


# =============================================================================
# Entrypoint
# =============================================================================

def run(run_id: str | None = None) -> str:
    """Executa o pipeline de registro completo.

    Args:
        run_id: ID do run MLflow. None → usa o run LSTM mais recente.

    Returns:
        Versão registrada no Model Registry.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    if run_id is None:
        run_id = _get_latest_run_id(cfg.mlflow.experiment_name)

    return register(run_id=run_id)


if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Registra o modelo LSTM no MLflow Registry.")
    parser.add_argument("--run-id", type=str, default=None, help="run_id do MLflow.")
    args = parser.parse_args()

    version = run(run_id=args.run_id)
    print(f"\nModelo registrado como versão: {version}")
    print(f"Status: Staging (aprovação manual necessária para Production)")
