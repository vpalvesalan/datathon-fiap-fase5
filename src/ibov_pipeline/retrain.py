# src/ibov_pipeline/retrain.py
"""Pipeline de retraining com avaliação champion-challenger.

Implementa o GAP 07 do guia do Datathon: retraining automatizado com
gate de promoção baseado em delta de MAPE no holdout imutável.

Fluxo:
    1. Carrega o modelo champion da Production no MLflow Registry.
    2. Treina um challenger com os dados mais recentes.
    3. Avalia ambos no holdout imutável.
    4. Promove o challenger para Production apenas se:
       mape_champion - mape_challenger >= promotion_delta_mape (padrão: 0.5%).
    5. Loga o resultado da comparação no MLflow para auditoria.

Uso:
    python -m src.ibov_pipeline.retrain
    python -m src.ibov_pipeline.retrain --force   # Promove mesmo sem melhora

Uso programático:
    from src.ibov_pipeline.retrain import run
    promoted = run()
"""
from __future__ import annotations

import logging
from datetime import datetime

import joblib
import mlflow
import mlflow.keras
import numpy as np
from mlflow.tracking import MlflowClient

from src.ibov_pipeline.config import cfg
from src.ibov_pipeline.feature_engineering import load_artifacts
from src.ibov_pipeline.register_model import register
from src.ibov_pipeline.train_lstm import compute_metrics, train

logger = logging.getLogger(__name__)


# =============================================================================
# Champion
# =============================================================================

def load_champion() -> tuple | None:
    """Carrega o modelo champion da Production no MLflow Registry.

    Returns:
        Tupla (model, version_str) ou None se não houver modelo em Production.
    """
    client = MlflowClient()
    model_name = cfg.registry.model_name

    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        logger.warning(
            "Nenhum modelo em Production para '%s'. "
            "Qualquer challenger será promovido automaticamente.",
            model_name,
        )
        return None

    champion_version = versions[0]
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.keras.load_model(model_uri)

    logger.info(
        "Champion carregado: %s v%s (run=%s)",
        model_name, champion_version.version, champion_version.run_id,
    )
    return model, champion_version.version


def evaluate_champion(model, scaler) -> dict[str, float]:
    """Avalia o champion no holdout imutável.

    Args:
        model: Modelo Keras champion.
        scaler: MinMaxScaler para reverter normalização.

    Returns:
        Dicionário de métricas do champion.
    """
    X_holdout = np.load(cfg.data.holdout_x_path)
    y_holdout = np.load(cfg.data.holdout_y_path)

    y_pred = model.predict(X_holdout, verbose=0).flatten()
    metrics = compute_metrics(y_holdout, y_pred, scaler)
    logger.info(
        "Champion — holdout MAPE=%.4f%% | MAE=%.2f | RMSE=%.2f",
        metrics["mape"] * 100, metrics["mae"], metrics["rmse"],
    )
    return metrics


# =============================================================================
# Challenger
# =============================================================================

def train_challenger(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler,
) -> tuple[str, dict[str, float]]:
    """Treina o challenger com os dados mais recentes.

    Args:
        X_train, X_test: Tensores de entrada.
        y_train, y_test: Vetores de alvo.
        scaler: MinMaxScaler.

    Returns:
        Tupla (run_id, métricas de treino).
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_id, metrics = train(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        run_name=f"lstm-challenger-{timestamp}",
    )
    logger.info("Challenger treinado: run_id=%s | MAPE=%.4f%%", run_id, metrics["mape"] * 100)
    return run_id, metrics


def evaluate_challenger(run_id: str, scaler) -> dict[str, float]:
    """Avalia o challenger no holdout imutável.

    Args:
        run_id: ID do run do challenger no MLflow.
        scaler: MinMaxScaler para reverter normalização.

    Returns:
        Dicionário de métricas do challenger.
    """
    X_holdout = np.load(cfg.data.holdout_x_path)
    y_holdout = np.load(cfg.data.holdout_y_path)

    model_uri = f"runs:/{run_id}/keras_model"
    model = mlflow.keras.load_model(model_uri)

    y_pred = model.predict(X_holdout, verbose=0).flatten()
    metrics = compute_metrics(y_holdout, y_pred, scaler)
    logger.info(
        "Challenger — holdout MAPE=%.4f%% | MAE=%.2f | RMSE=%.2f",
        metrics["mape"] * 100, metrics["mae"], metrics["rmse"],
    )
    return metrics


# =============================================================================
# Gate de promoção
# =============================================================================

def should_promote(
    champion_metrics: dict[str, float] | None,
    challenger_metrics: dict[str, float],
    delta_threshold: float,
) -> bool:
    """Decide se o challenger deve ser promovido.

    Lógica:
        - Se não há champion → promove sempre.
        - Se champion existe → promove apenas se melhora MAPE em ≥ delta_threshold.

    Args:
        champion_metrics: Métricas do champion (None se não houver).
        challenger_metrics: Métricas do challenger.
        delta_threshold: Melhora mínima exigida no MAPE (ex: 0.005 = 0.5%).

    Returns:
        True se o challenger deve ser promovido.
    """
    if champion_metrics is None:
        logger.info("Sem champion em Production → challenger promovido automaticamente.")
        return True

    delta = champion_metrics["mape"] - challenger_metrics["mape"]
    logger.info(
        "Delta MAPE: champion=%.4f%% | challenger=%.4f%% | delta=%.4f%% (threshold=%.4f%%)",
        champion_metrics["mape"] * 100,
        challenger_metrics["mape"] * 100,
        delta * 100,
        delta_threshold * 100,
    )

    if delta >= delta_threshold:
        logger.info("✅ Challenger aprovado para promoção (delta ≥ threshold).")
        return True
    else:
        logger.info(
            "❌ Challenger não promovido (delta=%.4f%% < threshold=%.4f%%).",
            delta * 100, delta_threshold * 100,
        )
        return False


def promote_challenger(challenger_run_id: str) -> str:
    """Registra e promove o challenger para Production.

    Arquiva a versão anterior do champion.

    Args:
        challenger_run_id: run_id do challenger a promover.

    Returns:
        Nova versão em Production.
    """
    client = MlflowClient()
    model_name = cfg.registry.model_name

    # Registra challenger no Model Registry
    version = register(run_id=challenger_run_id)

    # Arquiva o champion atual
    current_prod = client.get_latest_versions(model_name, stages=["Production"])
    for v in current_prod:
        client.transition_model_version_stage(
            name=model_name,
            version=v.version,
            stage="Archived",
        )
        logger.info("Champion v%s → Archived.", v.version)

    # Promove challenger de Staging para Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=False,
    )
    logger.info("Challenger v%s → Production. ✅", version)
    return version


def log_comparison(
    champion_metrics: dict[str, float] | None,
    challenger_metrics: dict[str, float],
    promoted: bool,
    challenger_run_id: str,
) -> None:
    """Loga o resultado da comparação no MLflow para auditoria.

    Args:
        champion_metrics: Métricas do champion (ou None).
        challenger_metrics: Métricas do challenger.
        promoted: Se o challenger foi promovido.
        challenger_run_id: run_id do challenger.
    """
    with mlflow.start_run(run_id=challenger_run_id):
        if champion_metrics:
            mlflow.log_metrics({
                f"champion_{k}": v for k, v in champion_metrics.items()
            })
            mlflow.log_metric(
                "delta_mape",
                champion_metrics["mape"] - challenger_metrics["mape"],
            )
        mlflow.set_tag("retrain_promoted", str(promoted).lower())
        mlflow.set_tag("retrain_timestamp", datetime.utcnow().isoformat())


# =============================================================================
# Entrypoint
# =============================================================================

def run(force: bool = False) -> bool:
    """Executa o ciclo completo de champion-challenger.

    Args:
        force: Se True, promove o challenger mesmo sem melhora de MAPE.

    Returns:
        True se o challenger foi promovido para Production.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # 1. Carrega artefatos e scaler
    X_train, X_test, y_train, y_test, scaler = load_artifacts()

    # 2. Carrega e avalia o champion
    champion_result = load_champion()
    champion_metrics = None
    if champion_result is not None:
        champion_model, _ = champion_result
        champion_metrics = evaluate_champion(champion_model, scaler)

    # 3. Treina e avalia o challenger
    challenger_run_id, _ = train_challenger(X_train, X_test, y_train, y_test, scaler)
    challenger_metrics = evaluate_challenger(challenger_run_id, scaler)

    # 4. Gate de promoção
    promote = force or should_promote(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        delta_threshold=cfg.retrain.promotion_delta_mape,
    )

    # 5. Promoção (se aprovado)
    if promote:
        promote_challenger(challenger_run_id=challenger_run_id)

    # 6. Log de auditoria
    log_comparison(champion_metrics, challenger_metrics, promote, challenger_run_id)

    return promote


if __name__ == "__main__":
    import argparse

    from src.ibov_pipeline.logging_config import setup_logging

    setup_logging("retrain")

    parser = argparse.ArgumentParser(description="Executa o ciclo de retraining champion-challenger.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Promove o challenger mesmo sem melhora de MAPE.",
    )
    args = parser.parse_args()

    promoted = run(force=args.force)
    print(f"\nResultado: {'✅ Challenger promovido para Production' if promoted else '⏸️  Champion mantido'}")
