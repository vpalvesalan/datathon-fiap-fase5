# src/ibov_pipeline/train_baseline.py
"""Treina modelos baseline (Scikit-Learn) com MLflow tracking completo.

O baseline serve como piso de comparação para o LSTM.
Todos os modelos definidos em model_config.yaml são treinados e logados.

Uso:
    python -m src.ibov_pipeline.train_baseline

Uso programático:
    from src.ibov_pipeline.train_baseline import run
    results = run()
"""
from __future__ import annotations

import importlib
import logging

import mlflow
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from src.ibov_pipeline.config import cfg
from src.ibov_pipeline.feature_engineering import load_artifacts

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _load_class(dotted_path: str):
    """Importa dinamicamente uma classe a partir do caminho pontilhado.

    Args:
        dotted_path: Ex: 'sklearn.linear_model.Ridge'.

    Returns:
        Classe importada.
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _flatten_for_sklearn(X: np.ndarray) -> np.ndarray:
    """O sklearn não aceita tensores 3D — achata (n, time_step, 1) → (n, time_step).

    Args:
        X: Tensor 3D de entrada.

    Returns:
        Matriz 2D (n_samples, time_step).
    """
    return X.reshape(X.shape[0], -1)


def _compute_metrics(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    scaler,
) -> dict[str, float]:
    """Calcula MAE, RMSE e MAPE na escala original do IBOV.

    Args:
        y_true_scaled: Valores reais normalizados.
        y_pred_scaled: Previsões normalizadas.
        scaler: MinMaxScaler para inverter a normalização.

    Returns:
        Dicionário de métricas.
    """
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


# =============================================================================
# Treino de um único baseline
# =============================================================================

def train_single_baseline(
    model_name: str,
    model_class,
    model_params: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler,
) -> tuple[str, dict[str, float]]:
    """Treina um baseline e loga tudo no MLflow.

    Args:
        model_name: Nome descritivo para o run.
        model_class: Classe Scikit-Learn do modelo.
        model_params: Hiperparâmetros do modelo.
        X_train, X_test: Tensores de entrada (serão achatados).
        y_train, y_test: Vetores de alvo.
        scaler: MinMaxScaler para reverter normalização.

    Returns:
        Tupla (run_id, dict de métricas).
    """
    X_train_2d = _flatten_for_sklearn(X_train)
    X_test_2d = _flatten_for_sklearn(X_test)

    with mlflow.start_run(run_name=f"baseline-{model_name}") as run:
        # --- Parâmetros ---
        mlflow.log_params(model_params)
        mlflow.log_param("model_class", model_class.__name__)
        mlflow.log_param("n_features", X_train_2d.shape[1])
        mlflow.log_param("n_samples_train", X_train_2d.shape[0])
        mlflow.log_param("time_step", cfg.features.time_step)

        # --- Tags de governança ---
        mlflow.set_tags({
            "model_type": "regression",
            "framework": "scikit-learn",
            "owner": cfg.registry.owner,
            "phase": "datathon-fase05",
            "pipeline_stage": "baseline",
            "risk_level": cfg.registry.risk_level,
        })

        # --- Treino ---
        model = model_class(**model_params)
        model.fit(X_train_2d, y_train)
        y_pred = model.predict(X_test_2d)

        # --- Métricas na escala real do IBOV ---
        metrics = _compute_metrics(y_test, y_pred, scaler)
        mlflow.log_metrics(metrics)

        # --- Artefato do modelo ---
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(
            "Baseline %s: MAE=%.2f pts | RMSE=%.2f pts | MAPE=%.4f%%",
            model_name, metrics["mae"], metrics["rmse"], metrics["mape"] * 100,
        )

        return run.info.run_id, metrics


# =============================================================================
# Entrypoint
# =============================================================================

def run() -> list[dict]:
    """Treina todos os baselines definidos em model_config.yaml.

    Returns:
        Lista de dicionários com {model_name, run_id, metrics}.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_test, y_train, y_test, scaler = load_artifacts()

    results = []
    for spec in cfg.baseline.models:
        model_class = _load_class(spec.class_)
        run_id, metrics = train_single_baseline(
            model_name=spec.name,
            model_class=model_class,
            model_params=spec.params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
        )
        results.append({"model_name": spec.name, "run_id": run_id, "metrics": metrics})

    # Exibe tabela de comparação ao final
    logger.info("\n=== Comparativo de Baselines ===")
    for r in sorted(results, key=lambda x: x["metrics"]["mape"]):
        logger.info(
            "%-20s | MAPE: %.4f%% | MAE: %.2f | RMSE: %.2f",
            r["model_name"],
            r["metrics"]["mape"] * 100,
            r["metrics"]["mae"],
            r["metrics"]["rmse"],
        )

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run()
