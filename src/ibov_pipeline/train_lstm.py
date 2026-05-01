# src/ibov_pipeline/train_lstm.py
"""Treina o modelo LSTM com MLflow tracking, busca de hiperparâmetros via
grid search com nested runs, captura de lineage (git SHA + dataset hash) e
avaliação no holdout imutável.

Fluxo:
    1. Carrega artefatos do feature_engineering (treino + teste + holdout).
    2. Se tuner.enabled=true → grid search com 1 run filho por trial.
    3. Caso contrário → treino único com hiperparâmetros fixos do YAML.
    4. Toda corrida loga: params, métricas test, métricas holdout, tags de
       governança (Nível 2 do Microsoft MLOps Maturity Model).

Uso:
    python -m src.ibov_pipeline.train_lstm

Uso programático:
    from src.ibov_pipeline.train_lstm import run
    run_id, metrics = run()
"""
from __future__ import annotations

import hashlib
import itertools
import logging
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

from src.ibov_pipeline.config import cfg
from src.ibov_pipeline.feature_engineering import load_artifacts

logger = logging.getLogger(__name__)


# =============================================================================
# Construção do modelo
# =============================================================================

def build_lstm(
    time_step: int,
    units_1: int,
    units_2: int,
    dropout_1: float,
    dropout_2: float,
    learning_rate: float,
) -> Sequential:
    """Constrói e compila a arquitetura LSTM de 2 camadas.

    Args:
        time_step: Tamanho da janela de entrada.
        units_1: Neurônios da primeira camada LSTM.
        units_2: Neurônios da segunda camada LSTM.
        dropout_1: Taxa de dropout após a primeira camada.
        dropout_2: Taxa de dropout após a segunda camada.
        learning_rate: Learning rate do otimizador Adam.

    Returns:
        Modelo Keras compilado.
    """
    model = Sequential(name="ibov-lstm")
    model.add(LSTM(units=units_1, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(dropout_1))
    model.add(LSTM(units=units_2, return_sequences=False))
    model.add(Dropout(dropout_2))
    model.add(Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=cfg.lstm.loss,
    )
    logger.info(
        "LSTM construído: units=[%d, %d] | dropout=[%.1f, %.1f] | lr=%.4f",
        units_1, units_2, dropout_1, dropout_2, learning_rate,
    )
    return model


# =============================================================================
# Lineage e governança (GAP 05 do guia do Datathon)
# =============================================================================

def _git_sha() -> str:
    """Retorna o SHA do commit atual; 'no-git' se não houver repositório."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha or "no-git"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def compute_lineage_tags(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_version: str = "0.1.0",
) -> dict[str, str]:
    """Captura tags de lineage e governança exigidas pelo Nível 2 MLOps.

    Args:
        X_train: Tensor de treino (usado para hash do dataset).
        y_train: Alvo de treino (usado para hash do dataset).
        model_version: Versão semântica do modelo.

    Returns:
        Dicionário de tags pronto para mlflow.set_tags.
    """
    dataset_hash = hashlib.sha256(
        X_train.tobytes() + y_train.tobytes()
    ).hexdigest()[:12]

    return {
        "git_sha": _git_sha(),
        "dataset_hash": dataset_hash,
        "training_data_version": dataset_hash,
        "model_version": model_version,
        "model_type": "regression",
        "framework": "tensorflow-keras",
        "owner": cfg.registry.owner,
        "phase": "datathon-fase05",
        "pipeline_stage": "lstm",
        "risk_level": cfg.registry.risk_level,
        "fairness_checked": "false",
    }


# =============================================================================
# Métricas
# =============================================================================

def compute_metrics(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    scaler,
) -> dict[str, float]:
    """Calcula MAE, RMSE, MAPE e Acurácia Direcional na escala real do IBOV.

    Args:
        y_true_scaled: Valores reais normalizados.
        y_pred_scaled: Previsões normalizadas.
        scaler: MinMaxScaler para reverter normalização.

    Returns:
        Dicionário com mae, rmse, mape e directional_accuracy.
    """
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(y_pred[1:] - y_true[:-1])
    directional_accuracy = float(np.mean(direction_true == direction_pred))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "directional_accuracy": directional_accuracy,
    }


def evaluate_on_holdout(model, scaler) -> dict[str, float]:
    """Avalia o modelo no conjunto holdout imutável.

    Returns:
        Métricas prefixadas com 'holdout_'.

    Raises:
        FileNotFoundError: Se os artefatos de holdout não existirem.
    """
    X_hold = np.load(cfg.data.holdout_x_path)
    y_hold = np.load(cfg.data.holdout_y_path)

    y_pred = model.predict(X_hold, verbose=0).flatten()
    raw = compute_metrics(y_hold, y_pred, scaler)
    return {f"holdout_{k}": v for k, v in raw.items()}


# =============================================================================
# Treino interno — assume um run MLflow ativo
# =============================================================================

def _train_in_active_run(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler,
    hyperparams: dict[str, Any],
) -> tuple[Sequential, dict[str, float]]:
    """Treina o LSTM e loga tudo no run MLflow ATIVO (não abre run novo).

    Pré-condição: deve haver um run MLflow ativo. Use train() para
    gerenciamento automático.

    Returns:
        Tupla (modelo treinado, métricas test + holdout).
    """
    if mlflow.active_run() is None:
        raise RuntimeError(
            "_train_in_active_run requer um mlflow.start_run ativo. "
            "Use train() para gerenciamento automático ou abra um run antes."
        )

    mlflow.log_params(hyperparams)
    mlflow.log_param("time_step", cfg.features.time_step)
    mlflow.log_param("epochs_max", cfg.lstm.epochs)
    mlflow.log_param("batch_size", cfg.lstm.batch_size)
    mlflow.log_param("patience", cfg.lstm.patience)
    mlflow.log_param("n_samples_train", len(X_train))
    mlflow.log_param("n_samples_test", len(X_test))

    model = build_lstm(time_step=cfg.features.time_step, **hyperparams)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=cfg.lstm.patience,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        epochs=cfg.lstm.epochs,
        batch_size=cfg.lstm.batch_size,
        validation_split=cfg.split.validation_split,
        callbacks=[early_stop],
        verbose=1,
    )

    epochs_trained = len(history.history["loss"])
    mlflow.log_param("epochs_trained", epochs_trained)
    mlflow.log_metric("final_train_loss", history.history["loss"][-1])
    mlflow.log_metric("final_val_loss", history.history["val_loss"][-1])

    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = compute_metrics(y_test, y_pred, scaler)
    mlflow.log_metrics(metrics)

    try:
        holdout_metrics = evaluate_on_holdout(model, scaler)
        mlflow.log_metrics(holdout_metrics)
        metrics.update(holdout_metrics)
    except FileNotFoundError:
        logger.warning("Holdout não encontrado — pulando avaliação no holdout.")

    logger.info(
        "Trial: MAE=%.2f | RMSE=%.2f | MAPE=%.4f%% | DirAcc=%.2f%%",
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"] * 100,
        metrics["directional_accuracy"] * 100,
    )

    return model, metrics


def _persist_model(model: Sequential) -> Path:
    """Salva o artefato Keras e loga no MLflow run ativo."""
    model_path = Path("data/processed/ibov/model_lstm.keras")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    mlflow.log_artifact(str(model_path), artifact_path="model")
    mlflow.keras.log_model(model, artifact_path="keras_model")
    return model_path


# =============================================================================
# Treino com gestão de run (API pública usada por retrain.py)
# =============================================================================

def train(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler,
    hyperparams: dict | None = None,
    run_name: str = "lstm-train",
) -> tuple[str, dict[str, float]]:
    """Treina o LSTM em um run MLflow gerenciado por esta função.

    Esta é a API pública consumida por retrain.py.

    Args:
        X_train, X_test: Tensores de entrada.
        y_train, y_test: Vetores de alvo.
        scaler: MinMaxScaler para reverter normalização.
        hyperparams: Hiperparâmetros. None → usa os do model_config.yaml.
        run_name: Nome do run no MLflow.

    Returns:
        Tupla (run_id, dict de métricas test + holdout).
    """
    hp = hyperparams or {
        "units_1": cfg.lstm.units_1,
        "units_2": cfg.lstm.units_2,
        "dropout_1": cfg.lstm.dropout_1,
        "dropout_2": cfg.lstm.dropout_2,
        "learning_rate": cfg.lstm.learning_rate,
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(compute_lineage_tags(X_train, y_train))
        model, metrics = _train_in_active_run(
            X_train, X_test, y_train, y_test, scaler, hp,
        )
        _persist_model(model)
        return run.info.run_id, metrics


# =============================================================================
# Busca de hiperparâmetros — Grid Search com Nested Runs
# =============================================================================

def _build_search_grid() -> list[dict[str, Any]]:
    """Constrói a grade de busca a partir de cfg.tuner.

    O produto cartesiano de (units_1, units_2, dropout_1, dropout_2,
    learning_rate) é truncado por amostragem regular se exceder
    cfg.tuner.max_trials, garantindo cobertura sem explosão combinatória.
    """
    tc = cfg.tuner

    units_options = list(range(tc.units_min, tc.units_max + 1, tc.units_step))

    dropout_options: list[float] = []
    v = tc.dropout_min
    while v <= tc.dropout_max + 1e-9:
        dropout_options.append(round(v, 4))
        v += tc.dropout_step

    grid = [
        {
            "units_1": u1,
            "units_2": u2,
            "dropout_1": d1,
            "dropout_2": d2,
            "learning_rate": lr,
        }
        for u1, u2, d1, d2, lr in itertools.product(
            units_options, units_options,
            dropout_options, dropout_options,
            tc.learning_rates,
        )
    ]

    if len(grid) > tc.max_trials:
        step = max(1, len(grid) // tc.max_trials)
        grid = grid[::step][: tc.max_trials]

    return grid


def run_hyperparameter_search(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler,
) -> tuple[str, str, dict[str, Any], dict[str, float]]:
    """Busca sistemática com MLflow nested runs.

    Cada combinação vira 1 run filho rastreável e comparável no MLflow UI.
    Otimização pelo MAPE no test set; o melhor modelo é persistido como
    artefato no run filho campeão e o resumo agregado no run pai.

    Returns:
        Tupla (parent_run_id, best_run_id, best_params, best_metrics).
    """
    grid = _build_search_grid()
    logger.info("Iniciando grid search com %d trials", len(grid))

    best_mape = float("inf")
    best_params: dict[str, Any] = {}
    best_run_id: str = ""
    best_metrics: dict[str, float] = {}

    base_lineage = compute_lineage_tags(X_train, y_train)

    with mlflow.start_run(run_name="hyperparameter-search") as parent_run:
        mlflow.set_tags({**base_lineage, "pipeline_stage": "hparam-search"})
        mlflow.set_tag("search_type", "grid")
        mlflow.log_param("n_trials", len(grid))
        mlflow.log_param("optimization_metric", "mape")

        for i, hp in enumerate(grid):
            with mlflow.start_run(
                run_name=f"trial-{i:02d}",
                nested=True,
            ) as child_run:
                mlflow.set_tags({
                    **base_lineage,
                    "pipeline_stage": "hparam-trial",
                    "trial_index": str(i),
                    "parent_run_id": parent_run.info.run_id,
                })

                model, metrics = _train_in_active_run(
                    X_train, X_test, y_train, y_test, scaler, hp,
                )

                if metrics["mape"] < best_mape:
                    best_mape = metrics["mape"]
                    best_params = hp
                    best_run_id = child_run.info.run_id
                    best_metrics = metrics
                    _persist_model(model)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({
            f"best_{k}": v for k, v in best_metrics.items()
            if isinstance(v, (int, float))
        })
        mlflow.set_tag("best_run_id", best_run_id)

        logger.info(
            "Busca concluída: melhor MAPE=%.4f%% | run_id=%s | params=%s",
            best_mape * 100, best_run_id, best_params,
        )

    return parent_run.info.run_id, best_run_id, best_params, best_metrics


# =============================================================================
# Entrypoint
# =============================================================================

def run() -> tuple[str, dict[str, float]]:
    """Executa o pipeline de treino LSTM completo.

    - Se cfg.tuner.enabled=True → grid search com nested runs; retorna o
      run filho campeão.
    - Caso contrário → treino único com hiperparâmetros fixos do YAML.

    Returns:
        Tupla (run_id, dict de métricas test + holdout).
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_test, y_train, y_test, scaler = load_artifacts()

    if cfg.tuner.enabled:
        _, best_run_id, _, best_metrics = run_hyperparameter_search(
            X_train, X_test, y_train, y_test, scaler,
        )
        return best_run_id, best_metrics

    return train(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        run_name="lstm-train",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_id, metrics = run()
    print(f"\nRun ID: {run_id}")
    print(f"Métricas: {metrics}")
