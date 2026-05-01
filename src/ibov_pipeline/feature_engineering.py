# src/ibov_pipeline/feature_engineering.py
"""Engenharia de features para o pipeline LSTM do IBOV.

Responsabilidades:
    1. Normalizar a série com MinMaxScaler.
    2. Gerar janelas deslizantes (sliding windows) no formato [samples, time_steps, 1].
    3. Dividir cronologicamente em treino, teste e holdout imutável.
    4. Persistir tensores (.npy) e scaler (.joblib) para uso pelo trainer.

Uso:
    python -m src.ibov_pipeline.feature_engineering

Uso programático:
    from src.ibov_pipeline.feature_engineering import run
    X_train, X_test, y_train, y_test, scaler = run()
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.ibov_pipeline.config import cfg

logger = logging.getLogger(__name__)


# =============================================================================
# Funções públicas
# =============================================================================

def fit_scaler(series: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Ajusta e aplica o MinMaxScaler à série.

    Args:
        series: Array 1D ou 2D com os valores de fechamento.

    Returns:
        Tupla (série normalizada como 2D array, scaler ajustado).
    """
    data = series.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=tuple(cfg.features.feature_range))  # type: ignore[arg-type]
    scaled = scaler.fit_transform(data)
    logger.info(
        "Scaler ajustado: min=%.2f, max=%.2f → range %s",
        float(scaler.data_min_[0]),
        float(scaler.data_max_[0]),
        cfg.features.feature_range,
    )
    return scaled, scaler


def create_sequences(
    scaled_data: np.ndarray,
    time_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera janelas deslizantes para o LSTM.

    Cada amostra X[i] contém os `time_step` valores anteriores;
    y[i] contém o valor seguinte (t+1).

    Args:
        scaled_data: Array 2D normalizado (n_samples, 1).
        time_step: Tamanho da janela de entrada.

    Returns:
        Tupla (X, y) onde X tem shape (n, time_step, 1) e y tem shape (n,).

    Raises:
        ValueError: Se não houver amostras suficientes para a janela.
    """
    flat = scaled_data.flatten()
    if len(flat) <= time_step:
        raise ValueError(
            f"Série ({len(flat)} pontos) menor que time_step ({time_step}). "
            "Reduza time_step ou aumente o período de coleta."
        )

    X, y = [], []
    for i in range(len(flat) - time_step):
        X.append(flat[i : i + time_step])
        y.append(flat[i + time_step])

    X_arr = np.array(X).reshape(-1, time_step, 1)   # [samples, time_steps, features]
    y_arr = np.array(y)

    logger.info("Janelas criadas: X=%s, y=%s", X_arr.shape, y_arr.shape)
    return X_arr, y_arr


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    holdout_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide cronologicamente em treino, teste e holdout.

    Ordem temporal preservada: [--- treino ---][--- teste ---][--- holdout ---]
    O holdout é separado PRIMEIRO (do final) para garantir que nunca vaza
    para o treino, mesmo em múltiplas execuções.

    Args:
        X: Tensor de entrada (n, time_step, 1).
        y: Vetor de alvo (n,).
        train_ratio: Proporção de amostras para treino.
        holdout_ratio: Proporção de amostras para holdout imutável.

    Returns:
        (X_train, X_test, y_train, y_test, X_holdout, y_holdout)
    """
    n = len(X)
    holdout_cut = int(n * (1 - holdout_ratio))
    train_cut = int(holdout_cut * train_ratio)

    X_train, y_train = X[:train_cut], y[:train_cut]
    X_test, y_test = X[train_cut:holdout_cut], y[train_cut:holdout_cut]
    X_holdout, y_holdout = X[holdout_cut:], y[holdout_cut:]

    logger.info(
        "Split cronológico: treino=%d | teste=%d | holdout=%d",
        len(X_train), len(X_test), len(X_holdout),
    )
    return X_train, X_test, y_train, y_test, X_holdout, y_holdout


def save_artifacts(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    scaler: MinMaxScaler,
) -> None:
    """Persiste todos os artefatos do feature engineering.

    Args:
        X_train, X_test, y_train, y_test: Conjuntos de treino e teste.
        X_holdout, y_holdout: Holdout imutável.
        scaler: Scaler ajustado.
    """
    # Cria diretórios se não existirem
    for path_str in [
        cfg.data.processed_x_path,
        cfg.data.scaler_path,
        cfg.data.holdout_x_path,
    ]:
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    # Salva tensores de treino/teste
    np.save(cfg.data.processed_x_path.replace("X.npy", "X_train.npy"), X_train)
    np.save(cfg.data.processed_x_path.replace("X.npy", "X_test.npy"), X_test)
    np.save(cfg.data.processed_y_path.replace("y.npy", "y_train.npy"), y_train)
    np.save(cfg.data.processed_y_path.replace("y.npy", "y_test.npy"), y_test)

    # Salva holdout (imutável — usado apenas no register_model e retrain)
    np.save(cfg.data.holdout_x_path, X_holdout)
    np.save(cfg.data.holdout_y_path, y_holdout)

    # Salva scaler
    joblib.dump(scaler, cfg.data.scaler_path)

    logger.info("Artefatos salvos com sucesso.")
    logger.info("  X_train: %s → %s", X_train.shape, cfg.data.processed_x_path)
    logger.info("  scaler  → %s", cfg.data.scaler_path)
    logger.info("  holdout → %s", cfg.data.holdout_x_path)


def load_artifacts() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler
]:
    """Carrega os artefatos processados do disco.

    Returns:
        (X_train, X_test, y_train, y_test, scaler)

    Raises:
        FileNotFoundError: Se os artefatos não existirem.
    """
    base = Path(cfg.data.processed_x_path).parent

    def _load(name: str) -> np.ndarray:
        path = base / name
        if not path.exists():
            raise FileNotFoundError(
                f"Artefato não encontrado: {path}. "
                "Execute feature_engineering.py primeiro."
            )
        return np.load(path)

    X_train = _load("X_train.npy")
    X_test = _load("X_test.npy")
    y_train = _load("y_train.npy")
    y_test = _load("y_test.npy")
    scaler: MinMaxScaler = joblib.load(cfg.data.scaler_path)

    logger.info(
        "Artefatos carregados: X_train=%s, X_test=%s",
        X_train.shape, X_test.shape,
    )
    return X_train, X_test, y_train, y_test, scaler


# =============================================================================
# Entrypoint
# =============================================================================

def run() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Executa o pipeline de feature engineering completo.

    1. Carrega o CSV bruto.
    2. Normaliza com MinMaxScaler.
    3. Cria janelas deslizantes.
    4. Divide cronologicamente.
    5. Persiste artefatos.

    Returns:
        (X_train, X_test, y_train, y_test, scaler)
    """
    from src.ibov_pipeline.make_dataset import load_raw

    df: pd.DataFrame = load_raw(cfg.data.raw_path)
    series = df["Close"].values.astype(float)

    scaled, scaler = fit_scaler(series)

    X, y = create_sequences(scaled, time_step=cfg.features.time_step)

    X_train, X_test, y_train, y_test, X_holdout, y_holdout = chronological_split(
        X, y,
        train_ratio=cfg.split.train_ratio,
        holdout_ratio=cfg.split.holdout_ratio,
    )

    save_artifacts(X_train, X_test, y_train, y_test, X_holdout, y_holdout, scaler)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run()