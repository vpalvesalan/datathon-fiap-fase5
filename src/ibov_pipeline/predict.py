"""Helper de inferência consumido pelo IBOVForecastTool do agente.

Carrega o modelo champion do **MLflow Model Registry** (Stage Production)
com fallback para o artefato local `data/processed/ibov/model_lstm.keras`.
Esse encapsulamento evita que o agente precise conhecer detalhes do
pipeline de treino — pede só "qual a previsão para amanhã?".

Uso:
    from src.ibov_pipeline.predict import predict_next_day
    result = predict_next_day()
    # {"predicted_close": 130_500.45, "last_close": 129_800.22, "delta_pct": 0.54}
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ibov_pipeline.config import cfg

logger = logging.getLogger(__name__)


def _load_champion_model():
    """Tenta carregar do MLflow Registry (Production); fallback para arquivo local."""
    try:
        import mlflow

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        uri = f"models:/{cfg.registry.model_name}/Production"
        model = mlflow.keras.load_model(uri)
        logger.info("Champion carregado do MLflow Registry: %s", uri)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Falha ao carregar do Registry (%s). Usando artefato local.", exc,
        )
        from tensorflow.keras.models import load_model

        local_path = Path(cfg.data.processed_x_path).parent / "model_lstm.keras"
        if not local_path.exists():
            raise FileNotFoundError(
                f"Nem Registry nem artefato local ({local_path}) disponíveis. "
                "Rode `python -m src.ibov_pipeline.train_lstm` primeiro."
            ) from exc
        return load_model(local_path)


def predict_next_day(
    recent_close_window: np.ndarray | None = None,
) -> dict[str, float]:
    """Prevê o preço de fechamento do próximo pregão do IBOV.

    Args:
        recent_close_window: Array 1D dos últimos N preços de fechamento.
            Se None, lê os últimos `cfg.features.time_step` pontos de
            `data/raw/ibov/ibov_close.csv`.

    Returns:
        Dicionário com `predicted_close`, `last_close` e `delta_pct`.

    Raises:
        ValueError: Se a janela for menor que `cfg.features.time_step`.
        FileNotFoundError: Se nenhum modelo (Registry ou local) puder ser carregado.
    """
    model = _load_champion_model()
    scaler = joblib.load(cfg.data.scaler_path)

    if recent_close_window is None:
        df = pd.read_csv(cfg.data.raw_path, index_col="Date", parse_dates=True)
        recent_close_window = df["Close"].tail(cfg.features.time_step).values

    if len(recent_close_window) < cfg.features.time_step:
        raise ValueError(
            f"Janela tem {len(recent_close_window)} pontos; "
            f"esperado ≥ {cfg.features.time_step}."
        )

    window = np.asarray(recent_close_window[-cfg.features.time_step:], dtype=float)
    scaled = scaler.transform(window.reshape(-1, 1))
    X = scaled.reshape(1, cfg.features.time_step, 1)

    pred_scaled = model.predict(X, verbose=0)
    pred_real = float(scaler.inverse_transform(pred_scaled)[0, 0])
    last_close = float(window[-1])
    delta_pct = (pred_real - last_close) / last_close * 100.0

    logger.info(
        "Previsão IBOV: %.2f → %.2f (%+.2f%%)",
        last_close, pred_real, delta_pct,
    )
    return {
        "predicted_close": pred_real,
        "last_close": last_close,
        "delta_pct": delta_pct,
    }


if __name__ == "__main__":
    from src.ibov_pipeline.logging_config import setup_logging

    setup_logging("predict")
    out = predict_next_day()
    print(f"\nPróximo fechamento IBOV (estimado): {out['predicted_close']:,.2f} pts")
    print(f"Último fechamento conhecido         : {out['last_close']:,.2f} pts")
    print(f"Variação prevista                   : {out['delta_pct']:+.2f}%")
