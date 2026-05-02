"""Helper de inferência consumido pelo IBOVForecastTool do agente.

Carrega o modelo champion do **MLflow Model Registry** (Stage Production)
com fallback para o artefato local `data/processed/ibov/model_lstm.keras`.
Para a janela de entrada, **baixa dados frescos do yfinance** a cada chamada
— se o download falhar, cai para o CSV local com aviso explícito.

Esse encapsulamento evita que o agente precise conhecer detalhes do
pipeline de treino — pede só "qual a previsão para amanhã?".

Uso:
    from src.ibov_pipeline.predict import predict_next_day
    r = predict_next_day()
    # {"predicted_close": 130_500.45, "last_close": 129_800.22, "delta_pct": 0.54,
    #  "as_of_date": "2026-05-02", "predicted_for_date": "2026-05-04",
    #  "data_source": "yfinance", "warning": None}
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ibov_pipeline.config import cfg

logger = logging.getLogger(__name__)


# =============================================================================
# Modelo
# =============================================================================

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


# =============================================================================
# Janela de entrada — yfinance fresco (preferido) → CSV local (fallback)
# =============================================================================

def _next_business_day(d: date) -> date:
    """Próximo dia útil (Mon–Fri). Não considera feriados B3."""
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:  # 5=sab, 6=dom
        nxt += timedelta(days=1)
    return nxt


def _fetch_from_yfinance(time_step: int) -> tuple[np.ndarray, date]:
    """Baixa do yfinance os últimos `time_step` pregões de Close. Sobe exceção em falha."""
    import yfinance as yf

    # Margem de calendário para garantir time_step pregões mesmo com feriados
    n_calendar = int(time_step * 1.6) + 15
    end = date.today() + timedelta(days=1)  # inclusivo
    start = end - timedelta(days=n_calendar)

    df = yf.download(
        cfg.data.symbol,
        start=str(start),
        end=str(end),
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance retornou vazio para {cfg.data.symbol}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()
    if len(close) < time_step:
        raise RuntimeError(
            f"yfinance devolveu {len(close)} pregões — esperado ≥ {time_step}."
        )

    window = close.tail(time_step).to_numpy(dtype=float)
    as_of = close.index[-1].date() if hasattr(close.index[-1], "date") else close.index[-1]
    return window, as_of


def _fetch_from_local_csv(time_step: int) -> tuple[np.ndarray, date]:
    """Lê os últimos `time_step` pregões de Close do CSV local. Sobe exceção em falha."""
    raw = Path(cfg.data.raw_path)
    if not raw.exists():
        raise FileNotFoundError(
            f"CSV local ausente em {raw}. Rode `python -m src.ibov_pipeline.make_dataset`."
        )

    df = pd.read_csv(raw, index_col="Date", parse_dates=True).sort_index()
    if len(df) < time_step:
        raise RuntimeError(
            f"CSV local tem {len(df)} pregões — esperado ≥ {time_step}."
        )

    window = df["Close"].tail(time_step).to_numpy(dtype=float)
    last = df.index[-1]
    as_of = last.date() if hasattr(last, "date") else last
    return window, as_of


def _fetch_recent_close(time_step: int) -> tuple[np.ndarray, date, str, str | None]:
    """Tenta yfinance primeiro; cai para CSV local se falhar.

    Returns:
        (window, as_of_date, data_source, warning_or_None)
    """
    try:
        window, as_of = _fetch_from_yfinance(time_step)
        logger.info("Janela obtida do yfinance — último fechamento em %s.", as_of)
        return window, as_of, "yfinance", None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Download do yfinance falhou (%s). Tentando CSV local...", exc)
        window, as_of = _fetch_from_local_csv(time_step)
        warning = (
            f"Download do yfinance falhou ({type(exc).__name__}: {exc}). "
            f"Usando CSV local — último fechamento em {as_of.isoformat()}, "
            f"previsão refere-se a {_next_business_day(as_of).isoformat()}."
        )
        logger.warning(warning)
        return window, as_of, "local_csv", warning


# =============================================================================
# Previsão
# =============================================================================

def predict_next_day(
    recent_close_window: np.ndarray | None = None,
) -> dict:
    """Prevê o preço de fechamento do próximo pregão do IBOV.

    Args:
        recent_close_window: Janela explícita (1D). Se None, baixa do yfinance
            e cai para CSV local em caso de falha.

    Returns:
        Dicionário com:
            - predicted_close (float)
            - last_close (float)
            - delta_pct (float)
            - as_of_date (str ISO ou None se janela for custom)
            - predicted_for_date (str ISO ou None se janela for custom)
            - data_source ("yfinance" | "local_csv" | "user_input")
            - warning (str ou None)

    Raises:
        ValueError: Se a janela for menor que `cfg.features.time_step`.
        FileNotFoundError: Se nem yfinance nem CSV local funcionarem.
    """
    model = _load_champion_model()
    scaler = joblib.load(cfg.data.scaler_path)

    as_of: date | None = None
    data_source = "user_input"
    warning: str | None = None

    if recent_close_window is None:
        recent_close_window, as_of, data_source, warning = _fetch_recent_close(
            cfg.features.time_step,
        )

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

    predicted_for = _next_business_day(as_of) if as_of else None

    logger.info(
        "Previsão IBOV: %.2f → %.2f (%+.2f%%) | as_of=%s | for=%s | source=%s",
        last_close, pred_real, delta_pct,
        as_of.isoformat() if as_of else "?",
        predicted_for.isoformat() if predicted_for else "?",
        data_source,
    )

    return {
        "predicted_close": pred_real,
        "last_close": last_close,
        "delta_pct": delta_pct,
        "as_of_date": as_of.isoformat() if as_of else None,
        "predicted_for_date": predicted_for.isoformat() if predicted_for else None,
        "data_source": data_source,
        "warning": warning,
    }


if __name__ == "__main__":
    from src.ibov_pipeline.logging_config import setup_logging

    setup_logging("predict")
    out = predict_next_day()

    if out.get("warning"):
        print(f"\n⚠️  {out['warning']}\n")

    if out.get("predicted_for_date"):
        print(f"Previsão para {out['predicted_for_date']} "
              f"(fonte: {out['data_source']}, baseada em {out['as_of_date']})")
    else:
        print("Previsão (janela custom, sem data conhecida)")

    print(f"  Próximo fechamento estimado : {out['predicted_close']:,.2f} pts")
    print(f"  Último fechamento conhecido : {out['last_close']:,.2f} pts")
    print(f"  Variação prevista            : {out['delta_pct']:+.2f}%")
