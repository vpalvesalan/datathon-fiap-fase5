"""Fixtures sintéticas compartilhadas para todos os testes.

Política (alinhada ao GAP 04 + GAP 08 do guia do Datathon):
- NUNCA usar dados reais nos testes — apenas séries geradas com seed fixo.
- Tensores pequenos para inferência/treino smoke (custo ~ms).
- Nada que dispare yfinance, MLflow remoto ou rede.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler


@pytest.fixture
def synthetic_close_series() -> np.ndarray:
    """500 pontos de preço sintético do IBOV (tendência + ruído gaussiano)."""
    rng = np.random.default_rng(42)
    trend = np.linspace(80_000, 130_000, 500)
    noise = rng.normal(0, 500, 500)
    return trend + noise


@pytest.fixture
def synthetic_ohlc_df() -> pd.DataFrame:
    """DataFrame com Close + DatetimeIndex (formato pós make_dataset.py).

    O index precisa se chamar 'Date' para passar no RAW_SCHEMA do pandera.
    """
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC", name="Date")
    trend = np.linspace(80_000, 130_000, n)
    noise = rng.normal(0, 500, n)
    return pd.DataFrame({"Close": trend + noise}, index=dates)


@pytest.fixture
def fitted_scaler(synthetic_close_series: np.ndarray) -> MinMaxScaler:
    """MinMaxScaler ajustado em dados sintéticos (range [0, 1])."""
    scaler = MinMaxScaler()
    scaler.fit(synthetic_close_series.reshape(-1, 1))
    return scaler


@pytest.fixture
def tiny_lstm_data() -> tuple[np.ndarray, np.ndarray]:
    """Tensores mínimos para smoke test de build/predict (sem custo de treino)."""
    rng = np.random.default_rng(42)
    X = rng.random((20, 60, 1)).astype(np.float32)
    y = rng.random(20).astype(np.float32)
    return X, y
