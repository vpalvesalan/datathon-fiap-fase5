"""Testes das funções puras de predict.py.

Cobre `_next_business_day`, `_fetch_from_local_csv` e o caminho de fallback
de `_fetch_recent_close`. Não testa yfinance live (vai pra integração).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ibov_pipeline.predict import (
    _fetch_from_local_csv,
    _fetch_recent_close,
    _next_business_day,
)


# =============================================================================
# _next_business_day
# =============================================================================

@pytest.mark.parametrize("d_in, d_out", [
    (date(2026, 5, 4), date(2026, 5, 5)),    # seg → ter
    (date(2026, 5, 7), date(2026, 5, 8)),    # qui → sex
    (date(2026, 5, 8), date(2026, 5, 11)),   # sex → seg (pula fim de semana)
    (date(2026, 5, 9), date(2026, 5, 11)),   # sab → seg
    (date(2026, 5, 10), date(2026, 5, 11)),  # dom → seg
])
def test_next_business_day(d_in, d_out):
    assert _next_business_day(d_in) == d_out


# =============================================================================
# _fetch_from_local_csv
# =============================================================================

def _write_csv(path, n_rows: int) -> None:
    """Cria um CSV no formato de data/raw/ibov/ibov_close.csv."""
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    df = pd.DataFrame({"Close": np.linspace(100_000, 130_000, n_rows)}, index=dates)
    df.index.name = "Date"
    df.to_csv(path)


def test_fetch_local_csv_returns_window_and_date(monkeypatch, tmp_path):
    csv = tmp_path / "ibov.csv"
    _write_csv(csv, n_rows=100)

    from src.ibov_pipeline import config as cfg_mod
    monkeypatch.setattr(cfg_mod.cfg.data, "raw_path", str(csv))

    window, as_of = _fetch_from_local_csv(time_step=60)

    assert window.shape == (60,)
    assert isinstance(as_of, date)
    # Última data deve ser o último business day no range gerado
    assert as_of == pd.date_range("2024-01-01", periods=100, freq="B")[-1].date()


def test_fetch_local_csv_raises_when_missing(monkeypatch, tmp_path):
    from src.ibov_pipeline import config as cfg_mod
    monkeypatch.setattr(cfg_mod.cfg.data, "raw_path", str(tmp_path / "nope.csv"))

    with pytest.raises(FileNotFoundError, match="CSV local ausente"):
        _fetch_from_local_csv(time_step=60)


def test_fetch_local_csv_raises_when_too_short(monkeypatch, tmp_path):
    csv = tmp_path / "short.csv"
    _write_csv(csv, n_rows=30)

    from src.ibov_pipeline import config as cfg_mod
    monkeypatch.setattr(cfg_mod.cfg.data, "raw_path", str(csv))

    with pytest.raises(RuntimeError, match="esperado ≥ 60"):
        _fetch_from_local_csv(time_step=60)


# =============================================================================
# _fetch_recent_close — fallback path quando yfinance falha
# =============================================================================

def test_fetch_recent_close_falls_back_to_csv(monkeypatch, tmp_path):
    """Se _fetch_from_yfinance levantar exceção, deve cair para CSV local
    e retornar warning explícito mencionando data prevista."""
    csv = tmp_path / "ibov.csv"
    _write_csv(csv, n_rows=100)

    from src.ibov_pipeline import config as cfg_mod
    from src.ibov_pipeline import predict as predict_mod

    monkeypatch.setattr(cfg_mod.cfg.data, "raw_path", str(csv))

    def _broken(*_, **__):
        raise RuntimeError("offline / sem rede")

    monkeypatch.setattr(predict_mod, "_fetch_from_yfinance", _broken)

    window, as_of, source, warning = _fetch_recent_close(time_step=60)

    assert window.shape == (60,)
    assert source == "local_csv"
    assert warning is not None
    assert "yfinance" in warning.lower()
    # Warning deve mencionar a data prevista (next business day)
    expected_next = _next_business_day(as_of).isoformat()
    assert expected_next in warning


def test_fetch_recent_close_uses_yfinance_when_available(monkeypatch):
    """Quando yfinance funciona, retorna source='yfinance' sem warning."""
    from src.ibov_pipeline import predict as predict_mod

    fake_window = np.linspace(100_000, 130_000, 60)
    fake_date = date(2026, 5, 1)

    def _ok(time_step):
        assert time_step == 60
        return fake_window, fake_date

    monkeypatch.setattr(predict_mod, "_fetch_from_yfinance", _ok)

    window, as_of, source, warning = _fetch_recent_close(time_step=60)

    assert source == "yfinance"
    assert warning is None
    assert as_of == fake_date
    np.testing.assert_array_equal(window, fake_window)
