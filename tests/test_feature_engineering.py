"""Testes de feature_engineering.py — sliding windows, scaler, splits.

Cobre os critérios do guia (GAP 04):
- Schema/shape correto
- Sem nulls após transformação
- Range respeitado
- Determinismo / consistência cronológica
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.ibov_pipeline.feature_engineering import (
    chronological_split,
    create_sequences,
    fit_scaler,
    load_artifacts,
    save_artifacts,
)


# =============================================================================
# fit_scaler
# =============================================================================

def test_fit_scaler_output_shape(synthetic_close_series):
    scaled, scaler = fit_scaler(synthetic_close_series)
    assert scaled.shape == (len(synthetic_close_series), 1)
    assert isinstance(scaler, MinMaxScaler)


def test_fit_scaler_range(synthetic_close_series):
    """Valores escalados devem cair em [0, 1]."""
    scaled, _ = fit_scaler(synthetic_close_series)
    assert scaled.min() >= 0.0 - 1e-9
    assert scaled.max() <= 1.0 + 1e-9


def test_fit_scaler_inverse_roundtrip(synthetic_close_series):
    """inverse_transform deve recuperar os valores originais (até float precision)."""
    scaled, scaler = fit_scaler(synthetic_close_series)
    recovered = scaler.inverse_transform(scaled).flatten()
    np.testing.assert_allclose(recovered, synthetic_close_series, rtol=1e-9)


# =============================================================================
# create_sequences
# =============================================================================

def test_create_sequences_shape(synthetic_close_series):
    """X deve ser (n - time_step, time_step, 1); y deve ser (n - time_step,)."""
    scaled, _ = fit_scaler(synthetic_close_series)
    X, y = create_sequences(scaled, time_step=60)
    assert X.shape == (len(synthetic_close_series) - 60, 60, 1)
    assert y.shape == (len(synthetic_close_series) - 60,)


def test_create_sequences_no_nulls(synthetic_close_series):
    scaled, _ = fit_scaler(synthetic_close_series)
    X, y = create_sequences(scaled, time_step=60)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_create_sequences_window_correctness():
    """X[i] deve ser a janela de tamanho time_step terminando em i; y[i] = próximo."""
    arr = np.arange(100).reshape(-1, 1).astype(float)
    X, y = create_sequences(arr, time_step=10)

    np.testing.assert_array_equal(X[0, :, 0], np.arange(10))
    assert y[0] == 10
    np.testing.assert_array_equal(X[1, :, 0], np.arange(1, 11))
    assert y[1] == 11
    # Última janela: X[-1] = [89,...,98], y[-1] = 99
    np.testing.assert_array_equal(X[-1, :, 0], np.arange(89, 99))
    assert y[-1] == 99


def test_create_sequences_raises_for_short_series():
    """Série menor ou igual ao time_step deve lançar ValueError."""
    short = np.arange(30).reshape(-1, 1).astype(float)
    with pytest.raises(ValueError):
        create_sequences(short, time_step=60)


# =============================================================================
# chronological_split
# =============================================================================

def test_chronological_split_no_leakage():
    """Holdout deve vir do FINAL — não pode haver vazamento futuro→passado."""
    X = np.arange(100).reshape(100, 1, 1).astype(float)
    y = np.arange(100).astype(float)
    X_tr, X_te, y_tr, y_te, X_h, y_h = chronological_split(
        X, y, train_ratio=0.8, holdout_ratio=0.1,
    )
    # Última janela do holdout deve ser a última observação
    assert y_h[-1] == 99
    # Treino vem antes de teste; teste antes de holdout
    assert y_tr[-1] < y_te[0] < y_h[0]


def test_chronological_split_total_count_preserved():
    """Soma dos splits == total de amostras."""
    X = np.arange(100).reshape(100, 1, 1).astype(float)
    y = np.arange(100).astype(float)
    X_tr, X_te, y_tr, y_te, X_h, y_h = chronological_split(
        X, y, train_ratio=0.8, holdout_ratio=0.1,
    )
    assert len(X_tr) + len(X_te) + len(X_h) == 100
    assert len(y_tr) + len(y_te) + len(y_h) == 100


def test_chronological_split_holdout_size():
    """holdout_ratio=0.1 sobre n=100 deve render 10 amostras de holdout."""
    X = np.arange(100).reshape(100, 1, 1).astype(float)
    y = np.arange(100).astype(float)
    _, _, _, _, X_h, _ = chronological_split(X, y, train_ratio=0.8, holdout_ratio=0.1)
    assert len(X_h) == 10


# =============================================================================
# Roundtrip save/load
# =============================================================================

def test_save_load_artifacts_roundtrip(tmp_path, monkeypatch):
    """save_artifacts + load_artifacts devem preservar tensores e scaler."""
    from src.ibov_pipeline import config as cfg_mod

    base = tmp_path / "processed"
    base.mkdir()
    holdout = tmp_path / "holdout"
    holdout.mkdir()

    monkeypatch.setattr(cfg_mod.cfg.data, "processed_x_path", str(base / "X.npy"))
    monkeypatch.setattr(cfg_mod.cfg.data, "processed_y_path", str(base / "y.npy"))
    monkeypatch.setattr(cfg_mod.cfg.data, "scaler_path", str(base / "scaler.joblib"))
    monkeypatch.setattr(cfg_mod.cfg.data, "holdout_x_path", str(holdout / "X_holdout.npy"))
    monkeypatch.setattr(cfg_mod.cfg.data, "holdout_y_path", str(holdout / "y_holdout.npy"))

    X_tr = np.zeros((50, 60, 1))
    X_te = np.zeros((20, 60, 1))
    y_tr = np.zeros(50)
    y_te = np.zeros(20)
    X_h = np.zeros((10, 60, 1))
    y_h = np.zeros(10)
    scaler = MinMaxScaler().fit(np.array([[0.0], [1.0]]))

    save_artifacts(X_tr, X_te, y_tr, y_te, X_h, y_h, scaler)
    X_tr_l, X_te_l, y_tr_l, y_te_l, scaler_l = load_artifacts()

    np.testing.assert_array_equal(X_tr_l, X_tr)
    np.testing.assert_array_equal(X_te_l, X_te)
    np.testing.assert_array_equal(y_tr_l, y_tr)
    assert isinstance(scaler_l, MinMaxScaler)
