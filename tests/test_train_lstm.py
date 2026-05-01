"""Testes das funções públicas (puras) de train_lstm.py.

Cobre: compute_metrics, compute_lineage_tags, build_lstm, evaluate_on_holdout
e _build_search_grid.

Não testa: train(), _train_in_active_run, run_hyperparameter_search, run() —
essas precisam de MLflow ativo e vivem em outra camada de testes (integração).
"""
from __future__ import annotations

import numpy as np

from src.ibov_pipeline.train_lstm import (
    _build_search_grid,
    build_lstm,
    compute_lineage_tags,
    compute_metrics,
    evaluate_on_holdout,
)


# =============================================================================
# compute_metrics
# =============================================================================

def test_compute_metrics_returns_required_keys(fitted_scaler):
    """Dict de saída deve ter exatamente as 4 métricas esperadas."""
    y_true = np.linspace(0.1, 0.9, 50)
    y_pred = y_true + 0.001
    m = compute_metrics(y_true, y_pred, fitted_scaler)
    assert set(m.keys()) == {"mae", "rmse", "mape", "directional_accuracy"}


def test_compute_metrics_perfect_prediction(fitted_scaler):
    """Previsão = real → erros (mae, rmse, mape) próximos de zero."""
    y_true = np.linspace(0.1, 0.9, 50)
    m = compute_metrics(y_true, y_true.copy(), fitted_scaler)
    assert m["mae"] < 1e-6
    assert m["rmse"] < 1e-6
    assert m["mape"] < 1e-6


def test_compute_metrics_directional_accuracy_perfect(fitted_scaler):
    """Previsão idêntica → directional_accuracy = 1.0."""
    y_true = np.linspace(0.1, 0.9, 50)
    m = compute_metrics(y_true, y_true.copy(), fitted_scaler)
    assert m["directional_accuracy"] == 1.0


def test_compute_metrics_metrics_are_floats(fitted_scaler):
    """Todas as métricas devem ser floats nativos (não numpy scalars) — JSON-friendly."""
    y_true = np.linspace(0.1, 0.9, 30)
    y_pred = y_true + 0.01
    m = compute_metrics(y_true, y_pred, fitted_scaler)
    for v in m.values():
        assert isinstance(v, float)


# =============================================================================
# compute_lineage_tags
# =============================================================================

REQUIRED_LINEAGE_KEYS = {
    "git_sha",
    "dataset_hash",
    "training_data_version",
    "model_version",
    "model_type",
    "framework",
    "owner",
    "phase",
    "pipeline_stage",
    "risk_level",
    "fairness_checked",
}


def test_compute_lineage_tags_required_keys(tiny_lstm_data):
    """Tags devem cobrir todo o schema obrigatório do GAP 05."""
    X, y = tiny_lstm_data
    tags = compute_lineage_tags(X, y)
    assert REQUIRED_LINEAGE_KEYS.issubset(set(tags.keys()))


def test_compute_lineage_tags_dataset_hash_deterministic(tiny_lstm_data):
    """Mesmos tensores → mesmo dataset_hash."""
    X, y = tiny_lstm_data
    t1 = compute_lineage_tags(X, y)
    t2 = compute_lineage_tags(X, y)
    assert t1["dataset_hash"] == t2["dataset_hash"]


def test_compute_lineage_tags_dataset_hash_changes_with_data(tiny_lstm_data):
    """Tensores diferentes → dataset_hash diferente (sensibilidade)."""
    X, y = tiny_lstm_data
    t1 = compute_lineage_tags(X, y)
    t2 = compute_lineage_tags(X + 1.0, y)
    assert t1["dataset_hash"] != t2["dataset_hash"]


def test_compute_lineage_tags_all_values_are_strings(tiny_lstm_data):
    """MLflow só aceita string em set_tags — todos os values devem ser str."""
    X, y = tiny_lstm_data
    tags = compute_lineage_tags(X, y)
    for k, v in tags.items():
        assert isinstance(v, str), f"tag '{k}' não é string: {type(v).__name__}"


# =============================================================================
# build_lstm
# =============================================================================

def test_build_lstm_output_shape(tiny_lstm_data):
    """Modelo construído deve aceitar (n, time_step, 1) e prever (n, 1)."""
    X, _ = tiny_lstm_data
    model = build_lstm(
        time_step=60, units_1=8, units_2=8,
        dropout_1=0.1, dropout_2=0.1, learning_rate=0.01,
    )
    pred = model.predict(X, verbose=0)
    assert pred.shape == (X.shape[0], 1)


def test_build_lstm_predict_in_finite_range(tiny_lstm_data):
    """Previsões não devem ser NaN/Inf — sanity check de inferência."""
    X, _ = tiny_lstm_data
    model = build_lstm(60, 4, 4, 0.1, 0.1, 0.01)
    pred = model.predict(X, verbose=0)
    assert np.isfinite(pred).all()


# =============================================================================
# evaluate_on_holdout
# =============================================================================

def test_evaluate_on_holdout_returns_prefixed_metrics(
    monkeypatch, tmp_path, fitted_scaler,
):
    """evaluate_on_holdout deve retornar dict com chaves prefixadas 'holdout_'."""
    from src.ibov_pipeline import config as cfg_mod

    rng = np.random.default_rng(0)
    X_h = rng.random((10, 60, 1)).astype(np.float32)
    y_h = rng.random(10).astype(np.float32)

    xp = tmp_path / "X_h.npy"
    yp = tmp_path / "y_h.npy"
    np.save(xp, X_h)
    np.save(yp, y_h)

    monkeypatch.setattr(cfg_mod.cfg.data, "holdout_x_path", str(xp))
    monkeypatch.setattr(cfg_mod.cfg.data, "holdout_y_path", str(yp))

    model = build_lstm(60, 4, 4, 0.1, 0.1, 0.01)
    metrics = evaluate_on_holdout(model, fitted_scaler)

    expected = {
        "holdout_mae",
        "holdout_rmse",
        "holdout_mape",
        "holdout_directional_accuracy",
    }
    assert set(metrics.keys()) == expected


# =============================================================================
# _build_search_grid
# =============================================================================

def test_build_search_grid_respects_max_trials():
    """A grade gerada não deve exceder cfg.tuner.max_trials."""
    from src.ibov_pipeline.config import cfg
    grid = _build_search_grid()
    assert len(grid) <= cfg.tuner.max_trials


def test_build_search_grid_each_trial_has_all_keys():
    """Cada trial deve conter as 5 chaves de hiperparâmetros."""
    grid = _build_search_grid()
    if grid:
        required = {"units_1", "units_2", "dropout_1", "dropout_2", "learning_rate"}
        assert required == set(grid[0].keys())
