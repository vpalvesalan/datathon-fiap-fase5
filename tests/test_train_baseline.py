"""Testes de helpers públicos de train_baseline.py.

Não testa train_single_baseline / run() — esses precisam de MLflow ativo
e estão na camada de testes de integração.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from src.ibov_pipeline.train_baseline import _flatten_for_sklearn, _load_class


# =============================================================================
# _load_class
# =============================================================================

def test_load_class_returns_correct_class():
    cls = _load_class("sklearn.linear_model.LinearRegression")
    assert cls is LinearRegression


def test_load_class_works_for_ridge():
    cls = _load_class("sklearn.linear_model.Ridge")
    assert cls is Ridge


def test_load_class_instance_is_usable():
    """Classe importada deve ser instanciável e ter API sklearn."""
    cls = _load_class("sklearn.linear_model.LinearRegression")
    model = cls()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


# =============================================================================
# _flatten_for_sklearn
# =============================================================================

def test_flatten_for_sklearn_shape():
    """(n, time_step, 1) deve virar (n, time_step)."""
    X = np.zeros((50, 60, 1))
    flat = _flatten_for_sklearn(X)
    assert flat.shape == (50, 60)


def test_flatten_for_sklearn_preserves_values():
    """Achatamento não deve corromper a ordem dos valores."""
    X = np.arange(60).reshape(1, 60, 1).astype(float)
    flat = _flatten_for_sklearn(X)
    np.testing.assert_array_equal(flat[0], np.arange(60))


def test_flatten_for_sklearn_handles_batch():
    """Shape preservado em batch maior."""
    rng = np.random.default_rng(0)
    X = rng.random((100, 60, 1))
    flat = _flatten_for_sklearn(X)
    assert flat.shape == (100, 60)
    # Conteúdo da primeira janela deve ser idêntico ao original (sem 3ª dim)
    np.testing.assert_array_equal(flat[0], X[0, :, 0])
