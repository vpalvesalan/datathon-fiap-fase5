"""Testes de carga e normalização de paths em config.py."""
from __future__ import annotations

from pathlib import Path

from src.ibov_pipeline.config import PROJECT_ROOT, _resolve, cfg


# =============================================================================
# Resolução de paths
# =============================================================================

def test_resolve_relative_becomes_absolute():
    """Path relativo deve virar absoluto, ancorado em PROJECT_ROOT."""
    result = _resolve("data/x.npy")
    assert Path(result).is_absolute()
    assert result == str(PROJECT_ROOT / "data" / "x.npy")


def test_resolve_absolute_path_unchanged():
    """Path absoluto não deve ser modificado."""
    abs_path = "/tmp/abs.npy"
    assert _resolve(abs_path) == abs_path


def test_resolve_url_passthrough():
    """Strings com esquema URL não devem ser tocadas."""
    assert _resolve("http://mlflow.local/api") == "http://mlflow.local/api"
    assert _resolve("sqlite:///mlflow.db") == "sqlite:///mlflow.db"
    assert _resolve("s3://bucket/path") == "s3://bucket/path"


# =============================================================================
# Instância global cfg
# =============================================================================

def test_cfg_data_paths_are_absolute():
    """Todos os paths de data/* devem ser absolutos pós-load."""
    assert Path(cfg.data.raw_path).is_absolute()
    assert Path(cfg.data.processed_x_path).is_absolute()
    assert Path(cfg.data.processed_y_path).is_absolute()
    assert Path(cfg.data.scaler_path).is_absolute()
    assert Path(cfg.data.holdout_x_path).is_absolute()
    assert Path(cfg.data.holdout_y_path).is_absolute()


def test_cfg_tracking_uri_is_absolute():
    """tracking_uri sem esquema URL deve ser absoluto."""
    if "://" not in cfg.mlflow.tracking_uri:
        assert Path(cfg.mlflow.tracking_uri).is_absolute()


def test_cfg_tuner_directory_is_absolute():
    assert Path(cfg.tuner.directory).is_absolute()


def test_cfg_basic_invariants():
    """Sanity check em valores essenciais do config."""
    assert cfg.features.time_step > 0
    assert 0 < cfg.split.train_ratio < 1
    assert 0 < cfg.split.holdout_ratio < 1
    assert cfg.lstm.units_1 > 0
    assert cfg.lstm.batch_size > 0
