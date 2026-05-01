"""Testes de validação de schema (pandera) e I/O em make_dataset.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

from src.ibov_pipeline.make_dataset import load_raw, save_raw, validate_schema


# =============================================================================
# Validação de schema
# =============================================================================

def test_validate_schema_accepts_valid_data(synthetic_ohlc_df):
    """DataFrame com Close > 0, sem nulls, deve passar."""
    result = validate_schema(synthetic_ohlc_df)
    assert "Close" in result.columns
    assert len(result) == len(synthetic_ohlc_df)


def test_validate_schema_rejects_null_close():
    """Null em Close deve disparar pandera SchemaError."""
    df = pd.DataFrame(
        {"Close": [100.0, None, 102.0]},
        index=pd.date_range("2022-01-01", periods=3, freq="B", tz="UTC"),
    )
    df.index.name = "Date"
    with pytest.raises(pa.errors.SchemaError):
        validate_schema(df)


def test_validate_schema_rejects_non_positive_close():
    """Preço ≤ 0 deve disparar pandera SchemaError."""
    df = pd.DataFrame(
        {"Close": [100.0, -50.0, 102.0]},
        index=pd.date_range("2022-01-01", periods=3, freq="B", tz="UTC"),
    )
    df.index.name = "Date"
    with pytest.raises(pa.errors.SchemaError):
        validate_schema(df)


# =============================================================================
# I/O
# =============================================================================

def test_save_raw_creates_file(tmp_path, synthetic_ohlc_df):
    csv_path = tmp_path / "ibov_close.csv"
    save_raw(synthetic_ohlc_df, csv_path)
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0


def test_save_raw_creates_parent_dirs(tmp_path, synthetic_ohlc_df):
    """save_raw deve criar diretórios pai se não existirem."""
    nested = tmp_path / "deep" / "nested" / "ibov.csv"
    save_raw(synthetic_ohlc_df, nested)
    assert nested.exists()


def test_load_raw_roundtrip(tmp_path, synthetic_ohlc_df):
    """save + load devem preservar a coluna Close."""
    csv_path = tmp_path / "ibov_close.csv"
    save_raw(synthetic_ohlc_df, csv_path)
    loaded = load_raw(csv_path)

    assert "Close" in loaded.columns
    assert len(loaded) == len(synthetic_ohlc_df)
    # Tolerância de float devido a serialização CSV
    assert abs(loaded["Close"].iloc[0] - synthetic_ohlc_df["Close"].iloc[0]) < 1e-6
