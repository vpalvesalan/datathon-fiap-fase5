# src/ibov_pipeline/make_dataset.py
"""Coleta os dados do IBOV via yfinance, valida o schema e persiste em CSV.

Uso direto:
    python -m src.ibov_pipeline.make_dataset

Uso programático:
    from src.ibov_pipeline.make_dataset import run
    df = run()
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import yfinance as yf
from pandera.pandas import Column, DataFrameSchema

from src.ibov_pipeline.config import cfg

logger = logging.getLogger(__name__)

# =============================================================================
# Schema de validação — contrato de dados do CSV bruto
# =============================================================================

# Checa preço final acima de 0 e ausência de Nulos.
RAW_SCHEMA = DataFrameSchema(
    columns={
        "Close": Column(
            float,
            checks=[
                pa.Check.gt(0, error="Preço de fechamento deve ser positivo."),
                pa.Check(lambda s: s.isna().sum() == 0, error="Close não pode ter nulos."),
            ],
        )
    },
    index=pa.Index(pd.DatetimeTZDtype(tz="UTC"), name="Date", coerce=True),
    coerce=True,
)


# =============================================================================
# Funções principais
# =============================================================================

def download_ibov(
    symbol: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Baixa a série histórica do IBOV via yfinance.

    Args:
        symbol: Ticker (ex: '^BVSP').
        start_date: Data de início no formato 'YYYY-MM-DD'.
        end_date: Data de fim. None usa a data de hoje.

    Returns:
        DataFrame com coluna 'Close' e índice DatetimeIndex.

    Raises:
        ValueError: Se o download retornar DataFrame vazio.
    """
    end = end_date or str(date.today())
    logger.info("Baixando %s de %s até %s...", symbol, start_date, end)

    raw = yf.download(symbol, start=start_date, end=end, progress=False)

    if raw.empty:
        raise ValueError(
            f"yfinance retornou DataFrame vazio para {symbol} "
            f"({start_date} → {end}). Verifique o ticker e as datas."
        )

    # MultiIndex de colunas (yfinance >= 0.2) → achatar
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].dropna()
    logger.info("Download concluído: %d registros.", len(df))
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame contra o contrato de schema definido.

    Args:
        df: DataFrame bruto.

    Returns:
        DataFrame validado (mesmo objeto se passou).

    Raises:
        pandera.errors.SchemaError: Se alguma validação falhar.
    """
    logger.info("Validando schema...")
    validated = RAW_SCHEMA.validate(df)
    logger.info("Schema validado com sucesso.")
    return validated


def save_raw(df: pd.DataFrame, path: str | Path) -> None:
    """Salva o DataFrame bruto em CSV, criando diretórios se necessário.

    Args:
        df: DataFrame validado.
        path: Caminho de destino.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest)
    logger.info("CSV salvo em: %s", dest)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Carrega o CSV bruto do disco.

    Args:
        path: Caminho para o CSV.

    Returns:
        DataFrame com índice DatetimeIndex.
    """
    return pd.read_csv(path, index_col="Date", parse_dates=True)


# =============================================================================
# Entrypoint
# =============================================================================

def run(force_download: bool = False) -> pd.DataFrame:
    """Executa o pipeline de coleta completo.

    Se o arquivo já existir e force_download=False, carrega do disco.
    Caso contrário, baixa, valida e salva.

    Args:
        force_download: Forçar novo download mesmo com arquivo existente.

    Returns:
        DataFrame com a série histórica do IBOV.
    """
    raw_path = Path(cfg.data.raw_path)

    if raw_path.exists() and not force_download:
        logger.info("Arquivo local encontrado em %s. Carregando...", raw_path)
        return load_raw(raw_path)

    df = download_ibov(
        symbol=cfg.data.symbol,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    df = validate_schema(df)
    save_raw(df, raw_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    df = run()
    print(f"\nRegistros coletados: {len(df)}")
    print(df.head())
