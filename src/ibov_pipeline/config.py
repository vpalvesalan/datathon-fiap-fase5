# src/ibov_pipeline/config.py
"""Carrega e valida o model_config.yaml usando Pydantic Settings.

Uso:
    from src.ibov_pipeline.config import cfg
    print(cfg.data.symbol)
    print(cfg.lstm.units_1)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Modelos de validação (um por seção do YAML)
# =============================================================================

class DataConfig(BaseModel):
    """Configurações de ingestão, caminhos de arquivos (raw/processed) e tickers."""

    symbol: str = "^BVSP"
    start_date: str = "2019-01-01"
    end_date: str | None = None           # None → usa data de hoje
    raw_path: str = "data/raw/ibov/ibov_close.csv"
    processed_x_path: str = "data/processed/ibov/X.npy"
    processed_y_path: str = "data/processed/ibov/y.npy"
    scaler_path: str = "data/processed/ibov/scaler.joblib"
    holdout_x_path: str = "data/holdout_test/X_holdout.npy"
    holdout_y_path: str = "data/holdout_test/y_holdout.npy"


class FeaturesConfig(BaseModel):
    """Parâmetros de engenharia de features, como tamanho da janela temporal e escala."""

    time_step: int = Field(60, gt=0)
    feature_range: list[float] = [0.0, 1.0]

    @field_validator("feature_range")
    @classmethod
    def validate_range(cls, v: list[float]) -> list[float]:
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("feature_range deve ser [min, max] com min < max.")
        return v


class SplitConfig(BaseModel):
    """Proporções para divisão de dados entre treino, teste e holdout (validação final)."""

    train_ratio: float = Field(0.80, gt=0, lt=1)
    holdout_ratio: float = Field(0.10, gt=0, lt=1)
    validation_split: float = Field(0.20, gt=0, lt=1)


class BaselineModelSpec(BaseModel):
    """Especificação técnica de um modelo scikit-learn para comparação."""

    name: str
    class_: str = Field(alias="class")
    params: dict[str, Any] = {}

    model_config = {"populate_by_name": True}


class BaselineConfig(BaseModel):
    """Lista de modelos de baseline a serem executados no pipeline."""

    models: list[BaselineModelSpec] = []


class LSTMConfig(BaseModel):
    """Hiperparâmetros da arquitetura e do treinamento da rede neural LSTM."""

    # Campos OBRIGATÓRIOS
    units_1: int = Field(..., gt=0)
    units_2: int = Field(..., gt=0)
    dropout_1: float = Field(..., ge=0, le=1)
    dropout_2: float = Field(..., ge=0, le=1)
    learning_rate: float = Field(..., gt=0)
    
    # Campos com VALOR PADRÃO (Podem ser omitidos no YAML)
    loss: str = "mean_squared_error"
    epochs: int = Field(100, gt=0)
    batch_size: int = Field(32, gt=0)
    patience: int = Field(10, gt=0)


class TunerConfig(BaseModel):
    """Parâmetros para o processo de ajuste automático (Hyperparameter Tuning)."""

    enabled: bool = False
    max_trials: int = 10
    executions_per_trial: int = 1
    directory: str = "data/tuning_dir"
    project_name: str = "ibov_lstm_tuning"
    units_min: int = 32
    units_max: int = 128
    units_step: int = 32
    dropout_min: float = 0.1
    dropout_max: float = 0.5
    dropout_step: float = 0.1
    learning_rates: list[float] = [0.01, 0.001, 0.0001]


class MLflowConfig(BaseModel):
    """Configurações para log de métricas e versionamento de experimentos no MLflow."""

    experiment_name: str = "ibov-forecasting"
    tracking_uri: str = "mlruns"


class RegistryConfig(BaseModel):
    """Metadados para registro do modelo em catálogo, incluindo nível de risco e dono."""

    model_name: str = "ibov-lstm-predictor"
    risk_level: str = "medium"
    owner: str = "grupo-xx"
    model_config = {"protected_namespaces": ()}
    @field_validator("risk_level")
    @classmethod
    def validate_risk(cls, v: str) -> str:
        allowed = {"low", "medium", "high", "critical"}
        if v not in allowed:
            raise ValueError(f"risk_level deve ser um de {allowed}.")
        return v


class RetrainConfig(BaseModel):
    """Critérios de performance e drift para gatilhos de retreinamento automático."""

    promotion_delta_mape: float = Field(0.005, gt=0)
    retrain_trigger_psi: float = Field(0.20, gt=0)


# =============================================================================
# Config raiz
# =============================================================================

class Config(BaseModel):
    """Estrutura raiz que consolida todas as seções de configuração do projeto."""

    data: DataConfig
    features: FeaturesConfig
    split: SplitConfig
    baseline: BaselineConfig
    lstm: LSTMConfig
    tuner: TunerConfig
    mlflow: MLflowConfig
    registry: RegistryConfig
    retrain: RetrainConfig


# =============================================================================
# Raiz do projeto e resolução de paths
# =============================================================================

# config.py vive em <root>/src/ibov_pipeline/config.py → parents[2] = <root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _resolve(p: str) -> str:
    """Resolve um path relativo contra PROJECT_ROOT.

    - Strings com esquema URL (ex: 'http://', 'sqlite://') são preservadas.
    - Paths absolutos são preservados.
    - Paths relativos viram absolutos ancorados em PROJECT_ROOT.

    Garante que o pipeline funcione independente do cwd (notebook, CI, DVC).
    """
    if "://" in p:
        return p
    path = Path(p)
    return str(path) if path.is_absolute() else str(PROJECT_ROOT / path)


# =============================================================================
# Carregamento
# =============================================================================

def _find_config_path() -> Path:
    """Localiza o model_config.yaml subindo a árvore de diretórios."""

    # Permite override via variável de ambiente
    env_path = os.getenv("MODEL_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    # Busca padrão: sobe até encontrar configs/model_config.yaml
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / "configs" / "model_config.yaml"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "model_config.yaml não encontrado. "
        "Defina MODEL_CONFIG_PATH ou coloque o arquivo em configs/model_config.yaml."
    )


def load_config(path: Path | str | None = None) -> Config:
    """Carrega e valida o model_config.yaml.

    Args:
        path: Caminho explícito para o YAML. Se None, busca automaticamente.

    Returns:
        Config validada com Pydantic. Todos os paths são absolutos.
    """
    config_path = Path(path) if path else _find_config_path()

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config.model_validate(raw)

    # Normaliza paths relativos contra PROJECT_ROOT. Sem isso, rodar de
    # notebooks/ ou de CI cria diretórios fantasmas (notebooks/data/, etc).
    for field in (
        "raw_path",
        "processed_x_path",
        "processed_y_path",
        "scaler_path",
        "holdout_x_path",
        "holdout_y_path",
    ):
        setattr(cfg.data, field, _resolve(getattr(cfg.data, field)))

    cfg.tuner.directory = _resolve(cfg.tuner.directory)
    cfg.mlflow.tracking_uri = _resolve(cfg.mlflow.tracking_uri)

    return cfg


# Instância global — importar com `from src.ibov_pipeline.config import cfg`
cfg: Config = load_config()
