"""Configuração centralizada de logging para o pipeline IBOV.

Cada entrypoint (`make_dataset`, `feature_engineering`, `train_baseline`,
`train_lstm`, `register_model`, `retrain`) chama `setup_logging(run_name)`
no `__main__` para obter:

- **Console (stdout)**: log INFO+ formatado para humanos / CI.
- **Arquivo `logs/<run_name>_<YYYYMMDD-HHMMSS>.log`**: trace completo
  do run, persistente, para auditoria e debug pós-mortem.

Os arquivos são por execução (timestamp no nome), facilitando associar
um log a um run_id do MLflow no momento do incidente. O diretório `logs/`
é ignorado pelo git (ver `.gitignore`).

Uso:
    from src.ibov_pipeline.logging_config import setup_logging

    if __name__ == "__main__":
        setup_logging("train_lstm")
        run()
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.ibov_pipeline.config import PROJECT_ROOT

LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logging(run_name: str, level: int = logging.INFO) -> Path:
    """Configura logging dual (console + arquivo) para um run do pipeline.

    Args:
        run_name: Identificador do executor (ex: 'train_lstm', 'baseline').
        level: Nível mínimo de log (default INFO).

    Returns:
        Path do arquivo de log criado, útil para incluir em mlflow.log_artifact().
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = LOG_DIR / f"{run_name}_{timestamp}.log"

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)
    # Idempotência: limpa handlers prévios (evita duplicação se chamado 2x)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root.info("Logging iniciado | arquivo: %s", log_file)
    return log_file
