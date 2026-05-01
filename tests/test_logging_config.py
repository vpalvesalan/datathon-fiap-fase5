"""Testes do logging dual (console + arquivo) em logging_config.py."""
from __future__ import annotations

import logging
from pathlib import Path

from src.ibov_pipeline.logging_config import LOG_DIR, setup_logging


def test_setup_logging_creates_file():
    """setup_logging deve criar um arquivo .log no LOG_DIR."""
    log_path = setup_logging("pytest_smoke")
    assert log_path.exists()
    assert log_path.parent == LOG_DIR
    assert log_path.suffix == ".log"
    log_path.unlink()  # cleanup


def test_setup_logging_writes_to_file():
    """Mensagens via logger devem aparecer no arquivo, não só no console."""
    log_path = setup_logging("pytest_write")
    logging.getLogger("test_module").info("payload de auditoria 12345")

    # Força flush dos handlers
    for h in logging.getLogger().handlers:
        h.flush()

    content = log_path.read_text(encoding="utf-8")
    assert "payload de auditoria 12345" in content
    assert "test_module" in content
    log_path.unlink()


def test_setup_logging_idempotent():
    """Chamar 2x não deve duplicar handlers (evita logs em duplicidade)."""
    setup_logging("pytest_idem_1")
    n1 = len(logging.getLogger().handlers)
    log_path = setup_logging("pytest_idem_2")
    n2 = len(logging.getLogger().handlers)

    assert n1 == n2  # mesmo número de handlers após segunda chamada
    log_path.unlink()
    # cleanup do primeiro
    for f in LOG_DIR.glob("pytest_idem_1_*.log"):
        f.unlink()


def test_setup_logging_filename_contains_run_name():
    """O nome do arquivo deve conter o run_name informado."""
    log_path = setup_logging("custom_run_label")
    assert "custom_run_label" in log_path.name
    log_path.unlink()


def test_log_dir_is_under_project_root():
    """LOG_DIR deve estar ancorado em PROJECT_ROOT (não em cwd)."""
    from src.ibov_pipeline.config import PROJECT_ROOT
    assert LOG_DIR.parent == PROJECT_ROOT
