"""Testes do agente — config + tools + smoke do agente.

Os testes que requerem Groq (LLM real) ou ChromaDB com PDFs são marcados
como integration e pulam quando as deps não estão instaladas / configuradas.
"""
from __future__ import annotations

import pytest


# =============================================================================
# Config do agente — sem deps pesadas
# =============================================================================

def test_agent_config_loads():
    from src.agent_pipeline.config import agent_cfg

    assert agent_cfg.llm.provider == "groq"
    assert agent_cfg.llm.temperature >= 0.0
    assert agent_cfg.embeddings.chunk_size > 0
    assert agent_cfg.rag.retrieval_k > 0


def test_agent_config_paths_are_absolute():
    from pathlib import Path
    from src.agent_pipeline.config import agent_cfg

    assert Path(agent_cfg.vector_store.persist_directory).is_absolute()
    assert Path(agent_cfg.golden_set.path).is_absolute()
    assert Path(agent_cfg.agent_docs.raw_path).is_absolute()


# =============================================================================
# Calculator tool — puro Python, sem deps externas
# =============================================================================

langchain = pytest.importorskip("langchain", reason="langchain não instalado")


def test_calculator_basic_arithmetic():
    from src.agent_pipeline.tools import _calculator_tool

    tool = _calculator_tool()
    out = tool.func("(130000 - 128500) / 128500 * 100")
    assert "Resultado:" in out


def test_calculator_blocks_non_arithmetic():
    from src.agent_pipeline.tools import _calculator_tool

    tool = _calculator_tool()
    out = tool.func("__import__('os').system('rm -rf /')")
    assert "Erro" in out


def test_get_default_tools_has_at_least_three():
    """O guia exige ≥ 3 tools. Garantimos isso na assinatura."""
    from src.agent_pipeline.tools import get_default_tools

    tools = get_default_tools()
    assert len(tools) >= 3
    names = {t.name for t in tools}
    # As 4 tools intencionais devem estar todas presentes
    assert {"ibov_forecast", "macro_rag", "calculator", "market_context"}.issubset(names)


def test_each_tool_has_description():
    """Todas as tools devem ter descrição não-vazia (LangChain usa pra rotear)."""
    from src.agent_pipeline.tools import get_default_tools

    for tool in get_default_tools():
        assert tool.description and len(tool.description) > 20


# =============================================================================
# Agente completo — integração (pula sem GROQ_API_KEY)
# =============================================================================

@pytest.mark.integration
def test_create_agent_requires_api_key(monkeypatch):
    """Sem GROQ_API_KEY, deve falhar com erro claro."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from src.agent_pipeline.react_agent import create_copiloto_agent

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        create_copiloto_agent()
