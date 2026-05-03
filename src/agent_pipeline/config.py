"""Carga e validação do agent_config.yaml com Pydantic Settings.

Espelha o padrão do `src.ibov_pipeline.config` para manter consistência.
Paths relativos são resolvidos contra PROJECT_ROOT — independente de cwd.

Uso:
    from src.agent_pipeline.config import agent_cfg
    print(agent_cfg.llm.model_name)
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.ibov_pipeline.config import PROJECT_ROOT, _resolve


# =============================================================================
# Modelos por seção do YAML
# =============================================================================

class RateLimitConfig(BaseModel):
    """Configuração de rate limit do provedor LLM.

    Usado por scripts de evaluation para pausar entre iterações e respeitar
    a janela de tokens/min do provedor (especialmente Groq free tier).
    """
    tokens_per_minute: int = Field(12000, gt=0)
    estimated_tokens_per_call: int = Field(5000, gt=0)
    safety_margin: float = Field(1.5, gt=0)


class LLMConfig(BaseModel):
    provider: str = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, gt=0)
    max_iterations: int = Field(10, gt=0)
    rate_limit: RateLimitConfig = RateLimitConfig()


class ObservabilityConfig(BaseModel):
    """Tracing do agente — MLflow primário, Langfuse opcional."""
    mlflow_experiment_name: str = "agent-tracing"
    enable_mlflow_tracing: bool = True


class EmbeddingsConfig(BaseModel):
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = Field(1000, gt=0)
    chunk_overlap: int = Field(200, ge=0)


class VectorStoreConfig(BaseModel):
    persist_directory: str = "data/processed/agent_db"
    collection_name: str = "ibov_macro_docs"


class RAGConfig(BaseModel):
    retrieval_k: int = Field(4, gt=0)


class GuardrailsConfig(BaseModel):
    max_input_length: int = Field(4096, gt=0)
    language: str = "pt"
    pii_entities: list[str] = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "BR_CPF"]


class GoldenSetConfig(BaseModel):
    path: str = "data/golden_set/perguntas.json"


class AgentDocsConfig(BaseModel):
    raw_path: str = "data/raw/agent_docs"


class EvaluationConfig(BaseModel):
    ragas_output: str = "evaluation/results/ragas_metrics.json"
    judge_output: str = "evaluation/results/llm_judge.json"
    ab_test_output: str = "evaluation/results/ab_test_prompts.json"
    benchmark_output: str = "evaluation/results/benchmark_llm.json"


# =============================================================================
# Config raiz
# =============================================================================

class AgentConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    rag: RAGConfig = RAGConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()
    golden_set: GoldenSetConfig = GoldenSetConfig()
    agent_docs: AgentDocsConfig = AgentDocsConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


# =============================================================================
# Carregamento
# =============================================================================

def _find_config_path() -> Path:
    """Localiza agent_config.yaml. AGENT_CONFIG_PATH override é respeitado."""
    env_path = os.getenv("AGENT_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    # Padrão: src/agent_pipeline/configs/agent_config.yaml
    default = PROJECT_ROOT / "src" / "agent_pipeline" / "configs" / "agent_config.yaml"
    if default.exists():
        return default

    raise FileNotFoundError(
        "agent_config.yaml não encontrado. Defina AGENT_CONFIG_PATH ou "
        "coloque em src/agent_pipeline/configs/agent_config.yaml."
    )


def load_agent_config(path: Path | str | None = None) -> AgentConfig:
    """Carrega e valida o YAML; normaliza paths para absolutos."""
    config_path = Path(path) if path else _find_config_path()

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = AgentConfig.model_validate(raw)

    # Normaliza paths contra PROJECT_ROOT (mesmo padrão do ibov_pipeline)
    cfg.vector_store.persist_directory = _resolve(cfg.vector_store.persist_directory)
    cfg.golden_set.path = _resolve(cfg.golden_set.path)
    cfg.agent_docs.raw_path = _resolve(cfg.agent_docs.raw_path)
    cfg.evaluation.ragas_output = _resolve(cfg.evaluation.ragas_output)
    cfg.evaluation.judge_output = _resolve(cfg.evaluation.judge_output)
    cfg.evaluation.ab_test_output = _resolve(cfg.evaluation.ab_test_output)
    cfg.evaluation.benchmark_output = _resolve(cfg.evaluation.benchmark_output)

    return cfg


# Instância global — `from src.agent_pipeline.config import agent_cfg`
agent_cfg: AgentConfig = load_agent_config()
