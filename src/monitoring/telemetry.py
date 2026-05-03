"""Telemetria — MLflow Tracing (primário) + Langfuse (opcional) + Prometheus.

GAP 01 do guia: dashboard de observabilidade ponta a ponta.

Arquitetura de tracing:
- **MLflow Tracing** (primário) — `mlflow.langchain.autolog()` instrumenta
  TODAS as chamadas LangChain (LLM, tools, retriever) automaticamente. Os
  traces aparecem no MLflow UI sob o experiment configurado em
  `agent_cfg.observability.mlflow_experiment_name`. Mesma URI do treino LSTM —
  single pane of glass para o ciclo MLOps inteiro.
- **Langfuse** (fallback opcional) — ativa em paralelo se LANGFUSE_PUBLIC_KEY
  estiver no env. Útil se a equipe quiser dashboards específicos de LLM
  (custos por modelo, A/B de prompts) sem perder o tracing local do MLflow.
- **Prometheus** — métricas operacionais custom (queries/min, latência, erros).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# =============================================================================
# MLflow Tracing — backend primário
# =============================================================================

def setup_mlflow_tracing(experiment_name: str | None = None) -> bool:
    """Habilita tracing automático de LangChain via MLflow.

    Após chamar, qualquer `agent.invoke()` / `llm.invoke()` / retriever
    é registrado como trace hierárquico no MLflow UI, com inputs, outputs,
    latência e tokens por span.

    Args:
        experiment_name: Override do experimento. None usa o do agent_cfg.

    Returns:
        True se autolog foi habilitado, False em caso de erro.
    """
    try:
        import mlflow

        from src.agent_pipeline.config import agent_cfg
        from src.ibov_pipeline.config import cfg

        if not agent_cfg.observability.enable_mlflow_tracing:
            logger.info("MLflow tracing desabilitado em agent_config.yaml.")
            return False

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        exp_name = experiment_name or agent_cfg.observability.mlflow_experiment_name
        mlflow.set_experiment(exp_name)

        # Auto-instrumenta todas as classes LangChain (ChatGroq, AgentExecutor,
        # Retriever, Tools). Idempotente — chamadas repetidas não duplicam.
        mlflow.langchain.autolog()

        logger.info(
            "MLflow tracing ativo | tracking_uri=%s | experiment=%s",
            cfg.mlflow.tracking_uri, exp_name,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow tracing indisponível (%s).", exc)
        return False


# =============================================================================
# Langfuse — instrumentação automática via callbacks LangChain (FALLBACK)
# =============================================================================

def get_langfuse_callback():
    """Retorna o CallbackHandler do Langfuse, ou None se não configurado.

    Habilita-se apenas quando LANGFUSE_PUBLIC_KEY e LANGFUSE_SECRET_KEY
    estão no env. Roda em paralelo ao MLflow tracing — nenhum conflito.
    Cole esse handler em `config.callbacks` ao invocar o agente.
    """
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None

    try:
        # Compatibilidade entre versões do langfuse:
        # v3+: from langfuse.langchain import CallbackHandler
        # v2.x: from langfuse.callback import CallbackHandler
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError:
            from langfuse.callback import CallbackHandler
        return CallbackHandler()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse indisponível (%s).", exc)
        return None


# =============================================================================
# Prometheus — métricas custom (operacionais)
# =============================================================================

# Definidas como módulo-level para que múltiplos endpoints reusem o mesmo
# registry. Lazy import para não pesar startup quando prometheus_client
# não estiver instalado.

_PROM_METRICS: dict = {}


def _ensure_prometheus():
    if _PROM_METRICS:
        return _PROM_METRICS

    try:
        from prometheus_client import Counter, Histogram

        _PROM_METRICS["agent_queries_total"] = Counter(
            "copiloto_agent_queries_total",
            "Total de queries enviadas ao agente.",
            ["model", "status"],  # status = ok | blocked | error
        )
        _PROM_METRICS["agent_latency_seconds"] = Histogram(
            "copiloto_agent_latency_seconds",
            "Latência de uma query do agente (segundos).",
        )
        _PROM_METRICS["lstm_predictions_total"] = Counter(
            "copiloto_lstm_predictions_total",
            "Total de previsões LSTM servidas.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prometheus não instalado: %s", exc)

    return _PROM_METRICS


def record_agent_query(model: str, status: str, duration_s: float) -> None:
    """Conveniência para o endpoint /agent/query."""
    metrics = _ensure_prometheus()
    if "agent_queries_total" in metrics:
        metrics["agent_queries_total"].labels(model=model, status=status).inc()
        metrics["agent_latency_seconds"].observe(duration_s)


def record_lstm_prediction() -> None:
    """Conveniência para o endpoint /predict."""
    metrics = _ensure_prometheus()
    if "lstm_predictions_total" in metrics:
        metrics["lstm_predictions_total"].inc()
