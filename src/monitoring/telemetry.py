"""Telemetria — Langfuse (qualidade LLM) + Prometheus (operacional).

GAP 01 do guia: dashboard de observabilidade ponta a ponta.

- Langfuse: faithfulness, latência por chamada, custo por modelo, traces
  do agente ReAct (Thought→Action→Observation auditável). Habilitado
  automaticamente quando LANGFUSE_PUBLIC_KEY está no env.
- Prometheus: contadores e histogramas operacionais expostos em /metrics
  pelo FastAPI (futuro), usados pelo Grafana via docker-compose.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# =============================================================================
# Langfuse — instrumentação automática via callbacks LangChain
# =============================================================================

def get_langfuse_callback():
    """Retorna o CallbackHandler do Langfuse, ou None se não configurado.

    Cole esse handler em config.callbacks ao chamar o agente para que
    todos os passos (LLM, tool calls, retriever) virem traces auditáveis.
    """
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        logger.info("Langfuse não configurado (sem env vars).")
        return None

    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse indisponível: %s", exc)
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
