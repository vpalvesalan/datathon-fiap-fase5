"""FastAPI — expõe os dois domínios em um único processo HTTP.

Endpoints:
- GET  /health              → smoke check, retorna status dos domínios.
- POST /predict             → previsão direta do LSTM (sem agente).
- POST /agent/query         → pergunta ao agente Copiloto, com guardrails.
- GET  /metrics             → métricas Prometheus (operacionais).
- GET  /chat                → UI Gradio (chat + linha de raciocínio).

Fluxo de /agent/query:
    user_input → InputGuardrail → agente (LLM + tools) → OutputGuardrail → resposta

Tracing:
- **MLflow** (primário) — `mlflow.langchain.autolog()` instrumenta tudo
  automaticamente no startup. Traces ficam visíveis no MLflow UI sob o
  experiment `agent_cfg.observability.mlflow_experiment_name`.
- **Langfuse** (fallback opcional) — se LANGFUSE_PUBLIC_KEY estiver no env,
  callbacks são adicionados em paralelo. Sem env vars, só o MLflow roda.

Uso local:
    uvicorn src.serving.app:app --reload --port 7860
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException

from src.serving.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP (Antes do yield) ---
    from src.ibov_pipeline.logging_config import setup_logging
    from src.monitoring.telemetry import setup_mlflow_tracing

    setup_logging("api_serving")
    setup_mlflow_tracing()
    logger.info("API inicializada e instrumentada.")
    
    yield  # A API escuta as requisições neste ponto
    
    # --- SHUTDOWN (Depois do yield) ---
    logger.info("Desligando a API...")


app = FastAPI(
    title="Copiloto IBOV — API",
    description=(
        "Serviço unificado: previsão LSTM (Domínio 1) e Agente Copiloto "
        "com RAG e tools (Domínio 2). Ver `/docs` para schema interativo."
    ),
    version="0.1.0",
    lifespan=lifespan, # Injeta o gerenciador de ciclo de vida moderno
)


# Singletons criados sob demanda (lazy) para acelerar startup ----------------

_input_guard = None
_output_guard = None
_agent_executor = None
_langfuse_handler = None


def _get_input_guard():
    global _input_guard
    if _input_guard is None:
        from src.security.guardrails import InputGuardrail
        _input_guard = InputGuardrail()
    return _input_guard


def _get_output_guard():
    global _output_guard
    if _output_guard is None:
        from src.security.guardrails import OutputGuardrail
        _output_guard = OutputGuardrail()
    return _output_guard


def _get_agent_executor():
    global _agent_executor
    if _agent_executor is None:
        from src.agent_pipeline.react_agent import create_copiloto_agent
        _agent_executor = create_copiloto_agent()
    return _agent_executor


def _get_langfuse_callbacks() -> list:
    """Retorna callbacks Langfuse SE as credenciais estiverem no env.

    O tracing primário é via MLflow (autolog em startup). Langfuse é uma
    camada adicional opcional — útil se a equipe quiser dashboards específicos
    de LLM. Roda em paralelo ao MLflow, sem conflito.
    """
    global _langfuse_handler
    if _langfuse_handler is None:
        from src.monitoring.telemetry import get_langfuse_callback
        _langfuse_handler = get_langfuse_callback()
        if _langfuse_handler is None:
            return []
    return [_langfuse_handler]


# Endpoints ------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Smoke check — não toca em modelos."""
    return HealthResponse()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Previsão direta do LSTM (sem passar pelo agente)."""
    from src.ibov_pipeline.predict import predict_next_day
    from src.monitoring.telemetry import record_lstm_prediction

    try:
        import numpy as np
        window = np.asarray(req.recent_close_window, dtype=float) if req.recent_close_window else None
        out = predict_next_day(recent_close_window=window)
        record_lstm_prediction()
        return PredictResponse(**out)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest) -> AgentQueryResponse:
    """Pergunta ao agente Copiloto com guardrails de input/output."""
    import time
    from src.agent_pipeline.config import agent_cfg
    from src.monitoring.telemetry import record_agent_query

    t0 = time.perf_counter()
    model_name = agent_cfg.llm.model_name

    # 1) Input guardrail
    ok, reason = _get_input_guard().validate(req.question)
    if not ok:
        record_agent_query(model=model_name, status="blocked", duration_s=time.perf_counter() - t0)
        return AgentQueryResponse(
            answer="",
            intermediate_steps=[],
            model="blocked",
            blocked=True,
            block_reason=reason,
        )

    # 2) Agente
    try:
        callbacks = _get_langfuse_callbacks()
        result = _get_agent_executor().invoke(
            {"input": req.question},
            config={"callbacks": callbacks} if callbacks else None,
        )
    except Exception as exc:  # noqa: BLE001
        record_agent_query(model=model_name, status="error", duration_s=time.perf_counter() - t0)
        logger.exception("Erro no agente.")
        raise HTTPException(status_code=500, detail=f"Erro interno: {exc}")

    raw_answer = result.get("output", "")
    steps = result.get("intermediate_steps", [])

    # 3) Output guardrail (sanitização de PII)
    try:
        clean_answer = _get_output_guard().sanitize(raw_answer)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Output guardrail falhou (%s) — retornando bruto.", exc)
        clean_answer = raw_answer

    record_agent_query(model=model_name, status="ok", duration_s=time.perf_counter() - t0)
    return AgentQueryResponse(
        answer=clean_answer,
        intermediate_steps=[str(s) for s in steps],
        model=model_name,
    )


# ============================================================================
# Métricas Prometheus — endpoint /metrics consumido pelo Grafana local
# ============================================================================

try:
    from prometheus_client import make_asgi_app
    app.mount("/metrics", make_asgi_app())
    logger.info("Endpoint /metrics (Prometheus) montado.")
except ImportError:
    logger.warning("prometheus_client não instalado — /metrics indisponível.")

# ============================================================================
# Montagem da Interface Gráfica (Gradio)
# ============================================================================
import gradio as gr

from src.serving.gradio_app import build_gradio_blocks 
app = gr.mount_gradio_app(app, build_gradio_blocks(), path="/chat")