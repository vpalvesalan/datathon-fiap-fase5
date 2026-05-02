"""Modelos Pydantic dos endpoints — fronteira de validação da API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    domains: dict[str, str] = Field(
        default_factory=lambda: {
            "lstm_predict": "ok",
            "agent_query": "ok",
        }
    )


class PredictRequest(BaseModel):
    """Input opcional para sobrescrever a janela default (últimos 60 do CSV)."""

    recent_close_window: list[float] | None = Field(
        default=None,
        description="Lista 1D dos últimos N preços de fechamento. "
                    "Omitir para usar o CSV bruto.",
    )


class PredictResponse(BaseModel):
    predicted_close: float
    last_close: float
    delta_pct: float
    model_version: str | None = None


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)


class AgentQueryResponse(BaseModel):
    answer: str
    intermediate_steps: list[Any] = Field(default_factory=list)
    model: str
    blocked: bool = False
    block_reason: str | None = None
