"""Rate limiter para respeitar o budget de tokens/min do provedor LLM.

Estratégia simples: estimar tokens por chamada e pausar entre chamadas o
suficiente para a média não exceder o limite. Mais previsível que rastrear
tokens reais via metadata (que exigiria parsing de cada response e somar
ao longo de uma janela deslizante de 60s).

Para o caso do Datathon (Groq free tier, 12k tok/min), a estimativa é
conservadora o suficiente. Quando migrar para tier pago, basta aumentar
`tokens_per_minute` no `agent_config.yaml::llm.rate_limit`.

Uso:
    from evaluation._rate_limit import from_config

    limiter = from_config()
    for item in items:
        limiter.wait_if_needed()    # bloqueia o necessário
        do_llm_call(item)
"""
from __future__ import annotations

import logging
import time

from src.agent_pipeline.config import agent_cfg

logger = logging.getLogger(__name__)


class TokenRateLimiter:
    """Pausa entre chamadas para respeitar tokens/minuto do provedor.

    Algoritmo: cada chamada custa ~`estimated_tokens_per_call * safety_margin`
    tokens. O budget por segundo é `tokens_per_minute / 60`. Logo, o intervalo
    mínimo entre chamadas é `tokens_consumidos / budget_por_seg`.

    Args:
        tokens_per_minute: Limite duro do provedor.
        estimated_tokens_per_call: Estimativa de quanto cada iteração consome.
        safety_margin: Folga (1.5 = 50% acima do estimado, conservador).
    """

    def __init__(
        self,
        tokens_per_minute: int,
        estimated_tokens_per_call: int,
        safety_margin: float = 1.5,
    ):
        self.tokens_per_minute = tokens_per_minute
        self.estimated_tokens_per_call = estimated_tokens_per_call
        self.safety_margin = safety_margin
        self._last_call_at: float | None = None

    @property
    def seconds_per_call(self) -> float:
        """Intervalo mínimo (em segundos) entre chamadas."""
        budget_per_sec = self.tokens_per_minute / 60.0
        return (self.estimated_tokens_per_call * self.safety_margin) / budget_per_sec

    def wait_if_needed(self) -> float:
        """Bloqueia o necessário para respeitar o budget. Retorna segs dormidos.

        A primeira chamada nunca pausa (ainda há budget total disponível).
        Da segunda em diante, calcula o tempo decorrido desde a última e
        completa o restante via time.sleep.
        """
        now = time.monotonic()

        if self._last_call_at is None:
            self._last_call_at = now
            return 0.0

        elapsed = now - self._last_call_at
        deficit = self.seconds_per_call - elapsed

        if deficit > 0:
            logger.info(
                "Rate limit: aguardando %.1fs para respeitar %d tok/min "
                "(estimado %d tok/chamada × margem %.1f).",
                deficit,
                self.tokens_per_minute,
                self.estimated_tokens_per_call,
                self.safety_margin,
            )
            time.sleep(deficit)
            now = time.monotonic()

        self._last_call_at = now
        return max(0.0, deficit)


def from_config() -> TokenRateLimiter:
    """Cria um TokenRateLimiter a partir de `agent_cfg.llm.rate_limit`."""
    rl = agent_cfg.llm.rate_limit
    return TokenRateLimiter(
        tokens_per_minute=rl.tokens_per_minute,
        estimated_tokens_per_call=rl.estimated_tokens_per_call,
        safety_margin=rl.safety_margin,
    )
