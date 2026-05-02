"""Ferramentas customizadas do agente ReAct (≥ 3 obrigatórias pelo guia).

Implementa 4 tools intencionalmente — redundância para o Demo Day caso
uma tool dependente de rede falhe ao vivo:

1. ibov_forecast    — chama o LSTM via Model Registry (Domínio 1).
2. macro_rag        — busca em relatórios Focus/Copom (RAG ChromaDB).
3. calculator       — aritmética determinística (evita alucinação numérica).
4. market_context   — cotações live de USD/BRL, S&P500, Brent (yfinance).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# =============================================================================
# Tool 1 — Previsão LSTM (Domínio 1)
# =============================================================================

def _ibov_forecast_tool():
    from langchain_core.tools import Tool

    def _run(_: str = "") -> str:
        from src.ibov_pipeline.predict import predict_next_day
        try:
            r = predict_next_day()
            return (
                f"Previsão LSTM para o próximo pregão do IBOV: "
                f"{r['predicted_close']:,.2f} pontos "
                f"(último fechamento: {r['last_close']:,.2f}, "
                f"variação prevista: {r['delta_pct']:+.2f}%). "
                f"OBS: a acurácia direcional histórica do modelo univariado "
                f"é ~50% — use como filtro de viés, não como sinal autônomo."
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ibov_forecast_tool falhou")
            return f"Erro ao consultar a previsão LSTM: {exc}"

    return Tool(
        name="ibov_forecast",
        description=(
            "Consulta o modelo LSTM (Domínio 1) para obter a previsão de "
            "preço do IBOV para o próximo pregão. NÃO recebe argumentos. "
            "Retorna preço previsto, último fechamento e variação percentual."
        ),
        func=_run,
    )


# =============================================================================
# Tool 2 — RAG sobre relatórios macro
# =============================================================================

def _rag_tool():
    from langchain_core.tools import Tool

    def _run(query: str) -> str:
        from src.agent_pipeline.rag_retriever import get_retriever
        try:
            retriever = get_retriever()
            docs = retriever.invoke(query)
            if not docs:
                return "Nenhum trecho relevante encontrado nos relatórios."
            return "\n\n---\n\n".join(
                f"[Fonte: {Path(d.metadata.get('source', '?')).name} — pg. {d.metadata.get('page', '?')}]\n"
                f"{d.page_content.strip()}"
                for d in docs
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("macro_rag falhou")
            return f"Erro ao consultar a base macro: {exc}"

    from pathlib import Path
    return Tool(
        name="macro_rag",
        description=(
            "Busca em relatórios Focus do BC, atas Copom e análises macro "
            "internas. Use para perguntas sobre expectativa de Selic, "
            "inflação, PIB ou contexto macroeconômico. Argumento: query em "
            "português livre."
        ),
        func=_run,
    )


# =============================================================================
# Tool 3 — Calculadora aritmética (anti-alucinação numérica)
# =============================================================================

_SAFE_EXPR = re.compile(r"^[\d\s+\-*/().]+$")


def _calculator_tool():
    from langchain_core.tools import Tool

    def _run(expression: str) -> str:
        if not _SAFE_EXPR.match(expression):
            return (
                "Erro: apenas dígitos e operadores aritméticos básicos "
                "(+ - * / ( ) .) são permitidos."
            )
        try:
            result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
            return f"Resultado: {result}"
        except Exception as exc:  # noqa: BLE001
            return f"Erro de cálculo: {exc}"

    return Tool(
        name="calculator",
        description=(
            "Calculadora aritmética determinística. Use para variações "
            "percentuais, somas, divisões — nunca tente fazer matemática "
            "no raciocínio. Argumento: expressão como '(130000 - 128500) / 128500 * 100'."
        ),
        func=_run,
    )


# =============================================================================
# Tool 4 — Contexto de mercado live (yfinance)
# =============================================================================

def _market_context_tool():
    from langchain_core.tools import Tool

    def _run(_: str = "") -> str:
        import yfinance as yf

        tickers = {
            "USD/BRL": "BRL=X",
            "S&P 500": "^GSPC",
            "Petróleo Brent": "BZ=F",
        }
        out = []
        for name, sym in tickers.items():
            try:
                hist = yf.Ticker(sym).history(period="2d")["Close"]
                if len(hist) < 2:
                    out.append(f"{name}: indisponível (dados insuficientes)")
                    continue
                last, prev = float(hist.iloc[-1]), float(hist.iloc[-2])
                delta = (last - prev) / prev * 100.0
                out.append(f"{name}: {last:,.2f} ({delta:+.2f}% vs. dia anterior)")
            except Exception as exc:  # noqa: BLE001
                out.append(f"{name}: indisponível ({exc})")
        return "Contexto de mercado (último pregão):\n" + "\n".join(out)

    return Tool(
        name="market_context",
        description=(
            "Cotações ao vivo de USD/BRL, S&P 500 e Petróleo Brent. "
            "Use para contextualizar movimentos do IBOV (correlação com fluxo "
            "internacional). NÃO recebe argumentos."
        ),
        func=_run,
    )


# =============================================================================
# Conjunto padrão
# =============================================================================

def get_default_tools() -> list:
    """Retorna as 4 tools padrão do Copiloto IBOV."""
    return [
        _ibov_forecast_tool(),
        _rag_tool(),
        _calculator_tool(),
        _market_context_tool(),
    ]
