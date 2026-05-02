"""LLM-as-a-judge — avaliação independente com ≥ 3 critérios.

Diferente do RAGAS (que mede aspectos do RAG), o judge avalia a resposta
final do agente sob 3 lentes de negócio:

1. **factual_correctness** — a resposta está factualmente correta?
2. **business_relevance**  — útil para o caso de uso (analista IBOV)?
3. **no_hallucination**    — não inventa dados ausentes do contexto?

O modelo-juiz roda numa configuração diferente do agente principal
(temperatura 0, modelo grande) para reduzir viés do auto-julgamento.

Uso:
    python -m evaluation.llm_judge
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.agent_pipeline.config import agent_cfg

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """Você é um juiz imparcial avaliando a resposta de um \
assistente financeiro para uma pergunta sobre o IBOV.

Avalie a resposta sob 3 critérios. Cada critério vale de 1 (péssimo) a 5 (excelente).

Critérios:
1. **factual_correctness**: a resposta está factualmente correta dado o ground truth?
2. **business_relevance**: a resposta é útil para um analista de mercado tomar decisão?
3. **no_hallucination**: a resposta evita inventar números/datas/nomes ausentes do contexto?

PERGUNTA: {question}

RESPOSTA AVALIADA: {answer}

GROUND TRUTH (referência): {ground_truth}

Responda APENAS com um JSON válido no formato:
{{"factual_correctness": <1-5>, "business_relevance": <1-5>, "no_hallucination": <1-5>, "rationale": "<frase curta justificando>"}}"""


def _judge_llm():
    """Modelo-juiz (Llama 3.3 70B com temperatura 0)."""
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY não definido.")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=api_key)


def _judge_one(judge, question: str, answer: str, ground_truth: str) -> dict:
    """Avalia uma resposta. Retorna dict com 3 scores + rationale."""
    prompt = JUDGE_PROMPT.format(
        question=question, answer=answer, ground_truth=ground_truth,
    )
    raw = judge.invoke(prompt).content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Tenta extrair JSON do texto
        import re
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        logger.warning("Juiz retornou JSON inválido: %s", raw[:200])
        return {
            "factual_correctness": 0,
            "business_relevance": 0,
            "no_hallucination": 0,
            "rationale": "PARSING_ERROR",
        }


def run_judge(
    golden_set_path: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Avalia o agente no golden set com LLM-as-judge.

    Returns:
        Dicionário com médias por critério e detalhes por pergunta.
    """
    from src.agent_pipeline.react_agent import create_copiloto_agent

    golden_path = Path(golden_set_path or agent_cfg.golden_set.path)
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    agent = create_copiloto_agent()
    judge = _judge_llm()

    details = []
    for i, item in enumerate(golden, 1):
        q, gt = item["pergunta"], item.get("resposta_esperada", "")
        logger.info("[%d/%d] avaliando", i, len(golden))

        answer = agent.invoke({"input": q}).get("output", "")
        score = _judge_one(judge, q, answer, gt)
        details.append({"pergunta": q, "answer": answer, **score})

    # Médias
    n = max(len(details), 1)
    avgs = {
        "factual_correctness_avg": sum(d.get("factual_correctness", 0) for d in details) / n,
        "business_relevance_avg": sum(d.get("business_relevance", 0) for d in details) / n,
        "no_hallucination_avg": sum(d.get("no_hallucination", 0) for d in details) / n,
        "n": n,
    }

    out = {"averages": avgs, "details": details}
    out_path = Path(output_path or agent_cfg.evaluation.judge_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    logger.info("LLM-as-judge: %s | salvo em %s", avgs, out_path)
    return out


if __name__ == "__main__":
    from src.ibov_pipeline.logging_config import setup_logging

    setup_logging("llm_judge")
    result = run_judge()
    print("\n=== LLM-as-Judge — médias ===")
    for k, v in result["averages"].items():
        print(f"  {k}: {v}")
