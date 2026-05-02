"""RAGAS — 4 métricas obrigatórias do guia (Etapa 3).

Referência: Es et al. (2024) — RAGAS: Automated Evaluation of RAG.
            https://arxiv.org/abs/2309.15217

Métricas avaliadas:
- faithfulness       — a resposta está ancorada no contexto recuperado?
- answer_relevancy   — a resposta endereça a pergunta?
- context_precision  — os trechos recuperados são úteis?
- context_recall     — o ground truth está coberto pelos trechos?

Uso:
    python -m evaluation.ragas_eval
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent_pipeline.config import agent_cfg

logger = logging.getLogger(__name__)


def evaluate_rag_pipeline(
    golden_set_path: str | None = None,
    output_path: str | None = None,
) -> dict[str, float]:
    """Roda o agente em cada pergunta do golden_set e calcula RAGAS.

    Returns:
        Dicionário com as 4 métricas RAGAS.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from src.agent_pipeline.rag_retriever import get_retriever
    from src.agent_pipeline.react_agent import create_copiloto_agent

    golden_path = Path(golden_set_path or agent_cfg.golden_set.path)
    if not golden_path.exists():
        raise FileNotFoundError(
            f"Golden set ausente em {golden_path}. "
            "Crie data/golden_set/perguntas.json com ≥ 20 pares antes de avaliar."
        )

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    if len(golden) < 20:
        logger.warning(
            "Golden set tem %d entradas — guia pede ≥ 20.", len(golden),
        )

    agent = create_copiloto_agent()
    retriever = get_retriever()

    rows = []
    for i, item in enumerate(golden, 1):
        q = item["pergunta"]
        logger.info("[%d/%d] %s", i, len(golden), q[:80])

        # Captura contexto separadamente (RAGAS precisa)
        ctx_docs = retriever.invoke(q)
        contexts = [d.page_content for d in ctx_docs]

        result = agent.invoke({"input": q})
        answer = result.get("output", "")

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item.get("resposta_esperada", ""),
        })

    dataset = Dataset.from_list(rows)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    metrics = {
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall": float(scores["context_recall"]),
        "n_questions": len(golden),
    }

    out_path = Path(output_path or agent_cfg.evaluation.ragas_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("RAGAS: %s | salvo em %s", metrics, out_path)
    return metrics


if __name__ == "__main__":
    from src.ibov_pipeline.logging_config import setup_logging

    setup_logging("ragas_eval")
    results = evaluate_rag_pipeline()
    print(f"\n=== RAGAS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
