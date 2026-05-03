"""RAGAS — 4 métricas obrigatórias do guia (Etapa 3).

Referência: Es et al. (2024) — RAGAS: Automated Evaluation of RAG.
            https://arxiv.org/abs/2309.15217

Métricas avaliadas:
- faithfulness       — a resposta está ancorada no contexto recuperado?
- answer_relevancy   — a resposta endereça a pergunta?
- context_precision  — os trechos recuperados são úteis?
- context_recall     — o ground truth está coberto pelos trechos?

Persistência incremental:
- Cada resposta gerada é gravada IMEDIATAMENTE em
  `evaluation/results/ragas_responses.jsonl` (uma linha JSON por pergunta).
- Se o script crashar (rate limit, CTRL+C, OOM), o progresso já está em disco.
- Re-executar o script detecta as perguntas já respondidas e PULA — você
  retoma de onde parou (ex: dia seguinte após reset do daily limit).
- Para reprocessar do zero, apague o arquivo `.jsonl` antes.

Uso:
    python -m evaluation.ragas_eval                # roda / retoma
    rm evaluation/results/ragas_responses.jsonl    # reset duro
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.agent_pipeline.config import agent_cfg

logger = logging.getLogger(__name__)


# =============================================================================
# Persistência incremental — JSONL append-only
# =============================================================================

def _responses_path() -> Path:
    """Caminho do arquivo incremental de respostas (sibling do metrics.json)."""
    return Path(agent_cfg.evaluation.ragas_output).parent / "ragas_responses.jsonl"


def _load_existing(path: Path) -> list[dict]:
    """Lê todas as linhas já gravadas. Tolerante a linhas malformadas (skip)."""
    if not path.exists():
        return []

    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Linha %d do JSONL inválida (%s) — pulando.", line_no, exc)
    return rows


def _append_row(path: Path, row: dict) -> None:
    """Append durável: flush + fsync para sobreviver a crash do processo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# =============================================================================
# Pipeline
# =============================================================================

def evaluate_rag_pipeline(
    golden_set_path: str | None = None,
    output_path: str | None = None,
) -> dict[str, float]:
    """Roda o agente em cada pergunta do golden_set e calcula RAGAS.

    Comportamento incremental: respostas são gravadas em JSONL conforme
    geradas. Re-executar pula perguntas já respondidas.

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

    from evaluation._rate_limit import from_config as _build_limiter

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

    # --- Retomada: lê o que já existe em disco ---
    jsonl_path = _responses_path()
    existing = _load_existing(jsonl_path)
    answered: set[str] = {r["question"] for r in existing}
    rows: list[dict] = list(existing)

    if existing:
        logger.info(
            "Retomando avaliação — %d/%d perguntas já em %s",
            len(existing), len(golden), jsonl_path,
        )

    pending = [item for item in golden if item["pergunta"] not in answered]
    if not pending:
        logger.info("Todas as %d perguntas já estavam respondidas. Pulando para RAGAS.", len(golden))
    else:
        agent = create_copiloto_agent()
        retriever = get_retriever()
        limiter = _build_limiter()

        logger.info(
            "Rate limit ativo: %d tok/min, %.1fs entre chamadas (estimado).",
            agent_cfg.llm.rate_limit.tokens_per_minute,
            limiter.seconds_per_call,
        )
        logger.info("%d perguntas pendentes nesta execução.", len(pending))

        for i, item in enumerate(pending, 1):
            q = item["pergunta"]
            try:
                limiter.wait_if_needed()
                logger.info("[%d/%d pendentes] %s", i, len(pending), q[:80])

                ctx_docs = retriever.invoke(q)
                contexts = [d.page_content for d in ctx_docs]

                result = agent.invoke({"input": q})
                answer = result.get("output", "")

                row = {
                    "question": q,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": item.get("resposta_esperada", ""),
                }
                _append_row(jsonl_path, row)   # ← durável ANTES de continuar
                rows.append(row)

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Falha na pergunta '%s' (%s: %s). %d respostas já em %s. "
                    "Re-execute o script para retomar.",
                    q[:80], type(exc).__name__, exc,
                    len(rows), jsonl_path,
                )
                raise   # propaga — o que já foi salvo está em disco

    # --- RAGAS sobre TODAS as respostas (existentes + novas) ---
    if not rows:
        raise RuntimeError("Nenhuma resposta para avaliar.")

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
        "n_questions": len(rows),
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
