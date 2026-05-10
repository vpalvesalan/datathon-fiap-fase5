"""RAGAS — 4 métricas obrigatórias do guia (Etapa 3).

Referência: Es et al. (2024) — RAGAS: Automated Evaluation of RAG.
            https://arxiv.org/abs/2309.15217

Métricas avaliadas:
- faithfulness       — a resposta está ancorada no contexto recuperado?
- answer_relevancy   — a resposta endereça a pergunta?
- context_precision  — os trechos recuperados são úteis?
- context_recall     — o ground truth está coberto pelos trechos?

Persistência incremental com scores por pergunta:
- Cada resposta gerada é gravada IMEDIATAMENTE em
  `evaluation/results/ragas_responses.jsonl` (respostas, sem scores).
- Após compilar dataset, calcula RAGAS e salva scores em
  `evaluation/results/ragas_responses_with_scores.jsonl` (com 4 métricas por pergunta).
- Salva as médias em `evaluation/results/ragas_metrics.json`.
- Se o script crashar (rate limit, CTRL+C, OOM), o progresso de respostas está em disco.
- Re-executar detecta perguntas já respondidas e PULA — você retoma de onde parou.
- Para reprocessar do zero, apague ambos os `.jsonl` antes.

LLM utilizado: Groq (llama-3.1-8b-instant) para embeddings/avaliação.

Uso:
    python -m evaluation.ragas_eval                           # roda / retoma
    rm evaluation/results/ragas_responses.jsonl               # reset duro
    rm evaluation/results/ragas_responses_with_scores.jsonl
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
    """Caminho do arquivo incremental de respostas (sem scores)."""
    return Path(agent_cfg.evaluation.ragas_output).parent / "ragas_responses.jsonl"


def _responses_with_scores_path() -> Path:
    """Caminho do arquivo com scores por pergunta."""
    return Path(agent_cfg.evaluation.ragas_output).parent / "ragas_responses_with_scores.jsonl"


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


def _load_existing_scores(path: Path) -> dict[str, dict]:
    """Lê scores já calculados (dict com question como chave)."""
    if not path.exists():
        return {}

    scores_dict = {}
    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                scores_dict[row["question"]] = row
            except json.JSONDecodeError as exc:
                logger.warning("Linha %d do JSONL inválida (%s) — pulando.", line_no, exc)
    return scores_dict


def _append_scores(path: Path, row: dict) -> None:
    """Salva scores de forma durável."""
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
    """Roda o agente em cada pergunta do golden_set e calcula RAGAS com Groq.

    Comportamento incremental:
    1. Respostas gravadas em ragas_responses.jsonl conforme geradas.
    2. Scores calculados e gravados em ragas_responses_with_scores.jsonl.
    3. Médias salvas em ragas_metrics.json.

    Returns:
        Dicionário com as 4 métricas RAGAS (médias).
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from langchain_groq import ChatGroq

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

                result = agent.invoke({"input": q})
                answer = result.get("output", "")
                steps = result.get("intermediate_steps", [])

                contexts = [
                    f"[{action.tool}] {str(observation)}"
                    for action, observation in steps
                ]

                if not contexts:
                    ctx_docs = retriever.invoke(q)
                    contexts = [d.page_content for d in ctx_docs]
                    logger.info("Pergunta resolvida sem tool — fallback ao retriever.")

                row = {
                    "question": q,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": item.get("resposta_esperada", ""),
                }
                _append_row(jsonl_path, row)
                rows.append(row)

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Falha na pergunta '%s' (%s: %s). %d respostas já em %s. "
                    "Re-execute o script para retomar.",
                    q[:80], type(exc).__name__, exc,
                    len(rows), jsonl_path,
                )
                raise

    # --- RAGAS sobre TODAS as respostas com Groq (não OpenAI) ---
    if not rows:
        raise RuntimeError("Nenhuma resposta para avaliar.")

    logger.info("Calculando RAGAS com Groq (llama-3.1-8b-instant) + Multilíngue embeddings...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY não definido. Defina a variável de ambiente.")

    # Usar Groq como LLM para RAGAS (não OpenAI)
    groq_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key=api_key)

    # Usar multilíngue embeddings (leve, português funciona bem, ~30-50s em CPU)
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    dataset = Dataset.from_list(rows)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=groq_llm,
        embeddings=embeddings,
    )

    # --- Extrair scores por pergunta e salvar incrementalmente ---
    scores_path = _responses_with_scores_path()
    existing_scores = _load_existing_scores(scores_path)

    # scores é um Dataset com colunas: faithfulness, answer_relevancy, context_precision, context_recall
    # + as colunas originais: question, answer, contexts, ground_truth
    for idx, row_data in enumerate(scores):
        question = row_data["question"]

        # Se já calculado, pular
        if question in existing_scores:
            logger.debug("Score para '%s' já existe, pulando.", question[:60])
            continue

        score_row = {
            "question": question,
            "answer": row_data.get("answer", ""),
            "faithfulness": float(row_data.get("faithfulness", 0)),
            "answer_relevancy": float(row_data.get("answer_relevancy", 0)),
            "context_precision": float(row_data.get("context_precision", 0)),
            "context_recall": float(row_data.get("context_recall", 0)),
        }
        _append_scores(scores_path, score_row)
        existing_scores[question] = score_row

    logger.info("✓ Scores salvos em %s", scores_path)

    # --- Calcular e salvar MÉDIAS ---
    all_scores = list(existing_scores.values())
    if not all_scores:
        logger.warning("Nenhum score para calcular médias.")
        return {}

    metrics = {
        "faithfulness": float(sum(s["faithfulness"] for s in all_scores) / len(all_scores)),
        "answer_relevancy": float(sum(s["answer_relevancy"] for s in all_scores) / len(all_scores)),
        "context_precision": float(sum(s["context_precision"] for s in all_scores) / len(all_scores)),
        "context_recall": float(sum(s["context_recall"] for s in all_scores) / len(all_scores)),
        "n_questions": len(all_scores),
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
