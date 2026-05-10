"""LLM-as-a-judge — avaliação independente com ≥ 3 critérios.

Diferente do RAGAS (que mede aspectos do RAG), o judge avalia a resposta
final do agente sob 3 lentes de negócio:

1. **factual_correctness** — a resposta está factualmente correta?
2. **business_relevance**  — útil para o caso de uso (analista IBOV)?
3. **no_hallucination**    — não inventa dados ausentes do contexto?

O modelo-juiz roda numa configuração diferente do agente principal
(temperatura 0, modelo grande) para reduzir viés do auto-julgamento.

Persistência incremental:
- Cada resposta avaliada é gravada IMEDIATAMENTE em
  `evaluation/results/judge_responses.jsonl` (uma linha JSON por pergunta).
- Se exceder rate limit ou crashar, o progresso já está em disco.
- Re-executar detecta perguntas já avaliadas e PULA — você retoma de onde parou.
- Para reprocessar do zero, apague o arquivo `.jsonl` antes.

Uso:
    python -m evaluation.llm_judge                  # roda / retoma
    rm evaluation/results/judge_responses.jsonl     # reset duro
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.agent_pipeline.config import agent_cfg

logger = logging.getLogger(__name__)

MODEL = "llama-3.1-8b-instant" #openai/gpt-oss-120b" #"llama-3.3-70b-versatile"

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


# =============================================================================
# Persistência incremental — JSONL append-only
# =============================================================================

def _responses_path() -> Path:
    """Caminho do arquivo incremental de respostas avaliadas."""
    return Path(agent_cfg.evaluation.judge_output).parent / "judge_responses.jsonl"


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
# Avaliação do juiz
# =============================================================================

def _judge_llm():
    """Modelo-juiz ({MODEL} com temperatura 0)."""
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY não definido.")
    logger.info(f"Modelo Juiz selecionado: {MODEL}.")
    return ChatGroq(model=MODEL, temperature=0.0, api_key=api_key)


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

    Comportamento incremental: respostas são gravadas em JSONL conforme
    geradas. Re-executar pula perguntas já avaliadas.

    Returns:
        Dicionário com médias por critério e detalhes por pergunta.
    """
    from src.agent_pipeline.react_agent import create_copiloto_agent
    from evaluation._rate_limit import from_config as _build_limiter

    golden_path = Path(golden_set_path or agent_cfg.golden_set.path)
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    # --- Retomada: lê o que já existe em disco ---
    jsonl_path = _responses_path()
    existing = _load_existing(jsonl_path)
    answered: set[str] = {r["pergunta"] for r in existing}
    rows: list[dict] = list(existing)

    if existing:
        logger.info(
            "Retomando avaliação — %d/%d perguntas já em %s",
            len(existing), len(golden), jsonl_path,
        )

    pending = [item for item in golden if item["pergunta"] not in answered]
    if not pending:
        logger.info("Todas as %d perguntas já estavam avaliadas.", len(golden))
    else:
        agent = create_copiloto_agent()
        judge = _judge_llm()
        limiter = _build_limiter()

        logger.info(
            "Rate limit ativo: %d tok/min, %.1fs entre chamadas (estimado).",
            agent_cfg.llm.rate_limit.tokens_per_minute,
            limiter.seconds_per_call,
        )
        logger.info("%d perguntas pendentes nesta execução.", len(pending))

        for i, item in enumerate(pending, 1):
            q, gt = item["pergunta"], item.get("resposta_esperada", "")
            try:
                limiter.wait_if_needed()
                logger.info("[%d/%d pendentes] %s", i, len(pending), q[:80])

                answer = agent.invoke({"input": q}).get("output", "")
                score = _judge_one(judge, q, answer, gt)

                row = {
                    "pergunta": q,
                    "answer": answer,
                    **score,
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
                raise   # propaga — o que já foi salvo está em disco

    # --- Calcula médias sobre TODAS as respostas (existentes + novas) ---
    if not rows:
        raise RuntimeError("Nenhuma resposta para avaliar.")

    n = len(rows)
    avgs = {
        "factual_correctness_avg": sum(d.get("factual_correctness", 0) for d in rows) / n,
        "business_relevance_avg": sum(d.get("business_relevance", 0) for d in rows) / n,
        "no_hallucination_avg": sum(d.get("no_hallucination", 0) for d in rows) / n,
        "n": n,
    }

    out = {"averages": avgs, "details": rows}
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
