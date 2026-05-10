"""A/B test de prompts — compara ≥ 3 system prompts no golden set.

Implementa o critério da Etapa 3: "Compara ≥ 3 variações de system prompt
no golden set e reporta delta nas métricas RAGAS".

Persistência incremental:
- Cada prompt avaliado é gravado IMEDIATAMENTE em
  `evaluation/results/ab_test_partial_results.json`.
- Se exceder rate limit ou crashar, o progresso já está em disco.
- Re-executar detecta prompts já avaliados e PULA — você retoma de onde parou.
- Para reprocessar do zero, apague o arquivo `.json` antes.

Uso:
    python -m evaluation.ab_test_prompts                  # roda / retoma
    rm evaluation/results/ab_test_partial_results.json    # reset duro
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.agent_pipeline.config import agent_cfg

logger = logging.getLogger(__name__)


# =============================================================================
# Persistência incremental
# =============================================================================

def _partial_results_path() -> Path:
    """Caminho do arquivo incremental de resultados parciais."""
    return Path(agent_cfg.evaluation.ab_test_output).parent / "ab_test_partial_results.json"


def _load_partial() -> dict[str, dict]:
    """Lê resultados já processados (dict com prompt_name como chave)."""
    path = _partial_results_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.warning("Arquivo parcial corrompido (%s) — ignorando.", exc)
        return {}


def _save_partial(partial: dict[str, dict]) -> None:
    """Salva resultados parciais de forma durável."""
    path = _partial_results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(partial, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


# 3 variantes de system prompt para comparar -----------------------------------

PROMPT_VARIANTS: dict[str, str] = {
    "baseline_strict": (
        "Você é um analista financeiro. Responda APENAS com base nas tools. "
        "Se não souber, diga 'não sei'."
    ),
    "step_by_step": (
        "Você é um analista financeiro. Pense passo a passo: primeiro identifique "
        "que dados precisa; depois acione as tools; por fim sintetize. SEMPRE "
        "cite as fontes que recuperou."
    ),
    "domain_expert": (
        "Você é um economista sênior especializado no mercado brasileiro com "
        "20 anos de experiência analisando o IBOV. Combine sinais quantitativos "
        "(LSTM) com contexto macro (Focus, Copom). Seja conciso e preciso. "
        "Reconheça limitações dos modelos."
    ),
}


def _evaluate_with_prompt(prompt_name: str, system_prompt: str, golden: list) -> dict:
    """Roda RAGAS com um system prompt customizado."""
    from datasets import Dataset
    from langchain_classic.agents import AgentExecutor, create_react_agent
    from langchain_classic.prompts import PromptTemplate
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from src.agent_pipeline.rag_retriever import get_retriever
    from src.agent_pipeline.react_agent import _build_llm
    from src.agent_pipeline.tools import get_default_tools

    template = (
        "Você tem acesso às seguintes ferramentas:\n\n{tools}\n\n"
        "Use o formato:\nQuestion: a pergunta\nThought: ...\n"
        "Action: uma de [{tool_names}]\nAction Input: ...\nObservation: ...\n"
        "Thought: já sei\nFinal Answer: a resposta\n\n"
        f"INSTRUÇÕES: {system_prompt}\n\n"
        "Question: {input}\n{agent_scratchpad}"
    )
    prompt = PromptTemplate.from_template(template)
    tools = get_default_tools()
    agent = create_react_agent(llm=_build_llm(), tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools, max_iterations=agent_cfg.llm.max_iterations,
        handle_parsing_errors=True,
    )

    retriever = get_retriever()
    rows = []
    for item in golden:
        q = item["pergunta"]
        contexts = [d.page_content for d in retriever.invoke(q)]
        answer = executor.invoke({"input": q}).get("output", "")
        rows.append({
            "question": q, "answer": answer, "contexts": contexts,
            "ground_truth": item.get("resposta_esperada", ""),
        })

    scores = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {
        "prompt_name": prompt_name,
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall": float(scores["context_recall"]),
    }


def run_ab_test(output_path: str | None = None) -> dict:
    """Roda os 3+ prompts com persistência incremental.

    Comportamento: cada prompt é avaliado e salvo imediatamente em
    ab_test_partial_results.json. Re-executar pula prompts já avaliados.

    Returns:
        Dicionário com resultados finais e vencedor.
    """
    from evaluation._rate_limit import from_config as _build_limiter

    golden_path = Path(agent_cfg.golden_set.path)
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    # --- Retomada: lê o que já existe em disco ---
    partial = _load_partial()
    completed: set[str] = set(partial.keys())
    pending = [name for name in PROMPT_VARIANTS if name not in completed]

    if completed:
        logger.info(
            "Retomando A/B test — %d/%d variantes já em %s",
            len(completed), len(PROMPT_VARIANTS), _partial_results_path(),
        )

    if pending:
        limiter = _build_limiter()
        logger.info(
            "Rate limit ativo: %d tok/min, %.1fs entre chamadas (estimado).",
            agent_cfg.llm.rate_limit.tokens_per_minute,
            limiter.seconds_per_call,
        )
        logger.info("%d variantes pendentes nesta execução.", len(pending))

        for i, name in enumerate(pending, 1):
            sysp = PROMPT_VARIANTS[name]
            try:
                limiter.wait_if_needed()
                logger.info("[%d/%d pendentes] avaliando variante: %s", i, len(pending), name)

                result = _evaluate_with_prompt(name, sysp, golden)
                partial[name] = result
                _save_partial(partial)

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Falha na variante '%s' (%s: %s). %d variantes já em %s. "
                    "Re-execute o script para retomar.",
                    name, type(exc).__name__, exc,
                    len(partial), _partial_results_path(),
                )
                raise

    # --- Compila resultado final ---
    if not partial:
        raise RuntimeError("Nenhuma variante para avaliar.")

    results = list(partial.values())
    winner = max(results, key=lambda r: r["faithfulness"])

    output = {
        "results": results,
        "winner": winner["prompt_name"],
        "criterion": "faithfulness",
    }

    out_path = Path(output_path or agent_cfg.evaluation.ab_test_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("A/B test concluído | vencedor: %s | salvo em %s",
                winner["prompt_name"], out_path)
    return output


if __name__ == "__main__":
    from src.ibov_pipeline.logging_config import setup_logging

    setup_logging("ab_test_prompts")
    out = run_ab_test()
    print("\n=== A/B Test de Prompts ===")
    print(f"Vencedor (faithfulness): {out['winner']}")
    for r in out["results"]:
        print(f"  {r['prompt_name']:18s} | faith={r['faithfulness']:.3f} "
              f"rel={r['answer_relevancy']:.3f}")
