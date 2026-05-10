"""Benchmark de ≥ 3 configurações de LLM (rubrica Etapa 2).

Compara modelos disponíveis no Groq sob 4 dimensões:
- faithfulness (RAGAS)
- answer_relevancy (RAGAS)
- latência média (s)
- tokens consumidos (custo proxy)

Persistência incremental:
- Cada configuração avaliada é gravada IMEDIATAMENTE em
  `evaluation/results/benchmark_partial_results.json`.
- Se exceder rate limit ou crashar, o progresso já está em disco.
- Re-executar detecta configs já avaliadas e PULA — você retoma de onde parou.
- Para reprocessar do zero, apague o arquivo `.json` antes.

Uso:
    python -m evaluation.benchmark_llm                  # roda / retoma
    rm evaluation/results/benchmark_partial_results.json     # reset duro
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from src.agent_pipeline.config import agent_cfg

logger = logging.getLogger(__name__)


# =============================================================================
# Persistência incremental
# =============================================================================

def _partial_results_path() -> Path:
    """Caminho do arquivo incremental de resultados parciais."""
    return Path(agent_cfg.evaluation.benchmark_output).parent / "benchmark_partial_results.json"


def _load_partial() -> dict[str, dict]:
    """Lê resultados já processados (dict com config_name como chave)."""
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


# 3 configurações intencionalmente diferentes ---------------------------------

LLM_CONFIGS: list[dict] = [
    {"name": "qwen3-32b",   "model": "qwen/qwen3-32b",     "temperature": 0.0},
    {"name": "llama-3.3-70b",  "model": "llama-3.3-70b-versatile",  "temperature": 0.0},
    {"name": "gpt-oss-120b", "model": "openai/gpt-oss-120b", "temperature": 0.0},
]


def _evaluate_config(cfg_dict: dict, golden: list) -> dict:
    """Avalia uma configuração de LLM contra o golden set."""
    import os
    from datasets import Dataset
    from langchain_groq import ChatGroq
    from langchain_classic.agents import AgentExecutor, create_react_agent
    from langchain_classic.prompts import PromptTemplate
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    from src.agent_pipeline.rag_retriever import get_retriever
    from src.agent_pipeline.tools import get_default_tools

    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=cfg_dict["model"], temperature=cfg_dict["temperature"], api_key=api_key)

    template = (
        "Ferramentas:\n{tools}\n\n"
        "Question: {input}\nThought: ...\nAction: [{tool_names}]\n"
        "Action Input: ...\nObservation: ...\nFinal Answer: ...\n"
        "{agent_scratchpad}"
    )
    prompt = PromptTemplate.from_template(template)
    tools = get_default_tools()
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools, max_iterations=agent_cfg.llm.max_iterations,
        handle_parsing_errors=True,
    )

    retriever = get_retriever()
    rows = []
    latencies = []

    for item in golden:
        q = item["pergunta"]
        contexts = [d.page_content for d in retriever.invoke(q)]

        t0 = time.perf_counter()
        answer = executor.invoke({"input": q}).get("output", "")
        latencies.append(time.perf_counter() - t0)

        rows.append({
            "question": q, "answer": answer, "contexts": contexts,
            "ground_truth": item.get("resposta_esperada", ""),
        })

    scores = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy],
    )

    return {
        "config_name": cfg_dict["name"],
        "model": cfg_dict["model"],
        "temperature": cfg_dict["temperature"],
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "latency_avg_s": sum(latencies) / max(len(latencies), 1),
        "n_questions": len(golden),
    }


def run_benchmark(output_path: str | None = None) -> dict:
    """Roda benchmark de configs com persistência incremental.

    Comportamento: cada config é avaliada e salva imediatamente em
    benchmark_partial_results.json. Re-executar pula configs já avaliadas.

    Returns:
        Dicionário com resultados de todas as configs.
    """
    from evaluation._rate_limit import from_config as _build_limiter

    golden_path = Path(agent_cfg.golden_set.path)
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    # --- Retomada: lê o que já existe em disco ---
    partial = _load_partial()
    completed: set[str] = set(partial.keys())
    pending = [c for c in LLM_CONFIGS if c["name"] not in completed]

    if completed:
        logger.info(
            "Retomando benchmark — %d/%d configs já em %s",
            len(completed), len(LLM_CONFIGS), _partial_results_path(),
        )

    if pending:
        limiter = _build_limiter()
        logger.info(
            "Rate limit ativo: %d tok/min, %.1fs entre chamadas (estimado).",
            agent_cfg.llm.rate_limit.tokens_per_minute,
            limiter.seconds_per_call,
        )
        logger.info("%d configs pendentes nesta execução.", len(pending))

        for i, c in enumerate(pending, 1):
            try:
                limiter.wait_if_needed()
                logger.info("[%d/%d pendentes] benchmarking: %s", i, len(pending), c["name"])

                result = _evaluate_config(c, golden)
                partial[c["name"]] = result
                _save_partial(partial)

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Falha em '%s' (%s: %s). %d configs já em %s. "
                    "Re-execute o script para retomar.",
                    c["name"], type(exc).__name__, exc,
                    len(partial), _partial_results_path(),
                )
                raise

    # --- Compila resultado final ---
    if not partial:
        raise RuntimeError("Nenhuma config para avaliar.")

    results = list(partial.values())
    output = {"results": results}

    out_path = Path(output_path or agent_cfg.evaluation.benchmark_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Benchmark concluído | salvo em %s", out_path)
    return output


if __name__ == "__main__":
    from src.ibov_pipeline.logging_config import setup_logging

    setup_logging("benchmark_llm")
    out = run_benchmark()
    print("\n=== Benchmark LLM ===")
    for r in out["results"]:
        if "error" in r:
            print(f"  {r['config_name']:25s} | ERRO: {r['error']}")
        else:
            print(f"  {r['config_name']:25s} | faith={r.get('faithfulness', 0):.3f} "
                  f"lat={r.get('latency_avg_s', 0):.2f}s")
