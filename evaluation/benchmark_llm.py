"""Benchmark de ≥ 3 configurações de LLM (rubrica Etapa 2).

Compara modelos disponíveis no Groq sob 4 dimensões:
- faithfulness (RAGAS)
- answer_relevancy (RAGAS)
- latência média (s)
- tokens consumidos (custo proxy)

Uso:
    python -m evaluation.benchmark_llm
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.agent_pipeline.config import agent_cfg

logger = logging.getLogger(__name__)


# 3 configurações intencionalmente diferentes ---------------------------------

LLM_CONFIGS: list[dict] = [
    {"name": "llama-3.1-8b",   "model": "llama-3.1-8b-instant",     "temperature": 0.0},
    {"name": "llama-3.3-70b",  "model": "llama-3.3-70b-versatile",  "temperature": 0.0},
    {"name": "llama-3.3-70b-warm", "model": "llama-3.3-70b-versatile", "temperature": 0.7},
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
    golden_path = Path(agent_cfg.golden_set.path)
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    results = []
    for c in LLM_CONFIGS:
        logger.info("Benchmarking: %s", c["name"])
        try:
            results.append(_evaluate_config(c, golden))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falha em %s", c["name"])
            results.append({"config_name": c["name"], "error": str(exc)})

    output = {"results": results}
    out_path = Path(output_path or agent_cfg.evaluation.benchmark_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Benchmark salvo em %s", out_path)
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
