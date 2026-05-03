# System Card — Copiloto Analítico do IBOV

> Sistema = LSTM (Domínio 1) + Agente LLM com RAG e tools (Domínio 2)
> + Guardrails + Telemetria + Governança.

## 1. Visão geral

O Copiloto IBOV cruza um modelo quantitativo (LSTM) com contexto
macroeconômico (RAG sobre Focus do BC, atas Copom e análises) para apoiar
analistas e gestores na compreensão de movimentos do índice — **NÃO**
para decidir trades autonomamente.

## 2. Arquitetura

```
Cliente HTTP
    │
    ▼
FastAPI (src/serving/app.py)
    ├──► /predict  ───► LSTM (MLflow Registry → Production)
    └──► /agent/query
            ├─ InputGuardrail (regex, OWASP LLM01)
            ├─ Agente ReAct (Groq Llama 3.3 70B)
            │     ├─ ibov_forecast      (LSTM via tool)
            │     ├─ macro_rag          (ChromaDB + multilingual MiniLM)
            │     ├─ calculator         (anti-alucinação numérica)
            │     └─ market_context     (yfinance live)
            └─ OutputGuardrail (Presidio, OWASP LLM06)
```

## 3. Componentes e versões

| Componente | Tecnologia | Onde |
|---|---|---|
| LLM | Llama 3.3 70B via Groq | `src/agent_pipeline/react_agent.py` |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (HF) | `rag_retriever.py` |
| Vector store | ChromaDB persistente | `data/processed/agent_db/` |
| Orquestração | LangChain ReAct | `react_agent.py` |
| Serving | FastAPI + uvicorn | `src/serving/` |
| Guardrails | Regex + Presidio | `src/security/guardrails.py` |
| Telemetria | Langfuse + Prometheus | `src/monitoring/telemetry.py` |
| Drift | PSI manual + Evidently opcional | `src/monitoring/drift_detection.py` |

## 4. Decisões arquiteturais

### Por que Groq em vez de vLLM/BentoML?
O guia menciona vLLM/BentoML como exemplos de "LLM serving". Optamos por Groq
porque: (a) free tier generoso, (b) latência competitiva (10× mais rápida que
GPU local em modelos 70B), (c) zero infra de GPU, (d) satisfaz o requisito
"LLM via API". A quantização é feita pela própria Groq — externalizada.

### Por que Llama 3.3 70B e não outro modelo?

Testamos Llama 4 Scout 17B; falhou em ReAct por loop; voltamos ao Llama 3.3 70B.

### Iteração empírica do OutputGuardrail (allow list)

O `pt_core_news_sm` (modelo NER do Presidio em PT) gerou dois tipos de
falsos positivos em testes manuais:

1. **Termos financeiros** (Brent, S&P, Wall Street) — aparecem em respostas
   legítimas com qualquer LLM. Adicionados ao `allow_list` como mitigação
   contínua.

2. **Termos do framework LangChain** (Agent, AgentExecutor, Action) — só
   surgem em mensagens fallback do orquestrador, ex: *"Agent stopped due
   to iteration limit"* quando o LLM falha em ReAct. Observamos isso em
   um experimento com Llama 4 Scout 17B (loop por baixa capacidade); com
   Llama 3.3 70B não ocorre, mas mantemos no `allow_list` como defesa em
   camada para outras falhas transitórias (rate limit, timeout).

Lista em `src/security/guardrails.py::FINANCIAL_ALLOW_LIST`. Ajuste
empírico, não cego — documentado e fácil de estender.

### Por que multilingual MiniLM em vez do MiniLM-L6?
Os documentos de RAG (Focus, Copom) são em PT-BR. O MiniLM-L6 é treinado em
inglês — embeddings degradam em PT. O multilingual preserva semântica entre
línguas com custo computacional similar.

### Por que 4 tools (não 3)?
Redundância para o Demo Day. Se uma tool dependente de rede falhar ao vivo
(ex: yfinance fora do ar), as outras 3 mantêm a demo funcional.

## 5. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Prompt injection (LLM01) | InputGuardrail (10+ padrões regex) |
| Vazamento de PII (LLM06) | OutputGuardrail Presidio (CPF, e-mail, tel, nome) |
| Alucinação numérica (LLM09) | Tool `calculator` obrigatória para cálculos |
| Alucinação factual | Prompt obriga uso de `macro_rag` + cita fonte |
| LLM offline | `try/except` em todos os tools, retorna mensagem clara |

Mapeamento OWASP completo em `docs/OWASP_MAPPING.md`.

## 6. Avaliação

| Eixo | Métrica | Onde |
|---|---|---|
| RAG | faithfulness, answer_relevancy, context_precision, context_recall | `evaluation/ragas_eval.py` |
| Negócio | factual_correctness, business_relevance, no_hallucination | `evaluation/llm_judge.py` |
| Prompt | A/B test em ≥3 system prompts | `evaluation/ab_test_prompts.py` |
| Modelo | Benchmark em ≥3 configurações | `evaluation/benchmark_llm.py` |
| Drift | PSI sobre features e predições | `src/monitoring/drift_detection.py` |

Resultados em `evaluation/results/*.json`.

## 7. Observabilidade

- **Langfuse** captura trace de cada query: prompt, tool calls, tokens, latência.
- **Prometheus** expõe contadores operacionais (queries, latência, erros).
- **Logs persistentes** em `logs/<run>_<timestamp>.log` para auditoria forense.

## 8. Limitações conhecidas

1. O LSTM tem acurácia direcional ~50% — é filtro de viés, não sinal autônomo.
2. RAG depende da qualidade/atualidade dos PDFs em `data/raw/agent_docs/`.
3. Groq tem rate limit no free tier (e.g. 12k tokens / min).
4. Sem cache de respostas — cada query roda o pipeline completo.

## 9. Conformidade LGPD

Ver `docs/LGPD_PLAN.md`. Resumo: opera apenas sobre dados públicos do
mercado financeiro; nenhum dado pessoal é coletado/armazenado pelo sistema.
PII eventualmente presente em PDFs de RAG é removida no output via Presidio.

## 10. Manutenção

- **Re-ingestão de RAG**: `python -m src.agent_pipeline.rag_retriever`
- **Re-treino LSTM**: `python -m src.ibov_pipeline.retrain`
- **Re-avaliação RAGAS**: `python -m evaluation.ragas_eval`
- **Logs**: `logs/`, MLflow UI, Langfuse dashboard
