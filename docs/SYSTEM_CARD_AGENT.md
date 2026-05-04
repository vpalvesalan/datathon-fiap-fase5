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
                 ┌──────────────────────────────────────────┐
                 │  RENDER (cloud — auto-deploy on push)    │
                 │  Container Docker (FastAPI + Gradio)     │
                 │     ├──► /predict  → LSTM (Registry)     │
                 │     ├──► /agent/query                    │
                 │     │     ├─ InputGuardrail (LLM01)      │
                 │     │     ├─ Agente ReAct (Groq 70B)     │
                 │     │     │   ├─ ibov_forecast           │
                 │     │     │   ├─ macro_rag (ChromaDB)    │
                 │     │     │   ├─ calculator              │
                 │     │     │   └─ market_context          │
                 │     │     └─ OutputGuardrail (LLM06)     │
                 │     ├──► /chat    → UI Gradio            │
                 │     └──► /metrics → Prometheus           │
                 └──────────────────────────────────────────┘
                                  ▲
                                  │ scrape
                                  │
                 ┌──────────────────────────────────────────┐
                 │  LOCAL (docker-compose, demo + dev)      │
                 │  ├─ Prometheus (9090)                    │
                 │  ├─ Grafana    (3000)                    │
                 │  └─ MLflow UI  (5000)                    │
                 └──────────────────────────────────────────┘
```

## 3. Componentes e versões

| Componente | Tecnologia | Onde |
|---|---|---|
| LLM | Llama 3.3 70B via Groq | `src/agent_pipeline/react_agent.py` |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (HF) | `rag_retriever.py` |
| Vector store | ChromaDB persistente | `data/processed/agent_db/` |
| Orquestração | LangChain ReAct | `react_agent.py` |
| Serving | FastAPI + uvicorn (no Render, plano free) | `src/serving/`, `render.yaml` |
| UI | Gradio montado em `/chat` (mesmo container, streaming) | `src/serving/gradio_app.py` |
| Guardrails | Regex + Presidio | `src/security/guardrails.py` |
| Telemetria | MLflow Tracing (primário) + Langfuse (opcional) + Prometheus | `src/monitoring/telemetry.py` |
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

### Por que MLflow Tracing como backend primário (e Langfuse opcional)?
O guia menciona Langfuse/TruLens como exemplos de telemetria LLM. Adotamos
**MLflow Tracing** como primário e Langfuse como camada opcional pelos
seguintes motivos:

1. **Single pane of glass** — MLflow já gerencia experimentos, modelos e
   registry do LSTM. Reusar para tracing do agente unifica todo o ciclo de
   vida MLOps em uma UI só.
2. **Zero infra adicional** — `mlflow.langchain.autolog()` instrumenta
   tudo automaticamente. Sem conta nova, sem env vars adicionais, sem
   dependência de rede para cloud externo.
3. **LGPD / privacidade** — traces ficam no `mlruns/` local. Nenhum dado
   da query do usuário sai do nosso ambiente, ao contrário do Langfuse
   cloud (que exigiria DPA com terceiro).
4. **Sem rate limit** — Langfuse free tier limita ingestão; MLflow local
   não tem limite.

Langfuse permanece **disponível como fallback opcional**: se o time
quiser usar dashboards específicos de LLM (custo por modelo, A/B de
prompts), basta preencher `LANGFUSE_PUBLIC_KEY` no `.env` que ambos os
backends rodam em paralelo, sem conflito. Decisão reversível por
configuração.

### Por que Render como hospedagem (e não Hugging Face Spaces)?

Avaliamos HF Spaces e Render para hospedar o serving. Optamos por
**Render** pelos seguintes diferenciais que importam para o escopo MLOps:

| Critério | HF Spaces (free) | Render (free) |
|---|---|---|
| Multi-container (`docker-compose`) | ❌ só 1 container | ✅ blueprint suporta múltiplos |
| Persistent disk no free tier | ❌ volátil | ⚠️ paid (não usamos) |
| Cron jobs nativos | ❌ | ✅ (futuro retreino agendado) |
| Filosofia | "showcase de IA" | "PaaS genérico, MLOps-friendly" |
| Auto-deploy via Git push | ✅ | ✅ |

CD configurado via [`render.yaml`](../render.yaml) — push em `main`
dispara webhook nativo do Render que rebuilda a imagem e faz redeploy.
**Sem GitHub Actions de deploy** — menos secrets, menos código.

### Por que observabilidade local (Prometheus + Grafana)?

Stack via [`docker-compose.yml`](../docker-compose.yml) na máquina do
desenvolvedor. **Não roda no Render** porque:

1. Render free tier é single-container — não cabe Prometheus + Grafana + API.
2. Prometheus é "puxador" — pode fazer scrape do `/metrics` do Render via
   HTTPS, sem precisar estar na mesma rede.
3. Grafana Cloud free tier é alternativa válida para produção real, mas
   adiciona conta externa para o demo.

Padrão alinhado com produção real: observabilidade vive em estação
separada do produto. No Demo Day, dois terminais — um com `docker-compose
up`, outro com queries pra API hospedada — basta para mostrar o ciclo
completo (operacional via Grafana, qualidade LLM via MLflow Tracing).

### Por que Gradio em `/chat` no MESMO container do FastAPI?

Padrão oficial do Gradio: `gr.mount_gradio_app(fastapi_app, blocks,
path="/chat")`. Vantagens sobre container separado:

1. **Single deploy** — Render free tier suporta 1 web service por blueprint.
2. **Reuso de singletons** — guardrails, agent executor e callbacks Langfuse
   são compartilhados entre os endpoints REST e a UI Gradio (lazy import
   evita ciclo).
3. **Streaming progressivo** — UI usa `agent_executor.stream()` em vez de
   `invoke()`, mostrando passos do ReAct (Action / Observation) em tempo
   real ao usuário, não só a resposta final.

**Observações**:
* **Prompt:**
   * O agente estava tentando usar a ferramenta `macro_rag` novamente (em um loop infinito) quando não encontrava a informação requerida. Instrução explicita foi inclusa no prompt para evitar o consumo de tokens desnecessários: 
      > "Se você usar a ferramenta macro_rag e a informação necessária não estiver no texto retornado, NÃO tente usar a ferramenta novamente. Pare imediatamente e responda: 'Desculpe, não encontrei essa informação nos documentos disponíveis'."
   * Nomes de ferramentas no system prompt foram migrados de `backticks` para texto puro após observar que o LLM replicava o markup ao emitir Action:, causando falha no parser do LangChain. A versão final inclui regra explícita instruindo o modelo a não envolver tool names em qualquer caractere adicional.

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
| Drift | PSI sobre features e predições — stage `drift_check` no DVC | `src/monitoring/drift_detection.py`, `dvc.yaml` |

Resultados em `evaluation/results/*.json` e `data/processed/ibov/drift_report.json`.

## 7. Observabilidade

Implementação dividida em duas camadas conforme padrão de produção real
(produto vive em uma máquina, observabilidade em outra):

### Camada CLOUD (Render)
- **MLflow Tracing** — `mlflow.langchain.autolog()` instrumenta cada chamada
  LangChain (LLM, tools, retriever) ao startup. Traces ficam no `mlruns/`
  do container — voláteis entre deploys (decisão deliberada; ver §4).
- **Langfuse** (fallback opcional) — ativa em paralelo se
  `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` estiverem nas env vars do
  Render dashboard. Para persistir traces de produção sem usar disco.
- **Endpoint `/metrics`** — formato Prometheus, exposto publicamente para
  scrape externo.
- **Logs em `logs/<run>_<timestamp>.log`** — também voláteis no Render.

### Camada LOCAL (docker-compose)
- **Prometheus** (porta 9090) — faz scrape do `/metrics` do Render a cada
  15s (ver `infra/prometheus.yml`).
- **Grafana** (porta 3000) — dashboard "Copiloto IBOV — Operacional"
  provisionado automaticamente, com 8 painéis: queries/min, latência
  p50/p95/p99, taxa de bloqueio do guardrail, taxa de erros, distribuição
  por modelo, total de previsões LSTM.
- **MLflow UI** (porta 5000) — apontando para `mlruns/` local, mostra
  histórico de runs de treino + traces de queries feitas localmente.

Setup detalhado em [`docs/DEPLOYMENT.md`](DEPLOYMENT.md).

### Drift detection automático (GAP 06)
Stage `drift_check` no [`dvc.yaml`](../dvc.yaml) computa PSI entre `X_train`
e `X_test` toda vez que os tensores mudam. Resultado vai para
`data/processed/ibov/drift_report.json` (versionado pelo Git para audit
trail) e MLflow (nested run com `tag pipeline_stage='monitoring'`).
Threshold em `cfg.retrain.retrain_trigger_psi=0.20` aciona retreinamento.

## 8. Limitações conhecidas

1. O LSTM tem acurácia direcional ~50% — é filtro de viés, não sinal autônomo.
2. RAG depende da qualidade/atualidade dos PDFs em `data/raw/agent_docs/`.
3. Groq tem rate limit no free tier (e.g. 12k tokens / min).
4. Sem cache de respostas — cada query roda o pipeline completo.
5. **Feature Management sem materialização incremental.** O `feature_engineering.run()` reprocessa o dataset inteiro a cada `dvc repro`, mesmo quando apenas os pregões mais recentes mudaram. Não há feature store dedicado (Feast / Hopsworks / Tecton). Decisão deliberada: o IBOV tem ~1.700 pontos e o reprocessamento custa < 2s — o ROI de implementar materialização incremental seria marginal neste escopo. **Mitigação parcial existente**: (a) DVC versiona os tensores via cache hash-based, evitando recomputação quando deps não mudam; (b) funções de feature engineering (`fit_scaler`, `create_sequences`, `chronological_split`) são compartilhadas entre treino, inferência (`predict.py`) e testes — reuso real, não copy-paste; (c) `dataset_hash` (SHA-256 dos tensores) é tag MLflow obrigatória, dando lineage por run. Para produção em escala (datasets > 1M pontos ou dezenas de features), o caminho seria (i) detecção incremental por hash do CSV bruto + delta-merge no tail, ou (ii) feature store dedicado com materialização sob demanda.

## 9. Conformidade LGPD

Ver `docs/LGPD_PLAN.md`. Resumo: opera apenas sobre dados públicos do
mercado financeiro; nenhum dado pessoal é coletado/armazenado pelo sistema.
PII eventualmente presente em PDFs de RAG é removida no output via Presidio.

## 10. Manutenção

- **Re-ingestão de RAG**: `python -m src.agent_pipeline.rag_retriever`
- **Re-treino LSTM**: `python -m src.ibov_pipeline.retrain`
- **Re-avaliação RAGAS**: `python -m evaluation.ragas_eval`
- **Logs**: `logs/`, MLflow UI, Langfuse dashboard
