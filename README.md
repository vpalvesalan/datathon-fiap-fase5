# Copiloto Analítico do IBOV

> **Tech Challenge — Fase 5 (Datathon)** • FIAP MLET (Pós-Tech)
> Sistema MLOps end-to-end que cruza um modelo quantitativo (LSTM) com um agente
> generativo (LLM + RAG) para apoiar analistas na compreensão de movimentos do
> Índice Bovespa.

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Coverage](https://img.shields.io/badge/coverage-63%25-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Status](https://img.shields.io/badge/status-Datathon%20ready-success.svg)

---

## TL;DR

Dois domínios independentes, integrados via uma única tool:

1. **Domínio quantitativo** — LSTM treinado sobre o histórico do IBOV via yfinance,
   com tracking MLflow, busca de hiperparâmetros (nested runs), holdout imutável e
   detecção de drift via PSI.
2. **Domínio generativo** — Agente ReAct (LangChain + Groq Llama 3.3 70B) com 4 tools
   (forecast LSTM, RAG sobre Focus/Copom, calculadora, cotações live) e guardrails
   de input (anti-prompt-injection) e output (Presidio PII removal).

Tudo exposto via FastAPI + Gradio em um único container, deployado no Render via
`git push`. Observabilidade local (Prometheus + Grafana + MLflow UI) faz scrape do
endpoint `/metrics` da nuvem.

## Arquitetura

```
                 ┌──────────────────────────────────────────┐
                 │  RENDER (cloud — auto-deploy on push)    │
                 │  Container Docker (FastAPI + Gradio)     │
                 │     ├──► /predict      → LSTM (Registry) │
                 │     ├──► /agent/query  → ReAct + Tools   │
                 │     │       ├─ ibov_forecast             │
                 │     │       ├─ macro_rag (ChromaDB)      │
                 │     │       ├─ calculator                │
                 │     │       └─ market_context (yfinance) │
                 │     ├──► /chat         → UI Gradio       │
                 │     ├──► /metrics      → Prometheus fmt  │
                 │     └──► /docs         → OpenAPI         │
                 └──────────────────────────────────────────┘
                                  ▲
                                  │ scrape /metrics a cada 15s
                                  │
                 ┌──────────────────────────────────────────┐
                 │  LOCAL (docker-compose, dev + Demo Day)  │
                 │  ├─ Prometheus  (9090)                   │
                 │  ├─ Grafana     (3000) — 8 painéis       │
                 │  └─ MLflow UI   (5000) — runs + traces   │
                 └──────────────────────────────────────────┘
```

## Demo

| Recurso | Endpoint |
|---|---|
| API REST (OpenAPI) | `/docs` |
| Chat UI (Gradio com streaming de raciocínio) | `/chat` |
| Smoke check | `/health` |
| Métricas Prometheus | `/metrics` |

### Exemplos de perguntas para o Copiloto

Acesse via **Gradio** (`/chat`) ou via **API REST**:

```bash
# Via curl (substitua <host> pela URL do Render ou localhost:7860)
curl -X POST <host>/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual a previsão do IBOV para amanhã? O contexto macroeconômico (S&P, cotação dólar) justifica a variação?"}'
```

Perguntas que exercitam as 4 tools do agente:

| Pergunta | Tools acionadas |
|---|---|
| "Qual a previsão do IBOV para amanhã? O contexto macroeconômico (S&P, cotação dólar) justifica a variação?" | `ibov_forecast` + `market_context` |
| "O que o relatório Focus diz sobre as expectativas de inflação para os próximos meses?" | `macro_rag` |
| "Se o IBOV subir 2,5% amanhã partindo de 128.450 pontos, qual será o valor final?" | `calculator` |
| "Quais são os principais riscos macroeconômicos para o mercado brasileiro citados nos relatórios do Copom?" | `macro_rag` |

## Quick Start

### Pré-requisitos
- Python 3.12, Git, Docker (opcional para observabilidade)
- Chave de API do [Groq](https://console.groq.com/keys) (free tier suficiente)
- Opcional: chaves do [Langfuse](https://cloud.langfuse.com) (fallback de tracing)

### Setup
```bash
# 1. Clone e ambiente
git clone <repo-url>
cd datathon-techchallenge-fase-5

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download pt_core_news_sm   # NER PT-BR para Presidio
python -m ipykernel install --user --name ibov_llm_pipeline_env \
  --display-name "Python (IBOV Pipeline)"  # kernel para os notebooks

# 2. Variáveis de ambiente
cp .env.example .env
# Edite .env e preencha GROQ_API_KEY (obrigatório)

# 3. Pipeline de dados + treino (DVC orquestra tudo)
dvc init        # uma vez só
dvc repro       # roda: make_dataset → feature_engineering → drift_check
                #       → train_baseline → ingest_agent_docs → train_lstm

# 4. (Opcional) Executar notebooks analíticos
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name="ibov_llm_pipeline_env" \
  notebooks/01_eda_ibov.ipynb               # EDA do IBOV
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name="ibov_llm_pipeline_env" \
  notebooks/02_lstm_training_report.ipynb   # Relatório do treino (lê do MLflow)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name="ibov_llm_pipeline_env" \
  notebooks/03_llm_evaluation.ipynb         # Visualização da avaliação do LLM

# 5. Smoke test do agente (CLI, antes de subir o servidor)
python -m src.agent_pipeline.react_agent "Qual a previsão do IBOV para amanhã?"

# 6. Servir
uvicorn src.serving.app:app --host 0.0.0.0 --port 7860 --reload
# Acessar http://localhost:7860/chat
```

### Observabilidade local (opcional, para Demo Day)
```bash
docker compose up -d            # Prometheus (9090) + Grafana (3000)
mlflow ui --port 5000           # runs + traces
```

## Pipeline MLOps (DVC)

[`dvc.yaml`](dvc.yaml) define 6 stages com dependências e outputs declarativos:

```
make_dataset (yfinance + pandera schema)
    └─► feature_engineering (MinMaxScaler + sliding windows + chronological split)
            ├─► drift_check (PSI; logado no MLflow + drift_report.json)
            ├─► train_baseline (Sklearn: LinearRegression, Ridge, RandomForest)
            └─► ingest_agent_docs (PDFs Focus/Copom → embeddings PT-BR → ChromaDB)
                    │
                    └─► train_lstm (TF/Keras; grid search com nested runs MLflow)
                            └─► register_model (Staging no MLflow Registry)
```

`dvc repro` re-executa apenas o que foi invalidado por mudança de código,
configuração ou dados upstream.

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| Modelo quantitativo | TensorFlow / Keras (LSTM 2 camadas + Dropout) |
| Agente / orquestração | LangChain ReAct |
| LLM | Llama 3.3 70B via [Groq](https://groq.com) (free tier) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace, on-device) |
| Vector store | ChromaDB persistente |
| Serving | FastAPI + Uvicorn |
| UI | Gradio (montado em `/chat` no mesmo container) |
| Tracking & Registry | MLflow (runs + autolog de tracing LangChain) |
| Versionamento de dados | DVC |
| Métricas operacionais | Prometheus + Grafana (local) |
| Métricas LLM | MLflow Tracing (primário) + Langfuse (opcional) |
| Drift | PSI custom + integração MLflow |
| Avaliação RAG | RAGAS (4 métricas) + LLM-as-judge (3 critérios) + A/B prompts |
| Segurança | Regex anti-prompt-injection + Microsoft Presidio (PII PT-BR) |
| Deploy cloud | Render (auto-deploy via webhook GitHub) |
| CI | GitHub Actions (ruff + mypy + bandit + pytest com gate ≥60%) |
| Conteinerização | Docker multi-stage (usuário não-root, healthcheck) |

## Estrutura do Projeto

```
datathon-techchallenge-fase-5/
├── src/
│   ├── ibov_pipeline/          # Domínio 1 — LSTM
│   │   ├── make_dataset.py     # yfinance + pandera
│   │   ├── feature_engineering.py
│   │   ├── train_baseline.py   # Sklearn baselines
│   │   ├── train_lstm.py       # LSTM + grid search nested runs
│   │   ├── register_model.py   # MLflow Registry com tags de governança
│   │   ├── retrain.py          # Champion-challenger
│   │   ├── predict.py          # Helper consumido pela tool do agente
│   │   ├── logging_config.py   # Logging dual (console + arquivo)
│   │   ├── config.py           # Pydantic loader
│   │   └── configs/model_config.yaml
│   ├── agent_pipeline/         # Domínio 2 — Agente
│   │   ├── react_agent.py      # ChatGroq + ReAct
│   │   ├── tools.py            # 4 tools customizadas
│   │   ├── rag_retriever.py    # Ingestão PDF + ChromaDB
│   │   ├── config.py
│   │   └── configs/agent_config.yaml
│   ├── serving/
│   │   ├── app.py              # FastAPI + endpoints + montagem Gradio
│   │   ├── gradio_app.py       # UI com streaming de raciocínio
│   │   ├── schemas.py          # Pydantic request/response
│   │   └── Dockerfile          # Multi-stage, usuário não-root
│   ├── security/
│   │   └── guardrails.py       # InputGuardrail + OutputGuardrail (Presidio)
│   └── monitoring/
│       ├── drift_detection.py  # PSI + integração MLflow + JSON
│       └── telemetry.py        # MLflow Tracing + Prometheus + Langfuse
├── tests/                      # 84 testes, cobertura 63% (gate ≥60%)
│   └── README.md               # Guia da suite de testes (filosofia, fixtures, como rodar)
├── notebooks/
│   ├── 01_eda_ibov.ipynb       # EDA: distribuições, sazonalidade, ACF, ADF
│   ├── 02_lstm_training_report.ipynb   # Relatório lendo do MLflow
│   └── 03_llm_evaluation.ipynb # Visualização dos resultados de avaliação do LLM
├── evaluation/
│   ├── ragas_eval.py           # 4 métricas RAGAS (incremental + retomada)
│   ├── llm_judge.py            # ≥3 critérios de negócio
│   ├── ab_test_prompts.py      # ≥3 system prompts comparados
│   ├── benchmark_llm.py        # ≥3 configurações de LLM
│   └── _rate_limit.py          # Token rate limiter (Groq free tier)
├── infra/                      # Configs Prometheus + Grafana (local)
│   ├── prometheus.yml
│   └── grafana/
│       ├── provisioning/       # Datasource + dashboard auto-loaded
│       └── dashboards/copiloto.json
├── docs/                       # Governança e documentação
│   ├── MODEL_CARD_IBOV.md
│   ├── SYSTEM_CARD_AGENT.md
│   ├── OWASP_MAPPING.md        # 6 ameaças LLM mapeadas
│   ├── RED_TEAM_REPORT.md      # 6 cenários adversariais
│   ├── LGPD_PLAN.md
│   └── DEPLOYMENT.md
├── data/                       # gerenciado por DVC; não commitado
│   ├── raw/
│   │   ├── ibov/
│   │   └── agent_docs/         # PDFs (Focus, Copom, análises macro)
│   ├── processed/
│   │   ├── ibov/               # tensores + scaler + drift_report.json
│   │   └── agent_db/           # ChromaDB persistido
│   ├── holdout_test/           # imutável
│   └── golden_set/perguntas.json   # ≥20 pares para avaliação RAGAS
├── .github/workflows/ci.yml    # CI: lint → type → security → tests
├── docker-compose.yml          # Stack de observabilidade local
├── render.yaml                 # Blueprint Render (cloud)
├── dvc.yaml                    # DAG de dados
├── pytest.ini + .coveragerc    # Quality gate ≥60%
├── pyproject.toml / requirements.txt
└── arquitetura.txt
```

## Maturidade MLOps — Nível 2 (Microsoft Maturity Model)

Mapeamento explícito das 6 dimensões avaliadas pela banca:

| Dimensão | Nível 2 esperado | Implementação |
|---|---|---|
| **Experiment Management** | MLflow padronizado, params + metrics + artifacts | `train_lstm.py` com nested runs, lineage tags (`git_sha`, `dataset_hash`), métricas test + holdout |
| **Model Management** | Registry com versionamento e metadata obrigatória | `register_model.py` com 12 tags de governança, transição automática para Staging |
| **CI/CD** | GitHub Actions: lint → test → build → deploy | `ci.yml` (ruff + mypy + bandit + pytest 60%) → Render webhook auto-deploy |
| **Monitoring** | Métricas, drift, dashboard, alertas | Prometheus (operacional) + MLflow Tracing (LLM) + PSI (drift) → Grafana 8 painéis |
| **Data Management** | DVC, versionamento, dados sintéticos em dev | `dvc.yaml` + `dataset_hash` MLflow + fixtures sintéticas em `tests/conftest.py` |
| **Feature Management** | Features compartilhadas, materialização incremental | Funções compartilhadas (`fit_scaler`, `create_sequences`); materialização incremental documentada como limitação aceita (ver `SYSTEM_CARD_AGENT.md` §8) |

## CI/CD

### CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
A cada push e PR:
1. **ruff** — lint
2. **mypy** — type check
3. **bandit** — security scan
4. **pytest** com `--cov-fail-under=60` — quality gate

### CD (Render)
Auto-deploy nativo via webhook do GitHub — sem GitHub Actions adicionais. A cada
push em `main`, Render rebuilda a imagem ([`src/serving/Dockerfile`](src/serving/Dockerfile)),
passa pelo healthcheck `/health` e migra o tráfego.

Setup detalhado em [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Avaliação

```bash
# RAGAS (4 métricas, com persistência incremental + rate limiter)
python -m evaluation.ragas_eval

# LLM-as-judge (3 critérios de negócio)
python -m evaluation.llm_judge

# A/B test de prompts (≥3 variantes)
python -m evaluation.ab_test_prompts

# Benchmark de LLMs (≥3 configurações)
python -m evaluation.benchmark_llm

# Drift detection
python -m src.monitoring.drift_detection
```

Resultados em `evaluation/results/*.json`.

## Documentação Detalhada

| Documento | Conteúdo |
|---|---|
| [`MODEL_CARD_IBOV.md`](docs/MODEL_CARD_IBOV.md) | Padrão Mitchell et al. — dados, métricas, limitações do LSTM |
| [`SYSTEM_CARD_AGENT.md`](docs/SYSTEM_CARD_AGENT.md) | Arquitetura completa, decisões e lineage de iterações |
| [`OWASP_MAPPING.md`](docs/OWASP_MAPPING.md) | 6 ameaças OWASP Top 10 LLM com mitigações |
| [`RED_TEAM_REPORT.md`](docs/RED_TEAM_REPORT.md) | 6 cenários adversariais testados |
| [`LGPD_PLAN.md`](docs/LGPD_PLAN.md) | Plano de conformidade Lei 13.709/2018 |
| [`DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Guia de deploy (Render) e observabilidade local |
| [`arquitetura.txt`](arquitetura.txt) | Tree comentado da estrutura de diretórios |

## Limitações & Roadmap

Limitações conhecidas estão documentadas honestamente em
[`SYSTEM_CARD_AGENT.md`](docs/SYSTEM_CARD_AGENT.md) seção 8. Resumo:

- LSTM univariado tem acurácia direcional ≈ 50% — é **filtro de viés**, não
  sinal autônomo (uso pretendido alinhado com a Hipótese de Mercado Eficiente).
- RAG depende dos PDFs em `data/raw/agent_docs/` (curadoria humana).
- Groq free tier tem rate limit de 12k tokens/min e 1M tokens/dia; mitigado
  por `evaluation/_rate_limit.py` com retomada incremental.
- Sem materialização incremental de features; reprocesso completo é < 2s.
- Sem cache de respostas no agente — cada query roda o pipeline ReAct completo.

Roadmap não-bloqueante:
- Cron de drift via GitHub Actions Schedule (hoje é via `dvc repro` manual).
- Persistent disk no Render para preservar traces MLflow entre deploys.
- Feature store dedicado (Feast) se o domínio crescer para múltiplos ativos.

## Autor

**Alan Alves** — FIAP MLET Pós-Tech, Datathon Fase 5

DPO (encarregado LGPD): Ricardo Cataldi

## Licença

MIT — ver [`LICENSE`](LICENSE).

## Referências

- Yao et al. (2023). _ReAct: Synergizing Reasoning and Acting in Language Models._ ICLR.
- Es et al. (2024). _RAGAS: Automated Evaluation of Retrieval Augmented Generation._
- Mitchell et al. (2019). _Model Cards for Model Reporting._ FAT*.
- OWASP (2025). _Top 10 for Large Language Model Applications._
- Microsoft (2024). _MLOps Maturity Model._ Azure ML Documentation.
