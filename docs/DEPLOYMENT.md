# Deployment & Observabilidade

Este projeto separa **training (local)** de **serving (cloud)** com **observabilidade local**.

---

## 0. Rodando Localmente (do zero)

Fluxo completo para subir e visualizar a aplicação na máquina de desenvolvimento, sem depender do Render.

### Pré-requisitos

- Python 3.12, Docker Desktop rodando
- `GROQ_API_KEY` válida (obrigatório para o agente)
- Dependências instaladas (`pip install -r requirements.txt`) e modelo spaCy baixado (`python -m spacy download pt_core_news_sm`)

### Passo a passo

**1. Treinar o modelo e construir o banco vetorial**

```bash
dvc repro
# Executa: make_dataset → feature_engineering → drift_check
#          → train_baseline → ingest_agent_docs → train_lstm → register_model
# Resultado: data/processed/ibov/model_lstm.keras + data/processed/agent_db/
```

**2. Subir a API + Gradio**

```bash
# Terminal 1
uvicorn src.serving.app:app --host 0.0.0.0 --port 7860 --reload
```

Aguarde a mensagem `Application startup complete.` e acesse:

| Interface | URL |
|---|---|
| Chat (Gradio) | <http://localhost:7860/chat> |
| API interativa (Swagger) | <http://localhost:7860/docs> |
| Healthcheck | <http://localhost:7860/health> |

**3. Testar o agente**

Cole no Gradio ou via curl:

```bash
curl -X POST http://localhost:7860/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual a previsão do IBOV para amanhã? O contexto macroeconômico (S&P, cotação dólar) justifica a variação?"}'
```

**4. (Opcional) Observabilidade completa**

```bash
# Terminal 2 — Prometheus + Grafana
docker compose up -d

# Terminal 3 — MLflow UI
mlflow ui --port 5000
```

| Serviço | URL | Credenciais |
|---|---|---|
| Prometheus | <http://localhost:9090> | — |
| Grafana | <http://localhost:3000> | `admin` / `admin` |
| MLflow UI | <http://localhost:5000> | — |

O dashboard "Copiloto IBOV — Operacional" é provisionado automaticamente no Grafana. As métricas começam a aparecer após as primeiras requisições à API.

---

Arquitetura:
- **Local:** DVC treina → MLflow registra → Git versiona artefatos
- **Cloud (Render):** Puxa código + modelo treinado → Serve sem retreinar
- **Observabilidade:** MLflow/Prometheus/Grafana rodam localmente (não em Render)

```
┌──────────────────────────────────────────────────────────┐
│  LOCAL: Training + Development                           │
│  ├─ dvc repro           → treina modelo                  │
│  ├─ mlflow.log_model()  → registra em mlruns/            │
│  ├─ create_model_version() → versiona no MLflow Registry │
│  └─ git push (modelo + código vão juntos)                │
└──────────────────────────────────────────────────────────┘
            ↓ (webhook GitHub → dispara CI)
┌──────────────────────────────────────────────────────────┐
│  CI Pipeline (GitHub Actions)                            │
│  ├─ Lint, Type Check, Security Scan                      │
│  ├─ Run Tests (pytest)                                   │
│  ├─ Build Docker Image (validação)                       │
│  └─ Trigger Render Deploy (webhook)                      │
└──────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────┐
│  RENDER (cloud — auto-deploy on git push)                │
│  └─ Container Docker (código + modelo treinado)          │
│     ├─ GET  /health                                      │
│     ├─ POST /predict   (usa modelo em data/processed/)   │
│     ├─ POST /agent/query                                 │
│     ├─ GET  /metrics    ← scraped pelo Prometheus local  │
│     ├─ GET  /chat       ← UI Gradio                      │
│     └─ GET  /docs       ← OpenAPI                        │
└──────────────────────────────────────────────────────────┘
                         ▲
                         │ scrape /metrics a cada 15s
                         │ (Prometheus local apenas)
┌──────────────────────────────────────────────────────────┐
│  LOCAL: Observabilidade (docker-compose)                 │
│  ├─ Prometheus  (porta 9090)                             │
│  ├─ Grafana     (porta 3000) — dashboards provisionados  │
│  └─ MLflow UI   (porta 5000) — dev only                  │
└──────────────────────────────────────────────────────────┘
```

---

## 1. Deploy no Render (cloud)

### Setup inicial (1 vez)

1. Criar conta em <https://render.com> (free tier).
2. **New → Blueprint Instance → Connect GitHub** → escolher este repo.
3. Render detecta o [`render.yaml`](../render.yaml) e propõe o serviço `copiloto-ibov`. Clique **Create**.
4. Após o primeiro build (~5 min), abra o serviço → aba **Environment** e configure as variáveis sensíveis:

   | Variável | Origem |
   |---|---|
   | `GROQ_API_KEY` | <https://console.groq.com/keys> |
   | `LANGFUSE_PUBLIC_KEY` (opcional) | <https://cloud.langfuse.com> |
   | `LANGFUSE_SECRET_KEY` (opcional) | idem |
   | `LANGFUSE_HOST` (opcional) | `https://cloud.langfuse.com` |

5. Após salvar as env vars, o Render redeploya. Espere o **healthcheck** ficar verde.

### Ciclo contínuo: Treinar → Versionar → Deploy

```bash
# ============================================================
# FASE 1: LOCAL (Training)
# ============================================================

# 1.1 Treinar o modelo (DVC)
dvc repro

# 1.2 Registrar no MLflow (observabilidade local)
python -c "
import mlflow
from mlflow import MlflowClient

# Training deve fazer mlflow.keras.log_model(model, 'lstm-ibov-model')
# Depois registra versão no Model Registry:

client = MlflowClient()
result = client.create_model_version(
    name='lstm-ibov',
    source='runs:/[RUN_ID]/lstm-ibov-model',  # substitua RUN_ID do run anterior
    stage='Production'
)
print(f'✓ Registered model version {result.version}')
"

# 1.3 Validar modelo localmente (opcional)
uvicorn src.serving.app:app --port 7860 --reload
# Teste em http://localhost:7860/chat

# ============================================================
# FASE 2: VERSIONAMENTO (Git)
# ============================================================

# 2.1 Versioná artefatos (modelo + configs)
git add data/processed/ibov/model_lstm.keras \
        data/processed/ibov/scaler.joblib \
        data/processed/agent_db/ \
        data/processed/ibov/drift_report.json \
        dvc.lock

# 2.2 Commit com contexto
git commit -m "Re-treino v2: acurácia 0.94, PSI=0.15

- Drift detection: PASS (PSI < 0.20)
- MLflow run: [ID do run local]
- Modelo: lstm-ibov/Production"

# ============================================================
# FASE 3: DEPLOY (GitHub Actions → Render)
# ============================================================

# 3.1 Push (dispara CI + Render deploy)
git push origin main

# CI (GitHub Actions) automaticamente:
# - Lint, type-check, security scan
# - Run tests (pytest)
# - Build Docker (valida Dockerfile + cópia de data/processed/)
# - Trigger Render webhook (deploy)

# Render:
# - git pull (puxa código + modelo)
# - docker build (COPY data/processed/ibov/ para imagem)
# - docker run (uvicorn inicia)
# ✓ Serviço online em ~30-60s
```

**Fluxo automático após `git push`:**
```
git push
  ↓ (webhook GitHub)
GitHub Actions CI
  ├─ Validações passam
  ├─ Docker build OK
  └─ curl webhook Render
      ↓
Render
  ├─ git pull
  ├─ docker build (com data/processed/)
  ├─ docker run
  └─ ✓ Online
```

### Notas sobre o free tier do Render

- **Sleep após 15 min** de inatividade. Cold start ~30s.
- **Sem persistent disk** — `mlruns/` não sobe. Decisão deliberada: MLflow roda **localmente** para dev; em produção, servimos o modelo já treinado de `data/processed/` (imutável no container).
- **Sem Prometheus/Grafana em Render** — deliberado. Esses rodam localmente (docker-compose); métricas de produção podem ser scrapeadas localmente se configurado.

### Estrutura da imagem Docker

```dockerfile
# src/serving/Dockerfile (multi-stage build)
FROM python:3.12-slim AS builder
  # ... instala dependências ...

FROM python:3.12-slim AS runtime
  COPY src/ /app/src/                         # código
  COPY data/processed/ibov/ /app/data/...     # ← MODELO TREINADO
  COPY data/processed/agent_db/ /app/data/... # ← VECTOR STORE
  # ... inicia uvicorn ...
```

**O modelo é imutável na imagem.** Para retreinar, o fluxo é:
1. Local: `dvc repro` + `mlflow log`
2. Git: commit novo artefatos
3. Push: dispara novo build no Render com modelo atualizado

### CI: Monitorar mudanças em dados

O arquivo [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) monitora:
- Mudanças em `data/processed/ibov/model_lstm.keras` → dispara CI + deploy
- Mudanças em `data/processed/agent_db/` → dispara CI + deploy
- Mudanças em `src/serving/Dockerfile` → dispara CI (validação)

---

## 2. Observabilidade Local (Prometheus + Grafana + MLflow)

Stack local levantada via Docker Compose. **Não sobe para Render.**

### Subir a stack

```bash
# Terminal 1 — FastAPI (uvicorn) com endpoint /metrics
uvicorn src.serving.app:app --host 0.0.0.0 --port 7860 --reload

# Terminal 2 — Prometheus + Grafana
docker compose up -d

# Terminal 3 — MLflow UI (lê de mlruns/)
mlflow ui --port 5000
```

### Acessos

| Serviço | URL | Credenciais |
|---|---|---|
| FastAPI | <http://localhost:7860/docs> | — |
| Gradio Chat | <http://localhost:7860/chat> | — |
| Prometheus | <http://localhost:9090> | — |
| Grafana | <http://localhost:3000> | `admin` / `admin` (ou anonymous) |
| MLflow UI | <http://localhost:5000> | — |

### Dashboard Grafana

O dashboard "Copiloto IBOV — Operacional" é provisionado automaticamente em `Copiloto IBOV/copiloto`. Painéis:

1. Queries por minuto
2. Latência p95
3. Taxa de bloqueio (guardrail)
4. Taxa de erros internos
5. Queries/min ao longo do tempo (por status)
6. Latência p50/p95/p99
7. Total de previsões LSTM
8. Distribuição de queries por modelo

### MLflow no Local

Após rodar `dvc repro` + `mlflow.log_model()`, os runs aparecem em <http://localhost:5000>:
- **Experimento:** `ibov-forecasting`
- **Tags importantes:**
  - `pipeline_stage` (treino, avaliação, drift check)
  - `data_version` (data de coleta)
  - `model_version` (se registrado)

### Fazer scrape do Render também (opcional)

Para monitorar produção também no Grafana local:

1. Edite [`infra/prometheus.yml`](../infra/prometheus.yml), descomente a seção `copiloto-render`
2. Ajuste a URL para seu domínio do Render (ex: `https://copiloto-ibov.onrender.com/metrics`)
3. Restart Prometheus:

```bash
docker compose restart prometheus
```

Agora o Grafana mostra métricas de dev + produção com labels `env="local"` e `env="production"`.

### Derrubar a stack

```bash
docker compose down            # mantém volumes (dados e dashboards ficam)
docker compose down -v         # apaga volumes (limpa tudo)
```

---

## 3. CI vs CD — separação de responsabilidades

| Ambiente | Responsabilidade | Trigger |
|---|---|---|
| **GitHub Actions** ([`ci.yml`](../.github/workflows/ci.yml)) | Lint, type check, security scan, pytest (gate ≥ 60%), docker build validation | Push e PR (se caminhos monitorados mudarem) |
| **Render webhook** | Build imagem + redeploy automático | Push em `main` (após CI passar) |

**Caminhos monitorados no CI:**
- `src/` — código
- `tests/`, `evaluation/` — testes
- `data/processed/ibov/model_lstm.keras` — modelo
- `data/processed/agent_db/` — vector store
- `src/serving/Dockerfile` — imagem

Não há GitHub Actions explícito para deploy — o Render escuta webhook nativo do GitHub. **Menos secrets, menos código, fluxo mais simples.**

Para forçar deploy manual:
```bash
# No dashboard do Render: Service → Manual Deploy → Deploy Latest Commit
```

---

## 4. Drift detection automático (DVC)

O drift é checado automaticamente sempre que os artefatos do feature engineering mudarem:

```bash
dvc repro                    # roda o DAG inteiro
# OU forçar só o drift:
dvc repro drift_check
```

O resultado fica em `data/processed/ibov/drift_report.json` (versionado pelo Git, `cache: false` no DVC) e é também logado no MLflow.

Threshold em `cfg.retrain.retrain_trigger_psi=0.20` — acima disso, classifica como `retrain_required` e recomenda rodar `python -m src.ibov_pipeline.retrain`.

---

## Resumo: Decisões Arquiteturais

| Decisão | Razão |
|---|---|
| **Modelo + código no Git** | Simplicidade, reproducibilidade, compatível com Render free tier |
| **DVC para treinamento** | Versionamento de dados/artefatos, pipeline reproducível |
| **MLflow local** | Observabilidade em dev (métricas, parâmetros, drift) |
| **Render sem persistent disk** | Free tier; modelo é imutável (gerado localmente) |
| **GitHub webhook para deploy** | Menos infra, sem secrets extras |
| **CI valida Docker build** | Garante que imagem sobe com modelo novo sem erros |

