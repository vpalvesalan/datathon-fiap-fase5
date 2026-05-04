# Deployment & Observabilidade

Este projeto separa **serving (cloud)** de **observabilidade (local)**:

```
┌──────────────────────────────────────────────────────────┐
│  RENDER (cloud — auto-deploy on git push)                │
│  └─ Container Docker (FastAPI + Gradio + LSTM + Agente)  │
│     - GET  /health                                       │
│     - POST /predict                                      │
│     - POST /agent/query                                  │
│     - GET  /metrics       ← scraped pelo Prometheus      │
│     - GET  /chat          ← UI Gradio                    │
│     - GET  /docs          ← OpenAPI                      │
└──────────────────────────────────────────────────────────┘
                       ▲
                       │ scrape /metrics a cada 15s
                       │
┌──────────────────────────────────────────────────────────┐
│  LOCAL (docker-compose, demo + dev)                      │
│  ├─ Prometheus  (porta 9090)                             │
│  ├─ Grafana     (porta 3000) — dashboards provisionados  │
│  └─ MLflow UI   (porta 5000) — `mlflow ui`               │
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

### Ciclo contínuo (a cada push)

```bash
# 1. Local: gerar artefatos atualizados (modelo + vector store)
dvc repro

# 2. Versionar
git add data/processed/ibov/model_lstm.keras \
        data/processed/ibov/scaler.joblib \
        data/processed/agent_db/ \
        data/processed/ibov/drift_report.json
git commit -m "Re-treino + nova ingestão RAG"

# 3. Push — Render detecta e faz redeploy automaticamente
git push origin main
```

Render lê o webhook do GitHub, faz rebuild da imagem (Docker), passa pelo healthcheck `/health` e migra o tráfego.

### Notas sobre o free tier

- **Sleep após 15 min** de inatividade. Cold start ~30s. Para o Demo Day, faça uma query de aquecimento 1 min antes da apresentação.
- **Sem persistent disk** — `mlruns/` (MLflow tracing) é volátil. Traces de produção somem entre deploys (decisão arquitetural deliberada — ver `docs/SYSTEM_CARD_AGENT.md`).
- **Sem Prometheus/Grafana no Render** — deliberado. Esses rodam localmente; ver seção 2.

### Não quebrar nada que já existe

- `dvc.yaml`, treinamento, testes, evaluation: continuam locais. Render não roda treino.
- `data/raw/`: NÃO é copiado pra imagem (Dockerfile só copia `data/processed/`). Coleta yfinance acontece sob demanda no `predict_next_day()` quando necessário.

---

## 2. Observabilidade Local (Prometheus + Grafana + MLflow)

Stack local levantada via Docker Compose.

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

### Fazer scrape do Render também (opcional)

Edite [`infra/prometheus.yml`](../infra/prometheus.yml), descomente a seção `copiloto-render` e ajuste a URL para o seu domínio do Render. Restart Prometheus:

```bash
docker compose restart prometheus
```

Agora o Grafana mostra métricas tanto locais quanto de produção, com label `env="local"` ou `env="production"`.

### Derrubar a stack

```bash
docker compose down            # mantém volumes (dados ficam)
docker compose down -v         # apaga volumes (limpa tudo)
```

---

## 3. CI vs CD — separação de responsabilidades

| Ambiente | Responsabilidade | Trigger |
|---|---|---|
| **GitHub Actions** ([`ci.yml`](../.github/workflows/ci.yml)) | Lint, type check, security scan, pytest com gate ≥ 60% | Push e PR |
| **Render webhook** | Build da imagem + redeploy | Push em `main` (após CI passar) |

Não há GitHub Actions para deploy — o Render escuta o webhook nativo do GitHub. **Menos secrets, menos código, fluxo mais simples.**

Para forçar deploy manual fora do push:
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

O resultado fica em `data/processed/ibov/drift_report.json` (versionado pelo Git, `cache: false` no DVC) e é também logado no MLflow (experiment `ibov-forecasting`, runs com `tags.pipeline_stage='monitoring'`).

Threshold em `cfg.retrain.retrain_trigger_psi=0.20` — acima disso, classifica como `retrain_required` e a recomendação é rodar `python -m src.ibov_pipeline.retrain`.
