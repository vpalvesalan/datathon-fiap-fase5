# =============================================================================
# Dockerfile do serviço — multi-stage build para imagem mínima.
#
# Arquitetura: Modelo treinado localmente + versionado no Git vem junto.
# O treino NÃO acontece aqui; apenas serve o modelo já treinado.
#
# Build:  docker build -f src/serving/Dockerfile -t copiloto-ibov:latest .
# Run:    docker run -p 7860:7860 --env-file .env copiloto-ibov:latest
#
# Fluxo:
#   LOCAL: dvc repro → treina modelo → git commit data/processed/
#          ↓
#   RENDER: docker build (copia data/processed/) → docker run (serve)
# =============================================================================

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# -----------------------------------------------------------------------------

FROM python:3.12-slim AS runtime

WORKDIR /app

# Copia deps compiladas do builder (sem build tools no runtime)
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Código + artefatos de modelo (treinados localmente, versionados no Git)
# model_lstm.keras, scaler.joblib, drift_report.json
COPY data/processed/ibov/ /app/data/processed/ibov/
# vector store ChromaDB (RAG)
COPY data/processed/agent_db/ /app/data/processed/agent_db/
COPY src/ /app/src/

# Usuário não-root
RUN useradd -m -u 1000 apiuser && chown -R apiuser /app
USER apiuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
