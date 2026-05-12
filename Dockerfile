# =============================================================================
# Dockerfile do serviço — multi-stage build para imagem mínima e segura.
#
# Arquitetura: Modelo (LSTM) treinado localmente + versionado no Git.
# O container utiliza permissões de usuário não-root (apiuser) para rodar
# de forma segura em ambientes como Render ou Kubernetes.
#
# Build:  docker build -t copiloto-ibov:latest .
# Run:    docker run -p 7860:7860 --env-file .env copiloto-ibov:latest
#
# Fluxo:
#   LOCAL: dvc repro → treina modelo → git commit data/processed/
#          ↓
#   RENDER: docker build (instala deps em /usr/local) → docker run (serve como apiuser)
# =============================================================================

# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.12-slim AS runtime

# Criar o usuário ANTES de copiar os arquivos
RUN useradd -m -u 1000 apiuser
WORKDIR /app

# Copia as dependências do builder para o local global do sistema no runtime
# Assim elas ficam acessíveis para qualquer usuário (incluindo apiuser)
COPY --from=builder /install /usr/local

# Copia código e artefatos
COPY data/processed/ibov/ /app/data/processed/ibov/
COPY data/processed/agent_db/ /app/data/processed/agent_db/
COPY src/ /app/src/

# Garante que o apiuser é dono da pasta /app
RUN chown -R apiuser:apiuser /app

USER apiuser

# O PATH padrão /usr/local/bin já incluirá o uvicorn agora
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

# Recomendado: use 'python -m uvicorn' para evitar problemas de PATH
CMD ["python", "-m", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
