# ============================================================================
# GitHub Autonomous Agent - Dockerfile
# ============================================================================
# Multi-stage build para imagen optimizada
# 
# Build:
#   docker build -t github-autonomous-agent:latest .
#
# Run:
#   docker run -d -p 8030:8030 --env-file .env github-autonomous-agent:latest
#
# Optimizaciones:
#   - Multi-stage build para reducir tamaño de imagen
#   - Usuario no-root para seguridad
#   - Health check integrado
#   - Cache de dependencias optimizado
# ============================================================================

# Stage 1: Builder
FROM python:3.11-slim as builder

# Metadata
LABEL maintainer="GitHub Autonomous Agent Team"
LABEL description="GitHub Autonomous Agent - Agente autónomo para GitHub"
LABEL version="1.0.0"

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requirements
COPY requirements.txt requirements-prod.txt ./

# Instalar dependencias Python
RUN pip install --no-cache-dir --user -r requirements.txt -r requirements-prod.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Instalar solo dependencias runtime del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/storage && \
    chown -R appuser:appuser /app

# Copiar dependencias instaladas desde builder
COPY --from=builder /root/.local /home/appuser/.local

# Copiar código de la aplicación
COPY --chown=appuser:appuser . .

# Asegurar que scripts sean ejecutables
RUN chmod +x scripts/*.sh scripts/*.py 2>/dev/null || true

# Cambiar a usuario no-root
USER appuser

# Agregar .local/bin al PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8030 \
    HOST=0.0.0.0

# Exponer puerto
EXPOSE 8030

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8030/health || exit 1

# Comando por defecto
# Usa variables de entorno PORT y HOST si están disponibles
CMD ["sh", "-c", "uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8030}"]


