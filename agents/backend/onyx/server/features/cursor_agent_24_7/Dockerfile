# Multi-stage Dockerfile optimizado para AWS y Docker Compose
# ===========================================================
# Stage 1: Builder - Compilar dependencias
FROM python:3.11-slim as builder

WORKDIR /build

# Instalar dependencias del sistema necesarias para compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Instalar Flower para monitoreo de Celery (opcional)
RUN pip install --no-cache-dir --user flower

# Stage 2: Runtime - Imagen final optimizada
FROM python:3.11-slim

WORKDIR /app

# Instalar solo runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias instaladas desde builder
COPY --from=builder /root/.local /root/.local

# Asegurar que scripts en .local están en PATH
ENV PATH=/root/.local/bin:$PATH

# Copiar código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/data /app/logs

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8024/api/health || exit 1

# Exponer puertos
EXPOSE 8024 5555

# Comando por defecto (puede ser sobrescrito en docker-compose)
CMD ["python", "main.py", "--mode", "api", "--host", "0.0.0.0", "--port", "8024"]

