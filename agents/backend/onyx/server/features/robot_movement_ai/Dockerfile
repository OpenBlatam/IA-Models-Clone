# Dockerfile para Robot Movement AI v2.0
# Multi-stage build para optimizar tamaño de imagen

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Instalar solo runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias instaladas desde builder
COPY --from=builder /root/.local /root/.local

# Asegurar que scripts estén en PATH
ENV PATH=/root/.local/bin:$PATH

# Copiar código de la aplicación
COPY . .

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 robotuser && \
    chown -R robotuser:robotuser /app

USER robotuser

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8010 \
    HOST=0.0.0.0

# Exponer puerto
EXPOSE 8010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Comando por defecto
CMD ["python", "-m", "robot_movement_ai.main", "--host", "0.0.0.0", "--port", "8010"]




