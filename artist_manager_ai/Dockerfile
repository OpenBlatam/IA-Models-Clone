FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/data /app/backups /app/exports

# Exponer puerto
EXPOSE 8000

# Comando por defecto
# Nota: Ajustar según estructura del proyecto
CMD ["python", "-m", "uvicorn", "api.app_factory:app", "--host", "0.0.0.0", "--port", "8000"]

