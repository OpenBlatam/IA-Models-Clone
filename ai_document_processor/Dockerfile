FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    antiword \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p logs temp models

# Crear usuario no-root
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Exponer puerto
EXPOSE 8001

# Comando de inicio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]


