# Ultimate TruthGPT Dockerfile
# ============================
# 
# Dockerfile para la aplicación definitiva de TruthGPT con todas
# las características ultra avanzadas integradas.

# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_ultimate_truthgpt.txt .
RUN pip install --no-cache-dir -r requirements_ultimate_truthgpt.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('all')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache

# Set permissions
RUN chmod +x start_ultimate_truthgpt.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "start_ultimate_truthgpt.py"]

























