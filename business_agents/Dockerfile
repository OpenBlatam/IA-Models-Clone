FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/*

# Copy service code
COPY . /app

# Install minimal runtime deps (extend as needed)
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pydantic orjson ujson python-multipart aiofiles httpx tenacity redis prometheus-client PyJWT

EXPOSE 8000

CMD ["uvicorn", "ultimate_quantum_ai_app:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools", "--timeout-keep-alive", "5"]







