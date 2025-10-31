from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from typing import Callable

from typing import Any, List, Dict, Optional
import logging
import asyncio
"""Métricas Prometheus e instrumentación automática para FastAPI LLM API."""

instrumentator = Instrumentator()
REQUESTS = Counter('llm_requests_total', 'Total LLM API requests', ['endpoint'])
ERRORS = Counter('llm_errors_total', 'Total LLM API errors', ['endpoint'])
LATENCY = Histogram('llm_request_latency_seconds', 'LLM API request latency', ['endpoint'])

def register_custom_metric(metric: Callable) -> None:
    """Permite registrar métricas customizadas adicionales (Datadog, etc)."""
    # Ejemplo: metric puede ser un Counter, Gauge, etc.
    pass 