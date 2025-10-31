from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import structlog
import logging
import orjson
from typing import Optional, Dict

    from .logging_utils import logger
from typing import Any, List, Dict, Optional
import asyncio
"""
Logging estructurado y ultra-rápido para FastAPI LLM API.
- Solo orjson para serialización de logs (requisito).
- structlog cache_logger_on_first_use=True para máxima velocidad.

Ejemplo de uso:
    logger.info({"event": "test", "request_id": "abc"})
"""

def _orjson_renderer(_, __, event_dict) -> Any:
    return orjson.dumps(event_dict).decode()


def configure_logging(level: int = logging.INFO) -> None:
    """Configura structlog y logging estándar para máxima velocidad y limpieza."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            _orjson_renderer
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [logging.StreamHandler()]


def get_logger(context: Optional[Dict] = None) -> structlog.BoundLogger:
    """Devuelve un logger structlog con contexto extra (request_id, user, etc)."""
    if context:
        return structlog.get_logger("llm_inference_api").bind(**context)
    return structlog.get_logger("llm_inference_api")

logger = get_logger() 