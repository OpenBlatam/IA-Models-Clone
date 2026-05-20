"""
Prometheus Metrics Handler - Endpoint para métricas Prometheus
==============================================================

Handler para exponer métricas en formato Prometheus.
"""

import logging
from typing import Dict, Any
from fastapi import Response
from fastapi.responses import PlainTextResponse

from ..observability import MCPObservability

logger = logging.getLogger(__name__)


async def get_prometheus_metrics(
    observability: MCPObservability
) -> Response:
    """
    Obtener métricas en formato Prometheus.
    
    Args:
        observability: Observability manager
    
    Returns:
        Response con métricas en formato Prometheus
    """
    try:
        # Intentar obtener métricas de Prometheus si están disponibles
        if observability and observability.metrics:
            try:
                from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
                metrics_data = generate_latest()
                return Response(
                    content=metrics_data,
                    media_type=CONTENT_TYPE_LATEST
                )
            except ImportError:
                logger.warning("prometheus_client not available, using fallback")
            except Exception as e:
                logger.warning(f"Error generating Prometheus metrics: {e}")
        
        # Fallback: usar formato simple
        from ..utils.metrics_helpers import get_metrics_collector, format_prometheus_metrics
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        formatted = format_prometheus_metrics(metrics)
        
        return PlainTextResponse(
            content=formatted,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}", exc_info=True)
        return PlainTextResponse(
            content="# Error generating metrics\n",
            status_code=500,
            media_type="text/plain"
        )

