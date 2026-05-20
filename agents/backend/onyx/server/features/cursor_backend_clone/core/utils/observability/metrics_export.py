"""
Metrics Export - Exportación de Métricas
=========================================

Sistema para exportar métricas a sistemas externos (Prometheus, StatsD, etc.)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricExport:
    """Exportación de métrica"""
    name: str
    value: float
    metric_type: str  # counter, gauge, histogram, summary
    labels: Dict[str, str]
    timestamp: datetime


class MetricsExporter:
    """
    Exportador de métricas.
    
    Permite exportar métricas a sistemas externos.
    """
    
    def __init__(self):
        self.exporters: Dict[str, Callable] = {}
        self.enabled_exporters: set[str] = set()
    
    def register_exporter(
        self,
        name: str,
        exporter_func: Callable[[List[MetricExport]], None]
    ) -> None:
        """
        Registrar exportador.
        
        Args:
            name: Nombre del exportador
            exporter_func: Función que exporta métricas
        """
        self.exporters[name] = exporter_func
        logger.info(f"📤 Metrics exporter registered: {name}")
    
    def enable_exporter(self, name: str) -> None:
        """Habilitar exportador"""
        if name in self.exporters:
            self.enabled_exporters.add(name)
            logger.debug(f"📤 Exporter enabled: {name}")
        else:
            logger.warning(f"Exporter '{name}' not found")
    
    def disable_exporter(self, name: str) -> None:
        """Deshabilitar exportador"""
        self.enabled_exporters.discard(name)
        logger.debug(f"📤 Exporter disabled: {name}")
    
    async def export(
        self,
        metrics: List[MetricExport]
    ) -> None:
        """
        Exportar métricas a todos los exportadores habilitados.
        
        Args:
            metrics: Lista de métricas a exportar
        """
        for name in self.enabled_exporters:
            if name in self.exporters:
                try:
                    exporter = self.exporters[name]
                    if asyncio.iscoroutinefunction(exporter):
                        await exporter(metrics)
                    else:
                        exporter(metrics)
                except Exception as e:
                    logger.error(f"Error exporting metrics to {name}: {e}")


def format_prometheus(
    metrics: List[MetricExport]
) -> str:
    """
    Formatear métricas en formato Prometheus.
    
    Args:
        metrics: Lista de métricas
        
    Returns:
        String en formato Prometheus
    """
    lines = []
    
    for metric in metrics:
        # Construir labels
        labels_str = ""
        if metric.labels:
            label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
            labels_str = "{" + ",".join(label_pairs) + "}"
        
        # Formatear línea
        line = f"{metric.name}{labels_str} {metric.value}"
        if metric.timestamp:
            line += f" {int(metric.timestamp.timestamp() * 1000)}"
        
        lines.append(line)
    
    return "\n".join(lines)


def format_statsd(
    metrics: List[MetricExport]
) -> str:
    """
    Formatear métricas en formato StatsD.
    
    Args:
        metrics: Lista de métricas
        
    Returns:
        String en formato StatsD
    """
    lines = []
    
    for metric in metrics:
        # Construir tags
        tags = []
        if metric.labels:
            tags = [f"{k}={v}" for k, v in metric.labels.items()]
        
        # Tipo de métrica StatsD
        type_map = {
            "counter": "c",
            "gauge": "g",
            "histogram": "h",
            "summary": "ms"
        }
        statsd_type = type_map.get(metric.metric_type, "g")
        
        # Formatear línea
        tag_str = f"|#{','.join(tags)}" if tags else ""
        line = f"{metric.name}:{metric.value}|{statsd_type}{tag_str}"
        lines.append(line)
    
    return "\n".join(lines)


async def export_to_prometheus(
    metrics: List[MetricExport],
    endpoint: Optional[str] = None
) -> None:
    """
    Exportar métricas a Prometheus.
    
    Args:
        metrics: Lista de métricas
        endpoint: Endpoint opcional para enviar (si None, solo formatea)
    """
    formatted = format_prometheus(metrics)
    
    if endpoint:
        # Enviar a endpoint (requiere requests o httpx)
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(endpoint, content=formatted)
        except ImportError:
            logger.warning("httpx not available, cannot send to endpoint")
        except Exception as e:
            logger.error(f"Error sending to Prometheus endpoint: {e}")
    else:
        logger.debug(f"Prometheus metrics:\n{formatted}")


async def export_to_statsd(
    metrics: List[MetricExport],
    host: str = "localhost",
    port: int = 8125
) -> None:
    """
    Exportar métricas a StatsD.
    
    Args:
        metrics: Lista de métricas
        host: Host de StatsD
        port: Puerto de StatsD
    """
    formatted = format_statsd(metrics)
    
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(formatted.encode(), (host, port))
        sock.close()
    except Exception as e:
        logger.error(f"Error sending to StatsD: {e}")

