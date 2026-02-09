"""
Connectors Module - Conectores Externos
Integraciones con servicios externos, APIs de terceros, y adaptadores.
"""

from .base import BaseConnector
from .service import ConnectorService
from .registry import ConnectorRegistry

__all__ = ["BaseConnector", "ConnectorService", "ConnectorRegistry"]

