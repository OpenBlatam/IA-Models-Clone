"""Connector Registry - Registro de conectores"""
from typing import Dict, Optional
from .base import BaseConnector

class ConnectorRegistry:
    def __init__(self):
        self._connectors: Dict[str, BaseConnector] = {}
    def register(self, name: str, connector: BaseConnector) -> None:
        self._connectors[name] = connector
    def get(self, name: str) -> Optional[BaseConnector]:
        return self._connectors.get(name)

