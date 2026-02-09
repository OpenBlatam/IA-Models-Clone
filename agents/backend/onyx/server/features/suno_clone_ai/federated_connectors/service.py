"""Federated Connector Service - Servicio de conectores federados"""
from typing import Optional
from .base import BaseFederatedConnector
from connectors.service import ConnectorService
from httpx.service import HTTPService
from tracing.service import TracingService

class FederatedConnectorService:
    def __init__(self, connector_service: Optional[ConnectorService] = None, http_service: Optional[HTTPService] = None, tracing_service: Optional[TracingService] = None):
        self.connector_service = connector_service
        self.http_service = http_service
        self.tracing_service = tracing_service

