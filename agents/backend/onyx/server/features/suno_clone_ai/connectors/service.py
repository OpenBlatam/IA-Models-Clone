"""Connector Service - Servicio de conectores"""
from typing import Optional
from .base import BaseConnector
from .registry import ConnectorRegistry
from httpx.service import HTTPService
from configs.settings import Settings
from tracing.service import TracingService

class ConnectorService:
    def __init__(self, http_service: Optional[HTTPService] = None, tracing_service: Optional[TracingService] = None):
        self.registry = ConnectorRegistry()
        self.http_service = http_service
        self.tracing_service = tracing_service

