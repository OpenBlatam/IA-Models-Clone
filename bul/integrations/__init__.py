"""
BUL Integrations Module
======================

Integración con APIs externas para el sistema BUL.
"""

from .external_apis import (
    ExternalAPIManager,
    APICredentials,
    APIRequest,
    APIResponse,
    IntegrationType,
    APIMethod,
    CRMIntegration,
    EmailIntegration,
    StorageIntegration,
    AnalyticsIntegration,
    get_global_external_api_manager
)

__all__ = [
    "ExternalAPIManager",
    "APICredentials",
    "APIRequest",
    "APIResponse",
    "IntegrationType",
    "APIMethod",
    "CRMIntegration",
    "EmailIntegration",
    "StorageIntegration",
    "AnalyticsIntegration",
    "get_global_external_api_manager"
]
























