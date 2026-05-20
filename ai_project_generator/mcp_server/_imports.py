"""
MCP Server Imports - Gestión modular de imports
===============================================

Módulo para gestionar los imports del servidor MCP de forma
organizada y modular, separando TYPE_CHECKING de imports reales.
"""

import logging
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


# Definición de grupos de imports para mejor organización
IMPORT_GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "core": [
        ("server", "MCPServer, MCPRequest, MCPResponse"),
        ("exceptions", "MCPError, MCPAuthenticationError, MCPAuthorizationError, "
         "MCPResourceNotFoundError, MCPConnectorError, MCPOperationError, "
         "MCPValidationError, MCPRateLimitError, MCPContextLimitError"),
    ],
    "connectors": [
        ("connectors", "FileSystemConnector, DatabaseConnector, APIConnector, "
         "ConnectorRegistry, BaseConnector"),
    ],
    "manifests": [
        ("manifests", "ResourceManifest, ManifestRegistry, ManifestLoader"),
    ],
    "security": [
        ("security", "MCPSecurityManager, Scope, AccessPolicy"),
    ],
    "contracts": [
        ("contracts", "ContextFrame, PromptFrame, FrameSerializer"),
    ],
    "utilities": [
        ("rate_limiter", "RateLimiter"),
        ("cache", "MCPCache"),
        ("middleware", "MCPLoggingMiddleware, MCPCORSMiddleware"),
        ("helpers", "create_mcp_server, setup_rate_limits, setup_cache"),
        ("retry", "RetryConfig, retry_with_backoff, retryable"),
        ("circuit_breaker", "CircuitBreaker, CircuitState, circuit_breaker"),
    ],
    "features": [
        ("batch", "BatchProcessor, BatchRequest, BatchResponse, batch_query"),
        ("webhooks", "WebhookManager, Webhook, WebhookEvent, WebhookPayload"),
        ("transformers", "RequestTransformer, ResponseTransformer, "
         "add_timestamp_transformer, mask_sensitive_data_transformer, "
         "compress_context_transformer"),
        ("admin", "MCPAdmin"),
        ("streaming", "StreamResponse, StreamChunk, stream_query"),
        ("config", "MCPConfig"),
        ("profiling", "PerformanceProfiler, profile_function"),
        ("queue", "AsyncTaskQueue, Task, TaskStatus"),
        ("graphql", "MCPGraphQL"),
        ("plugins", "Plugin, PluginManager, ConnectorPlugin, MiddlewarePlugin"),
        ("compression", "ResponseCompressor, CompressionType, get_compression_type"),
        ("health", "HealthChecker, HealthCheck, HealthReport, HealthStatus"),
    ],
    "advanced": [
        ("versioning", "VersionedRouter, APIVersion, versioned_route"),
        ("discovery", "ServiceRegistry, ServiceInfo, MCPServiceStatus"),
        ("pooling", "ConnectionPool, DatabaseConnectionPool, HTTPConnectionPool"),
        ("metrics_dashboard", "MetricsDashboard"),
        ("request_queue", "RequestQueue, QueuedRequest, RequestPriority"),
        ("multitenancy", "TenantManager, Tenant, TenantContext, tenant_middleware"),
        ("events", "EventStore, EventPublisher, Event, EventType"),
        ("locking", "LockManager, DistributedLock"),
        ("docs", "APIDocumentation"),
        ("interceptors", "RequestInterceptor, ResponseInterceptor"),
        ("cqrs", "CQRSBus, Command, Query, CommandHandler, QueryHandler"),
        ("saga", "SagaOrchestrator, Saga, SagaStep, SagaStepStatus"),
        ("message_queue", "MessageQueue, Message, MessagePriority"),
        ("advanced_cache", "AdvancedCache, CacheStrategy, CacheEntry"),
        ("validation", "AdvancedValidator, ValidationRule, validate_request"),
        ("load_balancer", "LoadBalancer, BackendServer, LoadBalanceStrategy"),
        ("gateway", "APIGateway, GatewayRoute"),
        ("websocket", "WebSocketManager, WebSocketMessage"),
        ("analytics", "AnalyticsCollector, AnalyticsEvent"),
        ("testing", "MCPTestClient, create_mock_connector, create_mock_manifest, assert_mcp_response"),
        ("backup", "BackupManager, BackupMetadata"),
        ("migration", "MigrationManager, Migration"),
        ("user_rate_limit", "UserRateLimiter"),
        ("throttling", "Throttler, AdaptiveThrottler, ThrottleConfig"),
        ("monitoring", "MonitoringSystem, Alert, AlertRule, AlertLevel"),
        ("feature_flags", "FeatureFlagManager, FeatureFlag, FeatureFlagStatus, feature_flag"),
        ("quotas", "QuotaManager, ResourceQuota, QuotaLimit"),
        ("ip_rate_limit", "IPRateLimiter"),
        ("advanced_logging", "StructuredLogger, RequestLogger, LogLevel"),
        ("optimization", "PerformanceOptimizer"),
    ],
}


def get_type_checking_imports() -> str:
    """
    Genera imports para TYPE_CHECKING.
    
    Returns:
        String con todos los imports para TYPE_CHECKING.
    """
    imports = []
    for group_name, modules in IMPORT_GROUPS.items():
        imports.append(f"    # {group_name.upper()}")
        for module_path, symbols in modules:
            imports.append(f"    from .{module_path} import {symbols}")
    return "\n".join(imports)


def get_runtime_imports() -> List[Tuple[str, str, str]]:
    """
    Genera lista de imports para runtime.
    
    Returns:
        Lista de tuplas (module_path, symbols, group_name) para imports reales.
    """
    imports = []
    for group_name, modules in IMPORT_GROUPS.items():
        for module_path, symbols in modules:
            imports.append((module_path, symbols, group_name))
    return imports


def get_all_symbols() -> List[str]:
    """
    Obtener lista de todos los símbolos que se importan.
    
    Returns:
        Lista de nombres de símbolos.
    """
    symbols = []
    for group_name, modules in IMPORT_GROUPS.items():
        for module_path, symbols_str in modules:
            symbol_list = [s.strip() for s in symbols_str.split(",")]
            symbols.extend(symbol_list)
    return symbols


def import_module_safely(
    module_path: str,
    symbols: str,
    group_name: str,
    namespace: Dict[str, Any]
) -> None:
    """
    Importar un módulo de forma segura con manejo de errores.
    
    Args:
        module_path: Ruta del módulo (ej: "server").
        symbols: Símbolos a importar (ej: "MCPServer, MCPRequest").
        group_name: Nombre del grupo para logging.
        namespace: Namespace donde asignar los imports (None si falla).
    """
    try:
        module = __import__(f".{module_path}", fromlist=symbols.split(","), level=1)
        symbol_list = [s.strip() for s in symbols.split(",")]
        
        for symbol_name in symbol_list:
            if hasattr(module, symbol_name):
                namespace[symbol_name] = getattr(module, symbol_name)
            else:
                logger.warning(
                    f"Symbol '{symbol_name}' not found in module '{module_path}'"
                )
    except ImportError as e:
        logger.debug(f"{group_name} not available ({module_path}): {e}")
    except Exception as e:
        logger.warning(f"Error importing {group_name} ({module_path}): {e}")

