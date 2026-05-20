"""
Advanced Utilities for Music Analyzer AI
=========================================

Advanced microservices, serverless, and enterprise features
"""

from .advanced_middleware import setup_advanced_middleware
from .oauth2_security import oauth2_security, get_current_active_user, require_scope, require_role, User
from .async_workers import WorkerManager, WorkerType
from .message_broker import MessageBrokerManager, BrokerType
from .serverless_optimizer import get_serverless_config, serverless_handler
from .structured_logging import CentralizedLogging
from .owasp_security import owasp_validator, ddos_protection
from .database_adapters import DatabaseManager
from .elasticsearch_client import elasticsearch_client
from .memcached_client import memcached_client
from .kong_gateway import kong_gateway_manager
from .aws_api_gateway import aws_api_gateway_manager
from .service_mesh import service_mesh_manager
from .service_discovery import service_discovery
from .inter_service_comm import service_registry

__all__ = [
    "setup_advanced_middleware",
    "oauth2_security",
    "get_current_active_user",
    "require_scope",
    "require_role",
    "User",
    "WorkerManager",
    "WorkerType",
    "MessageBrokerManager",
    "BrokerType",
    "get_serverless_config",
    "serverless_handler",
    "CentralizedLogging",
    "owasp_validator",
    "ddos_protection",
    "DatabaseManager",
    "elasticsearch_client",
    "memcached_client",
    "kong_gateway_manager",
    "aws_api_gateway_manager",
    "service_mesh_manager",
    "service_discovery",
    "service_registry",
]




