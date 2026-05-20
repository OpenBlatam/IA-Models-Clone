"""
MCP Server Handlers - Endpoint handlers
"""

from .resources import list_resources, get_resource
from .operations import query_resource
from .health import health_check
from .metrics import get_metrics, get_stats
from .info import get_server_info
from .version import get_version
from .prometheus import get_prometheus_metrics

__all__ = [
    "list_resources",
    "get_resource",
    "query_resource",
    "health_check",
    "get_metrics",
    "get_stats",
    "get_server_info",
    "get_version",
    "get_prometheus_metrics"
]

