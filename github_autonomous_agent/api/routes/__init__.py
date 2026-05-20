"""API routes."""

from . import (
    agent_routes, github_routes, task_routes, llm_routes,
    websocket_routes, stats_routes, llm_health, llm_models, llm_analytics, llm_optimization,
    batch_routes, audit_routes, notification_routes, monitoring_routes,
    auth_routes, plugin_routes, config_routes, feature_flags_routes, webhook_routes, scheduler_routes,
    queue_routes, batch_processor_routes, analytics_routes, search_routes,
    validation_routes
)

__all__ = [
    "agent_routes",
    "github_routes",
    "task_routes",
    "llm_routes",
    "websocket_routes",
    "stats_routes",
    "llm_health",
    "llm_models",
    "llm_analytics",
    "llm_optimization",
    "batch_routes",
    "audit_routes",
    "notification_routes",
    "monitoring_routes",
    "auth_routes",
    "plugin_routes",
    "config_routes",
    "feature_flags_routes",
    "webhook_routes",
    "scheduler_routes",
    "queue_routes",
    "batch_processor_routes",
    "analytics_routes",
    "search_routes",
    "validation_routes",
]
