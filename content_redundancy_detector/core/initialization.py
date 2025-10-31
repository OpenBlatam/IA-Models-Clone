"""
Application Initialization
Centralized initialization logic for all application components
"""

import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.config import settings
from core.logging_config import setup_logging, get_logger

logger = None


def initialize_logging():
    """Initialize logging system"""
    global logger
    logger = setup_logging()
    return logger


def initialize_advanced_features() -> dict:
    """Initialize advanced microservices features"""
    features_status = {
        "observability": False,
        "circuit_breaker": False,
        "async_workers": False,
        "serverless": False,
        "api_gateway": False
    }
    
    try:
        from observability import setup_observability
        setup_observability(settings.app_name, settings.app_version)
        features_status["observability"] = True
        logger.info("Observability initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize observability: {e}")
    
    try:
        from circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
        get_circuit_breaker("ai_ml_service", CircuitBreakerConfig(
            name="ai_ml_service",
            failure_threshold=5,
            timeout=60
        ))
        get_circuit_breaker("database", CircuitBreakerConfig(
            name="database",
            failure_threshold=3,
            timeout=30
        ))
        features_status["circuit_breaker"] = True
        logger.info("Circuit breakers initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize circuit breakers: {e}")
    
    try:
        from async_workers import get_celery_app
        celery_app = get_celery_app()
        if celery_app:
            features_status["async_workers"] = True
            logger.info("Celery workers configured")
    except Exception as e:
        logger.warning(f"Celery not available: {e}")
    
    try:
        from serverless_optimizer import optimize_for_serverless, is_serverless, warm_up
        if is_serverless():
            optimize_for_serverless()
            warm_up()
            features_status["serverless"] = True
            logger.info("Serverless optimizations applied")
    except Exception as e:
        logger.warning(f"Serverless optimizations not available: {e}")
    
    return features_status


def initialize_business_services():
    """Initialize business logic services"""
    services_status = {}
    
    try:
        from webhooks import webhook_manager
        services_status["webhook_manager"] = webhook_manager
        logger.info("Webhook manager initialized")
    except ImportError:
        class FallbackWebhookManager:
            async def send_webhook(self, *args, **kwargs):
                return {"status": "disabled"}
            async def start(self): pass
            async def stop(self): pass
        services_status["webhook_manager"] = FallbackWebhookManager()
        logger.warning("Webhook manager fallback created")
    
    # Initialize other services
    service_modules = [
        ("batch_processor", "batch_processor"),
        ("analytics", "analytics_engine"),
        ("export", "export_manager"),
        ("ai_ml_enhanced", "ai_ml_engine"),
        ("real_time_engine", "real_time_engine"),
        ("cloud_integration", "cloud_manager"),
        ("security_advanced", "security_manager"),
        ("monitoring_advanced", "monitoring_system"),
        ("automation_engine", "automation_engine"),
        ("realtime_analysis", "initialize_realtime_engine"),
        ("multimodal_analysis", "initialize_multimodal_engine"),
        ("custom_model_training", "initialize_custom_training_engine"),
        ("advanced_analytics_dashboard", "initialize_analytics_dashboard")
    ]
    
    for module_name, service_name in service_modules:
        try:
            module = __import__(module_name, fromlist=[service_name])
            service = getattr(module, service_name)
            services_status[module_name] = service
            logger.info(f"{module_name} initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize {module_name}: {e}")
    
    return services_status


async def startup_services(services: dict):
    """Start up all services"""
    webhook_manager = services.get("webhook_manager")
    if webhook_manager and hasattr(webhook_manager, "start"):
        await webhook_manager.start()
    
    # Start other async services
    async_services = [
        ("ai_ml_enhanced", "initialize"),
        ("real_time_engine", "start"),
        ("monitoring_advanced", "start_monitoring"),
        ("automation_engine", "start")
    ]
    
    for service_key, method_name in async_services:
        service = services.get(service_key)
        if service and hasattr(service, method_name):
            try:
                method = getattr(service, method_name)
                if callable(method):
                    result = method()
                    if hasattr(result, "__await__"):
                        await result
                logger.info(f"{service_key}.{method_name} started")
            except Exception as e:
                logger.error(f"Failed to start {service_key}.{method_name}: {e}")
    
    # Initialize async functions
    init_functions = [
        ("realtime_analysis", "initialize_realtime_engine"),
        ("multimodal_analysis", "initialize_multimodal_engine"),
        ("custom_model_training", "initialize_custom_training_engine"),
        ("advanced_analytics_dashboard", "initialize_analytics_dashboard")
    ]
    
    for module_key, func_name in init_functions:
        func = services.get(module_key)
        if func and callable(func):
            try:
                result = func()
                if hasattr(result, "__await__"):
                    await result
                logger.info(f"{func_name} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize {func_name}: {e}")


async def shutdown_services(services: dict):
    """Shutdown all services"""
    webhook_manager = services.get("webhook_manager")
    if webhook_manager and hasattr(webhook_manager, "stop"):
        await webhook_manager.stop()
    
    # Shutdown other async services
    async_services = [
        ("real_time_engine", "stop"),
        ("monitoring_advanced", "stop_monitoring"),
        ("automation_engine", "stop")
    ]
    
    for service_key, method_name in async_services:
        service = services.get(service_key)
        if service and hasattr(service, method_name):
            try:
                method = getattr(service, method_name)
                if callable(method):
                    result = method()
                    if hasattr(result, "__await__"):
                        await result
                logger.info(f"{service_key}.{method_name} stopped")
            except Exception as e:
                logger.error(f"Failed to stop {service_key}.{method_name}: {e}")
    
    # Cleanup services
    cleanup_services = [
        ("batch_processor", "cleanup_old_batches"),
        ("analytics", "cleanup_old_reports"),
        ("export", "cleanup_old_exports")
    ]
    
    for service_key, method_name in cleanup_services:
        service = services.get(service_key)
        if service and hasattr(service, method_name):
            try:
                method = getattr(service, method_name)
                if callable(method):
                    method()
                logger.info(f"{service_key}.{method_name} completed")
            except Exception as e:
                logger.error(f"Failed to cleanup {service_key}.{method_name}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    global logger
    
    # Startup
    logger = initialize_logging()
    logger.info("=" * 50)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Server: {settings.host}:{settings.port}")
    logger.info("=" * 50)
    
    # Initialize advanced features
    features_status = initialize_advanced_features()
    logger.info(f"Advanced features status: {features_status}")
    
    # Initialize business services
    services = initialize_business_services()
    
    # Start up services
    await startup_services(services)
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    
    # Shutdown services
    await shutdown_services(services)
    
    logger.info("Application shutdown complete")





