"""
Ultra-Modular Microservices System
=================================

Complete microservices architecture with independent service containers.
"""

from .document_processor_service import DocumentProcessorService
from .ai_service import AIService
from .transform_service import TransformService
from .validation_service import ValidationService
from .cache_service import CacheService
from .file_service import FileService
from .notification_service import NotificationService
from .metrics_service import MetricsService
from .api_gateway_service import APIGatewayService
from .message_bus_service import MessageBusService

__version__ = "4.0.0"
__author__ = "AI Document Processor Team"
__description__ = "Ultra-modular microservices for AI document processing"

__all__ = [
    "DocumentProcessorService",
    "AIService",
    "TransformService",
    "ValidationService",
    "CacheService",
    "FileService",
    "NotificationService",
    "MetricsService",
    "APIGatewayService",
    "MessageBusService",
]

















