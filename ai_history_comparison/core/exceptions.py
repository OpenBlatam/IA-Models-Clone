"""
Custom exceptions for the AI History Comparison System

This module defines all custom exceptions used throughout the system
to provide better error handling and debugging capabilities.
"""

from typing import Any, Dict, Optional


class AIHistoryException(Exception):
    """Base exception for all AI History Comparison System errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(AIHistoryException):
    """Raised when there's a configuration error"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key


class ValidationError(AIHistoryException):
    """Raised when data validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value


class AnalysisError(AIHistoryException):
    """Raised when analysis fails"""
    
    def __init__(self, message: str, analyzer_name: Optional[str] = None, input_data: Optional[Any] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "ANALYSIS_ERROR", details)
        self.analyzer_name = analyzer_name
        self.input_data = input_data


class ComparisonError(AIHistoryException):
    """Raised when comparison operations fail"""
    
    def __init__(self, message: str, comparison_type: Optional[str] = None, items: Optional[list] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "COMPARISON_ERROR", details)
        self.comparison_type = comparison_type
        self.items = items


class ProcessingError(AIHistoryException):
    """Raised when data processing fails"""
    
    def __init__(self, message: str, processor_name: Optional[str] = None, stage: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PROCESSING_ERROR", details)
        self.processor_name = processor_name
        self.stage = stage


class StorageError(AIHistoryException):
    """Raised when storage operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, resource: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "STORAGE_ERROR", details)
        self.operation = operation
        self.resource = resource


class IntegrationError(AIHistoryException):
    """Raised when external integrations fail"""
    
    def __init__(self, message: str, service_name: Optional[str] = None, endpoint: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "INTEGRATION_ERROR", details)
        self.service_name = service_name
        self.endpoint = endpoint


class AuthenticationError(AIHistoryException):
    """Raised when authentication fails"""
    
    def __init__(self, message: str, user_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTHENTICATION_ERROR", details)
        self.user_id = user_id


class AuthorizationError(AIHistoryException):
    """Raised when authorization fails"""
    
    def __init__(self, message: str, user_id: Optional[str] = None, resource: Optional[str] = None, action: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)
        self.user_id = user_id
        self.resource = resource
        self.action = action


class RateLimitError(AIHistoryException):
    """Raised when rate limits are exceeded"""
    
    def __init__(self, message: str, limit: Optional[int] = None, window: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.limit = limit
        self.window = window


class TimeoutError(AIHistoryException):
    """Raised when operations timeout"""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "TIMEOUT_ERROR", details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class ResourceNotFoundError(AIHistoryException):
    """Raised when a requested resource is not found"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RESOURCE_NOT_FOUND", details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class DuplicateResourceError(AIHistoryException):
    """Raised when trying to create a duplicate resource"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DUPLICATE_RESOURCE", details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class WorkflowError(AIHistoryException):
    """Raised when workflow operations fail"""
    
    def __init__(self, message: str, workflow_id: Optional[str] = None, step: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "WORKFLOW_ERROR", details)
        self.workflow_id = workflow_id
        self.step = step


class NotificationError(AIHistoryException):
    """Raised when notification operations fail"""
    
    def __init__(self, message: str, channel: Optional[str] = None, recipient: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "NOTIFICATION_ERROR", details)
        self.channel = channel
        self.recipient = recipient


class MonitoringError(AIHistoryException):
    """Raised when monitoring operations fail"""
    
    def __init__(self, message: str, metric_name: Optional[str] = None, component: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MONITORING_ERROR", details)
        self.metric_name = metric_name
        self.component = component


class CacheError(AIHistoryException):
    """Raised when cache operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CACHE_ERROR", details)
        self.operation = operation
        self.key = key


class SerializationError(AIHistoryException):
    """Raised when serialization/deserialization fails"""
    
    def __init__(self, message: str, data_type: Optional[str] = None, format: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SERIALIZATION_ERROR", details)
        self.data_type = data_type
        self.format = format


class EncryptionError(AIHistoryException):
    """Raised when encryption/decryption operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, algorithm: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "ENCRYPTION_ERROR", details)
        self.operation = operation
        self.algorithm = algorithm


class NetworkError(AIHistoryException):
    """Raised when network operations fail"""
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "NETWORK_ERROR", details)
        self.url = url
        self.status_code = status_code


class DatabaseError(AIHistoryException):
    """Raised when database operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, table: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", details)
        self.operation = operation
        self.table = table


class FileSystemError(AIHistoryException):
    """Raised when file system operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, path: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "FILESYSTEM_ERROR", details)
        self.operation = operation
        self.path = path


class AIProviderError(AIHistoryException):
    """Raised when AI provider operations fail"""
    
    def __init__(self, message: str, provider: Optional[str] = None, model: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AI_PROVIDER_ERROR", details)
        self.provider = provider
        self.model = model


class ModelError(AIHistoryException):
    """Raised when model operations fail"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, version: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_ERROR", details)
        self.model_name = model_name
        self.version = version


class DataQualityError(AIHistoryException):
    """Raised when data quality issues are detected"""
    
    def __init__(self, message: str, quality_metric: Optional[str] = None, threshold: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATA_QUALITY_ERROR", details)
        self.quality_metric = quality_metric
        self.threshold = threshold


class ComplianceError(AIHistoryException):
    """Raised when compliance violations are detected"""
    
    def __init__(self, message: str, standard: Optional[str] = None, violation_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "COMPLIANCE_ERROR", details)
        self.standard = standard
        self.violation_type = violation_type


class SecurityError(AIHistoryException):
    """Raised when security violations are detected"""
    
    def __init__(self, message: str, threat_type: Optional[str] = None, severity: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SECURITY_ERROR", details)
        self.threat_type = threat_type
        self.severity = severity


class PerformanceError(AIHistoryException):
    """Raised when performance thresholds are exceeded"""
    
    def __init__(self, message: str, metric: Optional[str] = None, threshold: Optional[float] = None, actual_value: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PERFORMANCE_ERROR", details)
        self.metric = metric
        self.threshold = threshold
        self.actual_value = actual_value


class DependencyError(AIHistoryException):
    """Raised when dependency operations fail"""
    
    def __init__(self, message: str, dependency_name: Optional[str] = None, version: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DEPENDENCY_ERROR", details)
        self.dependency_name = dependency_name
        self.version = version


class VersionError(AIHistoryException):
    """Raised when version compatibility issues occur"""
    
    def __init__(self, message: str, current_version: Optional[str] = None, required_version: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VERSION_ERROR", details)
        self.current_version = current_version
        self.required_version = required_version


class FeatureNotAvailableError(AIHistoryException):
    """Raised when a requested feature is not available"""
    
    def __init__(self, message: str, feature_name: Optional[str] = None, reason: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "FEATURE_NOT_AVAILABLE", details)
        self.feature_name = feature_name
        self.reason = reason


class MaintenanceError(AIHistoryException):
    """Raised when system is in maintenance mode"""
    
    def __init__(self, message: str, maintenance_window: Optional[str] = None, estimated_duration: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MAINTENANCE_ERROR", details)
        self.maintenance_window = maintenance_window
        self.estimated_duration = estimated_duration





















