"""
Domain Exceptions - Excepciones de Dominio
========================================

Excepciones específicas del dominio para el sistema de comparación
de historial de IA.
"""

from typing import Optional, Dict, Any


class DomainException(Exception):
    """
    Excepción base del dominio.
    
    Todas las excepciones específicas del dominio deben heredar de esta clase.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ValidationException(DomainException):
    """
    Excepción de validación.
    
    Se lanza cuando los datos no cumplen con las reglas de validación del dominio.
    """
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field
        self.value = value


class NotFoundException(DomainException):
    """
    Excepción de recurso no encontrado.
    
    Se lanza cuando se intenta acceder a un recurso que no existe.
    """
    
    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(message, error_code="NOT_FOUND")
        self.resource_type = resource_type
        self.resource_id = resource_id


class DuplicateException(DomainException):
    """
    Excepción de duplicado.
    
    Se lanza cuando se intenta crear un recurso que ya existe.
    """
    
    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} with id '{resource_id}' already exists"
        super().__init__(message, error_code="DUPLICATE")
        self.resource_type = resource_type
        self.resource_id = resource_id


class BusinessRuleException(DomainException):
    """
    Excepción de regla de negocio.
    
    Se lanza cuando se viola una regla de negocio del dominio.
    """
    
    def __init__(self, message: str, rule_name: Optional[str] = None):
        super().__init__(message, error_code="BUSINESS_RULE_VIOLATION")
        self.rule_name = rule_name


class AnalysisException(DomainException):
    """
    Excepción de análisis.
    
    Se lanza cuando ocurre un error durante el análisis de contenido.
    """
    
    def __init__(self, message: str, analysis_type: Optional[str] = None):
        super().__init__(message, error_code="ANALYSIS_ERROR")
        self.analysis_type = analysis_type


class ComparisonException(DomainException):
    """
    Excepción de comparación.
    
    Se lanza cuando ocurre un error durante la comparación de entradas.
    """
    
    def __init__(self, message: str, entry_1_id: Optional[str] = None, entry_2_id: Optional[str] = None):
        super().__init__(message, error_code="COMPARISON_ERROR")
        self.entry_1_id = entry_1_id
        self.entry_2_id = entry_2_id


class QualityAssessmentException(DomainException):
    """
    Excepción de evaluación de calidad.
    
    Se lanza cuando ocurre un error durante la evaluación de calidad.
    """
    
    def __init__(self, message: str, entry_id: Optional[str] = None):
        super().__init__(message, error_code="QUALITY_ASSESSMENT_ERROR")
        self.entry_id = entry_id


class RepositoryException(DomainException):
    """
    Excepción de repositorio.
    
    Se lanza cuando ocurre un error en el acceso a datos.
    """
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(message, error_code="REPOSITORY_ERROR")
        self.operation = operation


class ServiceException(DomainException):
    """
    Excepción de servicio.
    
    Se lanza cuando ocurre un error en un servicio de aplicación.
    """
    
    def __init__(self, message: str, service_name: Optional[str] = None):
        super().__init__(message, error_code="SERVICE_ERROR")
        self.service_name = service_name


class ConfigurationException(DomainException):
    """
    Excepción de configuración.
    
    Se lanza cuando hay un error en la configuración del sistema.
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, error_code="CONFIGURATION_ERROR")
        self.config_key = config_key


class RateLimitException(DomainException):
    """
    Excepción de límite de tasa.
    
    Se lanza cuando se excede el límite de tasa de operaciones.
    """
    
    def __init__(self, message: str, limit: Optional[int] = None, window: Optional[int] = None):
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED")
        self.limit = limit
        self.window = window


class AuthenticationException(DomainException):
    """
    Excepción de autenticación.
    
    Se lanza cuando falla la autenticación.
    """
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, error_code="AUTHENTICATION_ERROR")


class AuthorizationException(DomainException):
    """
    Excepción de autorización.
    
    Se lanza cuando falla la autorización.
    """
    
    def __init__(self, message: str = "Authorization failed", resource: Optional[str] = None):
        super().__init__(message, error_code="AUTHORIZATION_ERROR")
        self.resource = resource




