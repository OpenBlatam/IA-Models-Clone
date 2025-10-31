"""
Error Handler - Manejo avanzado de errores y logging
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Niveles de severidad de errores."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categorías de errores."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    EXPORT = "export"
    QUALITY = "quality"
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"


@dataclass
class ErrorInfo:
    """Información detallada de error."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class ErrorHandler:
    """Manejador avanzado de errores."""
    
    def __init__(self):
        self.error_log = []
        self.error_counts = {}
        self.alert_thresholds = {
            ErrorSeverity.HIGH: 10,
            ErrorSeverity.CRITICAL: 1
        }
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configurar logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configurar sistema de logging."""
        # Crear handler para archivo
        file_handler = logging.FileHandler('logs/errors.log')
        file_handler.setLevel(logging.ERROR)
        
        # Crear formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Agregar handler al logger
        self.logger.addHandler(file_handler)
    
    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> ErrorInfo:
        """
        Manejar error con información detallada.
        
        Args:
            error: Excepción capturada
            category: Categoría del error
            severity: Severidad del error
            context: Contexto adicional
            user_id: ID del usuario
            request_id: ID de la petición
            
        Returns:
            Información del error
        """
        # Generar ID único para el error
        error_id = self._generate_error_id()
        
        # Crear información del error
        error_info = ErrorInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            details=self._extract_error_details(error),
            stack_trace=traceback.format_exc(),
            user_id=user_id,
            request_id=request_id,
            context=context or {}
        )
        
        # Registrar error
        self._log_error(error_info)
        
        # Actualizar contadores
        self._update_error_counts(error_info)
        
        # Verificar alertas
        self._check_alerts(error_info)
        
        return error_info
    
    def _generate_error_id(self) -> str:
        """Generar ID único para el error."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _extract_error_details(self, error: Exception) -> Dict[str, Any]:
        """Extraer detalles del error."""
        details = {
            "type": type(error).__name__,
            "module": getattr(error, '__module__', 'unknown'),
            "args": getattr(error, 'args', [])
        }
        
        # Agregar detalles específicos según el tipo de error
        if hasattr(error, 'code'):
            details['code'] = error.code
        
        if hasattr(error, 'status_code'):
            details['status_code'] = error.status_code
        
        return details
    
    def _log_error(self, error_info: ErrorInfo):
        """Registrar error en logs."""
        # Agregar a log interno
        self.error_log.append(error_info)
        
        # Mantener solo los últimos 1000 errores
        if len(self.error_log) > 1000:
            self.error_log = self.error_log[-1000:]
        
        # Log según severidad
        log_message = self._format_error_message(error_info)
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _format_error_message(self, error_info: ErrorInfo) -> str:
        """Formatear mensaje de error para logging."""
        return (
            f"Error {error_info.error_id} - "
            f"Category: {error_info.category.value}, "
            f"Severity: {error_info.severity.value}, "
            f"Message: {error_info.message}, "
            f"User: {error_info.user_id or 'N/A'}, "
            f"Request: {error_info.request_id or 'N/A'}"
        )
    
    def _update_error_counts(self, error_info: ErrorInfo):
        """Actualizar contadores de errores."""
        key = f"{error_info.category.value}_{error_info.severity.value}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def _check_alerts(self, error_info: ErrorInfo):
        """Verificar si se deben enviar alertas."""
        threshold = self.alert_thresholds.get(error_info.severity)
        if threshold:
            count = self.error_counts.get(
                f"{error_info.category.value}_{error_info.severity.value}", 0
            )
            if count >= threshold:
                self._send_alert(error_info, count)
    
    def _send_alert(self, error_info: ErrorInfo, count: int):
        """Enviar alerta por error crítico."""
        alert_message = (
            f"ALERTA: {count} errores de severidad {error_info.severity.value} "
            f"en categoría {error_info.category.value}"
        )
        
        self.logger.critical(alert_message)
        
        # Aquí se podría integrar con sistemas de alertas externos
        # como Slack, email, etc.
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de errores."""
        total_errors = len(self.error_log)
        
        # Contar por categoría
        category_counts = {}
        for error in self.error_log:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Contar por severidad
        severity_counts = {}
        for error in self.error_log:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Errores recientes (últimas 24 horas)
        recent_errors = [
            error for error in self.error_log
            if (datetime.now() - error.timestamp).total_seconds() < 86400
        ]
        
        return {
            "total_errors": total_errors,
            "recent_errors": len(recent_errors),
            "category_counts": category_counts,
            "severity_counts": severity_counts,
            "error_counts": self.error_counts,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorInfo]:
        """Obtener errores por categoría."""
        return [error for error in self.error_log if error.category == category]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorInfo]:
        """Obtener errores por severidad."""
        return [error for error in self.error_log if error.severity == severity]
    
    def get_recent_errors(self, hours: int = 24) -> List[ErrorInfo]:
        """Obtener errores recientes."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [error for error in self.error_log if error.timestamp > cutoff_time]
    
    def clear_old_errors(self, days: int = 7):
        """Limpiar errores antiguos."""
        cutoff_time = datetime.now() - timedelta(days=days)
        self.error_log = [error for error in self.error_log if error.timestamp > cutoff_time]
        
        self.logger.info(f"Errores antiguos limpiados (más de {days} días)")


class ExportError(Exception):
    """Excepción base para errores de exportación."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ValidationError(ExportError):
    """Error de validación."""
    pass


class ProcessingError(ExportError):
    """Error de procesamiento."""
    pass


class QualityError(ExportError):
    """Error de calidad."""
    pass


class SystemError(ExportError):
    """Error del sistema."""
    pass


# Instancia global del manejador de errores
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Obtener la instancia global del manejador de errores."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_export_error(
    error: Exception,
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> ErrorInfo:
    """Función de conveniencia para manejar errores."""
    handler = get_error_handler()
    return handler.handle_error(error, category, severity, context, user_id, request_id)




