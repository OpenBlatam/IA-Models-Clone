"""
Validación de configuración al inicio de la aplicación con mejoras.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ConfigValidationError(Exception):
    """
    Error de validación de configuración con detalles mejorados.
    
    Attributes:
        errors: Lista de errores encontrados
        warnings: Lista de advertencias encontradas
    """
    
    def __init__(
        self,
        message: str,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ):
        """
        Inicializar error de validación.
        
        Args:
            message: Mensaje de error
            errors: Lista de errores (opcional)
            warnings: Lista de advertencias (opcional)
        """
        super().__init__(message)
        self.errors = errors or []
        self.warnings = warnings or []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir error a diccionario.
        
        Returns:
            Diccionario con información del error
        """
        return {
            "message": str(self),
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


def validate_configuration() -> Dict[str, Any]:
    """
    Validar configuración de la aplicación.
    
    Returns:
        Diccionario con resultados de validación
        
    Raises:
        ConfigValidationError: Si hay errores críticos de configuración
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # Validar GitHub token
    if not settings.GITHUB_TOKEN or not settings.GITHUB_TOKEN.strip():
        errors.append("GITHUB_TOKEN no está configurado. Es requerido para operaciones con GitHub.")
    elif len(settings.GITHUB_TOKEN.strip()) < 20:
        warnings.append("GITHUB_TOKEN parece ser inválido (muy corto, mínimo 20 caracteres).")
    elif not settings.GITHUB_TOKEN.startswith(("ghp_", "github_pat_", "gho_", "ghu_", "ghs_", "ghr_")):
        warnings.append("GITHUB_TOKEN no tiene el formato esperado de token de GitHub.")
    
    # Validar Redis (si se usa Celery)
    if settings.CELERY_BROKER_URL.startswith("redis://"):
        if "localhost" in settings.CELERY_BROKER_URL and not settings.DEBUG:
            warnings.append("Redis configurado para localhost. Considera usar una URL de producción.")
    
    # Validar base de datos
    if settings.DATABASE_URL.startswith("sqlite://") and not settings.DEBUG:
        warnings.append("SQLite en uso. Para producción, considera usar PostgreSQL.")
    
    # Validar storage paths
    try:
        storage_path = Path(settings.STORAGE_PATH)
        if not storage_path.exists():
            logger.debug(f"Creando directorio de storage: {storage_path}")
            storage_path.mkdir(parents=True, exist_ok=True)
        elif not storage_path.is_dir():
            errors.append(f"STORAGE_PATH existe pero no es un directorio: {settings.STORAGE_PATH}")
    except Exception as e:
        errors.append(f"No se puede acceder a STORAGE_PATH '{settings.STORAGE_PATH}': {e}")
    
    try:
        tasks_path = Path(settings.TASKS_STORAGE_PATH)
        if not tasks_path.exists():
            logger.debug(f"Creando directorio de tareas: {tasks_path}")
            tasks_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        warnings.append(f"No se puede crear TASKS_STORAGE_PATH '{settings.TASKS_STORAGE_PATH}': {e}")
    
    try:
        logs_path = Path(settings.LOGS_STORAGE_PATH)
        if not logs_path.exists():
            logger.debug(f"Creando directorio de logs: {logs_path}")
            logs_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        warnings.append(f"No se puede crear LOGS_STORAGE_PATH '{settings.LOGS_STORAGE_PATH}': {e}")
    
    # Validar LLM (si está habilitado)
    if settings.LLM_ENABLED:
        if not settings.OPENROUTER_API_KEY or not settings.OPENROUTER_API_KEY.strip():
            errors.append("LLM_ENABLED=True pero OPENROUTER_API_KEY no está configurado.")
        elif len(settings.OPENROUTER_API_KEY.strip()) < 20:
            warnings.append("OPENROUTER_API_KEY parece ser inválido (muy corto).")
        if not settings.LLM_DEFAULT_MODELS:
            warnings.append("LLM_DEFAULT_MODELS está vacío. No se podrán usar modelos LLM.")
        elif len(settings.LLM_DEFAULT_MODELS) == 0:
            warnings.append("LLM_DEFAULT_MODELS está vacío. No se podrán usar modelos LLM.")
    
    # Validar secret key
    if not settings.SECRET_KEY or settings.SECRET_KEY == "change-me" or len(settings.SECRET_KEY) < 32:
        if settings.DEBUG:
            warnings.append("SECRET_KEY no es seguro. Genera uno nuevo para producción (mínimo 32 caracteres).")
        else:
            errors.append("SECRET_KEY no es seguro. Debe tener al menos 32 caracteres para producción.")
    
    # Validar CORS
    if "*" in settings.CORS_ORIGINS and not settings.DEBUG:
        warnings.append("CORS permite todos los orígenes (*). Considera restringir en producción.")
    
    # Validar timeouts
    if settings.GITHUB_API_TIMEOUT < 5:
        warnings.append("GITHUB_API_TIMEOUT muy bajo. Puede causar timeouts frecuentes.")
    
    if settings.TASK_TIMEOUT < 30:
        warnings.append("TASK_TIMEOUT muy bajo. Tareas complejas pueden fallar.")
    
    # Validar workers
    if settings.WORKER_CONCURRENCY > 16:
        warnings.append("WORKER_CONCURRENCY muy alto (>16). Puede causar problemas de recursos.")
    elif settings.WORKER_CONCURRENCY < 1:
        errors.append("WORKER_CONCURRENCY debe ser al menos 1.")
    
    # Validar circuit breaker
    if settings.CIRCUIT_BREAKER_MAX_FAILURES < 1:
        errors.append("CIRCUIT_BREAKER_MAX_FAILURES debe ser al menos 1.")
    if settings.CIRCUIT_BREAKER_TIMEOUT < 10:
        warnings.append("CIRCUIT_BREAKER_TIMEOUT muy bajo (<10s). Puede causar recuperaciones frecuentes.")
    
    # Resultado
    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "critical_errors": len(errors),
        "warnings_count": len(warnings)
    }
    
    # Log resultados
    if errors:
        logger.error(f"❌ Errores de configuración encontrados: {len(errors)}")
        for error in errors:
            logger.error(f"  - {error}")
    
    if warnings:
        logger.warning(f"⚠️  Advertencias de configuración: {len(warnings)}")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    if not errors and not warnings:
        logger.info("✅ Configuración validada correctamente")
    
    # Lanzar excepción si hay errores críticos
    if errors:
        error_message = f"Errores de configuración encontrados ({len(errors)}): {', '.join(errors[:3])}"
        if len(errors) > 3:
            error_message += f" y {len(errors) - 3} más..."
        raise ConfigValidationError(
            error_message,
            errors=errors,
            warnings=warnings
        )
    
    return result


def print_config_summary() -> None:
    """Imprimir resumen de configuración (sin valores sensibles)."""
    logger.info("📋 Resumen de Configuración:")
    logger.info(f"  - Host: {settings.HOST}:{settings.PORT}")
    logger.info(f"  - Debug: {settings.DEBUG}")
    logger.info(f"  - GitHub Token: {'✅ Configurado' if settings.GITHUB_TOKEN else '❌ No configurado'}")
    logger.info(f"  - LLM Enabled: {settings.LLM_ENABLED}")
    logger.info(f"  - LLM Models: {len(settings.LLM_DEFAULT_MODELS)} modelos")
    logger.info(f"  - Database: {settings.DATABASE_URL.split('://')[0]}")
    logger.info(f"  - Workers: {settings.WORKER_CONCURRENCY}")
    logger.info(f"  - CORS Origins: {len(settings.CORS_ORIGINS)} orígenes")

