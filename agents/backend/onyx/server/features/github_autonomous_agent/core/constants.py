"""
Constantes utilizadas en toda la aplicación.
"""

# Estados de tareas
class TaskStatus:
    """Estados posibles de una tarea."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    # Estados finales (no pueden cambiar después)
    FINAL_STATES = [COMPLETED, FAILED, CANCELLED]
    
    # Estados activos (en proceso)
    ACTIVE_STATES = [PENDING, RUNNING]
    
    # Todos los estados válidos
    ALL_STATES = [PENDING, RUNNING, COMPLETED, FAILED, CANCELLED]
    
    @classmethod
    def is_valid(cls, status: str) -> bool:
        """
        Verificar si un estado es válido.
        
        Args:
            status: Estado a verificar
            
        Returns:
            True si el estado es válido, False en caso contrario
        """
        if not status or not isinstance(status, str):
            return False
        return status in cls.ALL_STATES
    
    @classmethod
    def is_final(cls, status: str) -> bool:
        """
        Verificar si un estado es final (no puede cambiar).
        
        Args:
            status: Estado a verificar
            
        Returns:
            True si el estado es final, False en caso contrario
        """
        return status in cls.FINAL_STATES
    
    @classmethod
    def is_active(cls, status: str) -> bool:
        """
        Verificar si un estado es activo (en proceso).
        
        Args:
            status: Estado a verificar
            
        Returns:
            True si el estado es activo, False en caso contrario
        """
        return status in cls.ACTIVE_STATES


# Estados del agente
class AgentStatus:
    """Estados del agente."""
    RUNNING = "running"
    STOPPED = "stopped"
    PAUSED = "paused"
    ERROR = "error"
    
    # Todos los estados válidos
    ALL_STATES = [RUNNING, STOPPED, PAUSED, ERROR]
    
    @classmethod
    def is_valid(cls, status: str) -> bool:
        """
        Verificar si un estado es válido.
        
        Args:
            status: Estado a verificar
            
        Returns:
            True si el estado es válido, False en caso contrario
        """
        if not status or not isinstance(status, str):
            return False
        return status in cls.ALL_STATES


# Configuración de instrucciones
class InstructionConfig:
    """Configuración para validación de instrucciones."""
    MIN_LENGTH = 5
    MAX_LENGTH = 5000
    DEFAULT_BRANCH = "main"


# Configuración de Git
class GitConfig:
    """Configuración relacionada con Git."""
    MAX_BRANCH_NAME_LENGTH = 255
    INVALID_BRANCH_CHARS = ['~', '^', ':', '?', '*', '[', ' ', '..', '@{', '\\']
    DEFAULT_BASE_BRANCH = "main"


# Configuración de retry
class RetryConfig:
    """Configuración para retry logic."""
    DEFAULT_MAX_ATTEMPTS = 3
    DEFAULT_MIN_WAIT = 2.0
    DEFAULT_MAX_WAIT = 10.0


# Mensajes de error comunes
class ErrorMessages:
    """Mensajes de error estandarizados."""
    GITHUB_TOKEN_REQUIRED = "GitHub token es requerido"
    GITHUB_TOKEN_NOT_CONFIGURED = (
        "GitHub token no configurado. "
        "Configure GITHUB_TOKEN en las variables de entorno."
    )
    TASK_NOT_FOUND = "Tarea no encontrada"
    REPOSITORY_NOT_FOUND = "Repositorio no encontrado"
    INVALID_INSTRUCTION = "Instrucción inválida"
    INVALID_FILE_PATH = "Ruta de archivo inválida"
    INVALID_BRANCH_NAME = "Nombre de rama inválido"
    TASK_PROCESSOR_NOT_INITIALIZED = "Task processor no inicializado"
    STORAGE_ERROR = "Error en el almacenamiento"
    VALIDATION_ERROR = "Error de validación"
    RATE_LIMIT_EXCEEDED = "Límite de requests excedido. Por favor intenta más tarde."
    TIMEOUT_ERROR = "La operación excedió el tiempo máximo permitido"
    CIRCUIT_BREAKER_OPEN = "Circuit breaker abierto. El servicio no está disponible temporalmente."
    # LLM Service Errors
    LLM_SERVICE_UNAVAILABLE = "LLM service no disponible. Verifica que OPENROUTER_API_KEY esté configurada."
    LLM_PROMPT_EMPTY = "El prompt no puede estar vacío"
    LLM_MODEL_NOT_FOUND = "Modelo no encontrado o no disponible"
    LLM_GENERATION_FAILED = "Error al generar respuesta del modelo"
    LLM_INVALID_REQUEST = "Request inválido para el servicio LLM"
    LLM_RATE_LIMIT_EXCEEDED = "Límite de rate limit excedido para este modelo/usuario"
    LLM_CACHE_ERROR = "Error en el sistema de cache"
    LLM_STREAMING_ERROR = "Error durante el streaming de respuesta"
    LLM_TEST_NOT_FOUND = "Test no encontrado"
    LLM_WEBHOOK_NOT_FOUND = "Webhook no encontrado"
    LLM_PROMPT_NOT_FOUND = "Prompt no encontrado"
    LLM_TEST_SUITE_NOT_FOUND = "Test suite no encontrada"
    LLM_INVALID_VARIANT = "Variante inválida para A/B test"
    LLM_INVALID_WEBHOOK_URL = "URL de webhook inválida"
    LLM_INVALID_PROMPT_VERSION = "Versión de prompt inválida"


# Mensajes de éxito comunes
class SuccessMessages:
    """Mensajes de éxito estandarizados."""
    TASK_CREATED = "Tarea creada exitosamente"
    TASK_COMPLETED = "Tarea completada exitosamente"
    TASK_DELETED = "Tarea eliminada exitosamente"
    AGENT_STARTED = "Agente iniciado exitosamente"
    AGENT_STOPPED = "Agente detenido exitosamente"
    AGENT_PAUSED = "Agente pausado exitosamente"
    AGENT_RESUMED = "Agente reanudado exitosamente"
    REPOSITORY_CLONED = "Repositorio clonado exitosamente"
    FILE_CREATED = "Archivo creado exitosamente"
    FILE_UPDATED = "Archivo actualizado exitosamente"
    BRANCH_CREATED = "Rama creada exitosamente"
    PR_CREATED = "Pull request creado exitosamente"


