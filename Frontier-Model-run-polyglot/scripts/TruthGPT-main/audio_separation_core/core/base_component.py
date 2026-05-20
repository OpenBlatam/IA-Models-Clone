"""
Base Component - Componente base simplificado para todos los componentes de audio.

Refactorizado para:
- Eliminar complejidad innecesaria
- Consolidar funcionalidad común
- Seguir principios SOLID
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class BaseComponent(ABC):
    """
    Componente base simplificado.
    
    Responsabilidades:
    - Gestión básica del ciclo de vida (inicialización, limpieza)
    - Estado y salud del componente
    - Métricas básicas
    
    Eliminado:
    - Configuración compleja (se pasa como parámetros)
    - Métodos abstractos innecesarios
    - Validación redundante
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Inicializa el componente base.
        
        Args:
            name: Nombre del componente (opcional, usa nombre de clase por defecto)
        """
        self._name = name or self.__class__.__name__
        self._version = "1.0.0"
        self._initialized = False
        self._ready = False
        self._start_time: Optional[float] = None
        self._last_error: Optional[str] = None
    
    @property
    def name(self) -> str:
        """Nombre del componente."""
        return self._name
    
    @property
    def version(self) -> str:
        """Versión del componente."""
        return self._version
    
    @property
    def is_initialized(self) -> bool:
        """Indica si el componente está inicializado."""
        return self._initialized
    
    @property
    def is_ready(self) -> bool:
        """Indica si el componente está listo para usar."""
        return self._ready
    
    def initialize(self, **kwargs) -> bool:
        """
        Inicializa el componente.
        
        Args:
            **kwargs: Parámetros adicionales pasados a _do_initialize()
        
        Returns:
            True si la inicialización fue exitosa
        
        Raises:
            Exception: Si la inicialización falla
        """
        if self._initialized:
            return True
        
        try:
            self._start_time = time.time()
            self._do_initialize(**kwargs)
            self._initialized = True
            self._ready = True
            self._last_error = None
            return True
        except Exception as e:
            self._last_error = str(e)
            self._ready = False
            raise
    
    def cleanup(self) -> None:
        """
        Limpia los recursos del componente.
        
        Idempotente: seguro llamar múltiples veces.
        """
        if self._initialized:
            try:
                self._do_cleanup()
            except Exception:
                pass
            finally:
                self._initialized = False
                self._ready = False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del componente.
        
        Returns:
            Diccionario con información de estado
        """
        uptime = 0.0
        if self._start_time:
            uptime = time.time() - self._start_time
        
        health = "healthy"
        if not self._ready:
            health = "unhealthy"
        elif self._last_error:
            health = "degraded"
        
        return {
            "name": self._name,
            "version": self._version,
            "initialized": self._initialized,
            "ready": self._ready,
            "health": health,
            "uptime_seconds": uptime,
            "last_error": self._last_error,
        }
    
    def _ensure_ready(self) -> None:
        """
        Asegura que el componente esté listo.
        
        Raises:
            RuntimeError: Si el componente no está listo
        """
        if not self._initialized:
            self.initialize()
        
        if not self._ready:
            raise RuntimeError(f"{self._name} is not ready: {self._last_error}")
    
    def _set_error(self, error_message: str) -> None:
        """
        Establece un mensaje de error en el componente.
        
        Args:
            error_message: Mensaje de error
        """
        self._last_error = error_message
        self._ready = False
    
    def _clear_error(self) -> None:
        """Limpia el mensaje de error del componente."""
        self._last_error = None
        if self._initialized:
            self._ready = True
    
    def _handle_error(
        self,
        error: Exception,
        error_class: type,
        operation: str,
        **kwargs
    ) -> None:
        """
        Helper para manejar errores de manera consistente.
        
        Args:
            error: La excepción que ocurrió
            error_class: Clase de excepción a lanzar
            operation: Nombre de la operación que falló
            **kwargs: Argumentos adicionales para la excepción (ej: component)
        """
        self._set_error(str(error))
        error_message = f"{operation} failed: {error}"
        raise error_class(error_message, **kwargs) from error
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS ABSTRACTOS (implementar en subclases)
    # ════════════════════════════════════════════════════════════════════════════
    
    @abstractmethod
    def _do_initialize(self, **kwargs) -> None:
        """
        Implementación específica de inicialización.
        
        Llamado por initialize() después de validaciones básicas.
        
        Args:
            **kwargs: Parámetros adicionales pasados desde initialize()
        """
        pass
    
    def _do_cleanup(self) -> None:
        """
        Implementación específica de limpieza.
        
        Override si es necesario. Por defecto no hace nada.
        """
        pass
