"""
Clase base para conectores de redes sociales
Elimina duplicación de código entre conectores
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..core.base_service import BaseService
from ..core.exceptions import ConnectorError
from ..utils.error_handler import RetryHandler, RetryConfig, CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class BaseConnector(BaseService, ABC):
    """
    Clase base para todos los conectores de redes sociales
    
    Proporciona:
    - Retry handler configurado
    - Circuit breaker configurado
    - Manejo de errores consistente
    - Logging estructurado
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Inicializa el conector base
        
        Args:
            api_key: API key para el servicio (opcional)
            retry_config: Configuración de retry (opcional, usa defaults)
            circuit_breaker_config: Configuración de circuit breaker (opcional, usa defaults)
        """
        super().__init__()
        self.api_key = api_key
        
        # Configurar retry con defaults si no se proporciona
        if retry_config is None:
            retry_config = RetryConfig(
                max_attempts=3,
                base_delay=1.0,
                backoff_factor=2.0
            )
        self.retry_handler = RetryHandler(retry_config)
        
        # Configurar circuit breaker con defaults si no se proporciona
        if circuit_breaker_config is None:
            circuit_breaker_config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0
            )
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)
    
    @abstractmethod
    async def get_profile(self, username: str) -> Dict[str, Any]:
        """
        Obtiene información básica del perfil
        
        Args:
            username: Nombre de usuario o identificador
            
        Returns:
            Diccionario con información del perfil
            
        Raises:
            ConnectorError: Si hay error obteniendo el perfil
        """
        pass
    
    @abstractmethod
    async def get_videos(
        self,
        username: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtiene videos del perfil
        
        Args:
            username: Nombre de usuario o identificador
            limit: Límite de videos a obtener
            
        Returns:
            Lista de diccionarios con información de videos
            
        Raises:
            ConnectorError: Si hay error obteniendo los videos
        """
        pass
    
    async def _execute_with_retry(
        self,
        operation: str,
        func,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecuta una operación con retry y circuit breaker
        
        Args:
            operation: Nombre de la operación (para logging)
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            Resultado de la función
            
        Raises:
            ConnectorError: Si la operación falla después de todos los reintentos
        """
        self._log_operation(operation, **kwargs)
        
        try:
            return await self.retry_handler.execute_async(
                lambda: self.circuit_breaker.call_async(func, *args, **kwargs)
            )
        except Exception as e:
            self._handle_error(
                e,
                operation,
                {"connector": self.__class__.__name__, **kwargs}
            )
            raise ConnectorError(
                f"Error en {operation}: {str(e)}",
                details={"connector": self.__class__.__name__}
            ) from e
    
    def _validate_api_key(self) -> None:
        """
        Valida que el API key esté presente si es requerido
        
        Raises:
            ConnectorError: Si el API key es requerido pero no está presente
        """
        if self.api_key is None:
            raise ConnectorError(
                f"API key requerida para {self.__class__.__name__}",
                details={"connector": self.__class__.__name__}
            )

