"""
Error Recovery System for Humanoid Devin Robot (Optimizado)
===========================================================

Sistema de recuperación automática de errores para el robot humanoide.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """
    Decorador para anotar excepciones con códigos de error y descripciones.
    
    Args:
        description: Descripción del error que se usará en el constructor.
    
    Usage:
        @ErrorCode(description="Invalid input provided")
        class MyException(Exception):
            def __init__(self):
                super().__init__(description)
    """
    def decorator(cls):
        # Almacenar la descripción en la clase
        cls._error_description = description
        return cls
    return decorator


class RecoveryStrategy(str, Enum):
    """Estrategias de recuperación."""
    RETRY = "retry"
    ROLLBACK = "rollback"
    ALTERNATIVE = "alternative"
    EMERGENCY_STOP = "emergency_stop"
    GRACEFUL_DEGRADE = "graceful_degrade"


@ErrorCode(description="Error in error recovery system")
class ErrorRecoveryError(Exception):
    """Excepción para errores del sistema de recuperación."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in error recovery system")
        super().__init__(message)
        self.message = message


class ErrorRecoverySystem:
    """
    Sistema de recuperación automática de errores.
    
    Detecta errores y aplica estrategias de recuperación automática.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_auto_recovery: bool = True
    ):
        """
        Inicializar sistema de recuperación de errores.
        
        Args:
            max_retries: Número máximo de reintentos
            retry_delay: Delay entre reintentos en segundos
            enable_auto_recovery: Habilitar recuperación automática
        """
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not isinstance(retry_delay, (int, float)) or retry_delay < 0:
            raise ValueError("retry_delay must be a non-negative number")
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_auto_recovery = enable_auto_recovery
        
        # Historial de errores y recuperaciones
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_history: List[Dict[str, Any]] = []
        
        # Estrategias de recuperación personalizadas
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # Estadísticas
        self.total_errors = 0
        self.total_recoveries = 0
        self.successful_recoveries = 0
        
        logger.info(
            f"Error recovery system initialized: "
            f"max_retries={max_retries}, auto_recovery={enable_auto_recovery}"
        )
    
    async def recover_from_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        robot_driver: Optional[Any] = None
    ) -> bool:
        """
        Intentar recuperarse de un error.
        
        Args:
            error: Excepción que ocurrió
            context: Contexto del error (acción, parámetros, etc.)
            robot_driver: Instancia del driver del robot (opcional)
            
        Returns:
            True si la recuperación fue exitosa, False en caso contrario
        """
        if not self.enable_auto_recovery:
            return False
        
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Registrar error
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_msg,
            "context": context
        }
        self.error_history.append(error_record)
        self.total_errors += 1
        
        logger.warning(f"Error detected: {error_type} - {error_msg}")
        
        # Determinar estrategia de recuperación
        strategy = self._determine_recovery_strategy(error, context)
        
        try:
            # Aplicar estrategia de recuperación
            success = await self._apply_recovery_strategy(
                strategy, error, context, robot_driver
            )
            
            # Registrar recuperación
            recovery_record = {
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy.value if isinstance(strategy, RecoveryStrategy) else str(strategy),
                "success": success,
                "error_type": error_type
            }
            self.recovery_history.append(recovery_record)
            self.total_recoveries += 1
            
            if success:
                self.successful_recoveries += 1
                logger.info(f"Recovery successful using strategy: {strategy}")
            else:
                logger.warning(f"Recovery failed using strategy: {strategy}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error during recovery: {e}", exc_info=True)
            return False
    
    def _determine_recovery_strategy(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> RecoveryStrategy:
        """
        Determinar estrategia de recuperación basada en el error.
        
        Args:
            error: Excepción
            context: Contexto
            
        Returns:
            Estrategia de recuperación
        """
        error_type = type(error).__name__
        action_type = context.get("action_type", "unknown")
        
        # Estrategias basadas en tipo de error
        if "Connection" in error_type or "Timeout" in error_type:
            return RecoveryStrategy.RETRY
        
        if "Validation" in error_type or "Value" in error_type:
            return RecoveryStrategy.ALTERNATIVE
        
        if "Safety" in error_type or "Collision" in error_type:
            return RecoveryStrategy.EMERGENCY_STOP
        
        if "Trajectory" in error_type or "Planning" in error_type:
            return RecoveryStrategy.ALTERNATIVE
        
        # Estrategia por defecto
        return RecoveryStrategy.RETRY
    
    async def _apply_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        error: Exception,
        context: Dict[str, Any],
        robot_driver: Optional[Any]
    ) -> bool:
        """
        Aplicar estrategia de recuperación.
        
        Args:
            strategy: Estrategia a aplicar
            error: Excepción original
            context: Contexto
            robot_driver: Driver del robot
            
        Returns:
            True si exitoso, False en caso contrario
        """
        if strategy == RecoveryStrategy.RETRY:
            return await self._retry_strategy(context, robot_driver)
        
        elif strategy == RecoveryStrategy.ROLLBACK:
            return await self._rollback_strategy(context, robot_driver)
        
        elif strategy == RecoveryStrategy.ALTERNATIVE:
            return await self._alternative_strategy(context, robot_driver)
        
        elif strategy == RecoveryStrategy.EMERGENCY_STOP:
            return await self._emergency_stop_strategy(robot_driver)
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADE:
            return await self._graceful_degrade_strategy(context, robot_driver)
        
        return False
    
    async def _retry_strategy(
        self,
        context: Dict[str, Any],
        robot_driver: Optional[Any]
    ) -> bool:
        """Estrategia de reintento."""
        action = context.get("action")
        if not action or not callable(action):
            return False
        
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
                result = await action()
                if result:
                    logger.info(f"Retry successful on attempt {attempt + 1}")
                    return True
            except Exception as e:
                logger.debug(f"Retry attempt {attempt + 1} failed: {e}")
        
        return False
    
    async def _rollback_strategy(
        self,
        context: Dict[str, Any],
        robot_driver: Optional[Any]
    ) -> bool:
        """Estrategia de rollback."""
        if not robot_driver:
            return False
        
        try:
            # Intentar volver a la última posición conocida
            if hasattr(robot_driver, 'get_last_safe_position'):
                safe_position = robot_driver.get_last_safe_position()
                if safe_position:
                    await robot_driver.set_joint_positions(safe_position)
                    logger.info("Rollback to safe position successful")
                    return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}", exc_info=True)
        
        return False
    
    async def _alternative_strategy(
        self,
        context: Dict[str, Any],
        robot_driver: Optional[Any]
    ) -> bool:
        """Estrategia alternativa."""
        action_type = context.get("action_type", "unknown")
        
        # Intentar acción alternativa más simple
        if action_type == "walk":
            # Intentar caminar más lento
            if robot_driver:
                try:
                    await robot_driver.walk(
                        direction=context.get("direction", "forward"),
                        distance=context.get("distance", 0.5) * 0.5,  # Mitad de distancia
                        speed=context.get("speed", 0.5) * 0.5  # Mitad de velocidad
                    )
                    return True
                except Exception:
                    pass
        
        return False
    
    async def _emergency_stop_strategy(
        self,
        robot_driver: Optional[Any]
    ) -> bool:
        """Estrategia de parada de emergencia."""
        if not robot_driver:
            return False
        
        try:
            # Parar todos los movimientos
            if hasattr(robot_driver, 'emergency_stop'):
                await robot_driver.emergency_stop()
                logger.warning("Emergency stop activated")
                return True
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}", exc_info=True)
        
        return False
    
    async def _graceful_degrade_strategy(
        self,
        context: Dict[str, Any],
        robot_driver: Optional[Any]
    ) -> bool:
        """Estrategia de degradación elegante."""
        # Reducir funcionalidad pero mantener operación básica
        logger.info("Graceful degradation activated")
        return True
    
    def register_recovery_strategy(
        self,
        error_type: str,
        strategy_func: Callable
    ) -> None:
        """
        Registrar estrategia de recuperación personalizada.
        
        Args:
            error_type: Tipo de error (nombre de clase)
            strategy_func: Función de recuperación
        """
        if not error_type or not isinstance(error_type, str):
            raise ValueError("error_type must be a non-empty string")
        if not callable(strategy_func):
            raise ValueError("strategy_func must be callable")
        
        self.recovery_strategies[error_type] = strategy_func
        logger.info(f"Recovery strategy registered for: {error_type}")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de recuperación.
        
        Returns:
            Dict con estadísticas
        """
        recovery_rate = (
            self.successful_recoveries / self.total_recoveries
            if self.total_recoveries > 0 else 0.0
        )
        
        return {
            "total_errors": self.total_errors,
            "total_recoveries": self.total_recoveries,
            "successful_recoveries": self.successful_recoveries,
            "recovery_rate": recovery_rate,
            "recent_errors": len([
                e for e in self.error_history
                if (datetime.now() - datetime.fromisoformat(e["timestamp"])).total_seconds() < 3600
            ])
        }
    
    def clear_history(self) -> None:
        """Limpiar historial de errores y recuperaciones."""
        self.error_history.clear()
        self.recovery_history.clear()
        logger.info("Error and recovery history cleared")

