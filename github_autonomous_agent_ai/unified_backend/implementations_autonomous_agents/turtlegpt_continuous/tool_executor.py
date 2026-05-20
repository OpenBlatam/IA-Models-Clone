"""
Tool Executor Module
====================

Ejecutor centralizado de herramientas con validación, manejo de errores y logging.
Proporciona una interfaz unificada para ejecutar herramientas del ToolRegistry.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..common.tools import ToolRegistry
from .error_handler import AgentError, ErrorSeverity
from .resilient_operations import resilient_call_async, CircuitBreaker

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Ejecutor centralizado de herramientas.
    
    Proporciona una interfaz unificada para ejecutar herramientas
    con validación, manejo de errores y logging.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        enable_circuit_breaker: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0
    ):
        """
        Inicializar ejecutor de herramientas.
        
        Args:
            tool_registry: Registro de herramientas
            enable_circuit_breaker: Si se debe usar circuit breaker
            circuit_breaker_threshold: Umbral de fallos para circuit breaker
            circuit_breaker_timeout: Tiempo de recuperación del circuit breaker
        """
        self.tool_registry = tool_registry
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Crear circuit breakers por herramienta
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        if enable_circuit_breaker:
            for tool_name in tool_registry.list_tools():
                self.circuit_breakers[tool_name] = CircuitBreaker(
                    failure_threshold=circuit_breaker_threshold,
                    recovery_timeout=circuit_breaker_timeout
                )
    
    def execute_tool(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        validate_args: bool = True,
        use_circuit_breaker: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar una herramienta (versión síncrona).
        
        Args:
            tool_name: Nombre de la herramienta
            args: Argumentos para la herramienta
            validate_args: Si se deben validar los argumentos
            use_circuit_breaker: Si se debe usar circuit breaker (None = usar default)
            
        Returns:
            Resultado de la ejecución
            
        Raises:
            AgentError: Si la herramienta no existe o falla
        """
        args = args or {}
        use_circuit_breaker = use_circuit_breaker if use_circuit_breaker is not None else self.enable_circuit_breaker
        
        # Validar que la herramienta existe
        tool = self.tool_registry.get(tool_name)
        if not tool:
            raise AgentError(
                f"Tool '{tool_name}' not found in registry",
                severity=ErrorSeverity.MEDIUM,
                context={"tool_name": tool_name, "available_tools": self.tool_registry.list_tools()}
            )
        
        # Validar argumentos si está habilitado
        if validate_args:
            validation_error = self._validate_tool_args(tool, args)
            if validation_error:
                raise AgentError(
                    f"Invalid arguments for tool '{tool_name}': {validation_error}",
                    severity=ErrorSeverity.MEDIUM,
                    context={"tool_name": tool_name, "args": args}
                )
        
        # Ejecutar herramienta
        try:
            start_time = datetime.now()
            
            # Usar circuit breaker si está habilitado
            if use_circuit_breaker and tool_name in self.circuit_breakers:
                result = self.circuit_breakers[tool_name].call(
                    tool.execute,
                    **args
                )
            else:
                result = tool.execute(**args)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.debug(
                f"Tool '{tool_name}' executed successfully in {elapsed:.2f}s",
                extra={"tool_name": tool_name, "elapsed": elapsed}
            )
            
            return {
                "tool": tool_name,
                "result": result,
                "success": True,
                "elapsed": elapsed,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0.0
            logger.error(
                f"Error executing tool '{tool_name}': {e}",
                exc_info=True,
                extra={"tool_name": tool_name, "args": args, "elapsed": elapsed}
            )
            
            raise AgentError(
                f"Tool execution failed: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context={"tool_name": tool_name, "args": args},
                original_error=e
            )
    
    async def execute_tool_async(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        validate_args: bool = True,
        use_circuit_breaker: Optional[bool] = None,
        max_retries: int = 0,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar una herramienta (versión asíncrona con retries y timeout).
        
        Args:
            tool_name: Nombre de la herramienta
            args: Argumentos para la herramienta
            validate_args: Si se deben validar los argumentos
            use_circuit_breaker: Si se debe usar circuit breaker (None = usar default)
            max_retries: Número máximo de reintentos
            timeout: Timeout en segundos
            
        Returns:
            Resultado de la ejecución
            
        Raises:
            AgentError: Si la herramienta no existe o falla
        """
        args = args or {}
        use_circuit_breaker = use_circuit_breaker if use_circuit_breaker is not None else self.enable_circuit_breaker
        
        # Validar que la herramienta existe
        tool = self.tool_registry.get(tool_name)
        if not tool:
            raise AgentError(
                f"Tool '{tool_name}' not found in registry",
                severity=ErrorSeverity.MEDIUM,
                context={"tool_name": tool_name, "available_tools": self.tool_registry.list_tools()}
            )
        
        # Validar argumentos si está habilitado
        if validate_args:
            validation_error = self._validate_tool_args(tool, args)
            if validation_error:
                raise AgentError(
                    f"Invalid arguments for tool '{tool_name}': {validation_error}",
                    severity=ErrorSeverity.MEDIUM,
                    context={"tool_name": tool_name, "args": args}
                )
        
        # Función interna para ejecutar la herramienta
        async def _execute():
            start_time = datetime.now()
            
            # Usar circuit breaker si está habilitado
            if use_circuit_breaker and tool_name in self.circuit_breakers:
                result = await self.circuit_breakers[tool_name].call_async(
                    self._execute_tool_async,
                    tool,
                    args
                )
            else:
                result = await self._execute_tool_async(tool, args)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.debug(
                f"Tool '{tool_name}' executed successfully in {elapsed:.2f}s",
                extra={"tool_name": tool_name, "elapsed": elapsed}
            )
            
            return {
                "tool": tool_name,
                "result": result,
                "success": True,
                "elapsed": elapsed,
                "timestamp": datetime.now().isoformat()
            }
        
        # Ejecutar con retries y timeout si están configurados
        try:
            if max_retries > 0 or timeout:
                result = await resilient_call_async(
                    _execute,
                    max_retries=max_retries,
                    timeout=timeout,
                    retryable_errors=(AgentError,)
                )
            else:
                result = await _execute()
            
            return result
            
        except Exception as e:
            logger.error(
                f"Error executing tool '{tool_name}': {e}",
                exc_info=True,
                extra={"tool_name": tool_name, "args": args}
            )
            
            raise AgentError(
                f"Tool execution failed: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context={"tool_name": tool_name, "args": args},
                original_error=e
            )
    
    async def _execute_tool_async(
        self,
        tool: Any,
        args: Dict[str, Any]
    ) -> Any:
        """
        Ejecutar herramienta de forma asíncrona.
        
        Args:
            tool: Herramienta a ejecutar
            args: Argumentos
            
        Returns:
            Resultado de la herramienta
        """
        # Si la herramienta tiene método async, usarlo
        if hasattr(tool, 'execute_async'):
            return await tool.execute_async(**args)
        
        # Si no, ejecutar síncronamente en un executor
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: tool.execute(**args)
            )
    
    def _validate_tool_args(self, tool: Any, args: Dict[str, Any]) -> Optional[str]:
        """
        Validar argumentos de una herramienta.
        
        Args:
            tool: Herramienta a validar
            args: Argumentos a validar
            
        Returns:
            Mensaje de error si hay problema, None si es válido
        """
        # Validación básica: verificar que la herramienta tiene método execute
        if not hasattr(tool, 'execute'):
            return "Tool does not have execute method"
        
        # Si la herramienta tiene método de validación, usarlo
        if hasattr(tool, 'validate_args'):
            try:
                tool.validate_args(**args)
                return None
            except Exception as e:
                return str(e)
        
        # Validación básica: verificar que args es un dict
        if not isinstance(args, dict):
            return "Args must be a dictionary"
        
        return None
    
    def get_available_tools(self) -> List[str]:
        """
        Obtener lista de herramientas disponibles.
        
        Returns:
            Lista de nombres de herramientas
        """
        return self.tool_registry.list_tools()
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtener información sobre una herramienta.
        
        Args:
            tool_name: Nombre de la herramienta
            
        Returns:
            Información de la herramienta o None si no existe
        """
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return None
        
        info = {
            "name": tool_name,
            "available": True
        }
        
        # Agregar información adicional si está disponible
        if hasattr(tool, 'description'):
            info["description"] = tool.description
        if hasattr(tool, 'parameters'):
            info["parameters"] = tool.parameters
        
        # Estado del circuit breaker si está disponible
        if tool_name in self.circuit_breakers:
            breaker = self.circuit_breakers[tool_name]
            info["circuit_breaker"] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count
            }
        
        return info
    
    def reset_circuit_breaker(self, tool_name: str) -> bool:
        """
        Resetear circuit breaker de una herramienta.
        
        Args:
            tool_name: Nombre de la herramienta
            
        Returns:
            True si se reseteó, False si no existe
        """
        if tool_name in self.circuit_breakers:
            self.circuit_breakers[tool_name].reset()
            logger.info(f"Circuit breaker reset for tool '{tool_name}'")
            return True
        return False


def create_tool_executor(
    tool_registry: ToolRegistry,
    enable_circuit_breaker: bool = True,
    circuit_breaker_threshold: int = 5,
    circuit_breaker_timeout: float = 60.0
) -> ToolExecutor:
    """
    Factory function para crear ToolExecutor.
    
    Args:
        tool_registry: Registro de herramientas
        enable_circuit_breaker: Si se debe usar circuit breaker
        circuit_breaker_threshold: Umbral de fallos para circuit breaker
        circuit_breaker_timeout: Tiempo de recuperación del circuit breaker
        
    Returns:
        Instancia de ToolExecutor
    """
    return ToolExecutor(
        tool_registry,
        enable_circuit_breaker,
        circuit_breaker_threshold,
        circuit_breaker_timeout
    )


