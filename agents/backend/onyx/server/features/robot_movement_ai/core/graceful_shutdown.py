"""
Graceful Shutdown System
========================

Sistema de apagado graceful.
"""

import logging
import asyncio
import signal
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ShutdownHandler:
    """Handler de shutdown."""
    handler_id: str
    name: str
    handler_func: Callable
    priority: int = 5  # 1-10, mayor = primero
    timeout: float = 30.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class GracefulShutdownManager:
    """
    Gestor de apagado graceful.
    
    Gestiona handlers de shutdown ordenados.
    """
    
    def __init__(self):
        """Inicializar gestor de shutdown."""
        self.handlers: Dict[str, ShutdownHandler] = {}
        self.shutdown_in_progress = False
        self.shutdown_started_at: Optional[datetime] = None
    
    def register_handler(
        self,
        handler_id: str,
        name: str,
        handler_func: Callable,
        priority: int = 5,
        timeout: float = 30.0,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ShutdownHandler:
        """
        Registrar handler de shutdown.
        
        Args:
            handler_id: ID único del handler
            name: Nombre
            handler_func: Función handler
            priority: Prioridad (1-10, mayor = primero)
            timeout: Timeout en segundos
            enabled: Si está habilitado
            metadata: Metadata adicional
            
        Returns:
            Handler registrado
        """
        handler = ShutdownHandler(
            handler_id=handler_id,
            name=name,
            handler_func=handler_func,
            priority=priority,
            timeout=timeout,
            enabled=enabled,
            metadata=metadata or {}
        )
        
        self.handlers[handler_id] = handler
        logger.info(f"Registered shutdown handler: {name} ({handler_id})")
        
        return handler
    
    async def shutdown(self, timeout: float = 60.0) -> Dict[str, Any]:
        """
        Ejecutar shutdown graceful.
        
        Args:
            timeout: Timeout total en segundos
            
        Returns:
            Resumen de shutdown
        """
        if self.shutdown_in_progress:
            return {"error": "Shutdown already in progress"}
        
        self.shutdown_in_progress = True
        self.shutdown_started_at = datetime.now()
        
        logger.info("Starting graceful shutdown...")
        
        # Ordenar handlers por prioridad (mayor primero)
        enabled_handlers = [
            h for h in self.handlers.values()
            if h.enabled
        ]
        enabled_handlers.sort(key=lambda x: x.priority, reverse=True)
        
        results = []
        start_time = asyncio.get_event_loop().time()
        
        for handler in enabled_handlers:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Shutdown timeout reached, stopping handlers")
                break
            
            try:
                logger.info(f"Executing shutdown handler: {handler.name}")
                
                if asyncio.iscoroutinefunction(handler.handler_func):
                    await asyncio.wait_for(
                        handler.handler_func(),
                        timeout=min(handler.timeout, timeout - elapsed)
                    )
                else:
                    loop = asyncio.get_event_loop()
                    await asyncio.wait_for(
                        loop.run_in_executor(None, handler.handler_func),
                        timeout=min(handler.timeout, timeout - elapsed)
                    )
                
                results.append({
                    "handler_id": handler.handler_id,
                    "name": handler.name,
                    "success": True
                })
                logger.info(f"Shutdown handler {handler.name} completed successfully")
            except asyncio.TimeoutError:
                results.append({
                    "handler_id": handler.handler_id,
                    "name": handler.name,
                    "success": False,
                    "error": "Timeout"
                })
                logger.error(f"Shutdown handler {handler.name} timed out")
            except Exception as e:
                results.append({
                    "handler_id": handler.handler_id,
                    "name": handler.name,
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"Shutdown handler {handler.name} failed: {e}")
        
        total_time = (datetime.now() - self.shutdown_started_at).total_seconds()
        success_count = sum(1 for r in results if r.get("success", False))
        
        logger.info(f"Graceful shutdown completed in {total_time:.2f}s")
        
        return {
            "total_handlers": len(enabled_handlers),
            "successful_handlers": success_count,
            "failed_handlers": len(enabled_handlers) - success_count,
            "total_time": total_time,
            "results": results
        }
    
    def setup_signal_handlers(self) -> None:
        """Configurar handlers de señales."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Signal handlers registered")
    
    def get_handler(self, handler_id: str) -> Optional[ShutdownHandler]:
        """Obtener handler por ID."""
        return self.handlers.get(handler_id)
    
    def list_handlers(self) -> List[ShutdownHandler]:
        """Listar todos los handlers."""
        return list(self.handlers.values())


# Instancia global
_graceful_shutdown_manager: Optional[GracefulShutdownManager] = None


def get_graceful_shutdown_manager() -> GracefulShutdownManager:
    """Obtener instancia global del gestor de shutdown."""
    global _graceful_shutdown_manager
    if _graceful_shutdown_manager is None:
        _graceful_shutdown_manager = GracefulShutdownManager()
    return _graceful_shutdown_manager






