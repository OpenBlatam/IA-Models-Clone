"""
Agent Lifecycle Module
======================

Gestión del ciclo de vida del agente: inicio, ejecución y detención.
Centraliza la lógica de inicialización, ejecución y limpieza.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentLifecycle:
    """
    Gestor del ciclo de vida del agente.
    
    Maneja las fases de inicio, ejecución y detención del agente
    de forma estructurada y robusta.
    """
    
    def __init__(
        self,
        agent_name: str,
        start_callback: Optional[Callable] = None,
        stop_callback: Optional[Callable] = None,
        cleanup_callback: Optional[Callable] = None
    ):
        """
        Inicializar gestor de ciclo de vida.
        
        Args:
            agent_name: Nombre del agente
            start_callback: Callback para inicio
            stop_callback: Callback para detención
            cleanup_callback: Callback para limpieza
        """
        self.agent_name = agent_name
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.cleanup_callback = cleanup_callback
        self.start_time: Optional[datetime] = None
        self.stop_time: Optional[datetime] = None
    
    async def start(
        self,
        main_loop: Callable,
        pre_start_hooks: Optional[list] = None,
        post_start_hooks: Optional[list] = None
    ) -> None:
        """
        Iniciar el agente.
        
        Args:
            main_loop: Función del loop principal
            pre_start_hooks: Hooks a ejecutar antes de iniciar
            post_start_hooks: Hooks a ejecutar después de iniciar
        """
        self.start_time = datetime.now()
        logger.info(f"Starting agent lifecycle: {self.agent_name}")
        
        # Ejecutar hooks pre-inicio
        if pre_start_hooks:
            for hook in pre_start_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception as e:
                    logger.warning(f"Error in pre-start hook: {e}")
        
        # Ejecutar callback de inicio
        if self.start_callback:
            try:
                if asyncio.iscoroutinefunction(self.start_callback):
                    await self.start_callback()
                else:
                    self.start_callback()
            except Exception as e:
                logger.error(f"Error in start callback: {e}", exc_info=True)
        
        # Ejecutar loop principal
        try:
            await main_loop()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            raise
        finally:
            # Ejecutar hooks post-inicio
            if post_start_hooks:
                for hook in post_start_hooks:
                    try:
                        if asyncio.iscoroutinefunction(hook):
                            await hook()
                        else:
                            hook()
                    except Exception as e:
                        logger.warning(f"Error in post-start hook: {e}")
            
            # Ejecutar limpieza
            await self.cleanup()
    
    async def stop(self) -> None:
        """Detener el agente."""
        if self.stop_time:
            logger.warning("Agent already stopped")
            return
        
        self.stop_time = datetime.now()
        logger.info(f"Stopping agent lifecycle: {self.agent_name}")
        
        # Ejecutar callback de detención
        if self.stop_callback:
            try:
                if asyncio.iscoroutinefunction(self.stop_callback):
                    await self.stop_callback()
                else:
                    self.stop_callback()
            except Exception as e:
                logger.error(f"Error in stop callback: {e}", exc_info=True)
    
    async def cleanup(self) -> None:
        """Limpiar recursos del agente."""
        logger.info(f"Cleaning up agent lifecycle: {self.agent_name}")
        
        # Ejecutar callback de limpieza
        if self.cleanup_callback:
            try:
                if asyncio.iscoroutinefunction(self.cleanup_callback):
                    await self.cleanup_callback()
                else:
                    self.cleanup_callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}", exc_info=True)
        
        # Calcular tiempo de ejecución
        if self.start_time:
            if self.stop_time:
                duration = (self.stop_time - self.start_time).total_seconds()
            else:
                duration = (datetime.now() - self.start_time).total_seconds()
            
            logger.info(f"Agent ran for {duration:.2f} seconds")
    
    def get_uptime(self) -> Optional[float]:
        """
        Obtener tiempo de ejecución en segundos.
        
        Returns:
            Tiempo de ejecución o None si no ha iniciado
        """
        if not self.start_time:
            return None
        
        if self.stop_time:
            return (self.stop_time - self.start_time).total_seconds()
        
        return (datetime.now() - self.start_time).total_seconds()
    
    def is_running(self) -> bool:
        """
        Verificar si el agente está ejecutándose.
        
        Returns:
            True si está ejecutándose
        """
        return self.start_time is not None and self.stop_time is None


def create_agent_lifecycle(
    agent_name: str,
    start_callback: Optional[Callable] = None,
    stop_callback: Optional[Callable] = None,
    cleanup_callback: Optional[Callable] = None
) -> AgentLifecycle:
    """
    Factory function para crear AgentLifecycle.
    
    Args:
        agent_name: Nombre del agente
        start_callback: Callback para inicio
        stop_callback: Callback para detención
        cleanup_callback: Callback para limpieza
        
    Returns:
        Instancia de AgentLifecycle
    """
    return AgentLifecycle(
        agent_name=agent_name,
        start_callback=start_callback,
        stop_callback=stop_callback,
        cleanup_callback=cleanup_callback
    )


