"""
Persistent Service - Servicio persistente 24/7
==============================================

Servicio que puede correr en background incluso cuando la computadora está "apagada"
(usando servicios del sistema o cloud).

Gestiona el ciclo de vida del agente como servicio del sistema con manejo
adecuado de señales y shutdown graceful.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional, TYPE_CHECKING

from .validation_utils import validate_not_none, validate_not_empty
from .error_handling import error_context

if TYPE_CHECKING:
    from .agent import CursorAgent

logger = logging.getLogger(__name__)


class PersistentService:
    """
    Servicio persistente que puede correr 24/7.
    
    Gestiona el agente como servicio del sistema con:
    - Manejo de señales (SIGINT, SIGTERM)
    - Shutdown graceful
    - Integración con servicios del sistema (Windows/Linux/macOS)
    """
    
    def __init__(
        self,
        agent: "CursorAgent",
        service_name: str = "cursor_agent_24_7"
    ) -> None:
        """
        Inicializar servicio persistente.
        
        Args:
            agent: Instancia del agente a ejecutar.
            service_name: Nombre del servicio (default: "cursor_agent_24_7").
        
        Raises:
            ValueError: Si agent es None o service_name está vacío.
        """
        validate_not_none(agent, "agent")
        validate_not_empty(service_name, "service_name")
        
        self.agent: "CursorAgent" = agent
        self.service_name: str = service_name.strip()
        self.running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._original_handlers: dict[int, signal.Handlers] = {}
        
        # Configurar handlers de señales
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """
        Configurar handlers de señales del sistema.
        
        Guarda los handlers originales para restaurarlos si es necesario.
        """
        try:
            # Guardar handlers originales
            self._original_handlers[signal.SIGINT] = signal.signal(
                signal.SIGINT,
                self._signal_handler
            )
            self._original_handlers[signal.SIGTERM] = signal.signal(
                signal.SIGTERM,
                self._signal_handler
            )
            
            # En Windows, también manejar SIGBREAK si está disponible
            if hasattr(signal, 'SIGBREAK'):
                try:
                    self._original_handlers[signal.SIGBREAK] = signal.signal(
                        signal.SIGBREAK,
                        self._signal_handler
                    )
                except (AttributeError, ValueError):
                    pass  # SIGBREAK no disponible en este sistema
            
            logger.debug("Signal handlers configured")
        
        except (ValueError, OSError) as e:
            logger.warning(f"Could not set up signal handlers: {e}")
    
    def _signal_handler(
        self,
        signum: int,
        frame: Optional[object]
    ) -> None:
        """
        Manejar señales del sistema.
        
        Args:
            signum: Número de la señal recibida.
            frame: Frame actual (opcional).
        """
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.info(f"📶 Received signal {signal_name} ({signum}), shutting down...")
        self._shutdown_event.set()
    
    async def run(self) -> None:
        """
        Ejecutar servicio persistente.
        
        Inicia el agente y mantiene el servicio corriendo hasta recibir
        una señal de shutdown.
        
        Raises:
            RuntimeError: Si hay error al iniciar el agente.
        """
        if self.running:
            logger.warning("Service is already running")
            return
        
        logger.info(f"🚀 Starting persistent service: {self.service_name}")
        self.running = True
        
        try:
            # Iniciar agente
            from .error_handling import safe_async_call
            
            result = await safe_async_call(
                self.agent.start,
                operation="starting agent in persistent service",
                logger_instance=logger,
                reraise=True
            )
            
            if result is not None:
                logger.info("✅ Agent started successfully")
            
            # Mantener servicio corriendo hasta recibir señal de shutdown
            try:
                await self._shutdown_event.wait()
                logger.info("Shutdown signal received")
            except asyncio.CancelledError:
                logger.info("Service cancelled")
        
        except Exception as e:
            logger.error(f"❌ Service error: {e}", exc_info=True)
            raise RuntimeError(f"Service error: {e}") from e
        
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """
        Cerrar servicio de forma graceful.
        
        Detiene el agente y limpia recursos.
        """
        if not self.running:
            return
        
        logger.info("🛑 Shutting down service...")
        self.running = False
        
        try:
            # Detener agente
            from .error_handling import safe_async_call
            
            result = await safe_async_call(
                self.agent.stop,
                operation="stopping agent in persistent service",
                logger_instance=logger,
                reraise=False
            )
            
            if result is not None:
                logger.info("✅ Agent stopped successfully")
            
            # Restaurar handlers de señales originales
            self._restore_signal_handlers()
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
        
        finally:
            logger.info("✅ Service stopped")
    
    def _restore_signal_handlers(self) -> None:
        """Restaurar handlers de señales originales"""
        try:
            for signum, handler in self._original_handlers.items():
                try:
                    signal.signal(signum, handler)
                except (ValueError, OSError) as e:
                    logger.debug(f"Could not restore signal handler for {signum}: {e}")
        except Exception as e:
            logger.warning(f"Error restoring signal handlers: {e}")
    
    def install_as_service(self) -> None:
        """
        Instalar como servicio del sistema.
        
        Proporciona instrucciones para instalar el servicio en diferentes
        sistemas operativos. La implementación real dependerá del sistema.
        
        TODO: Implementar instalación real como servicio:
        - Windows: usar NSSM o pywin32
        - Linux: usar systemd
        - macOS: usar launchd
        """
        logger.info("📦 Service installation not yet implemented")
        logger.info("💡 To run as service:")
        
        if sys.platform == "win32":
            logger.info("   - Windows: Use NSSM or Task Scheduler")
            logger.info("   - Example: nssm install CursorAgent24_7 python.exe <path_to_main.py>")
        elif sys.platform == "linux":
            logger.info("   - Linux: Create systemd service file")
            logger.info("   - Example: /etc/systemd/system/cursor-agent-24-7.service")
        elif sys.platform == "darwin":
            logger.info("   - macOS: Create launchd plist file")
            logger.info("   - Example: ~/Library/LaunchAgents/com.cursor.agent24_7.plist")
        else:
            logger.info("   - Platform-specific service installation required")
