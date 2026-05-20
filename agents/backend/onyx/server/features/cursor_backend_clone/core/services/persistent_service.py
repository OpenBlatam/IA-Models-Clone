"""
Persistent Service - Servicio persistente 24/7
==============================================

Servicio que puede correr en background incluso cuando la computadora está "apagada"
(usando servicios del sistema o cloud).
"""

import asyncio
import logging
import signal
import sys
from typing import Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class PersistentService:
    """Servicio persistente que puede correr 24/7"""
    
    def __init__(self, agent, service_name: str = "cursor_agent_24_7"):
        self.agent = agent
        self.service_name = service_name
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # Configurar handlers de señales
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Manejar señales del sistema"""
        logger.info(f"📶 Received signal {signum}, shutting down...")
        self._shutdown_event.set()
        
    async def run(self) -> None:
        """Ejecutar servicio persistente"""
        logger.info(f"🚀 Starting persistent service: {self.service_name}")
        self.running = True
        
        try:
            # Iniciar agente
            await self.agent.start()
            
            # Mantener servicio corriendo
            await self._shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"❌ Service error: {e}", exc_info=True)
        finally:
            await self.shutdown()
            
    async def shutdown(self) -> None:
        """Cerrar servicio"""
        if not self.running:
            return
            
        logger.info("🛑 Shutting down service...")
        self.running = False
        
        # Detener agente
        await self.agent.stop()
        
        logger.info("✅ Service stopped")
        
    def install_as_service(self) -> None:
        """Instalar como servicio del sistema"""
        # TODO: Implementar instalación como servicio
        # - Windows: usar NSSM o pywin32
        # - Linux: usar systemd
        # - macOS: usar launchd
        
        logger.info("📦 Service installation not yet implemented")
        logger.info("💡 To run as service:")
        logger.info("   - Windows: Use NSSM or Task Scheduler")
        logger.info("   - Linux: Create systemd service file")
        logger.info("   - macOS: Create launchd plist file")


