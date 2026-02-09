"""
Persistent Service
==================

Servicio persistente que mantiene el agente ejecutándose.
"""

import asyncio
import logging
from typing import Optional

from .agent import GitHubAutonomousAgent, AgentStatus

logger = logging.getLogger(__name__)


class PersistentService:
    """Servicio persistente para el agente."""
    
    def __init__(self):
        self.agent: Optional[GitHubAutonomousAgent] = None
        self._running = False
        
    async def start(self) -> None:
        """Iniciar el servicio."""
        if self._running:
            logger.warning("El servicio ya está en ejecución")
            return
            
        logger.info("🚀 Iniciando servicio persistente...")
        
        self.agent = GitHubAutonomousAgent()
        await self.agent.start()
        
        self._running = True
        logger.info("✅ Servicio iniciado correctamente")
        
    async def stop(self) -> None:
        """Detener el servicio. Solo se llama cuando el usuario presiona el botón de parar."""
        if not self._running:
            logger.info("El servicio ya está detenido")
            return
            
        logger.info("⏹️  Deteniendo servicio persistente (solicitado por el usuario)...")
        
        # Marcar como detenido PRIMERO para evitar reinicios
        self._running = False
        
        if self.agent:
            await self.agent.stop()
            
        logger.info("✅ Servicio detenido correctamente por el usuario")
        
    async def run_forever(self) -> None:
        """Ejecutar el servicio indefinidamente. Solo se detiene cuando se llama explícitamente a stop()."""
        logger.info("🔄 Servicio ejecutándose indefinidamente...")
        logger.info("⚠️  El servicio NO se detendrá automáticamente. Solo se detendrá cuando presiones el botón de parar.")
        
        try:
            # El servicio continúa ejecutándose indefinidamente
            # Solo se detiene cuando self._running es False (seteado por stop())
            while self._running:
                # Verificar que el agente sigue corriendo
                if self.agent and self.agent.status == AgentStatus.STOPPED:
                    # Si el agente fue detenido pero el servicio sigue corriendo, reiniciarlo
                    if self._running:  # Solo si el servicio no ha sido detenido explícitamente
                        logger.warning("Agente detenido inesperadamente, reiniciando...")
                        try:
                            await self.agent.start()
                        except Exception as e:
                            logger.error(f"Error al reiniciar agente: {e}", exc_info=True)
                            await asyncio.sleep(5)  # Esperar antes de reintentar
                
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            # KeyboardInterrupt solo se maneja si el usuario realmente quiere detener
            logger.info("⚠️  Interrupción recibida. El servicio continuará ejecutándose.")
            logger.info("⚠️  Para detener el servicio, usa el botón de parar en la interfaz web.")
            # NO detener automáticamente - solo loguear
            # await self.stop()  # COMENTADO: No detener automáticamente
        except Exception as e:
            # En caso de error, loguear pero continuar
            logger.error(f"Error en el servicio: {e}", exc_info=True)
            logger.warning("El servicio continuará ejecutándose a pesar del error")
            await asyncio.sleep(5)  # Esperar antes de continuar
            
    def is_running(self) -> bool:
        """Verificar si el servicio está en ejecución."""
        return self._running




