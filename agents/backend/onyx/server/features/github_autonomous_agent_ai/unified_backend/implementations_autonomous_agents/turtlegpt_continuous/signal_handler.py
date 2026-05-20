"""
Signal Handler Module
====================

Gestión centralizada de señales del sistema para detención graceful.
"""

import signal
import logging
from typing import Callable, Optional, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Tipos de señales soportadas."""
    SIGINT = signal.SIGINT
    SIGTERM = signal.SIGTERM


class SignalHandler:
    """
    Gestor centralizado de señales del sistema.
    
    Permite registrar callbacks para diferentes señales y manejar
    la detención graceful del agente.
    """
    
    def __init__(self):
        """Inicializar gestor de señales."""
        self._handlers: Dict[int, Callable] = {}
        self._original_handlers: Dict[int, Callable] = {}
        self._is_setup = False
    
    def register_handler(
        self,
        signal_type: SignalType,
        callback: Callable[[int, Optional], None],
        override_existing: bool = True
    ) -> None:
        """
        Registrar un handler para una señal.
        
        Args:
            signal_type: Tipo de señal
            callback: Función a llamar cuando se reciba la señal
            override_existing: Si sobrescribir handler existente
        """
        sig_num = signal_type.value
        
        if sig_num in self._handlers and not override_existing:
            logger.warning(f"Handler for signal {sig_num} already exists, skipping")
            return
        
        # Guardar handler original si existe
        try:
            original = signal.signal(sig_num, signal.SIG_DFL)
            if original != signal.SIG_DFL and original != signal.SIG_IGN:
                self._original_handlers[sig_num] = original
        except (ValueError, OSError) as e:
            logger.warning(f"Could not get original handler for signal {sig_num}: {e}")
        
        # Crear wrapper que llama al callback
        def signal_wrapper(signum, frame):
            logger.info(f"Received signal {signum} ({signal_type.name})")
            try:
                callback(signum, frame)
            except Exception as e:
                logger.error(f"Error in signal handler: {e}", exc_info=True)
        
        # Registrar handler
        try:
            signal.signal(sig_num, signal_wrapper)
            self._handlers[sig_num] = callback
            logger.debug(f"Registered handler for signal {sig_num} ({signal_type.name})")
        except (ValueError, OSError) as e:
            logger.error(f"Could not register handler for signal {sig_num}: {e}")
    
    def register_stop_handler(
        self,
        stop_callback: Callable[[], None],
        signals: Optional[list] = None
    ) -> None:
        """
        Registrar handler para detención graceful.
        
        Args:
            stop_callback: Función a llamar para detener el agente
            signals: Lista de señales a registrar (default: SIGINT, SIGTERM)
        """
        if signals is None:
            signals = [SignalType.SIGINT, SignalType.SIGTERM]
        
        def handler(signum, frame):
            logger.info(f"Stop signal received ({signum}), calling stop callback")
            try:
                stop_callback()
            except Exception as e:
                logger.error(f"Error in stop callback: {e}", exc_info=True)
        
        for sig_type in signals:
            self.register_handler(sig_type, handler)
    
    def restore_original_handlers(self) -> None:
        """Restaurar handlers originales del sistema."""
        for sig_num, original_handler in self._original_handlers.items():
            try:
                signal.signal(sig_num, original_handler)
                logger.debug(f"Restored original handler for signal {sig_num}")
            except (ValueError, OSError) as e:
                logger.warning(f"Could not restore handler for signal {sig_num}: {e}")
        
        self._handlers.clear()
        self._original_handlers.clear()
    
    def unregister_handler(self, signal_type: SignalType) -> None:
        """
        Desregistrar handler para una señal.
        
        Args:
            signal_type: Tipo de señal
        """
        sig_num = signal_type.value
        
        if sig_num in self._handlers:
            # Restaurar handler original si existe
            if sig_num in self._original_handlers:
                try:
                    signal.signal(sig_num, self._original_handlers[sig_num])
                    logger.debug(f"Restored original handler for signal {sig_num}")
                except (ValueError, OSError) as e:
                    logger.warning(f"Could not restore handler for signal {sig_num}: {e}")
            else:
                # Usar handler por defecto
                try:
                    signal.signal(sig_num, signal.SIG_DFL)
                except (ValueError, OSError) as e:
                    logger.warning(f"Could not reset handler for signal {sig_num}: {e}")
            
            del self._handlers[sig_num]
            if sig_num in self._original_handlers:
                del self._original_handlers[sig_num]
            
            logger.debug(f"Unregistered handler for signal {sig_num}")
    
    def cleanup(self) -> None:
        """Limpiar todos los handlers registrados."""
        self.restore_original_handlers()
        logger.debug("Signal handlers cleaned up")
    
    def get_registered_signals(self) -> list:
        """
        Obtener lista de señales con handlers registrados.
        
        Returns:
            Lista de números de señal
        """
        return list(self._handlers.keys())


def create_signal_handler() -> SignalHandler:
    """
    Factory function para crear SignalHandler.
    
    Returns:
        Instancia de SignalHandler
    """
    return SignalHandler()


