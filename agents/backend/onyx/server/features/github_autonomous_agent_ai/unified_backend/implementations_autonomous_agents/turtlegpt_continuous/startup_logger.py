"""
Startup Logger Module
====================

Logging estructurado para el inicio del agente.
Proporciona mensajes de inicio consistentes y bien formateados.
"""

import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class StartupLogger:
    """
    Logger especializado para el inicio del agente.
    
    Proporciona mensajes de inicio consistentes y bien formateados
    con información relevante sobre la configuración y estado del agente.
    """
    
    def __init__(self, agent_name: str, separator: str = "=", width: int = 60):
        """
        Inicializar logger de inicio.
        
        Args:
            agent_name: Nombre del agente
            separator: Carácter separador
            width: Ancho del banner
        """
        self.agent_name = agent_name
        self.separator = separator
        self.width = width
    
    def log_startup_banner(self, papers: Optional[List[str]] = None) -> None:
        """
        Loggear banner de inicio.
        
        Args:
            papers: Lista de papers integrados
        """
        banner = self.separator * self.width
        logger.info(banner)
        logger.info(f"Starting {self.agent_name}")
        logger.info(banner)
        
        if papers:
            papers_str = ", ".join(papers)
            logger.info(f"Papers integrated: {papers_str}")
        
        logger.info("Agent will run 24/7 until stopped")
        logger.info("Press Ctrl+C to stop")
        logger.info(banner)
    
    def log_configuration(
        self,
        config: Dict[str, Any],
        include_defaults: bool = False
    ) -> None:
        """
        Loggear configuración del agente.
        
        Args:
            config: Diccionario de configuración
            include_defaults: Si incluir valores por defecto
        """
        logger.info("Agent Configuration:")
        for key, value in config.items():
            if include_defaults or value is not None:
                logger.info(f"  {key}: {value}")
    
    def log_components_initialized(
        self,
        components: List[str]
    ) -> None:
        """
        Loggear componentes inicializados.
        
        Args:
            components: Lista de nombres de componentes
        """
        logger.info(f"Initialized {len(components)} components:")
        for component in components:
            logger.info(f"  ✓ {component}")
    
    def log_strategies_enabled(
        self,
        strategies: Dict[str, bool]
    ) -> None:
        """
        Loggear estrategias habilitadas.
        
        Args:
            strategies: Diccionario de estrategias y su estado
        """
        enabled = [name for name, enabled in strategies.items() if enabled]
        disabled = [name for name, enabled in strategies.items() if not enabled]
        
        if enabled:
            logger.info(f"Enabled strategies ({len(enabled)}): {', '.join(enabled)}")
        
        if disabled:
            logger.debug(f"Disabled strategies ({len(disabled)}): {', '.join(disabled)}")
    
    def log_ready(self) -> None:
        """Loggear que el agente está listo."""
        logger.info(f"{self.agent_name} is ready and running")
    
    def log_stopping(self) -> None:
        """Loggear que el agente se está deteniendo."""
        logger.info(f"Stopping {self.agent_name}...")
    
    def log_stopped(self) -> None:
        """Loggear que el agente se detuvo."""
        logger.info(f"{self.agent_name} stopped")
    
    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """
        Loggear error durante el inicio.
        
        Args:
            error: Excepción ocurrida
            context: Contexto adicional
        """
        context_msg = f" ({context})" if context else ""
        logger.error(f"Error during startup{context_msg}: {error}", exc_info=True)
    
    def log_warning(self, message: str, context: Optional[str] = None) -> None:
        """
        Loggear advertencia durante el inicio.
        
        Args:
            message: Mensaje de advertencia
            context: Contexto adicional
        """
        context_msg = f" ({context})" if context else ""
        logger.warning(f"{message}{context_msg}")


def create_startup_logger(
    agent_name: str,
    separator: str = "=",
    width: int = 60
) -> StartupLogger:
    """
    Factory function para crear StartupLogger.
    
    Args:
        agent_name: Nombre del agente
        separator: Carácter separador
        width: Ancho del banner
        
    Returns:
        Instancia de StartupLogger
    """
    return StartupLogger(agent_name, separator, width)


