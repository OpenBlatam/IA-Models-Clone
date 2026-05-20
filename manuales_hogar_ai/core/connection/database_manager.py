"""
Database Manager
================

Gestor especializado para conexiones de base de datos.
"""

import logging
import asyncio
from typing import Optional
from sqlalchemy import text

from ...core.base.service_base import BaseService
from ...database.session import _engine, close_db, init_db

logger = logging.getLogger(__name__)


class DatabaseManager(BaseService):
    """Gestor de conexiones de base de datos."""
    
    def __init__(self):
        """Inicializar gestor."""
        super().__init__(logger_name=__name__)
        self.engine = None
    
    async def initialize(self):
        """Inicializar conexión de base de datos."""
        try:
            init_db()
            self.engine = _engine
            self.log_info("Database connection initialized")
        except Exception as e:
            self.log_error(f"Failed to initialize database: {e}")
            raise
    
    async def health_check(self) -> bool:
        """
        Verificar salud de la conexión.
        
        Returns:
            True si está saludable
        """
        if not self.engine:
            return False
        
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            self.log_debug("Database health check: OK")
            return True
        except Exception as e:
            self.log_warning(f"Database health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Limpiar conexión."""
        try:
            await close_db()
            self.log_info("Database connection closed")
        except Exception as e:
            self.log_error(f"Error closing database: {e}")

