"""
Database Checker
================

Checker especializado para base de datos.
"""

from typing import Dict, Any
from ...core.base.service_base import BaseService
from ...database.session import get_database_url


class DatabaseChecker(BaseService):
    """Checker de base de datos."""
    
    def __init__(self):
        """Inicializar checker."""
        super().__init__(logger_name=__name__)
    
    async def check(self) -> Dict[str, Any]:
        """
        Verificar conectividad de base de datos.
        
        Returns:
            Diccionario con status y message
        """
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text
            
            database_url = get_database_url()
            engine = create_async_engine(
                database_url,
                pool_pre_ping=True,
                connect_args={"command_timeout": 5},
            )
            
            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()
            
            await engine.dispose()
            
            return {
                "status": "healthy",
                "message": "Database is accessible",
            }
        except Exception as e:
            self.log_error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Database error: {str(e)}",
            }

