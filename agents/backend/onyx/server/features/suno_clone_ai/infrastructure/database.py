"""
Database Infrastructure
Abstracción para diferentes backends de base de datos
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Tipos de base de datos soportados"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    DYNAMODB = "dynamodb"
    MONGODB = "mongodb"


class DatabaseManager(ABC):
    """Interfaz abstracta para gestión de base de datos"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Conecta a la base de datos"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Desconecta de la base de datos"""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Ejecuta una consulta"""
        pass
    
    @abstractmethod
    async def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """Ejecuta una transacción"""
        pass


class SQLDatabaseManager(DatabaseManager):
    """Implementación para bases de datos SQL"""
    
    def __init__(self, database_url: str, db_type: DatabaseType):
        self.database_url = database_url
        self.db_type = db_type
        self._connection = None
    
    async def connect(self) -> bool:
        """Conecta a la base de datos SQL"""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker
            
            if self.db_type == DatabaseType.SQLITE:
                engine = create_async_engine(
                    self.database_url.replace("sqlite://", "sqlite+aiosqlite://"),
                    echo=False
                )
            elif self.db_type == DatabaseType.POSTGRESQL:
                engine = create_async_engine(
                    self.database_url.replace("postgresql://", "postgresql+asyncpg://"),
                    echo=False
                )
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
            
            self._session_factory = sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
            self._engine = engine
            
            logger.info(f"Connected to {self.db_type.value} database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self):
        """Desconecta de la base de datos"""
        if self._engine:
            await self._engine.dispose()
            logger.info("Disconnected from database")
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Ejecuta una consulta SQL"""
        async with self._session_factory() as session:
            result = await session.execute(query, params or {})
            return [dict(row) for row in result]
    
    async def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """Ejecuta una transacción"""
        async with self._session_factory() as session:
            try:
                for op in operations:
                    await session.execute(op['query'], op.get('params', {}))
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Transaction failed: {e}")
                raise


class DynamoDBManager(DatabaseManager):
    """Implementación para DynamoDB"""
    
    def __init__(self, table_name: str, region: str = "us-east-1"):
        self.table_name = table_name
        self.region = region
        self._client = None
    
    async def connect(self) -> bool:
        """Conecta a DynamoDB"""
        try:
            from aws.services.dynamodb import DynamoDBService
            
            self._client = DynamoDBService(
                table_name=self.table_name,
                region_name=self.region
            )
            logger.info(f"Connected to DynamoDB table: {self.table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to DynamoDB: {e}")
            raise
    
    async def disconnect(self):
        """Desconecta de DynamoDB"""
        self._client = None
        logger.info("Disconnected from DynamoDB")
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Ejecuta una consulta en DynamoDB"""
        # DynamoDB usa su propia sintaxis de query
        # Esta es una implementación simplificada
        if params:
            key = params.get('key')
            if key:
                item = self._client.get_item(key)
                return [item] if item else []
        return []
    
    async def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """Ejecuta una transacción en DynamoDB"""
        # DynamoDB transactions
        items = []
        for op in operations:
            items.append(op.get('item', {}))
        
        if items:
            self._client.batch_write(items)
        return True


# Factory function
_database_manager: Optional[DatabaseManager] = None


def get_database() -> Optional[DatabaseManager]:
    """Obtiene la instancia global del gestor de base de datos"""
    return _database_manager


def create_database_manager(
    db_type: DatabaseType,
    connection_string: str,
    **kwargs
) -> DatabaseManager:
    """Crea un gestor de base de datos según el tipo"""
    if db_type in [DatabaseType.SQLITE, DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
        return SQLDatabaseManager(connection_string, db_type)
    elif db_type == DatabaseType.DYNAMODB:
        return DynamoDBManager(
            table_name=kwargs.get('table_name', connection_string),
            region=kwargs.get('region', 'us-east-1')
        )
    else:
        raise ValueError(f"Unsupported database type: {db_type}")















