"""
Database Per Service Pattern
=============================

Each microservice has its own database:
- Movement Service -> Movement DB
- Trajectory Service -> Trajectory DB
- Chat Service -> Chat DB
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import os

logger = logging.getLogger(__name__)


class DatabaseAdapter(ABC):
    """Abstract database adapter."""
    
    @abstractmethod
    async def connect(self):
        """Connect to database."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from database."""
        pass
    
    @abstractmethod
    async def query(self, query: str, params: Dict[str, Any] = None):
        """Execute query."""
        pass


class DynamoDBAdapter(DatabaseAdapter):
    """DynamoDB adapter for serverless."""
    
    def __init__(self, table_name: str, region: str = "us-east-1"):
        self.table_name = table_name
        self.region = region
        self._client = None
    
    async def connect(self):
        """Connect to DynamoDB."""
        try:
            import boto3
            self._client = boto3.client(
                "dynamodb",
                region_name=self.region
            )
            logger.info(f"Connected to DynamoDB table: {self.table_name}")
        except ImportError:
            logger.warning("boto3 not installed, DynamoDB disabled")
        except Exception as e:
            logger.error(f"Failed to connect to DynamoDB: {e}")
    
    async def disconnect(self):
        """Disconnect from DynamoDB."""
        self._client = None
    
    async def query(self, query: str, params: Dict[str, Any] = None):
        """Execute DynamoDB query."""
        if not self._client:
            raise Exception("DynamoDB not connected")
        
        # Simplified - in production use proper DynamoDB operations
        return {"items": []}


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL adapter."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._pool = None
    
    async def connect(self):
        """Connect to PostgreSQL."""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.connection_string)
            logger.info("Connected to PostgreSQL")
        except ImportError:
            logger.warning("asyncpg not installed, PostgreSQL disabled")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
    
    async def disconnect(self):
        """Disconnect from PostgreSQL."""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def query(self, query: str, params: Dict[str, Any] = None):
        """Execute PostgreSQL query."""
        if not self._pool:
            raise Exception("PostgreSQL not connected")
        
        async with self._pool.acquire() as connection:
            return await connection.fetch(query, *(params or {}).values())


class ServiceDatabase:
    """Database per service manager."""
    
    def __init__(self, service_name: str, db_type: str = "dynamodb"):
        self.service_name = service_name
        self.db_type = db_type
        self.adapter: Optional[DatabaseAdapter] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection."""
        if self._initialized:
            return
        
        if self.db_type == "dynamodb":
            table_name = os.getenv(f"{self.service_name.upper().replace('-', '_')}_TABLE", f"{self.service_name}-table")
            self.adapter = DynamoDBAdapter(table_name)
        elif self.db_type == "postgresql":
            connection_string = os.getenv(f"{self.service_name.upper().replace('-', '_')}_DATABASE_URL")
            if connection_string:
                self.adapter = PostgreSQLAdapter(connection_string)
            else:
                logger.warning(f"No database URL for {self.service_name}")
                return
        
        if self.adapter:
            await self.adapter.connect()
            self._initialized = True
            logger.info(f"Database initialized for {self.service_name}")
    
    async def query(self, query: str, params: Dict[str, Any] = None):
        """Execute query."""
        if not self._initialized:
            await self.initialize()
        
        if not self.adapter:
            return None
        
        return await self.adapter.query(query, params)
    
    async def close(self):
        """Close database connection."""
        if self.adapter:
            await self.adapter.disconnect()
            self._initialized = False


# Service-specific databases
def get_movement_database() -> ServiceDatabase:
    """Get movement service database."""
    return ServiceDatabase("movement-service", db_type="dynamodb")


def get_trajectory_database() -> ServiceDatabase:
    """Get trajectory service database."""
    return ServiceDatabase("trajectory-service", db_type="dynamodb")


def get_chat_database() -> ServiceDatabase:
    """Get chat service database."""
    return ServiceDatabase("chat-service", db_type="dynamodb")















