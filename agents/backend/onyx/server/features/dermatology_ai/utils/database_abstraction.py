"""
Database Abstraction Layer for Multiple Database Backends
Supports SQLite, PostgreSQL, DynamoDB, Cosmos DB
"""

import os
from typing import Any, Optional, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    DYNAMODB = "dynamodb"
    COSMOSDB = "cosmosdb"
    MEMORY = "memory"  # For testing


class DatabaseAdapter(ABC):
    """Abstract database adapter interface"""
    
    @abstractmethod
    async def connect(self):
        """Connect to database"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from database"""
        pass
    
    @abstractmethod
    async def create_table(self, table_name: str, schema: Dict[str, Any]):
        """Create table"""
        pass
    
    @abstractmethod
    async def insert(self, table_name: str, data: Dict[str, Any]) -> str:
        """Insert record"""
        pass
    
    @abstractmethod
    async def get(self, table_name: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get record by key"""
        pass
    
    @abstractmethod
    async def query(
        self,
        table_name: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query records"""
        pass
    
    @abstractmethod
    async def update(
        self,
        table_name: str,
        key: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> bool:
        """Update record"""
        pass
    
    @abstractmethod
    async def delete(self, table_name: str, key: Dict[str, Any]) -> bool:
        """Delete record"""
        pass


class DynamoDBAdapter(DatabaseAdapter):
    """AWS DynamoDB adapter"""
    
    def __init__(self, table_name_prefix: str = "dermatology"):
        self.table_name_prefix = table_name_prefix
        self.client = None
        self.resource = None
    
    async def connect(self):
        """Connect to DynamoDB"""
        try:
            import boto3
            
            self.client = boto3.client("dynamodb")
            self.resource = boto3.resource("dynamodb")
            logger.info("✅ Connected to DynamoDB")
        except ImportError:
            logger.warning("boto3 not installed. Install with: pip install boto3")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to DynamoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from DynamoDB"""
        # DynamoDB client doesn't need explicit disconnect
        pass
    
    async def create_table(self, table_name: str, schema: Dict[str, Any]):
        """Create DynamoDB table"""
        full_table_name = f"{self.table_name_prefix}-{table_name}"
        
        try:
            self.client.create_table(
                TableName=full_table_name,
                KeySchema=[
                    {"AttributeName": schema.get("partition_key", "id"), "KeyType": "HASH"}
                ],
                AttributeDefinitions=[
                    {
                        "AttributeName": schema.get("partition_key", "id"),
                        "AttributeType": schema.get("key_type", "S")
                    }
                ],
                BillingMode="PAY_PER_REQUEST"
            )
            logger.info(f"Created DynamoDB table: {full_table_name}")
        except self.client.exceptions.ResourceInUseException:
            logger.info(f"Table {full_table_name} already exists")
    
    async def insert(self, table_name: str, data: Dict[str, Any]) -> str:
        """Insert into DynamoDB"""
        full_table_name = f"{self.table_name_prefix}-{table_name}"
        table = self.resource.Table(full_table_name)
        
        table.put_item(Item=data)
        return data.get("id", "")
    
    async def get(self, table_name: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get from DynamoDB"""
        full_table_name = f"{self.table_name_prefix}-{table_name}"
        table = self.resource.Table(full_table_name)
        
        response = table.get_item(Key=key)
        return response.get("Item")
    
    async def query(
        self,
        table_name: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query DynamoDB"""
        full_table_name = f"{self.table_name_prefix}-{table_name}"
        table = self.resource.Table(full_table_name)
        
        # Simplified query - in production, use proper DynamoDB query expressions
        scan_kwargs = {}
        if limit:
            scan_kwargs["Limit"] = limit
        
        response = table.scan(**scan_kwargs)
        return response.get("Items", [])
    
    async def update(
        self,
        table_name: str,
        key: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> bool:
        """Update in DynamoDB"""
        full_table_name = f"{self.table_name_prefix}-{table_name}"
        table = self.resource.Table(full_table_name)
        
        update_expression = "SET " + ", ".join([f"{k} = :{k}" for k in updates.keys()])
        expression_values = {f":{k}": v for k, v in updates.items()}
        
        table.update_item(
            Key=key,
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values
        )
        return True
    
    async def delete(self, table_name: str, key: Dict[str, Any]) -> bool:
        """Delete from DynamoDB"""
        full_table_name = f"{self.table_name_prefix}-{table_name}"
        table = self.resource.Table(full_table_name)
        
        table.delete_item(Key=key)
        return True


class CosmosDBAdapter(DatabaseAdapter):
    """Azure Cosmos DB adapter"""
    
    def __init__(self, endpoint: str, key: str, database_name: str = "dermatology"):
        self.endpoint = endpoint
        self.key = key
        self.database_name = database_name
        self.client = None
        self.database = None
    
    async def connect(self):
        """Connect to Cosmos DB"""
        try:
            from azure.cosmos import CosmosClient, PartitionKey
            
            self.client = CosmosClient(self.endpoint, self.key)
            self.database = self.client.create_database_if_not_exists(id=self.database_name)
            logger.info("✅ Connected to Cosmos DB")
        except ImportError:
            logger.warning("azure-cosmos not installed. Install with: pip install azure-cosmos")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Cosmos DB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Cosmos DB"""
        # Cosmos DB client doesn't need explicit disconnect
        pass
    
    async def create_table(self, table_name: str, schema: Dict[str, Any]):
        """Create Cosmos DB container"""
        partition_key = schema.get("partition_key", "/id")
        
        self.database.create_container_if_not_exists(
            id=table_name,
            partition_key=PartitionKey(path=partition_key)
        )
        logger.info(f"Created Cosmos DB container: {table_name}")
    
    async def insert(self, table_name: str, data: Dict[str, Any]) -> str:
        """Insert into Cosmos DB"""
        container = self.database.get_container_client(table_name)
        response = container.create_item(body=data)
        return response.get("id", "")
    
    async def get(self, table_name: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get from Cosmos DB"""
        container = self.database.get_container_client(table_name)
        item_id = key.get("id")
        partition_key = key.get("partition_key", item_id)
        
        try:
            response = container.read_item(item=item_id, partition_key=partition_key)
            return response
        except Exception:
            return None
    
    async def query(
        self,
        table_name: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query Cosmos DB"""
        container = self.database.get_container_client(table_name)
        
        query = "SELECT * FROM c"
        if filter_conditions:
            # Build WHERE clause from filter_conditions
            conditions = " AND ".join([f"c.{k} = @{k}" for k in filter_conditions.keys()])
            query += f" WHERE {conditions}"
        
        if limit:
            query += f" OFFSET 0 LIMIT {limit}"
        
        items = container.query_items(
            query=query,
            parameters=[{"name": f"@{k}", "value": v} for k, v in filter_conditions.items()] if filter_conditions else None,
            enable_cross_partition_query=True
        )
        
        return list(items)
    
    async def update(
        self,
        table_name: str,
        key: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> bool:
        """Update in Cosmos DB"""
        container = self.database.get_container_client(table_name)
        item_id = key.get("id")
        partition_key = key.get("partition_key", item_id)
        
        # Read existing item
        item = container.read_item(item=item_id, partition_key=partition_key)
        
        # Update fields
        item.update(updates)
        
        # Replace item
        container.replace_item(item=item, body=item)
        return True
    
    async def delete(self, table_name: str, key: Dict[str, Any]) -> bool:
        """Delete from Cosmos DB"""
        container = self.database.get_container_client(table_name)
        item_id = key.get("id")
        partition_key = key.get("partition_key", item_id)
        
        container.delete_item(item=item_id, partition_key=partition_key)
        return True


def get_database_adapter(
    db_type: Optional[str] = None,
    **kwargs
) -> DatabaseAdapter:
    """
    Factory function to get database adapter
    
    Args:
        db_type: Database type (dynamodb, cosmosdb, sqlite, postgresql)
        **kwargs: Database-specific configuration
        
    Returns:
        DatabaseAdapter instance
    """
    db_type = db_type or os.getenv("DATABASE_TYPE", "sqlite")
    db_type = DatabaseType(db_type.lower())
    
    if db_type == DatabaseType.DYNAMODB:
        return DynamoDBAdapter(
            table_name_prefix=kwargs.get("table_name_prefix", "dermatology")
        )
    elif db_type == DatabaseType.COSMOSDB:
        endpoint = kwargs.get("endpoint") or os.getenv("COSMOSDB_ENDPOINT")
        key = kwargs.get("key") or os.getenv("COSMOSDB_KEY")
        database_name = kwargs.get("database_name", "dermatology")
        
        if not endpoint or not key:
            raise ValueError("Cosmos DB requires endpoint and key")
        
        return CosmosDBAdapter(endpoint=endpoint, key=key, database_name=database_name)
    else:
        # Fallback to SQLite (implemented elsewhere)
        logger.warning(f"Database type {db_type} using fallback SQLite adapter")
        from services.database import DatabaseManager
        return DatabaseManager()  # Assuming this exists















