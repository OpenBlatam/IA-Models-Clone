"""
Cloud Services Integration - Integración con servicios cloud
============================================================

Integración con servicios cloud gestionados:
- AWS DynamoDB
- Azure Cosmos DB
- GCP Firestore
- Redis Cloud
"""

import logging
from typing import Optional, Dict, Any, List, Protocol
from abc import ABC, abstractmethod

from .types import DatabaseKey, DatabaseValue, DatabaseQuery, JSONDict

logger = logging.getLogger(__name__)


class CloudDatabase(ABC):
    """Interfaz para bases de datos cloud"""
    
    @abstractmethod
    async def get(self, key: DatabaseKey) -> Optional[DatabaseValue]:
        """Obtiene un item"""
        pass
    
    @abstractmethod
    async def put(self, key: DatabaseKey, value: DatabaseValue) -> bool:
        """Guarda un item"""
        pass
    
    @abstractmethod
    async def delete(self, key: DatabaseKey) -> bool:
        """Elimina un item"""
        pass
    
    @abstractmethod
    async def query(
        self,
        index: str,
        condition: DatabaseQuery,
        limit: Optional[int] = None
    ) -> List[DatabaseValue]:
        """Consulta items"""
        pass


class DynamoDBClient(CloudDatabase):
    """Cliente para AWS DynamoDB"""
    
    def __init__(
        self,
        table_name: str,
        region: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None
    ) -> None:
        self.table_name = table_name
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self._client: Optional[Any] = None
    
    def _get_client(self) -> Any:
        """Obtiene cliente de DynamoDB"""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "dynamodb",
                    region_name=self.region,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key
                )
            except ImportError:
                logger.error("boto3 not available. Install with: pip install boto3")
                raise
        return self._client
    
    async def get(self, key: DatabaseKey) -> Optional[DatabaseValue]:
        """Obtiene un item de DynamoDB"""
        try:
            client = self._get_client()
            response = client.get_item(
                TableName=self.table_name,
                Key={"id": {"S": key}}
            )
            
            if "Item" in response:
                return self._unmarshal_item(response["Item"])
            return None
            
        except Exception as e:
            logger.error(f"DynamoDB get error: {e}")
            return None
    
    async def put(self, key: DatabaseKey, value: DatabaseValue) -> bool:
        """Guarda un item en DynamoDB"""
        try:
            client = self._get_client()
            key_str = str(key)
            item = {"id": {"S": key_str}, **self._marshal_item(value)}
            
            client.put_item(
                TableName=self.table_name,
                Item=item
            )
            return True
            
        except Exception as e:
            logger.error(f"DynamoDB put error: {e}")
            return False
    
    async def delete(self, key: DatabaseKey) -> bool:
        """Elimina un item de DynamoDB"""
        try:
            client = self._get_client()
            key_str = str(key)
            client.delete_item(
                TableName=self.table_name,
                Key={"id": {"S": key_str}}
            )
            return True
            
        except Exception as e:
            logger.error(f"DynamoDB delete error: {e}")
            return False
    
    async def query(
        self,
        index: str,
        condition: DatabaseQuery,
        limit: Optional[int] = None
    ) -> List[DatabaseValue]:
        """Consulta items de DynamoDB"""
        try:
            client = self._get_client()
            query_params = {
                "TableName": self.table_name,
                "IndexName": index,
                "KeyConditionExpression": condition.get("expression"),
                "ExpressionAttributeValues": condition.get("values", {})
            }
            
            if limit:
                query_params["Limit"] = limit
            
            response = client.query(**query_params)
            return [self._unmarshal_item(item) for item in response.get("Items", [])]
            
        except Exception as e:
            logger.error(f"DynamoDB query error: {e}")
            return []
    
    def _marshal_item(self, item: DatabaseValue) -> Dict[str, Any]:
        """Convierte item a formato DynamoDB"""
        # Implementación simplificada
        marshalled = {}
        for key, value in item.items():
            if isinstance(value, str):
                marshalled[key] = {"S": value}
            elif isinstance(value, int):
                marshalled[key] = {"N": str(value)}
            elif isinstance(value, bool):
                marshalled[key] = {"BOOL": value}
        return marshalled
    
    def _unmarshal_item(self, item: Dict[str, Any]) -> DatabaseValue:
        """Convierte item de formato DynamoDB"""
        # Implementación simplificada
        unmarshalled = {}
        for key, value in item.items():
            if "S" in value:
                unmarshalled[key] = value["S"]
            elif "N" in value:
                unmarshalled[key] = int(value["N"])
            elif "BOOL" in value:
                unmarshalled[key] = value["BOOL"]
        return unmarshalled


class CosmosDBClient(CloudDatabase):
    """Cliente para Azure Cosmos DB"""
    
    def __init__(
        self,
        endpoint: str,
        key: str,
        database_name: str,
        container_name: str
    ) -> None:
        self.endpoint = endpoint
        self.key = key
        self.database_name = database_name
        self.container_name = container_name
        self._client: Optional[Any] = None
    
    def _get_client(self) -> Any:
        """Obtiene cliente de Cosmos DB"""
        if self._client is None:
            try:
                from azure.cosmos import CosmosClient
                self._client = CosmosClient(self.endpoint, self.key)
            except ImportError:
                logger.error("azure-cosmos not available. Install with: pip install azure-cosmos")
                raise
        return self._client
    
    async def get(self, key: DatabaseKey) -> Optional[DatabaseValue]:
        """Obtiene un item de Cosmos DB"""
        try:
            client = self._get_client()
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            
            item = container.read_item(item=key, partition_key=key)
            return item
            
        except Exception as e:
            logger.error(f"Cosmos DB get error: {e}")
            return None
    
    async def put(self, key: DatabaseKey, value: DatabaseValue) -> bool:
        """Guarda un item en Cosmos DB"""
        try:
            client = self._get_client()
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            
            key_str = str(key)
            item: DatabaseValue = {"id": key_str, **value}
            container.upsert_item(item)
            return True
            
        except Exception as e:
            logger.error(f"Cosmos DB put error: {e}")
            return False
    
    async def delete(self, key: DatabaseKey) -> bool:
        """Elimina un item de Cosmos DB"""
        try:
            client = self._get_client()
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            
            key_str = str(key)
            container.delete_item(item=key_str, partition_key=key_str)
            return True
            
        except Exception as e:
            logger.error(f"Cosmos DB delete error: {e}")
            return False
    
    async def query(
        self,
        index: str,
        condition: DatabaseQuery,
        limit: Optional[int] = None
    ) -> List[DatabaseValue]:
        """Consulta items de Cosmos DB"""
        try:
            client = self._get_client()
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            
            query = f"SELECT * FROM c WHERE {condition.get('expression', '1=1')}"
            items = container.query_items(
                query=query,
                enable_cross_partition_query=True
            )
            
            results = []
            for item in items:
                results.append(item)
                if limit and len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Cosmos DB query error: {e}")
            return []


def get_cloud_database(
    provider: str,
    **kwargs: Any
) -> Optional[CloudDatabase]:
    """
    Obtiene cliente de base de datos cloud.
    
    Args:
        provider: Proveedor (dynamodb, cosmosdb)
        **kwargs: Configuración específica del proveedor
    
    Returns:
        Cliente de base de datos cloud
    """
    if provider.lower() == "dynamodb":
        return DynamoDBClient(**kwargs)
    elif provider.lower() == "cosmosdb":
        return CosmosDBClient(**kwargs)
    else:
        logger.warning(f"Unknown cloud database provider: {provider}")
        return None

