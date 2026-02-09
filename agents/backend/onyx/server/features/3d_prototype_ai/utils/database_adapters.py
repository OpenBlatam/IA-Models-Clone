"""
Database Adapters - Adaptadores para bases de datos cloud-native
================================================================

Soporta:
- AWS DynamoDB
- Azure Cosmos DB
- MongoDB (fallback)
"""

import logging
from typing import Dict, Optional, List, Any
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseAdapter(ABC):
    """Interfaz base para adaptadores de base de datos"""
    
    @abstractmethod
    async def get(self, key: str, table: str) -> Optional[Dict]:
        """Obtiene un item por clave"""
        pass
    
    @abstractmethod
    async def put(self, key: str, data: Dict, table: str) -> bool:
        """Guarda un item"""
        pass
    
    @abstractmethod
    async def query(self, table: str, **filters) -> List[Dict]:
        """Consulta items"""
        pass
    
    @abstractmethod
    async def delete(self, key: str, table: str) -> bool:
        """Elimina un item"""
        pass


class DynamoDBAdapter(DatabaseAdapter):
    """Adaptador para AWS DynamoDB"""
    
    def __init__(self, region: str = "us-east-1", table_prefix: str = "3d_prototype_"):
        self.region = region
        self.table_prefix = table_prefix
        self.client = None
        self._setup()
    
    def _setup(self):
        """Configura cliente de DynamoDB"""
        try:
            import boto3
            self.client = boto3.client('dynamodb', region_name=self.region)
            logger.info("DynamoDB client configured")
        except ImportError:
            logger.warning("boto3 not available. Install with: pip install boto3")
        except Exception as e:
            logger.error(f"Failed to setup DynamoDB: {e}")
    
    async def get(self, key: str, table: str) -> Optional[Dict]:
        """Obtiene un item de DynamoDB"""
        if not self.client:
            return None
        
        try:
            response = self.client.get_item(
                TableName=f"{self.table_prefix}{table}",
                Key={"id": {"S": key}}
            )
            
            if "Item" in response:
                return self._unmarshal_item(response["Item"])
            return None
        except Exception as e:
            logger.error(f"DynamoDB get error: {e}")
            return None
    
    async def put(self, key: str, data: Dict, table: str) -> bool:
        """Guarda un item en DynamoDB"""
        if not self.client:
            return False
        
        try:
            item = {"id": {"S": key}, **self._marshal_item(data)}
            item["updated_at"] = {"S": datetime.utcnow().isoformat()}
            
            self.client.put_item(
                TableName=f"{self.table_prefix}{table}",
                Item=item
            )
            return True
        except Exception as e:
            logger.error(f"DynamoDB put error: {e}")
            return False
    
    async def query(self, table: str, **filters) -> List[Dict]:
        """Consulta items en DynamoDB"""
        if not self.client:
            return []
        
        try:
            # Construir expresión de filtro
            filter_expression = None
            expression_attribute_values = {}
            
            if filters:
                conditions = []
                for i, (key, value) in enumerate(filters.items()):
                    attr_name = f"#{key}"
                    attr_value = f":val{i}"
                    conditions.append(f"{attr_name} = {attr_value}")
                    expression_attribute_values[attr_value] = self._marshal_value(value)
                
                filter_expression = " AND ".join(conditions)
            
            response = self.client.scan(
                TableName=f"{self.table_prefix}{table}",
                FilterExpression=filter_expression,
                ExpressionAttributeValues=expression_attribute_values if expression_attribute_values else None
            )
            
            return [self._unmarshal_item(item) for item in response.get("Items", [])]
        except Exception as e:
            logger.error(f"DynamoDB query error: {e}")
            return []
    
    async def delete(self, key: str, table: str) -> bool:
        """Elimina un item de DynamoDB"""
        if not self.client:
            return False
        
        try:
            self.client.delete_item(
                TableName=f"{self.table_prefix}{table}",
                Key={"id": {"S": key}}
            )
            return True
        except Exception as e:
            logger.error(f"DynamoDB delete error: {e}")
            return False
    
    def _marshal_item(self, item: Dict) -> Dict:
        """Convierte item Python a formato DynamoDB"""
        result = {}
        for key, value in item.items():
            result[key] = self._marshal_value(value)
        return result
    
    def _marshal_value(self, value: Any) -> Dict:
        """Convierte valor a formato DynamoDB"""
        if isinstance(value, str):
            return {"S": value}
        elif isinstance(value, (int, float)):
            return {"N": str(value)}
        elif isinstance(value, bool):
            return {"BOOL": value}
        elif isinstance(value, list):
            return {"L": [self._marshal_value(v) for v in value]}
        elif isinstance(value, dict):
            return {"M": self._marshal_item(value)}
        else:
            return {"S": str(value)}
    
    def _unmarshal_item(self, item: Dict) -> Dict:
        """Convierte item DynamoDB a formato Python"""
        result = {}
        for key, value in item.items():
            result[key] = self._unmarshal_value(value)
        return result
    
    def _unmarshal_value(self, value: Dict) -> Any:
        """Convierte valor DynamoDB a Python"""
        if "S" in value:
            return value["S"]
        elif "N" in value:
            return float(value["N"])
        elif "BOOL" in value:
            return value["BOOL"]
        elif "L" in value:
            return [self._unmarshal_value(v) for v in value["L"]]
        elif "M" in value:
            return self._unmarshal_item(value["M"])
        else:
            return value


class CosmosDBAdapter(DatabaseAdapter):
    """Adaptador para Azure Cosmos DB"""
    
    def __init__(self, endpoint: str, key: str, database: str = "3d_prototype"):
        self.endpoint = endpoint
        self.key = key
        self.database = database
        self.client = None
        self._setup()
    
    def _setup(self):
        """Configura cliente de Cosmos DB"""
        try:
            from azure.cosmos import CosmosClient, PartitionKey
            
            self.client = CosmosClient(self.endpoint, self.key)
            self.database_client = self.client.get_database_client(self.database)
            logger.info("Cosmos DB client configured")
        except ImportError:
            logger.warning("azure-cosmos not available. Install with: pip install azure-cosmos")
        except Exception as e:
            logger.error(f"Failed to setup Cosmos DB: {e}")
    
    async def get(self, key: str, table: str) -> Optional[Dict]:
        """Obtiene un item de Cosmos DB"""
        if not self.client:
            return None
        
        try:
            container = self.database_client.get_container_client(table)
            item = container.read_item(item=key, partition_key=key)
            return item
        except Exception as e:
            logger.error(f"Cosmos DB get error: {e}")
            return None
    
    async def put(self, key: str, data: Dict, table: str) -> bool:
        """Guarda un item en Cosmos DB"""
        if not self.client:
            return False
        
        try:
            container = self.database_client.get_container_client(table)
            data["id"] = key
            data["updated_at"] = datetime.utcnow().isoformat()
            container.upsert_item(data)
            return True
        except Exception as e:
            logger.error(f"Cosmos DB put error: {e}")
            return False
    
    async def query(self, table: str, **filters) -> List[Dict]:
        """Consulta items en Cosmos DB"""
        if not self.client:
            return []
        
        try:
            container = self.database_client.get_container_client(table)
            
            # Construir query
            query = "SELECT * FROM c WHERE 1=1"
            parameters = []
            
            for i, (key, value) in enumerate(filters.items()):
                query += f" AND c.{key} = @param{i}"
                parameters.append({"name": f"@param{i}", "value": value})
            
            items = container.query_items(
                query=query,
                parameters=parameters
            )
            
            return list(items)
        except Exception as e:
            logger.error(f"Cosmos DB query error: {e}")
            return []
    
    async def delete(self, key: str, table: str) -> bool:
        """Elimina un item de Cosmos DB"""
        if not self.client:
            return False
        
        try:
            container = self.database_client.get_container_client(table)
            container.delete_item(item=key, partition_key=key)
            return True
        except Exception as e:
            logger.error(f"Cosmos DB delete error: {e}")
            return False


class DatabaseManager:
    """Gestor de bases de datos que soporta múltiples adaptadores"""
    
    def __init__(self, adapter_type: str = "dynamodb", **kwargs):
        self.adapter: Optional[DatabaseAdapter] = None
        
        if adapter_type == "dynamodb":
            self.adapter = DynamoDBAdapter(**kwargs)
        elif adapter_type == "cosmosdb":
            self.adapter = CosmosDBAdapter(**kwargs)
        else:
            logger.warning(f"Unknown adapter type: {adapter_type}")
    
    async def get(self, key: str, table: str) -> Optional[Dict]:
        """Obtiene un item"""
        if self.adapter:
            return await self.adapter.get(key, table)
        return None
    
    async def put(self, key: str, data: Dict, table: str) -> bool:
        """Guarda un item"""
        if self.adapter:
            return await self.adapter.put(key, data, table)
        return False
    
    async def query(self, table: str, **filters) -> List[Dict]:
        """Consulta items"""
        if self.adapter:
            return await self.adapter.query(table, **filters)
        return []
    
    async def delete(self, key: str, table: str) -> bool:
        """Elimina un item"""
        if self.adapter:
            return await self.adapter.delete(key, table)
        return False




