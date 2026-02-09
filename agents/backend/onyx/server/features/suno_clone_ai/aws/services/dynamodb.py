"""
DynamoDB Service para almacenamiento serverless
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Intentar importar boto3
try:
    import boto3
    from boto3.dynamodb.conditions import Key, Attr
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available. Install with: pip install boto3")


class DynamoDBService:
    """
    Servicio para interactuar con DynamoDB
    Optimizado para operaciones serverless
    """
    
    def __init__(
        self,
        table_name: str,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None  # Para local testing
    ):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for DynamoDBService")
        
        self.table_name = table_name
        self.region_name = region_name
        
        # Crear cliente DynamoDB
        self.dynamodb = boto3.resource(
            'dynamodb',
            region_name=region_name,
            endpoint_url=endpoint_url
        )
        self.table = self.dynamodb.Table(table_name)
        
        # Cliente de bajo nivel para operaciones batch
        self.client = boto3.client(
            'dynamodb',
            region_name=region_name,
            endpoint_url=endpoint_url
        )
    
    def put_item(self, item: Dict[str, Any]) -> bool:
        """
        Inserta o actualiza un item
        
        Args:
            item: Diccionario con los datos del item
            
        Returns:
            True si fue exitoso
        """
        try:
            # Agregar timestamp si no existe
            if 'created_at' not in item:
                item['created_at'] = datetime.utcnow().isoformat()
            item['updated_at'] = datetime.utcnow().isoformat()
            
            self.table.put_item(Item=item)
            return True
        except ClientError as e:
            logger.error(f"DynamoDB put_item error: {e}")
            raise
    
    def get_item(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Obtiene un item por su clave
        
        Args:
            key: Diccionario con las claves primarias
            
        Returns:
            Item o None si no existe
        """
        try:
            response = self.table.get_item(Key=key)
            return response.get('Item')
        except ClientError as e:
            logger.error(f"DynamoDB get_item error: {e}")
            raise
    
    def delete_item(self, key: Dict[str, Any]) -> bool:
        """Elimina un item"""
        try:
            self.table.delete_item(Key=key)
            return True
        except ClientError as e:
            logger.error(f"DynamoDB delete_item error: {e}")
            raise
    
    def query(
        self,
        key_condition_expression,
        filter_expression=None,
        index_name: Optional[str] = None,
        limit: Optional[int] = None,
        exclusive_start_key: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Query items usando key condition
        
        Args:
            key_condition_expression: Expresión de condición de clave
            filter_expression: Expresión de filtro opcional
            index_name: Nombre del índice GSI/LSI
            limit: Límite de resultados
            exclusive_start_key: Clave para paginación
            
        Returns:
            Lista de items
        """
        try:
            kwargs = {
                'KeyConditionExpression': key_condition_expression
            }
            
            if filter_expression:
                kwargs['FilterExpression'] = filter_expression
            if index_name:
                kwargs['IndexName'] = index_name
            if limit:
                kwargs['Limit'] = limit
            if exclusive_start_key:
                kwargs['ExclusiveStartKey'] = exclusive_start_key
            
            response = self.table.query(**kwargs)
            return response.get('Items', [])
        except ClientError as e:
            logger.error(f"DynamoDB query error: {e}")
            raise
    
    def scan(
        self,
        filter_expression=None,
        limit: Optional[int] = None,
        exclusive_start_key: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan de la tabla (usar con precaución en tablas grandes)
        
        Args:
            filter_expression: Expresión de filtro
            limit: Límite de resultados
            exclusive_start_key: Clave para paginación
            
        Returns:
            Lista de items
        """
        try:
            kwargs = {}
            
            if filter_expression:
                kwargs['FilterExpression'] = filter_expression
            if limit:
                kwargs['Limit'] = limit
            if exclusive_start_key:
                kwargs['ExclusiveStartKey'] = exclusive_start_key
            
            response = self.table.scan(**kwargs)
            return response.get('Items', [])
        except ClientError as e:
            logger.error(f"DynamoDB scan error: {e}")
            raise
    
    def batch_write(self, items: List[Dict[str, Any]]) -> bool:
        """
        Escribe múltiples items en batch
        
        Args:
            items: Lista de items a escribir
            
        Returns:
            True si fue exitoso
        """
        try:
            with self.table.batch_writer() as batch:
                for item in items:
                    if 'created_at' not in item:
                        item['created_at'] = datetime.utcnow().isoformat()
                    item['updated_at'] = datetime.utcnow().isoformat()
                    batch.put_item(Item=item)
            return True
        except ClientError as e:
            logger.error(f"DynamoDB batch_write error: {e}")
            raise
    
    def update_item(
        self,
        key: Dict[str, Any],
        update_expression: str,
        expression_attribute_values: Dict[str, Any],
        expression_attribute_names: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Actualiza un item
        
        Args:
            key: Clave del item
            update_expression: Expresión de actualización
            expression_attribute_values: Valores para la expresión
            expression_attribute_names: Nombres de atributos (para palabras reservadas)
            
        Returns:
            Item actualizado
        """
        try:
            # Agregar updated_at automáticamente
            if '#updated_at' not in (expression_attribute_names or {}):
                if expression_attribute_names is None:
                    expression_attribute_names = {}
                expression_attribute_names['#updated_at'] = 'updated_at'
            
            if ':updated_at' not in expression_attribute_values:
                expression_attribute_values[':updated_at'] = datetime.utcnow().isoformat()
            
            if 'SET #updated_at = :updated_at' not in update_expression:
                update_expression += ', SET #updated_at = :updated_at'
            
            kwargs = {
                'Key': key,
                'UpdateExpression': update_expression,
                'ExpressionAttributeValues': expression_attribute_values,
                'ReturnValues': 'ALL_NEW'
            }
            
            if expression_attribute_names:
                kwargs['ExpressionAttributeNames'] = expression_attribute_names
            
            response = self.table.update_item(**kwargs)
            return response.get('Attributes', {})
        except ClientError as e:
            logger.error(f"DynamoDB update_item error: {e}")
            raise















