"""
AWS Adapters - Adaptadores para servicios AWS
============================================

Adaptadores para integrar con servicios AWS:
- DynamoDB para estado persistente
- ElastiCache Redis para caché
- S3 para almacenamiento de archivos
- CloudWatch para logging y métricas
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Intentar importar boto3
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    logger.warning("boto3 not installed, AWS adapters will not work")


class AWSStateManager:
    """
    Gestor de estado usando DynamoDB.
    
    Reemplaza el StateManager basado en archivos para entornos serverless.
    """
    
    def __init__(
        self,
        agent: Any,
        table_name: str = "cursor-agent-state",
        region: str = "us-east-1"
    ):
        """
        Inicializar gestor de estado con DynamoDB.
        
        Args:
            agent: Instancia del agente.
            table_name: Nombre de la tabla DynamoDB.
            region: Región AWS.
        """
        if not HAS_BOTO3:
            raise ImportError("boto3 is required for AWSStateManager")
        
        self.agent = agent
        self.table_name = table_name
        self.region = region
        
        # Cliente DynamoDB
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table(table_name)
        
        # Verificar/crear tabla
        self._ensure_table_exists()
        
        logger.info(f"AWSStateManager initialized with table: {table_name}")
    
    def _ensure_table_exists(self) -> None:
        """Asegurar que la tabla DynamoDB existe."""
        try:
            # Intentar describir la tabla
            self.table.meta.client.describe_table(TableName=self.table_name)
            logger.debug(f"Table {self.table_name} exists")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.warning(f"Table {self.table_name} does not exist. Create it manually or use Terraform/CDK.")
            else:
                raise
    
    async def save(self) -> None:
        """
        Guardar estado en DynamoDB.
        
        Raises:
            RuntimeError: Si hay error al guardar.
        """
        from .retry_handler import retry_dynamodb
        
        @retry_dynamodb
        async def _save_internal():
            try:
                state = {
                    "version": "1.0",
                    "timestamp": datetime.now().isoformat(),
                    "status": self.agent.status.value,
                    "running": self.agent.running,
                    "config": {
                        "check_interval": self.agent.config.check_interval,
                        "max_concurrent_tasks": self.agent.config.max_concurrent_tasks,
                        "task_timeout": self.agent.config.task_timeout,
                        "auto_restart": self.agent.config.auto_restart,
                        "persistent_storage": self.agent.config.persistent_storage,
                    },
                    "tasks": self._serialize_tasks(),
                }
                
                # Guardar en DynamoDB
                self.table.put_item(
                    Item={
                        "id": "agent_state",  # Partition key
                        "state": json.dumps(state),
                        "updated_at": datetime.now().isoformat(),
                        "ttl": int((datetime.now().timestamp() + 86400 * 30))  # 30 días TTL
                    }
                )
                
                logger.debug("State saved to DynamoDB")
            except Exception as e:
                logger.error(f"Error saving state to DynamoDB: {e}", exc_info=True)
                raise RuntimeError(f"Failed to save state: {e}") from e
        
        await _save_internal()
    
    async def load(self) -> None:
        """
        Cargar estado desde DynamoDB.
        """
        try:
            response = self.table.get_item(
                Key={"id": "agent_state"}
            )
            
            if "Item" not in response:
                logger.debug("No state found in DynamoDB, starting fresh")
                return
            
            state_data = json.loads(response["Item"]["state"])
            
            # Validar versión
            if state_data.get("version") != "1.0":
                logger.warning(f"State version mismatch: {state_data.get('version')}")
                return
            
            # Restaurar tareas
            if "tasks" in state_data:
                self._deserialize_tasks(state_data["tasks"])
            
            logger.info("State loaded from DynamoDB")
            
        except Exception as e:
            logger.warning(f"Error loading state from DynamoDB: {e}", exc_info=True)
    
    def _serialize_tasks(self) -> List[Dict[str, Any]]:
        """Serializar tareas."""
        return [
            {
                "id": task.id,
                "command": task.command,
                "status": task.status,
                "timestamp": task.timestamp.isoformat(),
                "result": task.result,
                "error": task.error,
            }
            for task in self.agent.tasks.values()
            if task.status in ("pending", "running")
        ]
    
    def _deserialize_tasks(self, tasks_data: List[Dict[str, Any]]) -> None:
        """Deserializar tareas."""
        from ..agent import Task
        
        for task_data in tasks_data:
            try:
                task = Task(
                    id=task_data["id"],
                    command=task_data["command"],
                    timestamp=datetime.fromisoformat(task_data["timestamp"]),
                    status=task_data.get("status", "pending"),
                    result=task_data.get("result"),
                    error=task_data.get("error"),
                )
                self.agent.tasks[task.id] = task
                
                if task.status == "pending":
                    self.agent.task_queue.put_nowait(task)
            
            except Exception as e:
                logger.warning(f"Error deserializing task {task_data.get('id')}: {e}")


class AWSCacheAdapter:
    """
    Adaptador de caché para AWS ElastiCache Redis o DynamoDB.
    """
    
    def __init__(
        self,
        cache_type: str = "elasticache",  # "elasticache" o "dynamodb"
        endpoint: Optional[str] = None,
        table_name: str = "cursor-agent-cache",
        region: str = "us-east-1"
    ):
        """
        Inicializar adaptador de caché.
        
        Args:
            cache_type: Tipo de caché ("elasticache" o "dynamodb").
            endpoint: Endpoint de Redis (para ElastiCache).
            table_name: Nombre de tabla DynamoDB (para DynamoDB cache).
            region: Región AWS.
        """
        if not HAS_BOTO3:
            raise ImportError("boto3 is required for AWSCacheAdapter")
        
        self.cache_type = cache_type
        self.region = region
        
        if cache_type == "elasticache":
            # Redis/ElastiCache
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=endpoint or os.getenv("REDIS_ENDPOINT", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"Redis cache connected to {endpoint}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, falling back to in-memory")
                self.cache_type = "memory"
                self._memory_cache: Dict[str, Any] = {}
        
        elif cache_type == "dynamodb":
            # DynamoDB
            self.dynamodb = boto3.resource('dynamodb', region_name=region)
            self.table = self.dynamodb.Table(table_name)
            self._ensure_table_exists()
            logger.info(f"DynamoDB cache initialized with table: {table_name}")
        
        else:
            # Fallback a memoria
            self.cache_type = "memory"
            self._memory_cache: Dict[str, Any] = {}
            logger.warning("Using in-memory cache (not persistent)")
    
    def _ensure_table_exists(self) -> None:
        """Asegurar que la tabla DynamoDB existe."""
        try:
            self.table.meta.client.describe_table(TableName=self.table.name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.warning(f"Table {self.table.name} does not exist. Create it manually.")
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché."""
        try:
            if self.cache_type == "elasticache":
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            
            elif self.cache_type == "dynamodb":
                response = self.table.get_item(Key={"key": key})
                if "Item" in response:
                    item = response["Item"]
                    # Verificar TTL
                    if "ttl" in item and item["ttl"] < datetime.now().timestamp():
                        await self.delete(key)
                        return None
                    return json.loads(item["value"])
                return None
            
            else:  # memory
                return self._memory_cache.get(key)
        
        except Exception as e:
            logger.warning(f"Error getting cache key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Establecer valor en caché."""
        try:
            if self.cache_type == "elasticache":
                value_str = json.dumps(value)
                if ttl:
                    self.redis_client.setex(key, ttl, value_str)
                else:
                    self.redis_client.set(key, value_str)
            
            elif self.cache_type == "dynamodb":
                item = {
                    "key": key,
                    "value": json.dumps(value),
                    "updated_at": datetime.now().isoformat()
                }
                if ttl:
                    item["ttl"] = int(datetime.now().timestamp() + ttl)
                
                self.table.put_item(Item=item)
            
            else:  # memory
                self._memory_cache[key] = value
        
        except Exception as e:
            logger.warning(f"Error setting cache key {key}: {e}")
    
    async def delete(self, key: str) -> None:
        """Eliminar clave del caché."""
        try:
            if self.cache_type == "elasticache":
                self.redis_client.delete(key)
            elif self.cache_type == "dynamodb":
                self.table.delete_item(Key={"key": key})
            else:  # memory
                self._memory_cache.pop(key, None)
        except Exception as e:
            logger.warning(f"Error deleting cache key {key}: {e}")
    
    async def clear(self) -> None:
        """Limpiar todo el caché."""
        try:
            if self.cache_type == "elasticache":
                self.redis_client.flushdb()
            elif self.cache_type == "dynamodb":
                # DynamoDB no tiene clear fácil, usar scan y delete
                response = self.table.scan()
                with self.table.batch_writer() as batch:
                    for item in response.get("Items", []):
                        batch.delete_item(Key={"key": item["key"]})
            else:  # memory
                self._memory_cache.clear()
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")


class CloudWatchLogger:
    """
    Logger estructurado para CloudWatch Logs.
    """
    
    def __init__(
        self,
        log_group: str = "cursor-agent-24-7",
        log_stream: Optional[str] = None,
        region: str = "us-east-1"
    ):
        """
        Inicializar logger de CloudWatch.
        
        Args:
            log_group: Grupo de logs de CloudWatch.
            log_stream: Stream de logs (opcional, se genera automáticamente).
            region: Región AWS.
        """
        if not HAS_BOTO3:
            logger.warning("boto3 not available, using standard logging")
            self.client = None
            return
        
        self.log_group = log_group
        self.region = region
        self.client = boto3.client('logs', region_name=region)
        
        # Crear log group si no existe
        try:
            self.client.create_log_group(logGroupName=log_group)
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                logger.warning(f"Could not create log group: {e}")
        
        # Crear log stream
        if not log_stream:
            import uuid
            log_stream = f"stream-{uuid.uuid4().hex[:8]}"
        
        self.log_stream = log_stream
        try:
            self.client.create_log_stream(
                logGroupName=log_group,
                logStreamName=log_stream
            )
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                logger.warning(f"Could not create log stream: {e}")
    
    def log(self, message: str, level: str = "INFO", extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Enviar log a CloudWatch.
        
        Args:
            message: Mensaje de log.
            level: Nivel de log (INFO, ERROR, WARNING, etc.).
            extra: Datos adicionales.
        """
        if not self.client:
            # Fallback a logging estándar
            getattr(logger, level.lower(), logger.info)(message, extra=extra)
            return
        
        try:
            log_event = {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "message": json.dumps({
                    "level": level,
                    "message": message,
                    "extra": extra or {}
                })
            }
            
            self.client.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
                logEvents=[log_event]
            )
        
        except Exception as e:
            # Fallback a logging estándar en caso de error
            logger.warning(f"Failed to send log to CloudWatch: {e}")
            getattr(logger, level.lower(), logger.info)(message, extra=extra)

