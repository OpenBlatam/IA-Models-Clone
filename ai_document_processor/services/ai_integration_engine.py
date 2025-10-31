"""
Motor de Integración AI
======================

Motor para integración con sistemas externos, APIs, bases de datos y servicios de terceros.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
import uuid
from pathlib import Path
import hashlib
import sqlite3
import psycopg2
import pymongo
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class IntegrationType(str, Enum):
    """Tipos de integración"""
    REST_API = "rest_api"
    GRAPHQL_API = "graphql_api"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    EMAIL = "email"
    MESSAGE_QUEUE = "message_queue"
    WEBHOOK = "webhook"
    FTP = "ftp"
    SFTP = "sftp"
    CUSTOM = "custom"

class ConnectionStatus(str, Enum):
    """Estado de conexión"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    UNKNOWN = "unknown"

class DataFormat(str, Enum):
    """Formatos de datos"""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    YAML = "yaml"
    BINARY = "binary"
    TEXT = "text"

@dataclass
class IntegrationConfig:
    """Configuración de integración"""
    id: str
    name: str
    integration_type: IntegrationType
    connection_config: Dict[str, Any] = field(default_factory=dict)
    authentication: Dict[str, Any] = field(default_factory=dict)
    data_mapping: Dict[str, Any] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=dict)
    timeout_config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class IntegrationConnection:
    """Conexión de integración"""
    config: IntegrationConfig
    status: ConnectionStatus
    last_connected: Optional[datetime] = None
    last_error: Optional[str] = None
    connection_pool: Any = None
    session: Any = None

@dataclass
class DataTransformation:
    """Transformación de datos"""
    id: str
    name: str
    source_format: DataFormat
    target_format: DataFormat
    transformation_rules: List[Dict[str, Any]] = field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class IntegrationResult:
    """Resultado de integración"""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    response_time: float = 0.0
    status_code: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class AIIntegrationEngine:
    """Motor de integración AI"""
    
    def __init__(self):
        self.integrations: Dict[str, IntegrationConnection] = {}
        self.data_transformations: Dict[str, DataTransformation] = {}
        self.connection_pools: Dict[str, Any] = {}
        self.http_sessions: Dict[str, aiohttp.ClientSession] = {}
        
        # Configuración
        self.default_timeout = 30
        self.max_retries = 3
        self.retry_delay = 1
        
    async def initialize(self):
        """Inicializa el motor de integración"""
        logger.info("Inicializando motor de integración AI...")
        
        # Cargar configuraciones de integración
        await self._load_integration_configs()
        
        # Cargar transformaciones de datos
        await self._load_data_transformations()
        
        # Inicializar conexiones
        await self._initialize_connections()
        
        logger.info("Motor de integración AI inicializado")
    
    async def _load_integration_configs(self):
        """Carga configuraciones de integración"""
        try:
            configs_file = Path("data/integration_configs.json")
            if configs_file.exists():
                with open(configs_file, 'r', encoding='utf-8') as f:
                    configs_data = json.load(f)
                
                for config_data in configs_data:
                    config = IntegrationConfig(
                        id=config_data["id"],
                        name=config_data["name"],
                        integration_type=IntegrationType(config_data["integration_type"]),
                        connection_config=config_data["connection_config"],
                        authentication=config_data["authentication"],
                        data_mapping=config_data["data_mapping"],
                        retry_config=config_data["retry_config"],
                        timeout_config=config_data["timeout_config"],
                        enabled=config_data["enabled"],
                        created_at=datetime.fromisoformat(config_data["created_at"]),
                        updated_at=datetime.fromisoformat(config_data["updated_at"])
                    )
                    
                    connection = IntegrationConnection(
                        config=config,
                        status=ConnectionStatus.UNKNOWN
                    )
                    
                    self.integrations[config.id] = connection
                
                logger.info(f"Cargadas {len(self.integrations)} configuraciones de integración")
            
        except Exception as e:
            logger.error(f"Error cargando configuraciones de integración: {e}")
    
    async def _load_data_transformations(self):
        """Carga transformaciones de datos"""
        try:
            transformations_file = Path("data/data_transformations.json")
            if transformations_file.exists():
                with open(transformations_file, 'r', encoding='utf-8') as f:
                    transformations_data = json.load(f)
                
                for transform_data in transformations_data:
                    transformation = DataTransformation(
                        id=transform_data["id"],
                        name=transform_data["name"],
                        source_format=DataFormat(transform_data["source_format"]),
                        target_format=DataFormat(transform_data["target_format"]),
                        transformation_rules=transform_data["transformation_rules"],
                        validation_rules=transform_data["validation_rules"],
                        created_at=datetime.fromisoformat(transform_data["created_at"])
                    )
                    
                    self.data_transformations[transformation.id] = transformation
                
                logger.info(f"Cargadas {len(self.data_transformations)} transformaciones de datos")
            
        except Exception as e:
            logger.error(f"Error cargando transformaciones de datos: {e}")
    
    async def _initialize_connections(self):
        """Inicializa conexiones"""
        try:
            for integration_id, connection in self.integrations.items():
                if connection.config.enabled:
                    await self._establish_connection(integration_id)
            
        except Exception as e:
            logger.error(f"Error inicializando conexiones: {e}")
    
    async def _establish_connection(self, integration_id: str):
        """Establece conexión"""
        try:
            connection = self.integrations[integration_id]
            config = connection.config
            
            connection.status = ConnectionStatus.CONNECTING
            
            if config.integration_type == IntegrationType.REST_API:
                await self._connect_rest_api(integration_id)
            elif config.integration_type == IntegrationType.DATABASE:
                await self._connect_database(integration_id)
            elif config.integration_type == IntegrationType.EMAIL:
                await self._connect_email(integration_id)
            elif config.integration_type == IntegrationType.MESSAGE_QUEUE:
                await self._connect_message_queue(integration_id)
            else:
                logger.warning(f"Tipo de integración no soportado: {config.integration_type}")
                connection.status = ConnectionStatus.ERROR
                connection.last_error = f"Tipo no soportado: {config.integration_type}"
            
        except Exception as e:
            logger.error(f"Error estableciendo conexión {integration_id}: {e}")
            connection = self.integrations[integration_id]
            connection.status = ConnectionStatus.ERROR
            connection.last_error = str(e)
    
    async def _connect_rest_api(self, integration_id: str):
        """Conecta a API REST"""
        try:
            connection = self.integrations[integration_id]
            config = connection.config
            
            # Crear sesión HTTP
            timeout = aiohttp.ClientTimeout(total=config.timeout_config.get("total", self.default_timeout))
            session = aiohttp.ClientSession(timeout=timeout)
            
            # Configurar autenticación
            auth = None
            if config.authentication.get("type") == "bearer":
                token = config.authentication.get("token")
                auth = aiohttp.BearerTokenAuth(token)
            elif config.authentication.get("type") == "basic":
                username = config.authentication.get("username")
                password = config.authentication.get("password")
                auth = aiohttp.BasicAuth(username, password)
            
            # Probar conexión
            base_url = config.connection_config.get("base_url")
            if base_url:
                test_endpoint = config.connection_config.get("test_endpoint", "/health")
                async with session.get(f"{base_url}{test_endpoint}", auth=auth) as response:
                    if response.status < 400:
                        connection.status = ConnectionStatus.CONNECTED
                        connection.last_connected = datetime.now()
                        connection.session = session
                        self.http_sessions[integration_id] = session
                    else:
                        connection.status = ConnectionStatus.ERROR
                        connection.last_error = f"HTTP {response.status}"
            
        except Exception as e:
            logger.error(f"Error conectando a API REST {integration_id}: {e}")
            connection = self.integrations[integration_id]
            connection.status = ConnectionStatus.ERROR
            connection.last_error = str(e)
    
    async def _connect_database(self, integration_id: str):
        """Conecta a base de datos"""
        try:
            connection = self.integrations[integration_id]
            config = connection.config
            
            db_type = config.connection_config.get("type")
            connection_string = config.connection_config.get("connection_string")
            
            if db_type == "sqlite":
                # SQLite no requiere conexión persistente
                connection.status = ConnectionStatus.CONNECTED
                connection.last_connected = datetime.now()
                
            elif db_type == "postgresql":
                # PostgreSQL
                conn = psycopg2.connect(connection_string)
                connection.status = ConnectionStatus.CONNECTED
                connection.last_connected = datetime.now()
                connection.connection_pool = conn
                
            elif db_type == "mongodb":
                # MongoDB
                client = pymongo.MongoClient(connection_string)
                # Probar conexión
                client.admin.command('ping')
                connection.status = ConnectionStatus.CONNECTED
                connection.last_connected = datetime.now()
                connection.connection_pool = client
                
            else:
                connection.status = ConnectionStatus.ERROR
                connection.last_error = f"Tipo de BD no soportado: {db_type}"
            
        except Exception as e:
            logger.error(f"Error conectando a base de datos {integration_id}: {e}")
            connection = self.integrations[integration_id]
            connection.status = ConnectionStatus.ERROR
            connection.last_error = str(e)
    
    async def _connect_email(self, integration_id: str):
        """Conecta a servicio de email"""
        try:
            connection = self.integrations[integration_id]
            config = connection.config
            
            # En implementación real, configurar SMTP/IMAP
            connection.status = ConnectionStatus.CONNECTED
            connection.last_connected = datetime.now()
            
        except Exception as e:
            logger.error(f"Error conectando a email {integration_id}: {e}")
            connection = self.integrations[integration_id]
            connection.status = ConnectionStatus.ERROR
            connection.last_error = str(e)
    
    async def _connect_message_queue(self, integration_id: str):
        """Conecta a cola de mensajes"""
        try:
            connection = self.integrations[integration_id]
            config = connection.config
            
            # En implementación real, configurar Redis/RabbitMQ
            connection.status = ConnectionStatus.CONNECTED
            connection.last_connected = datetime.now()
            
        except Exception as e:
            logger.error(f"Error conectando a cola de mensajes {integration_id}: {e}")
            connection = self.integrations[integration_id]
            connection.status = ConnectionStatus.ERROR
            connection.last_error = str(e)
    
    async def create_integration(
        self,
        name: str,
        integration_type: IntegrationType,
        connection_config: Dict[str, Any],
        authentication: Dict[str, Any] = None,
        data_mapping: Dict[str, Any] = None
    ) -> str:
        """Crea nueva integración"""
        try:
            integration_id = f"integration_{uuid.uuid4().hex[:8]}"
            
            config = IntegrationConfig(
                id=integration_id,
                name=name,
                integration_type=integration_type,
                connection_config=connection_config,
                authentication=authentication or {},
                data_mapping=data_mapping or {},
                retry_config={"max_retries": self.max_retries, "delay": self.retry_delay},
                timeout_config={"total": self.default_timeout}
            )
            
            connection = IntegrationConnection(
                config=config,
                status=ConnectionStatus.UNKNOWN
            )
            
            self.integrations[integration_id] = connection
            
            # Establecer conexión
            await self._establish_connection(integration_id)
            
            # Guardar configuración
            await self._save_integration_configs()
            
            logger.info(f"Integración creada: {integration_id}")
            return integration_id
            
        except Exception as e:
            logger.error(f"Error creando integración: {e}")
            raise
    
    async def send_data(
        self,
        integration_id: str,
        data: Any,
        endpoint: str = None,
        method: str = "POST",
        headers: Dict[str, str] = None
    ) -> IntegrationResult:
        """Envía datos a integración"""
        try:
            if integration_id not in self.integrations:
                return IntegrationResult(
                    success=False,
                    error_message=f"Integración no encontrada: {integration_id}"
                )
            
            connection = self.integrations[integration_id]
            
            if connection.status != ConnectionStatus.CONNECTED:
                return IntegrationResult(
                    success=False,
                    error_message=f"Conexión no disponible: {connection.status.value}"
                )
            
            start_time = time.time()
            
            if connection.config.integration_type == IntegrationType.REST_API:
                result = await self._send_data_rest_api(connection, data, endpoint, method, headers)
            elif connection.config.integration_type == IntegrationType.DATABASE:
                result = await self._send_data_database(connection, data)
            elif connection.config.integration_type == IntegrationType.EMAIL:
                result = await self._send_data_email(connection, data)
            elif connection.config.integration_type == IntegrationType.MESSAGE_QUEUE:
                result = await self._send_data_message_queue(connection, data)
            else:
                result = IntegrationResult(
                    success=False,
                    error_message=f"Tipo de integración no soportado: {connection.config.integration_type}"
                )
            
            result.response_time = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Error enviando datos: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _send_data_rest_api(
        self,
        connection: IntegrationConnection,
        data: Any,
        endpoint: str,
        method: str,
        headers: Dict[str, str]
    ) -> IntegrationResult:
        """Envía datos a API REST"""
        try:
            session = connection.session
            if not session:
                return IntegrationResult(
                    success=False,
                    error_message="Sesión HTTP no disponible"
                )
            
            base_url = connection.config.connection_config.get("base_url")
            url = f"{base_url}{endpoint}" if endpoint else base_url
            
            # Configurar headers
            request_headers = headers or {}
            if connection.config.authentication.get("type") == "bearer":
                token = connection.config.authentication.get("token")
                request_headers["Authorization"] = f"Bearer {token}"
            
            # Configurar autenticación
            auth = None
            if connection.config.authentication.get("type") == "basic":
                username = connection.config.authentication.get("username")
                password = connection.config.authentication.get("password")
                auth = aiohttp.BasicAuth(username, password)
            
            # Enviar request
            async with session.request(
                method=method,
                url=url,
                json=data if isinstance(data, (dict, list)) else None,
                data=data if isinstance(data, str) else None,
                headers=request_headers,
                auth=auth
            ) as response:
                
                response_data = None
                if response.content_type == "application/json":
                    response_data = await response.json()
                else:
                    response_data = await response.text()
                
                return IntegrationResult(
                    success=response.status < 400,
                    data=response_data,
                    status_code=response.status,
                    metadata={"url": url, "method": method}
                )
                
        except Exception as e:
            logger.error(f"Error enviando datos a API REST: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _send_data_database(self, connection: IntegrationConnection, data: Any) -> IntegrationResult:
        """Envía datos a base de datos"""
        try:
            config = connection.config
            db_type = config.connection_config.get("type")
            
            if db_type == "sqlite":
                return await self._send_data_sqlite(connection, data)
            elif db_type == "postgresql":
                return await self._send_data_postgresql(connection, data)
            elif db_type == "mongodb":
                return await self._send_data_mongodb(connection, data)
            else:
                return IntegrationResult(
                    success=False,
                    error_message=f"Tipo de BD no soportado: {db_type}"
                )
                
        except Exception as e:
            logger.error(f"Error enviando datos a base de datos: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _send_data_sqlite(self, connection: IntegrationConnection, data: Any) -> IntegrationResult:
        """Envía datos a SQLite"""
        try:
            db_path = connection.config.connection_config.get("database_path")
            table = connection.config.data_mapping.get("table")
            
            if not table:
                return IntegrationResult(
                    success=False,
                    error_message="Tabla no especificada en data_mapping"
                )
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            if isinstance(data, list):
                # Insertar múltiples registros
                placeholders = ", ".join(["?" for _ in data[0].keys()])
                columns = ", ".join(data[0].keys())
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                
                for record in data:
                    cursor.execute(query, list(record.values()))
            else:
                # Insertar un registro
                placeholders = ", ".join(["?" for _ in data.keys()])
                columns = ", ".join(data.keys())
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                cursor.execute(query, list(data.values()))
            
            conn.commit()
            conn.close()
            
            return IntegrationResult(
                success=True,
                data={"rows_affected": cursor.rowcount},
                metadata={"table": table}
            )
            
        except Exception as e:
            logger.error(f"Error enviando datos a SQLite: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _send_data_postgresql(self, connection: IntegrationConnection, data: Any) -> IntegrationResult:
        """Envía datos a PostgreSQL"""
        try:
            conn = connection.connection_pool
            table = connection.config.data_mapping.get("table")
            
            if not table:
                return IntegrationResult(
                    success=False,
                    error_message="Tabla no especificada en data_mapping"
                )
            
            cursor = conn.cursor()
            
            if isinstance(data, list):
                # Insertar múltiples registros
                placeholders = ", ".join(["%s" for _ in data[0].keys()])
                columns = ", ".join(data[0].keys())
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                
                for record in data:
                    cursor.execute(query, list(record.values()))
            else:
                # Insertar un registro
                placeholders = ", ".join(["%s" for _ in data.keys()])
                columns = ", ".join(data.keys())
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                cursor.execute(query, list(data.values()))
            
            conn.commit()
            
            return IntegrationResult(
                success=True,
                data={"rows_affected": cursor.rowcount},
                metadata={"table": table}
            )
            
        except Exception as e:
            logger.error(f"Error enviando datos a PostgreSQL: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _send_data_mongodb(self, connection: IntegrationConnection, data: Any) -> IntegrationResult:
        """Envía datos a MongoDB"""
        try:
            client = connection.connection_pool
            database_name = connection.config.connection_config.get("database")
            collection_name = connection.config.data_mapping.get("collection")
            
            if not collection_name:
                return IntegrationResult(
                    success=False,
                    error_message="Colección no especificada en data_mapping"
                )
            
            db = client[database_name]
            collection = db[collection_name]
            
            if isinstance(data, list):
                result = collection.insert_many(data)
                inserted_count = len(result.inserted_ids)
            else:
                result = collection.insert_one(data)
                inserted_count = 1
            
            return IntegrationResult(
                success=True,
                data={"inserted_count": inserted_count},
                metadata={"collection": collection_name}
            )
            
        except Exception as e:
            logger.error(f"Error enviando datos a MongoDB: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _send_data_email(self, connection: IntegrationConnection, data: Any) -> IntegrationResult:
        """Envía datos por email"""
        try:
            # En implementación real, usar smtplib
            logger.info(f"Enviando email: {data}")
            
            return IntegrationResult(
                success=True,
                data={"message": "Email enviado"},
                metadata={"type": "email"}
            )
            
        except Exception as e:
            logger.error(f"Error enviando email: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _send_data_message_queue(self, connection: IntegrationConnection, data: Any) -> IntegrationResult:
        """Envía datos a cola de mensajes"""
        try:
            # En implementación real, usar Redis/RabbitMQ
            logger.info(f"Enviando a cola de mensajes: {data}")
            
            return IntegrationResult(
                success=True,
                data={"message": "Mensaje enviado"},
                metadata={"type": "message_queue"}
            )
            
        except Exception as e:
            logger.error(f"Error enviando a cola de mensajes: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def receive_data(
        self,
        integration_id: str,
        query: Dict[str, Any] = None,
        endpoint: str = None
    ) -> IntegrationResult:
        """Recibe datos de integración"""
        try:
            if integration_id not in self.integrations:
                return IntegrationResult(
                    success=False,
                    error_message=f"Integración no encontrada: {integration_id}"
                )
            
            connection = self.integrations[integration_id]
            
            if connection.status != ConnectionStatus.CONNECTED:
                return IntegrationResult(
                    success=False,
                    error_message=f"Conexión no disponible: {connection.status.value}"
                )
            
            start_time = time.time()
            
            if connection.config.integration_type == IntegrationType.REST_API:
                result = await self._receive_data_rest_api(connection, endpoint)
            elif connection.config.integration_type == IntegrationType.DATABASE:
                result = await self._receive_data_database(connection, query)
            else:
                result = IntegrationResult(
                    success=False,
                    error_message=f"Tipo de integración no soportado para recepción: {connection.config.integration_type}"
                )
            
            result.response_time = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Error recibiendo datos: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _receive_data_rest_api(self, connection: IntegrationConnection, endpoint: str) -> IntegrationResult:
        """Recibe datos de API REST"""
        try:
            session = connection.session
            if not session:
                return IntegrationResult(
                    success=False,
                    error_message="Sesión HTTP no disponible"
                )
            
            base_url = connection.config.connection_config.get("base_url")
            url = f"{base_url}{endpoint}" if endpoint else base_url
            
            # Configurar autenticación
            auth = None
            if connection.config.authentication.get("type") == "basic":
                username = connection.config.authentication.get("username")
                password = connection.config.authentication.get("password")
                auth = aiohttp.BasicAuth(username, password)
            
            async with session.get(url, auth=auth) as response:
                if response.status < 400:
                    if response.content_type == "application/json":
                        data = await response.json()
                    else:
                        data = await response.text()
                    
                    return IntegrationResult(
                        success=True,
                        data=data,
                        status_code=response.status,
                        metadata={"url": url}
                    )
                else:
                    return IntegrationResult(
                        success=False,
                        error_message=f"HTTP {response.status}",
                        status_code=response.status
                    )
                    
        except Exception as e:
            logger.error(f"Error recibiendo datos de API REST: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _receive_data_database(self, connection: IntegrationConnection, query: Dict[str, Any]) -> IntegrationResult:
        """Recibe datos de base de datos"""
        try:
            config = connection.config
            db_type = config.connection_config.get("type")
            
            if db_type == "sqlite":
                return await self._receive_data_sqlite(connection, query)
            elif db_type == "postgresql":
                return await self._receive_data_postgresql(connection, query)
            elif db_type == "mongodb":
                return await self._receive_data_mongodb(connection, query)
            else:
                return IntegrationResult(
                    success=False,
                    error_message=f"Tipo de BD no soportado: {db_type}"
                )
                
        except Exception as e:
            logger.error(f"Error recibiendo datos de base de datos: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _receive_data_sqlite(self, connection: IntegrationConnection, query: Dict[str, Any]) -> IntegrationResult:
        """Recibe datos de SQLite"""
        try:
            db_path = connection.config.connection_config.get("database_path")
            table = connection.config.data_mapping.get("table")
            
            if not table:
                return IntegrationResult(
                    success=False,
                    error_message="Tabla no especificada en data_mapping"
                )
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Construir query
            sql_query = query.get("sql", f"SELECT * FROM {table}")
            limit = query.get("limit", 100)
            
            if "LIMIT" not in sql_query.upper():
                sql_query += f" LIMIT {limit}"
            
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            
            # Obtener nombres de columnas
            columns = [description[0] for description in cursor.description]
            
            # Convertir a lista de diccionarios
            data = [dict(zip(columns, row)) for row in rows]
            
            conn.close()
            
            return IntegrationResult(
                success=True,
                data=data,
                metadata={"table": table, "count": len(data)}
            )
            
        except Exception as e:
            logger.error(f"Error recibiendo datos de SQLite: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _receive_data_postgresql(self, connection: IntegrationConnection, query: Dict[str, Any]) -> IntegrationResult:
        """Recibe datos de PostgreSQL"""
        try:
            conn = connection.connection_pool
            table = connection.config.data_mapping.get("table")
            
            if not table:
                return IntegrationResult(
                    success=False,
                    error_message="Tabla no especificada en data_mapping"
                )
            
            cursor = conn.cursor()
            
            # Construir query
            sql_query = query.get("sql", f"SELECT * FROM {table}")
            limit = query.get("limit", 100)
            
            if "LIMIT" not in sql_query.upper():
                sql_query += f" LIMIT {limit}"
            
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            
            # Obtener nombres de columnas
            columns = [description[0] for description in cursor.description]
            
            # Convertir a lista de diccionarios
            data = [dict(zip(columns, row)) for row in rows]
            
            return IntegrationResult(
                success=True,
                data=data,
                metadata={"table": table, "count": len(data)}
            )
            
        except Exception as e:
            logger.error(f"Error recibiendo datos de PostgreSQL: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _receive_data_mongodb(self, connection: IntegrationConnection, query: Dict[str, Any]) -> IntegrationResult:
        """Recibe datos de MongoDB"""
        try:
            client = connection.connection_pool
            database_name = connection.config.connection_config.get("database")
            collection_name = connection.config.data_mapping.get("collection")
            
            if not collection_name:
                return IntegrationResult(
                    success=False,
                    error_message="Colección no especificada en data_mapping"
                )
            
            db = client[database_name]
            collection = db[collection_name]
            
            # Construir query
            mongo_query = query.get("query", {})
            limit = query.get("limit", 100)
            
            cursor = collection.find(mongo_query).limit(limit)
            data = list(cursor)
            
            return IntegrationResult(
                success=True,
                data=data,
                metadata={"collection": collection_name, "count": len(data)}
            )
            
        except Exception as e:
            logger.error(f"Error recibiendo datos de MongoDB: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def transform_data(
        self,
        transformation_id: str,
        data: Any
    ) -> IntegrationResult:
        """Transforma datos"""
        try:
            if transformation_id not in self.data_transformations:
                return IntegrationResult(
                    success=False,
                    error_message=f"Transformación no encontrada: {transformation_id}"
                )
            
            transformation = self.data_transformations[transformation_id]
            
            # Aplicar reglas de transformación
            transformed_data = await self._apply_transformation_rules(data, transformation.transformation_rules)
            
            # Validar datos transformados
            validation_result = await self._validate_data(transformed_data, transformation.validation_rules)
            
            if not validation_result:
                return IntegrationResult(
                    success=False,
                    error_message="Datos transformados no pasaron validación"
                )
            
            return IntegrationResult(
                success=True,
                data=transformed_data,
                metadata={"transformation_id": transformation_id}
            )
            
        except Exception as e:
            logger.error(f"Error transformando datos: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def _apply_transformation_rules(self, data: Any, rules: List[Dict[str, Any]]) -> Any:
        """Aplica reglas de transformación"""
        try:
            transformed_data = data
            
            for rule in rules:
                rule_type = rule.get("type")
                
                if rule_type == "field_mapping":
                    # Mapeo de campos
                    mapping = rule.get("mapping", {})
                    if isinstance(transformed_data, dict):
                        new_data = {}
                        for old_field, new_field in mapping.items():
                            if old_field in transformed_data:
                                new_data[new_field] = transformed_data[old_field]
                        transformed_data = new_data
                
                elif rule_type == "field_filter":
                    # Filtrar campos
                    fields = rule.get("fields", [])
                    if isinstance(transformed_data, dict):
                        filtered_data = {k: v for k, v in transformed_data.items() if k in fields}
                        transformed_data = filtered_data
                
                elif rule_type == "data_type_conversion":
                    # Conversión de tipos
                    conversions = rule.get("conversions", {})
                    if isinstance(transformed_data, dict):
                        for field, target_type in conversions.items():
                            if field in transformed_data:
                                try:
                                    if target_type == "int":
                                        transformed_data[field] = int(transformed_data[field])
                                    elif target_type == "float":
                                        transformed_data[field] = float(transformed_data[field])
                                    elif target_type == "str":
                                        transformed_data[field] = str(transformed_data[field])
                                    elif target_type == "bool":
                                        transformed_data[field] = bool(transformed_data[field])
                                except (ValueError, TypeError):
                                    pass  # Mantener valor original si conversión falla
                
                elif rule_type == "custom_function":
                    # Función personalizada
                    function_name = rule.get("function")
                    if function_name == "uppercase":
                        if isinstance(transformed_data, dict):
                            for key, value in transformed_data.items():
                                if isinstance(value, str):
                                    transformed_data[key] = value.upper()
                    elif function_name == "lowercase":
                        if isinstance(transformed_data, dict):
                            for key, value in transformed_data.items():
                                if isinstance(value, str):
                                    transformed_data[key] = value.lower()
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error aplicando reglas de transformación: {e}")
            return data
    
    async def _validate_data(self, data: Any, rules: List[Dict[str, Any]]) -> bool:
        """Valida datos"""
        try:
            for rule in rules:
                rule_type = rule.get("type")
                
                if rule_type == "required_fields":
                    # Campos requeridos
                    required_fields = rule.get("fields", [])
                    if isinstance(data, dict):
                        for field in required_fields:
                            if field not in data or data[field] is None:
                                return False
                
                elif rule_type == "field_types":
                    # Tipos de campos
                    field_types = rule.get("field_types", {})
                    if isinstance(data, dict):
                        for field, expected_type in field_types.items():
                            if field in data:
                                if expected_type == "int" and not isinstance(data[field], int):
                                    return False
                                elif expected_type == "str" and not isinstance(data[field], str):
                                    return False
                                elif expected_type == "float" and not isinstance(data[field], (int, float)):
                                    return False
                
                elif rule_type == "field_ranges":
                    # Rangos de campos
                    field_ranges = rule.get("field_ranges", {})
                    if isinstance(data, dict):
                        for field, range_config in field_ranges.items():
                            if field in data:
                                value = data[field]
                                if "min" in range_config and value < range_config["min"]:
                                    return False
                                if "max" in range_config and value > range_config["max"]:
                                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando datos: {e}")
            return False
    
    async def _save_integration_configs(self):
        """Guarda configuraciones de integración"""
        try:
            # Crear directorio de datos
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Convertir a formato serializable
            configs_data = []
            for connection in self.integrations.values():
                config = connection.config
                configs_data.append({
                    "id": config.id,
                    "name": config.name,
                    "integration_type": config.integration_type.value,
                    "connection_config": config.connection_config,
                    "authentication": config.authentication,
                    "data_mapping": config.data_mapping,
                    "retry_config": config.retry_config,
                    "timeout_config": config.timeout_config,
                    "enabled": config.enabled,
                    "created_at": config.created_at.isoformat(),
                    "updated_at": config.updated_at.isoformat()
                })
            
            # Guardar archivo
            configs_file = data_dir / "integration_configs.json"
            with open(configs_file, 'w', encoding='utf-8') as f:
                json.dump(configs_data, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Error guardando configuraciones de integración: {e}")
    
    async def get_integration_status(self, integration_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de integración"""
        try:
            if integration_id not in self.integrations:
                return None
            
            connection = self.integrations[integration_id]
            config = connection.config
            
            return {
                "id": config.id,
                "name": config.name,
                "type": config.integration_type.value,
                "status": connection.status.value,
                "enabled": config.enabled,
                "last_connected": connection.last_connected.isoformat() if connection.last_connected else None,
                "last_error": connection.last_error,
                "created_at": config.created_at.isoformat(),
                "updated_at": config.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de integración: {e}")
            return None
    
    async def get_integration_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard de integración"""
        try:
            # Estadísticas generales
            total_integrations = len(self.integrations)
            connected_integrations = len([c for c in self.integrations.values() if c.status == ConnectionStatus.CONNECTED])
            enabled_integrations = len([c for c in self.integrations.values() if c.config.enabled])
            
            # Distribución por tipo
            type_distribution = {}
            for connection in self.integrations.values():
                integration_type = connection.config.integration_type.value
                type_distribution[integration_type] = type_distribution.get(integration_type, 0) + 1
            
            # Distribución por estado
            status_distribution = {}
            for connection in self.integrations.values():
                status = connection.status.value
                status_distribution[status] = status_distribution.get(status, 0) + 1
            
            # Integraciones con errores
            error_integrations = [
                {
                    "id": connection.config.id,
                    "name": connection.config.name,
                    "type": connection.config.integration_type.value,
                    "error": connection.last_error
                }
                for connection in self.integrations.values()
                if connection.status == ConnectionStatus.ERROR
            ]
            
            return {
                "total_integrations": total_integrations,
                "connected_integrations": connected_integrations,
                "enabled_integrations": enabled_integrations,
                "type_distribution": type_distribution,
                "status_distribution": status_distribution,
                "error_integrations": error_integrations,
                "connection_rate": (connected_integrations / total_integrations * 100) if total_integrations > 0 else 0,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard de integración: {e}")
            return {"error": str(e)}
    
    async def test_integration(self, integration_id: str) -> IntegrationResult:
        """Prueba integración"""
        try:
            if integration_id not in self.integrations:
                return IntegrationResult(
                    success=False,
                    error_message=f"Integración no encontrada: {integration_id}"
                )
            
            connection = self.integrations[integration_id]
            
            # Reestablecer conexión
            await self._establish_connection(integration_id)
            
            if connection.status == ConnectionStatus.CONNECTED:
                return IntegrationResult(
                    success=True,
                    data={"message": "Conexión exitosa"},
                    metadata={"integration_id": integration_id}
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=connection.last_error or "Error de conexión desconocido"
                )
            
        except Exception as e:
            logger.error(f"Error probando integración: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def close_connections(self):
        """Cierra todas las conexiones"""
        try:
            # Cerrar sesiones HTTP
            for session in self.http_sessions.values():
                if session and not session.closed:
                    await session.close()
            
            # Cerrar conexiones de base de datos
            for connection in self.integrations.values():
                if connection.connection_pool:
                    if hasattr(connection.connection_pool, 'close'):
                        connection.connection_pool.close()
                    elif hasattr(connection.connection_pool, 'disconnect'):
                        connection.connection_pool.disconnect()
            
            logger.info("Conexiones cerradas")
            
        except Exception as e:
            logger.error(f"Error cerrando conexiones: {e}")

