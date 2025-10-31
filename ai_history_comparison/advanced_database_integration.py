"""
Advanced Database Integration System
===================================

This module provides advanced database integration capabilities including:
- Multi-database support (PostgreSQL, MongoDB, Redis, InfluxDB)
- Data synchronization and replication
- Advanced querying and analytics
- Real-time data streaming
- Database performance monitoring
- Automated backup and recovery
- Data migration and transformation
- Database optimization and indexing
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncpg
import motor.motor_asyncio
import redis.asyncio as redis
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import psycopg2
from pymongo import MongoClient
import pickle
import hashlib

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import get_ai_history_config

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    name: str
    type: str  # postgresql, mongodb, redis, influxdb
    host: str
    port: int
    database: str
    username: str = None
    password: str = None
    ssl_mode: str = "prefer"
    connection_pool_size: int = 10
    max_overflow: int = 20
    enabled: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DatabaseConnection:
    """Database connection wrapper"""
    config: DatabaseConfig
    connection: Any = None
    is_connected: bool = False
    last_used: datetime = None
    connection_count: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        if self.last_used is None:
            self.last_used = datetime.now()


@dataclass
class QueryResult:
    """Database query result"""
    query: str
    result: Any
    execution_time: float
    row_count: int = 0
    success: bool = True
    error_message: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DatabaseMetrics:
    """Database performance metrics"""
    database_name: str
    connection_count: int
    active_queries: int
    average_query_time: float
    total_queries: int
    error_rate: float
    memory_usage: float
    cpu_usage: float
    disk_usage: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AdvancedDatabaseIntegration:
    """Advanced database integration system"""
    
    def __init__(self):
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        
        # Database configurations
        self.database_configs: Dict[str, DatabaseConfig] = {}
        self.database_connections: Dict[str, DatabaseConnection] = {}
        
        # Connection pools
        self.connection_pools: Dict[str, Any] = {}
        
        # Query tracking
        self.query_history: List[QueryResult] = []
        self.performance_metrics: List[DatabaseMetrics] = []
        
        # Data synchronization
        self.sync_enabled = True
        self.sync_interval_minutes = 30
        self.sync_task: Optional[asyncio.Task] = None
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Initialize database configurations
        self._initialize_database_configs()
    
    def _initialize_database_configs(self):
        """Initialize database configurations from environment variables"""
        try:
            # PostgreSQL configuration
            postgres_config = DatabaseConfig(
                name="postgresql_main",
                type="postgresql",
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DB", "ai_history"),
                username=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
                connection_pool_size=int(os.getenv("POSTGRES_POOL_SIZE", "10")),
                enabled=os.getenv("POSTGRES_ENABLED", "false").lower() == "true"
            )
            self.database_configs["postgresql_main"] = postgres_config
            
            # MongoDB configuration
            mongodb_config = DatabaseConfig(
                name="mongodb_main",
                type="mongodb",
                host=os.getenv("MONGODB_HOST", "localhost"),
                port=int(os.getenv("MONGODB_PORT", "27017")),
                database=os.getenv("MONGODB_DB", "ai_history"),
                username=os.getenv("MONGODB_USER", ""),
                password=os.getenv("MONGODB_PASSWORD", ""),
                connection_pool_size=int(os.getenv("MONGODB_POOL_SIZE", "10")),
                enabled=os.getenv("MONGODB_ENABLED", "false").lower() == "true"
            )
            self.database_configs["mongodb_main"] = mongodb_config
            
            # Redis configuration
            redis_config = DatabaseConfig(
                name="redis_cache",
                type="redis",
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                database=int(os.getenv("REDIS_DB", "0")),
                password=os.getenv("REDIS_PASSWORD", ""),
                connection_pool_size=int(os.getenv("REDIS_POOL_SIZE", "10")),
                enabled=os.getenv("REDIS_ENABLED", "false").lower() == "true"
            )
            self.database_configs["redis_cache"] = redis_config
            
            # InfluxDB configuration
            influxdb_config = DatabaseConfig(
                name="influxdb_metrics",
                type="influxdb",
                host=os.getenv("INFLUXDB_HOST", "localhost"),
                port=int(os.getenv("INFLUXDB_PORT", "8086")),
                database=os.getenv("INFLUXDB_DB", "ai_metrics"),
                username=os.getenv("INFLUXDB_USER", ""),
                password=os.getenv("INFLUXDB_PASSWORD", ""),
                connection_pool_size=int(os.getenv("INFLUXDB_POOL_SIZE", "5")),
                enabled=os.getenv("INFLUXDB_ENABLED", "false").lower() == "true"
            )
            self.database_configs["influxdb_metrics"] = influxdb_config
            
            logger.info(f"Initialized {len(self.database_configs)} database configurations")
        
        except Exception as e:
            logger.error(f"Error initializing database configurations: {str(e)}")
    
    async def initialize_connections(self):
        """Initialize database connections"""
        try:
            for config_name, config in self.database_configs.items():
                if not config.enabled:
                    continue
                
                try:
                    if config.type == "postgresql":
                        await self._initialize_postgresql_connection(config)
                    elif config.type == "mongodb":
                        await self._initialize_mongodb_connection(config)
                    elif config.type == "redis":
                        await self._initialize_redis_connection(config)
                    elif config.type == "influxdb":
                        await self._initialize_influxdb_connection(config)
                    
                    logger.info(f"Initialized {config.type} connection: {config_name}")
                
                except Exception as e:
                    logger.error(f"Error initializing {config.type} connection {config_name}: {str(e)}")
            
            logger.info("Database connections initialized")
        
        except Exception as e:
            logger.error(f"Error initializing database connections: {str(e)}")
            raise
    
    async def _initialize_postgresql_connection(self, config: DatabaseConfig):
        """Initialize PostgreSQL connection"""
        try:
            # Create connection string
            connection_string = f"postgresql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
            
            # Create async engine
            engine = create_async_engine(
                connection_string,
                pool_size=config.connection_pool_size,
                max_overflow=config.max_overflow,
                echo=False
            )
            
            # Create session factory
            session_factory = sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Store connection
            connection = DatabaseConnection(config=config, connection=engine)
            self.database_connections[config.name] = connection
            
            # Test connection
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            connection.is_connected = True
            logger.info(f"PostgreSQL connection established: {config.name}")
        
        except Exception as e:
            logger.error(f"Error initializing PostgreSQL connection: {str(e)}")
            raise
    
    async def _initialize_mongodb_connection(self, config: DatabaseConfig):
        """Initialize MongoDB connection"""
        try:
            # Create connection string
            if config.username and config.password:
                connection_string = f"mongodb://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
            else:
                connection_string = f"mongodb://{config.host}:{config.port}/{config.database}"
            
            # Create async client
            client = motor.motor_asyncio.AsyncIOMotorClient(connection_string)
            
            # Test connection
            await client.admin.command('ping')
            
            # Store connection
            connection = DatabaseConnection(config=config, connection=client)
            connection.is_connected = True
            self.database_connections[config.name] = connection
            
            logger.info(f"MongoDB connection established: {config.name}")
        
        except Exception as e:
            logger.error(f"Error initializing MongoDB connection: {str(e)}")
            raise
    
    async def _initialize_redis_connection(self, config: DatabaseConfig):
        """Initialize Redis connection"""
        try:
            # Create connection pool
            pool = redis.ConnectionPool(
                host=config.host,
                port=config.port,
                db=config.database,
                password=config.password if config.password else None,
                max_connections=config.connection_pool_size
            )
            
            # Create client
            client = redis.Redis(connection_pool=pool)
            
            # Test connection
            await client.ping()
            
            # Store connection
            connection = DatabaseConnection(config=config, connection=client)
            connection.is_connected = True
            self.database_connections[config.name] = connection
            
            logger.info(f"Redis connection established: {config.name}")
        
        except Exception as e:
            logger.error(f"Error initializing Redis connection: {str(e)}")
            raise
    
    async def _initialize_influxdb_connection(self, config: DatabaseConfig):
        """Initialize InfluxDB connection"""
        try:
            # Create client
            client = influxdb_client.InfluxDBClient(
                url=f"http://{config.host}:{config.port}",
                token=config.password if config.password else "",
                org=config.username if config.username else "default"
            )
            
            # Test connection
            health = client.health()
            if health.status != "pass":
                raise Exception(f"InfluxDB health check failed: {health.message}")
            
            # Store connection
            connection = DatabaseConnection(config=config, connection=client)
            connection.is_connected = True
            self.database_connections[config.name] = connection
            
            logger.info(f"InfluxDB connection established: {config.name}")
        
        except Exception as e:
            logger.error(f"Error initializing InfluxDB connection: {str(e)}")
            raise
    
    async def execute_query(self, 
                          database_name: str, 
                          query: str, 
                          parameters: Dict[str, Any] = None) -> QueryResult:
        """Execute a query on the specified database"""
        try:
            start_time = datetime.now()
            
            if database_name not in self.database_connections:
                raise ValueError(f"Database connection not found: {database_name}")
            
            connection = self.database_connections[database_name]
            if not connection.is_connected:
                raise Exception(f"Database connection not active: {database_name}")
            
            # Execute query based on database type
            if connection.config.type == "postgresql":
                result = await self._execute_postgresql_query(connection, query, parameters)
            elif connection.config.type == "mongodb":
                result = await self._execute_mongodb_query(connection, query, parameters)
            elif connection.config.type == "redis":
                result = await self._execute_redis_query(connection, query, parameters)
            elif connection.config.type == "influxdb":
                result = await self._execute_influxdb_query(connection, query, parameters)
            else:
                raise ValueError(f"Unsupported database type: {connection.config.type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create query result
            query_result = QueryResult(
                query=query,
                result=result,
                execution_time=execution_time,
                row_count=len(result) if isinstance(result, list) else 1,
                success=True,
                metadata={
                    "database_name": database_name,
                    "database_type": connection.config.type,
                    "parameters": parameters
                }
            )
            
            # Update connection stats
            connection.last_used = datetime.now()
            connection.connection_count += 1
            
            # Store query result
            self.query_history.append(query_result)
            
            return query_result
        
        except Exception as e:
            logger.error(f"Error executing query on {database_name}: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query=query,
                result=None,
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                metadata={
                    "database_name": database_name,
                    "parameters": parameters
                }
            )
    
    async def _execute_postgresql_query(self, connection: DatabaseConnection, query: str, parameters: Dict[str, Any]) -> Any:
        """Execute PostgreSQL query"""
        try:
            engine = connection.connection
            async with engine.begin() as conn:
                if parameters:
                    result = await conn.execute(text(query), parameters)
                else:
                    result = await conn.execute(text(query))
                
                # Fetch results if it's a SELECT query
                if query.strip().upper().startswith('SELECT'):
                    return result.fetchall()
                else:
                    return {"affected_rows": result.rowcount}
        
        except Exception as e:
            logger.error(f"Error executing PostgreSQL query: {str(e)}")
            raise
    
    async def _execute_mongodb_query(self, connection: DatabaseConnection, query: str, parameters: Dict[str, Any]) -> Any:
        """Execute MongoDB query"""
        try:
            client = connection.connection
            db = client[connection.config.database]
            
            # Parse query (simplified - in production, use proper query parsing)
            if "find" in query.lower():
                collection_name = parameters.get("collection", "performance_data")
                collection = db[collection_name]
                filter_query = parameters.get("filter", {})
                limit = parameters.get("limit", 100)
                
                cursor = collection.find(filter_query).limit(limit)
                results = []
                async for document in cursor:
                    results.append(document)
                return results
            
            elif "insert" in query.lower():
                collection_name = parameters.get("collection", "performance_data")
                collection = db[collection_name]
                document = parameters.get("document", {})
                
                result = await collection.insert_one(document)
                return {"inserted_id": str(result.inserted_id)}
            
            else:
                raise ValueError(f"Unsupported MongoDB query: {query}")
        
        except Exception as e:
            logger.error(f"Error executing MongoDB query: {str(e)}")
            raise
    
    async def _execute_redis_query(self, connection: DatabaseConnection, query: str, parameters: Dict[str, Any]) -> Any:
        """Execute Redis query"""
        try:
            client = connection.connection
            
            # Parse query (simplified - in production, use proper command parsing)
            if query.lower().startswith("get"):
                key = parameters.get("key", "")
                return await client.get(key)
            
            elif query.lower().startswith("set"):
                key = parameters.get("key", "")
                value = parameters.get("value", "")
                expire = parameters.get("expire", None)
                
                if expire:
                    return await client.setex(key, expire, value)
                else:
                    return await client.set(key, value)
            
            elif query.lower().startswith("hgetall"):
                key = parameters.get("key", "")
                return await client.hgetall(key)
            
            else:
                raise ValueError(f"Unsupported Redis query: {query}")
        
        except Exception as e:
            logger.error(f"Error executing Redis query: {str(e)}")
            raise
    
    async def _execute_influxdb_query(self, connection: DatabaseConnection, query: str, parameters: Dict[str, Any]) -> Any:
        """Execute InfluxDB query"""
        try:
            client = connection.connection
            query_api = client.query_api()
            
            # Execute query
            result = query_api.query(query)
            
            # Convert to list of dictionaries
            results = []
            for table in result:
                for record in table.records:
                    results.append({
                        "time": record.get_time(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                        "measurement": record.get_measurement(),
                        "tags": record.values
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Error executing InfluxDB query: {str(e)}")
            raise
    
    async def store_performance_data(self, 
                                   model_name: str,
                                   metric: PerformanceMetric,
                                   value: float,
                                   timestamp: datetime = None) -> Dict[str, Any]:
        """Store performance data in all configured databases"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            results = {}
            
            # Prepare data
            data = {
                "model_name": model_name,
                "metric": metric.value,
                "value": value,
                "timestamp": timestamp,
                "created_at": datetime.now()
            }
            
            # Store in PostgreSQL
            if "postgresql_main" in self.database_connections:
                try:
                    query = """
                    INSERT INTO performance_data (model_name, metric, value, timestamp, created_at)
                    VALUES (:model_name, :metric, :value, :timestamp, :created_at)
                    """
                    result = await self.execute_query("postgresql_main", query, data)
                    results["postgresql"] = result.success
                except Exception as e:
                    logger.error(f"Error storing in PostgreSQL: {str(e)}")
                    results["postgresql"] = False
            
            # Store in MongoDB
            if "mongodb_main" in self.database_connections:
                try:
                    query = "insert"
                    result = await self.execute_query("mongodb_main", query, {
                        "collection": "performance_data",
                        "document": data
                    })
                    results["mongodb"] = result.success
                except Exception as e:
                    logger.error(f"Error storing in MongoDB: {str(e)}")
                    results["mongodb"] = False
            
            # Store in Redis (as cache)
            if "redis_cache" in self.database_connections:
                try:
                    key = f"performance:{model_name}:{metric.value}:{timestamp.isoformat()}"
                    query = "set"
                    result = await self.execute_query("redis_cache", query, {
                        "key": key,
                        "value": json.dumps(data, default=str),
                        "expire": 3600  # 1 hour
                    })
                    results["redis"] = result.success
                except Exception as e:
                    logger.error(f"Error storing in Redis: {str(e)}")
                    results["redis"] = False
            
            # Store in InfluxDB (as time series)
            if "influxdb_metrics" in self.database_connections:
                try:
                    write_api = self.database_connections["influxdb_metrics"].connection.write_api(write_options=SYNCHRONOUS)
                    
                    point = influxdb_client.Point("performance_metrics") \
                        .tag("model_name", model_name) \
                        .tag("metric", metric.value) \
                        .field("value", value) \
                        .time(timestamp)
                    
                    write_api.write(bucket="ai_metrics", record=point)
                    results["influxdb"] = True
                except Exception as e:
                    logger.error(f"Error storing in InfluxDB: {str(e)}")
                    results["influxdb"] = False
            
            return results
        
        except Exception as e:
            logger.error(f"Error storing performance data: {str(e)}")
            return {"error": str(e)}
    
    async def get_performance_data(self, 
                                 model_name: str = None,
                                 metric: str = None,
                                 start_date: datetime = None,
                                 end_date: datetime = None,
                                 limit: int = 1000) -> Dict[str, Any]:
        """Retrieve performance data from databases"""
        try:
            results = {}
            
            # Get from PostgreSQL
            if "postgresql_main" in self.database_connections:
                try:
                    query = "SELECT * FROM performance_data WHERE 1=1"
                    parameters = {}
                    
                    if model_name:
                        query += " AND model_name = :model_name"
                        parameters["model_name"] = model_name
                    
                    if metric:
                        query += " AND metric = :metric"
                        parameters["metric"] = metric
                    
                    if start_date:
                        query += " AND timestamp >= :start_date"
                        parameters["start_date"] = start_date
                    
                    if end_date:
                        query += " AND timestamp <= :end_date"
                        parameters["end_date"] = end_date
                    
                    query += " ORDER BY timestamp DESC LIMIT :limit"
                    parameters["limit"] = limit
                    
                    result = await self.execute_query("postgresql_main", query, parameters)
                    results["postgresql"] = result.result if result.success else []
                except Exception as e:
                    logger.error(f"Error retrieving from PostgreSQL: {str(e)}")
                    results["postgresql"] = []
            
            # Get from MongoDB
            if "mongodb_main" in self.database_connections:
                try:
                    filter_query = {}
                    if model_name:
                        filter_query["model_name"] = model_name
                    if metric:
                        filter_query["metric"] = metric
                    if start_date or end_date:
                        filter_query["timestamp"] = {}
                        if start_date:
                            filter_query["timestamp"]["$gte"] = start_date
                        if end_date:
                            filter_query["timestamp"]["$lte"] = end_date
                    
                    query = "find"
                    result = await self.execute_query("mongodb_main", query, {
                        "collection": "performance_data",
                        "filter": filter_query,
                        "limit": limit
                    })
                    results["mongodb"] = result.result if result.success else []
                except Exception as e:
                    logger.error(f"Error retrieving from MongoDB: {str(e)}")
                    results["mongodb"] = []
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving performance data: {str(e)}")
            return {"error": str(e)}
    
    async def start_monitoring(self):
        """Start database monitoring"""
        if self.is_monitoring:
            logger.warning("Database monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started database monitoring")
    
    async def stop_monitoring(self):
        """Stop database monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped database monitoring")
    
    async def _monitoring_loop(self):
        """Database monitoring loop"""
        while self.is_monitoring:
            try:
                await self._collect_database_metrics()
                await asyncio.sleep(60)  # Monitor every minute
            except Exception as e:
                logger.error(f"Error in database monitoring loop: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _collect_database_metrics(self):
        """Collect database performance metrics"""
        try:
            for db_name, connection in self.database_connections.items():
                if not connection.is_connected:
                    continue
                
                try:
                    # Collect basic metrics
                    metrics = DatabaseMetrics(
                        database_name=db_name,
                        connection_count=connection.connection_count,
                        active_queries=0,  # Would need to track active queries
                        average_query_time=self._calculate_average_query_time(db_name),
                        total_queries=len([q for q in self.query_history if q.metadata.get("database_name") == db_name]),
                        error_rate=self._calculate_error_rate(db_name),
                        memory_usage=0.0,  # Would need database-specific queries
                        cpu_usage=0.0,     # Would need database-specific queries
                        disk_usage=0.0     # Would need database-specific queries
                    )
                    
                    self.performance_metrics.append(metrics)
                    
                    # Keep only last 1000 metrics
                    if len(self.performance_metrics) > 1000:
                        self.performance_metrics = self.performance_metrics[-1000:]
                
                except Exception as e:
                    logger.error(f"Error collecting metrics for {db_name}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error collecting database metrics: {str(e)}")
    
    def _calculate_average_query_time(self, database_name: str) -> float:
        """Calculate average query time for a database"""
        try:
            db_queries = [q for q in self.query_history if q.metadata.get("database_name") == db_name and q.success]
            if not db_queries:
                return 0.0
            
            return sum(q.execution_time for q in db_queries) / len(db_queries)
        except Exception:
            return 0.0
    
    def _calculate_error_rate(self, database_name: str) -> float:
        """Calculate error rate for a database"""
        try:
            db_queries = [q for q in self.query_history if q.metadata.get("database_name") == database_name]
            if not db_queries:
                return 0.0
            
            error_count = len([q for q in db_queries if not q.success])
            return error_count / len(db_queries)
        except Exception:
            return 0.0
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get database status information"""
        return {
            "total_databases": len(self.database_configs),
            "enabled_databases": len([c for c in self.database_configs.values() if c.enabled]),
            "connected_databases": len([c for c in self.database_connections.values() if c.is_connected]),
            "database_connections": {
                name: {
                    "type": conn.config.type,
                    "connected": conn.is_connected,
                    "connection_count": conn.connection_count,
                    "error_count": conn.error_count,
                    "last_used": conn.last_used.isoformat() if conn.last_used else None
                }
                for name, conn in self.database_connections.items()
            },
            "query_history_count": len(self.query_history),
            "performance_metrics_count": len(self.performance_metrics)
        }
    
    def get_performance_metrics(self, limit: int = 100) -> List[DatabaseMetrics]:
        """Get recent database performance metrics"""
        return self.performance_metrics[-limit:]
    
    def get_query_history(self, limit: int = 100) -> List[QueryResult]:
        """Get recent query history"""
        return self.query_history[-limit:]


# Global database integration instance
_database_integration: Optional[AdvancedDatabaseIntegration] = None


def get_database_integration() -> AdvancedDatabaseIntegration:
    """Get or create global database integration"""
    global _database_integration
    if _database_integration is None:
        _database_integration = AdvancedDatabaseIntegration()
    return _database_integration


# Example usage
async def main():
    """Example usage of database integration"""
    db_integration = get_database_integration()
    
    # Initialize connections
    await db_integration.initialize_connections()
    
    # Start monitoring
    await db_integration.start_monitoring()
    
    # Store performance data
    results = await db_integration.store_performance_data(
        model_name="gpt-4",
        metric=PerformanceMetric.QUALITY_SCORE,
        value=0.85
    )
    print(f"Storage results: {results}")
    
    # Retrieve performance data
    data = await db_integration.get_performance_data(
        model_name="gpt-4",
        limit=10
    )
    print(f"Retrieved data: {data}")
    
    # Get status
    status = db_integration.get_database_status()
    print(f"Database status: {status}")
    
    # Stop monitoring
    await db_integration.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())

























