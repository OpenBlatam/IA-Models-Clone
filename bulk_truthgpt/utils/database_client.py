"""
Database Client
==============

Advanced database client for the Bulk TruthGPT system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import asyncpg
from asyncpg import Pool, Connection

logger = logging.getLogger(__name__)

class DatabaseClient:
    """
    Advanced database client with connection pooling.
    
    Features:
    - Connection pooling
    - Automatic retry
    - Transaction support
    - Query optimization
    - Performance monitoring
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "bulk_truthgpt",
        user: str = "postgres",
        password: str = "password",
        min_connections: int = 5,
        max_connections: int = 20
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections
        
        self.pool: Optional[Pool] = None
        self.connected = False
        
    async def initialize(self):
        """Initialize database connection."""
        logger.info("Initializing Database Client...")
        
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_connections,
                max_size=self.max_connections
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            self.connected = True
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info("Database Client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Database Client: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create database tables."""
        try:
            async with self.pool.acquire() as conn:
                # Tasks table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id VARCHAR(255) PRIMARY KEY,
                        request JSONB NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        priority INTEGER NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        started_at TIMESTAMP WITH TIME ZONE,
                        completed_at TIMESTAMP WITH TIME ZONE,
                        progress FLOAT DEFAULT 0.0,
                        documents_generated INTEGER DEFAULT 0,
                        errors JSONB DEFAULT '[]'::jsonb,
                        metadata JSONB DEFAULT '{}'::jsonb
                    )
                """)
                
                # Documents table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id VARCHAR(255) PRIMARY KEY,
                        task_id VARCHAR(255) NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        analysis JSONB DEFAULT '{}'::jsonb,
                        optimization JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        updated_at TIMESTAMP WITH TIME ZONE,
                        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                    )
                """)
                
                # Metrics table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        metric_id SERIAL PRIMARY KEY,
                        task_id VARCHAR(255),
                        metric_name VARCHAR(255) NOT NULL,
                        metric_value FLOAT NOT NULL,
                        metric_data JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                    )
                """)
                
                # Create indexes
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_task_id ON documents(task_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_task_id ON metrics(task_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)")
                
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    async def store_task(self, task_id: str, task_data: Dict[str, Any]) -> bool:
        """Store task in database."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO tasks (task_id, request, status, priority, created_at, started_at, completed_at, progress, documents_generated, errors, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (task_id) DO UPDATE SET
                        request = EXCLUDED.request,
                        status = EXCLUDED.status,
                        priority = EXCLUDED.priority,
                        started_at = EXCLUDED.started_at,
                        completed_at = EXCLUDED.completed_at,
                        progress = EXCLUDED.progress,
                        documents_generated = EXCLUDED.documents_generated,
                        errors = EXCLUDED.errors,
                        metadata = EXCLUDED.metadata
                """, 
                task_id,
                json.dumps(task_data.get("request", {})),
                task_data.get("status", "created"),
                task_data.get("priority", 5),
                task_data.get("created_at"),
                task_data.get("started_at"),
                task_data.get("completed_at"),
                task_data.get("progress", 0.0),
                task_data.get("documents_generated", 0),
                json.dumps(task_data.get("errors", [])),
                json.dumps(task_data.get("metadata", {}))
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store task {task_id}: {str(e)}")
            return False
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task from database."""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM tasks WHERE task_id = $1
                """, task_id)
                
                if not row:
                    return None
                
                return {
                    "task_id": row["task_id"],
                    "request": json.loads(row["request"]),
                    "status": row["status"],
                    "priority": row["priority"],
                    "created_at": row["created_at"].isoformat(),
                    "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                    "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                    "progress": float(row["progress"]),
                    "documents_generated": row["documents_generated"],
                    "errors": json.loads(row["errors"]),
                    "metadata": json.loads(row["metadata"])
                }
                
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {str(e)}")
            return None
    
    async def get_tasks(
        self, 
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get tasks from database."""
        try:
            async with self.pool.acquire() as conn:
                if status:
                    rows = await conn.fetch("""
                        SELECT * FROM tasks 
                        WHERE status = $1 
                        ORDER BY created_at DESC 
                        LIMIT $2 OFFSET $3
                    """, status, limit, offset)
                else:
                    rows = await conn.fetch("""
                        SELECT * FROM tasks 
                        ORDER BY created_at DESC 
                        LIMIT $1 OFFSET $2
                    """, limit, offset)
                
                tasks = []
                for row in rows:
                    tasks.append({
                        "task_id": row["task_id"],
                        "request": json.loads(row["request"]),
                        "status": row["status"],
                        "priority": row["priority"],
                        "created_at": row["created_at"].isoformat(),
                        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                        "progress": float(row["progress"]),
                        "documents_generated": row["documents_generated"],
                        "errors": json.loads(row["errors"]),
                        "metadata": json.loads(row["metadata"])
                    })
                
                return tasks
                
        except Exception as e:
            logger.error(f"Failed to get tasks: {str(e)}")
            return []
    
    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update task in database."""
        try:
            async with self.pool.acquire() as conn:
                # Build dynamic update query
                set_clauses = []
                values = []
                param_count = 1
                
                for key, value in updates.items():
                    if key in ["request", "errors", "metadata"]:
                        set_clauses.append(f"{key} = ${param_count}")
                        values.append(json.dumps(value))
                    else:
                        set_clauses.append(f"{key} = ${param_count}")
                        values.append(value)
                    param_count += 1
                
                values.append(task_id)
                
                query = f"""
                    UPDATE tasks 
                    SET {', '.join(set_clauses)}
                    WHERE task_id = ${param_count}
                """
                
                await conn.execute(query, *values)
                return True
                
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {str(e)}")
            return False
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete task from database."""
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Delete related documents
                    await conn.execute("DELETE FROM documents WHERE task_id = $1", task_id)
                    
                    # Delete related metrics
                    await conn.execute("DELETE FROM metrics WHERE task_id = $1", task_id)
                    
                    # Delete task
                    result = await conn.execute("DELETE FROM tasks WHERE task_id = $1", task_id)
                    
                    return result == "DELETE 1"
                    
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {str(e)}")
            return False
    
    async def store_document(self, document_id: str, document_data: Dict[str, Any]) -> bool:
        """Store document in database."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO documents (document_id, task_id, content, metadata, analysis, optimization, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (document_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        analysis = EXCLUDED.analysis,
                        optimization = EXCLUDED.optimization,
                        updated_at = EXCLUDED.updated_at
                """,
                document_id,
                document_data.get("task_id"),
                document_data.get("content", ""),
                json.dumps(document_data.get("metadata", {})),
                json.dumps(document_data.get("analysis", {})),
                json.dumps(document_data.get("optimization", {})),
                document_data.get("created_at"),
                datetime.utcnow().isoformat()
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store document {document_id}: {str(e)}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document from database."""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM documents WHERE document_id = $1
                """, document_id)
                
                if not row:
                    return None
                
                return {
                    "document_id": row["document_id"],
                    "task_id": row["task_id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]),
                    "analysis": json.loads(row["analysis"]),
                    "optimization": json.loads(row["optimization"]),
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
                }
                
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None
    
    async def get_task_documents(
        self, 
        task_id: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get documents for a task."""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM documents 
                    WHERE task_id = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2 OFFSET $3
                """, task_id, limit, offset)
                
                documents = []
                for row in rows:
                    documents.append({
                        "document_id": row["document_id"],
                        "task_id": row["task_id"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"]),
                        "analysis": json.loads(row["analysis"]),
                        "optimization": json.loads(row["optimization"]),
                        "created_at": row["created_at"].isoformat(),
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"Failed to get task documents: {str(e)}")
            return []
    
    async def search_documents(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search documents."""
        try:
            async with self.pool.acquire() as conn:
                # Build search query
                search_query = """
                    SELECT * FROM documents 
                    WHERE content ILIKE $1
                """
                params = [f"%{query}%"]
                param_count = 2
                
                if filters:
                    for key, value in filters.items():
                        if key == "task_id":
                            search_query += f" AND task_id = ${param_count}"
                            params.append(value)
                            param_count += 1
                        elif key == "created_after":
                            search_query += f" AND created_at >= ${param_count}"
                            params.append(value)
                            param_count += 1
                        elif key == "created_before":
                            search_query += f" AND created_at <= ${param_count}"
                            params.append(value)
                            param_count += 1
                
                search_query += f" ORDER BY created_at DESC LIMIT ${param_count}"
                params.append(limit)
                
                rows = await conn.fetch(search_query, *params)
                
                documents = []
                for row in rows:
                    documents.append({
                        "document_id": row["document_id"],
                        "task_id": row["task_id"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"]),
                        "analysis": json.loads(row["analysis"]),
                        "optimization": json.loads(row["optimization"]),
                        "created_at": row["created_at"].isoformat(),
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}")
            return []
    
    async def store_metric(
        self, 
        task_id: str, 
        metric_name: str, 
        metric_value: float,
        metric_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store metric in database."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO metrics (task_id, metric_name, metric_value, metric_data, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                task_id,
                metric_name,
                metric_value,
                json.dumps(metric_data or {}),
                datetime.utcnow()
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store metric: {str(e)}")
            return False
    
    async def get_metrics(
        self, 
        task_id: Optional[str] = None,
        metric_name: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get metrics from database."""
        try:
            async with self.pool.acquire() as conn:
                query = "SELECT * FROM metrics WHERE 1=1"
                params = []
                param_count = 1
                
                if task_id:
                    query += f" AND task_id = ${param_count}"
                    params.append(task_id)
                    param_count += 1
                
                if metric_name:
                    query += f" AND metric_name = ${param_count}"
                    params.append(metric_name)
                    param_count += 1
                
                query += f" ORDER BY created_at DESC LIMIT ${param_count}"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                metrics = []
                for row in rows:
                    metrics.append({
                        "metric_id": row["metric_id"],
                        "task_id": row["task_id"],
                        "metric_name": row["metric_name"],
                        "metric_value": float(row["metric_value"]),
                        "metric_data": json.loads(row["metric_data"]),
                        "created_at": row["created_at"].isoformat()
                    })
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return []
    
    async def count_tasks(self, status: Optional[str] = None) -> int:
        """Count tasks."""
        try:
            async with self.pool.acquire() as conn:
                if status:
                    result = await conn.fetchval("SELECT COUNT(*) FROM tasks WHERE status = $1", status)
                else:
                    result = await conn.fetchval("SELECT COUNT(*) FROM tasks")
                
                return result or 0
                
        except Exception as e:
            logger.error(f"Failed to count tasks: {str(e)}")
            return 0
    
    async def get_old_tasks(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Get old tasks for cleanup."""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT task_id FROM tasks 
                    WHERE created_at < $1 
                    AND status IN ('completed', 'failed', 'cancelled')
                """, cutoff_time)
                
                return [{"task_id": row["task_id"]} for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get old tasks: {str(e)}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            async with self.pool.acquire() as conn:
                # Task statistics
                task_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_tasks,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks,
                        COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing_tasks
                    FROM tasks
                """)
                
                # Document statistics
                doc_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_documents,
                        AVG(LENGTH(content)) as avg_content_length
                    FROM documents
                """)
                
                # Metric statistics
                metric_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_metrics,
                        COUNT(DISTINCT metric_name) as unique_metrics
                    FROM metrics
                """)
                
                return {
                    "tasks": dict(task_stats) if task_stats else {},
                    "documents": dict(doc_stats) if doc_stats else {},
                    "metrics": dict(metric_stats) if metric_stats else {}
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup database connection."""
        try:
            if self.pool:
                await self.pool.close()
            
            self.connected = False
            logger.info("Database Client cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Database Client: {str(e)}")











