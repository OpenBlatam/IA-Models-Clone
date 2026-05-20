"""
Database - Sistema de persistencia con base de datos
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Gestor de base de datos para persistencia"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Inicializar el gestor de base de datos.

        Args:
            db_path: Ruta al archivo de base de datos
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "addition_removal_ai.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Inicializar esquema de base de datos"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabla de operaciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    content_before TEXT,
                    content_after TEXT,
                    metadata TEXT,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de versiones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY,
                    content_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    operation_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (operation_id) REFERENCES operations(id)
                )
            """)
            
            # Tabla de historial
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    operation_id TEXT,
                    change_type TEXT,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (operation_id) REFERENCES operations(id)
                )
            """)
            
            # Índices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_operations_user 
                ON operations(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_operations_type 
                ON operations(operation_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_versions_content 
                ON versions(content_id)
            """)
            
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager para conexiones de base de datos"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_operation(
        self,
        operation_id: str,
        operation_type: str,
        content_before: str,
        content_after: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ):
        """
        Guardar una operación en la base de datos.

        Args:
            operation_id: ID único de la operación
            operation_type: Tipo de operación
            content_before: Contenido antes
            content_after: Contenido después
            metadata: Metadatos adicionales
            user_id: ID del usuario
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO operations 
                (id, operation_type, content_before, content_after, metadata, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                operation_id,
                operation_type,
                content_before,
                content_after,
                json.dumps(metadata) if metadata else None,
                user_id
            ))
            conn.commit()

    def get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener una operación por ID.

        Args:
            operation_id: ID de la operación

        Returns:
            Diccionario con la operación o None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM operations WHERE id = ?
            """, (operation_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    "id": row["id"],
                    "operation_type": row["operation_type"],
                    "content_before": row["content_before"],
                    "content_after": row["content_after"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    "user_id": row["user_id"],
                    "created_at": row["created_at"]
                }
            return None

    def save_version(
        self,
        content_id: str,
        content: str,
        operation_id: Optional[str] = None
    ) -> str:
        """
        Guardar una versión del contenido.

        Args:
            content_id: ID del contenido
            content: Contenido a guardar
            operation_id: ID de la operación relacionada

        Returns:
            ID de la versión guardada
        """
        import uuid
        version_id = str(uuid.uuid4())
        
        # Obtener número de versión
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(version_number) as max_version 
                FROM versions WHERE content_id = ?
            """, (content_id,))
            row = cursor.fetchone()
            version_number = (row["max_version"] or 0) + 1
            
            cursor.execute("""
                INSERT INTO versions 
                (id, content_id, content, version_number, operation_id)
                VALUES (?, ?, ?, ?, ?)
            """, (version_id, content_id, content, version_number, operation_id))
            conn.commit()
        
        return version_id

    def get_versions(self, content_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener versiones de un contenido.

        Args:
            content_id: ID del contenido
            limit: Número máximo de versiones

        Returns:
            Lista de versiones
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM versions 
                WHERE content_id = ? 
                ORDER BY version_number DESC 
                LIMIT ?
            """, (content_id, limit))
            
            versions = []
            for row in cursor.fetchall():
                versions.append({
                    "id": row["id"],
                    "content_id": row["content_id"],
                    "content": row["content"],
                    "version_number": row["version_number"],
                    "operation_id": row["operation_id"],
                    "created_at": row["created_at"]
                })
            return versions

    def get_recent_operations(
        self,
        user_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Obtener operaciones recientes.

        Args:
            user_id: ID del usuario (opcional)
            limit: Número máximo de operaciones

        Returns:
            Lista de operaciones
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute("""
                    SELECT * FROM operations 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (user_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM operations 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            
            operations = []
            for row in cursor.fetchall():
                operations.append({
                    "id": row["id"],
                    "operation_type": row["operation_type"],
                    "content_before": row["content_before"][:100] + "..." if row["content_before"] and len(row["content_before"]) > 100 else row["content_before"],
                    "content_after": row["content_after"][:100] + "..." if row["content_after"] and len(row["content_after"]) > 100 else row["content_after"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    "user_id": row["user_id"],
                    "created_at": row["created_at"]
                })
            return operations






