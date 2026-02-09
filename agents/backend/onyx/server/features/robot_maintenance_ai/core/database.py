"""
Database layer for persisting conversations and maintenance records.
Refactored to use context managers and reduce code duplication.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
from contextlib import contextmanager
import logging

from ..utils.file_helpers import get_iso_timestamp
from ..utils.json_helpers import safe_json_loads, safe_json_dumps_or_empty

logger = logging.getLogger(__name__)


class MaintenanceDatabase:
    """
    SQLite database for storing conversations and maintenance records.
    Refactored to use context managers for connection handling.
    """
    
    def __init__(self, db_path: str = "data/maintenance.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self, row_factory: Optional[type] = None) -> Iterator[sqlite3.Connection]:
        """
        Context manager for database connections.
        
        Args:
            row_factory: Optional row factory (e.g., sqlite3.Row)
        
        Yields:
            Database connection
        """
        conn = sqlite3.connect(self.db_path)
        if row_factory:
            conn.row_factory = row_factory
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    @contextmanager
    def _get_cursor(self, row_factory: Optional[type] = None) -> Iterator[sqlite3.Cursor]:
        """
        Context manager for database cursors.
        
        Args:
            row_factory: Optional row factory (e.g., sqlite3.Row)
        
        Yields:
            Database cursor
        """
        with self._get_connection(row_factory) as conn:
            cursor = conn.cursor()
            yield cursor
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    robot_type TEXT,
                    maintenance_type TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    message_count INTEGER DEFAULT 0
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    metadata TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS maintenance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    robot_type TEXT,
                    maintenance_type TEXT,
                    sensor_data TEXT,
                    prediction TEXT,
                    created_at TEXT
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_robot_type 
                ON conversations(robot_type)
            """)
    
    def save_conversation(
        self,
        conversation_id: str,
        robot_type: Optional[str] = None,
        maintenance_type: Optional[str] = None
    ):
        """Save or update conversation metadata."""
        now = get_iso_timestamp()
        
        with self._get_cursor() as cursor:
            # Check if conversation exists
            cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
            exists = cursor.fetchone()
            
            if exists:
                # Update existing
                cursor.execute("""
                    UPDATE conversations 
                    SET robot_type = ?, maintenance_type = ?, updated_at = ?,
                        message_count = (SELECT COUNT(*) FROM messages WHERE conversation_id = ?)
                    WHERE id = ?
                """, (robot_type, maintenance_type, now, conversation_id, conversation_id))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO conversations 
                    (id, robot_type, maintenance_type, created_at, updated_at, message_count)
                    VALUES (?, ?, ?, ?, ?, 
                        (SELECT COUNT(*) FROM messages WHERE conversation_id = ?))
                """, (conversation_id, robot_type, maintenance_type, now, now, conversation_id))
    
    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save a message to the database."""
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                conversation_id,
                role,
                content,
                safe_json_dumps_or_empty(metadata),
                get_iso_timestamp()
            ))
        
        # Update conversation metadata
        self.save_conversation(conversation_id)
    
    def _row_to_dict(self, row: sqlite3.Row, fields: List[str]) -> Dict[str, Any]:
        """
        Convert a database row to dictionary.
        
        Args:
            row: Database row
            fields: List of field names to extract
        
        Returns:
            Dictionary with row data
        """
        return {field: row[field] for field in fields if field in row.keys()}
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation messages from database."""
        with self._get_cursor(row_factory=sqlite3.Row) as cursor:
            cursor.execute("""
                SELECT role, content, metadata, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """, (conversation_id,))
            
            rows = cursor.fetchall()
            messages = []
            for row in rows:
                messages.append({
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": safe_json_loads(row["metadata"]),
                    "timestamp": row["timestamp"]
                })
        
        return messages
    
    def save_maintenance_record(
        self,
        robot_type: str,
        maintenance_type: str,
        sensor_data: Dict[str, Any],
        prediction: Dict[str, Any]
    ):
        """Save a maintenance prediction record."""
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO maintenance_records 
                (robot_type, maintenance_type, sensor_data, prediction, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                robot_type,
                maintenance_type,
                safe_json_dumps_or_empty(sensor_data),
                safe_json_dumps_or_empty(prediction),
                get_iso_timestamp()
            ))
    
    def get_maintenance_history(
        self,
        robot_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get maintenance history."""
        with self._get_cursor(row_factory=sqlite3.Row) as cursor:
            if robot_type:
                cursor.execute("""
                    SELECT * FROM maintenance_records
                    WHERE robot_type = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (robot_type, limit))
            else:
                cursor.execute("""
                    SELECT * FROM maintenance_records
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            records = []
            for row in rows:
                records.append({
                    "id": row["id"],
                    "robot_type": row["robot_type"],
                    "maintenance_type": row["maintenance_type"],
                    "sensor_data": safe_json_loads(row["sensor_data"], default={}),
                    "prediction": safe_json_loads(row["prediction"], default={}),
                    "created_at": row["created_at"]
                })
        
        return records
    
    def _fetch_conversations(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """
        Helper method to fetch conversations with a query.
        
        Args:
            query: SQL query
            params: Query parameters
        
        Returns:
            List of conversation dictionaries
        """
        with self._get_cursor(row_factory=sqlite3.Row) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            conversations = []
            for row in rows:
                conversations.append({
                    "id": row["id"],
                    "robot_type": row["robot_type"],
                    "maintenance_type": row["maintenance_type"],
                    "created_at": row.get("created_at", ""),
                    "updated_at": row.get("updated_at", ""),
                    "message_count": row.get("message_count", 0)
                })
        
        return conversations
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversations."""
        return self._fetch_conversations("SELECT * FROM conversations ORDER BY created_at DESC")
    
    def get_conversations_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Get conversations within date range."""
        return self._fetch_conversations("""
            SELECT * FROM conversations
            WHERE created_at >= ? AND created_at <= ?
            ORDER BY created_at DESC
        """, (start_date, end_date))
    
    def _fetch_messages(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """
        Helper method to fetch messages with a query.
        
        Args:
            query: SQL query
            params: Query parameters
        
        Returns:
            List of message dictionaries
        """
        with self._get_cursor(row_factory=sqlite3.Row) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                messages.append({
                    "id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": safe_json_loads(row["metadata"]),
                    "timestamp": row["timestamp"]
                })
        
        return messages
    
    def get_all_messages(self) -> List[Dict[str, Any]]:
        """Get all messages."""
        return self._fetch_messages("SELECT * FROM messages ORDER BY timestamp DESC")
    
    def get_messages_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Get messages within date range."""
        return self._fetch_messages("""
            SELECT * FROM messages
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        """, (start_date, end_date))
    
    def get_messages_by_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        return self.get_conversation(conversation_id)
    
    def get_all_maintenance_records(self) -> List[Dict[str, Any]]:
        """Get all maintenance records."""
        return self.get_maintenance_history(limit=10000)
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation and all its messages."""
        with self._get_cursor() as cursor:
            # Delete messages first (foreign key constraint)
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            
            # Delete conversation
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        
        logger.info(f"Deleted conversation {conversation_id}")



