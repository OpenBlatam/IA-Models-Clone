"""
🚀 TruthGPT SOTA Persistence Hub - System 5.9 Gold Standard
Unified storage interface for distributed system state.
"""

import sqlite3
import json
import logging
from typing import Any, Optional, Dict
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger("TruthGPT.SOTA.Persistence")

class PersistenceHub:
    """
    Industrial Persistence Layer.
    Handles global state, user preferences, and agentic memory with ACID compliance.
    """

    def __init__(self, db_path: str = "truthgpt_system.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the global system tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS service_registry (
                    service_id INTEGER PRIMARY KEY,
                    name TEXT,
                    status TEXT,
                    metadata TEXT
                )
            """)
            logger.info(f"✓ System Persistence Hub initialized at {self.db_path}")

    def set_state(self, key: str, value: Any):
        """Set a global system state variable."""
        val_str = json.dumps(value)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO global_state (key, value) VALUES (?, ?)", (key, val_str))

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a global system state variable."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT value FROM global_state WHERE key = ?", (key,)).fetchone()
            return json.loads(row[0]) if row else default

    def register_service(self, service_id: int, name: str, status: str, metadata: Dict[str, Any] = None):
        """Register a layer as a formal system service."""
        meta_str = json.dumps(metadata or {})
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO service_registry (service_id, name, status, metadata) VALUES (?, ?, ?, ?)", 
                         (service_id, name, status, meta_str))

# Singleton Instance
persistence_hub = PersistenceHub()
