"""
Servicio para gestionar canciones generadas
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import sqlite3
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


class SongService:
    """Servicio para gestionar canciones"""
    
    def __init__(self):
        self.db_path = settings.database_url.replace("sqlite:///", "")
        self._init_database()
    
    def _init_database(self):
        """Inicializa la base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabla de canciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS songs (
                    song_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    prompt TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT DEFAULT 'processing',
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de historial de chat
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT,
                    song_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_song(
        self,
        song_id: str,
        user_id: Optional[str],
        prompt: str,
        file_path: str,
        metadata: Optional[Dict] = None,
        status: str = "completed"
    ):
        """Guarda información de una canción"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO songs 
                (song_id, user_id, prompt, file_path, status, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                song_id,
                user_id,
                prompt,
                file_path,
                status,
                metadata_json,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Song saved: {song_id}")
            
        except Exception as e:
            logger.error(f"Error saving song: {e}")
            raise
    
    def get_song(self, song_id: str) -> Optional[Dict]:
        """Obtiene una canción por ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM songs WHERE song_id = ?", (song_id,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                song = dict(row)
                if song.get("metadata"):
                    song["metadata"] = json.loads(song["metadata"])
                return song
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting song: {e}")
            return None
    
    def list_songs(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """Lista canciones"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute("""
                    SELECT * FROM songs 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (user_id, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM songs 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            rows = cursor.fetchall()
            conn.close()
            
            songs = []
            for row in rows:
                song = dict(row)
                if song.get("metadata"):
                    song["metadata"] = json.loads(song["metadata"])
                songs.append(song)
            
            return songs
            
        except Exception as e:
            logger.error(f"Error listing songs: {e}")
            return []
    
    def delete_song(self, song_id: str) -> bool:
        """Elimina una canción"""
        try:
            # Obtener información de la canción
            song = self.get_song(song_id)
            if not song:
                return False
            
            # Eliminar archivo de audio
            file_path = Path(song["file_path"])
            if file_path.exists():
                file_path.unlink()
            
            # Eliminar de base de datos
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM songs WHERE song_id = ?", (song_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Song deleted: {song_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting song: {e}")
            return False
    
    def update_song_status(self, song_id: str, status: str, message: Optional[str] = None):
        """Actualiza el estado de una canción"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE songs 
                SET status = ?, updated_at = ?
                WHERE song_id = ?
            """, (status, datetime.now().isoformat(), song_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Song status updated: {song_id} -> {status}")
            
        except Exception as e:
            logger.error(f"Error updating song status: {e}")
    
    def save_chat_message(
        self,
        user_id: str,
        message: str,
        response: Optional[str] = None,
        song_id: Optional[str] = None
    ):
        """Guarda un mensaje del chat"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chat_history (user_id, message, response, song_id)
                VALUES (?, ?, ?, ?)
            """, (user_id, message, response, song_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving chat message: {e}")
    
    def get_chat_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Obtiene el historial de chat de un usuario"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM chat_history 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []

