"""
Servicio async optimizado para gestionar canciones generadas

Usa aiosqlite para operaciones de base de datos asíncronas y orjson para serialización rápida.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import aiosqlite
import orjson

try:
    from config.settings import settings
except ImportError:
    from ..config.settings import settings

logger = logging.getLogger(__name__)


class SongServiceAsync:
    """Servicio async optimizado para gestionar canciones"""
    
    def __init__(self):
        self.db_path = settings.database_url.replace("sqlite:///", "")
        self._connection_pool: Optional[aiosqlite.Connection] = None
    
    async def _get_connection(self) -> aiosqlite.Connection:
        """Obtiene una conexión de la pool (reutiliza conexiones)"""
        if self._connection_pool is None:
            self._connection_pool = await aiosqlite.connect(
                self.db_path,
                check_same_thread=False
            )
            # Optimizaciones de SQLite para máximo rendimiento
            await self._connection_pool.execute("PRAGMA journal_mode=WAL")
            await self._connection_pool.execute("PRAGMA synchronous=NORMAL")
            await self._connection_pool.execute("PRAGMA cache_size=10000")
            await self._connection_pool.execute("PRAGMA temp_store=MEMORY")
            await self._connection_pool.execute("PRAGMA mmap_size=268435456")  # 256MB
            await self._connection_pool.execute("PRAGMA optimize")
        return self._connection_pool
    
    async def _init_database(self) -> None:
        """Inicializa la base de datos con índices optimizados"""
        conn = await self._get_connection()
        
        # Tabla de canciones con índices
        await conn.execute("""
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
        
        # Índices para búsquedas rápidas
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_songs_user_id ON songs(user_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_songs_status ON songs(status)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_songs_created_at ON songs(created_at DESC)
        """)
        
        # Tabla de historial de chat con índice
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT,
                song_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_user_id ON chat_history(user_id, created_at DESC)
        """)
        
        await conn.commit()
        logger.info("Database initialized with optimizations")
    
    async def save_song(
        self,
        song_id: str,
        user_id: Optional[str],
        prompt: str,
        file_path: str,
        metadata: Optional[Dict] = None,
        status: str = "completed"
    ) -> None:
        """Guarda información de una canción (async)"""
        conn = await self._get_connection()
        
        # Usar orjson para serialización rápida
        metadata_json = orjson.dumps(metadata).decode() if metadata else None
        
        await conn.execute("""
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
        
        await conn.commit()
        logger.debug(f"Song saved: {song_id}")
    
    async def get_song(self, song_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene una canción por ID (async)"""
        conn = await self._get_connection()
        
        async with conn.execute(
            "SELECT * FROM songs WHERE song_id = ?", 
            (song_id,)
        ) as cursor:
            row = await cursor.fetchone()
            
            if row:
                song = dict(zip([col[0] for col in cursor.description], row))
                if song.get("metadata"):
                    # Usar orjson para deserialización rápida
                    song["metadata"] = orjson.loads(song["metadata"])
                return song
            
            return None
    
    async def list_songs(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Lista canciones con paginación optimizada (async)"""
        conn = await self._get_connection()
        
        if user_id:
            async with conn.execute("""
                SELECT * FROM songs 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (user_id, limit, offset)) as cursor:
                rows = await cursor.fetchall()
                columns = [col[0] for col in cursor.description]
        else:
            async with conn.execute("""
                SELECT * FROM songs 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset)) as cursor:
                rows = await cursor.fetchall()
                columns = [col[0] for col in cursor.description]
        
        # Deserialización optimizada con orjson
        songs = []
        for row in rows:
            song = dict(zip(columns, row))
            if song.get("metadata"):
                song["metadata"] = orjson.loads(song["metadata"])
            songs.append(song)
        
        return songs
    
    async def delete_song(self, song_id: str) -> bool:
        """Elimina una canción (async)"""
        song = await self.get_song(song_id)
        if not song:
            return False
        
        # Eliminar archivo de audio (no bloqueante)
        file_path = Path(song["file_path"])
        if file_path.exists():
            # Usar unlink en lugar de remove para mejor rendimiento
            file_path.unlink()
        
        # Eliminar de base de datos
        conn = await self._get_connection()
        await conn.execute("DELETE FROM songs WHERE song_id = ?", (song_id,))
        await conn.commit()
        
        logger.info(f"Song deleted: {song_id}")
        return True
    
    async def update_song_status(
        self, 
        song_id: str, 
        status: str, 
        message: Optional[str] = None
    ) -> None:
        """Actualiza el estado de una canción (async)"""
        conn = await self._get_connection()
        await conn.execute("""
            UPDATE songs 
            SET status = ?, updated_at = ?
            WHERE song_id = ?
        """, (status, datetime.now().isoformat(), song_id))
        await conn.commit()
    
    async def save_chat_message(
        self,
        user_id: str,
        message: str,
        response: Optional[str] = None,
        song_id: Optional[str] = None
    ) -> None:
        """Guarda un mensaje del chat (async)"""
        conn = await self._get_connection()
        await conn.execute("""
            INSERT INTO chat_history (user_id, message, response, song_id)
            VALUES (?, ?, ?, ?)
        """, (user_id, message, response, song_id))
        await conn.commit()
    
    async def get_chat_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene el historial de chat (async)"""
        conn = await self._get_connection()
        
        async with conn.execute("""
            SELECT * FROM chat_history 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (user_id, limit)) as cursor:
            rows = await cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    async def close(self) -> None:
        """Cierra la conexión de la pool"""
        if self._connection_pool:
            await self._connection_pool.close()
            self._connection_pool = None


# Instancia global async
_song_service_async: Optional[SongServiceAsync] = None


async def get_song_service_async() -> SongServiceAsync:
    """Obtiene la instancia global del servicio async"""
    global _song_service_async
    if _song_service_async is None:
        _song_service_async = SongServiceAsync()
        await _song_service_async._init_database()
    return _song_service_async

