"""
Advanced Data Manager - Sistema avanzado de gestión de datos
"""

import asyncio
import logging
import json
import sqlite3
import aiosqlite
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib
from pathlib import Path
import pickle
import gzip

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Tipos de datos."""
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    STRUCTURED = "structured"
    CACHE = "cache"
    METADATA = "metadata"


class StorageType(Enum):
    """Tipos de almacenamiento."""
    MEMORY = "memory"
    FILE = "file"
    DATABASE = "database"
    CACHE = "cache"
    TEMPORARY = "temporary"


@dataclass
class DataRecord:
    """Registro de datos."""
    record_id: str
    data_type: DataType
    storage_type: StorageType
    key: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    size_bytes: int = 0
    checksum: str = ""


@dataclass
class DataIndex:
    """Índice de datos."""
    index_id: str
    name: str
    fields: List[str]
    unique: bool = False
    created_at: datetime = field(default_factory=datetime.now)


class AdvancedDataManager:
    """
    Sistema avanzado de gestión de datos.
    """
    
    def __init__(self, data_directory: str = "data"):
        """Inicializar gestor de datos avanzado."""
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(exist_ok=True)
        
        # Almacenamiento en memoria
        self.memory_storage: Dict[str, DataRecord] = {}
        self.cache_storage: Dict[str, DataRecord] = {}
        
        # Base de datos
        self.db_path = self.data_directory / "data_manager.db"
        self.db_connection: Optional[aiosqlite.Connection] = None
        
        # Índices
        self.indexes: Dict[str, DataIndex] = {}
        
        # Configuración
        self.max_memory_records = 10000
        self.max_cache_records = 5000
        self.compression_enabled = True
        self.encryption_enabled = False
        
        # Estadísticas
        self.stats = {
            "total_records": 0,
            "memory_records": 0,
            "cache_records": 0,
            "database_records": 0,
            "total_size_bytes": 0,
            "operations_count": 0,
            "start_time": datetime.now()
        }
        
        logger.info("AdvancedDataManager inicializado")
    
    async def initialize(self):
        """Inicializar el gestor de datos."""
        try:
            # Inicializar base de datos
            await self._initialize_database()
            
            # Cargar datos existentes
            await self._load_existing_data()
            
            # Iniciar limpieza automática
            asyncio.create_task(self._periodic_cleanup())
            
            logger.info("AdvancedDataManager inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar AdvancedDataManager: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el gestor de datos."""
        try:
            # Guardar datos en memoria
            await self._save_memory_data()
            
            # Cerrar conexión de base de datos
            if self.db_connection:
                await self.db_connection.close()
            
            logger.info("AdvancedDataManager cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar AdvancedDataManager: {e}")
    
    async def _initialize_database(self):
        """Inicializar base de datos."""
        try:
            self.db_connection = await aiosqlite.connect(self.db_path)
            
            # Crear tabla principal
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS data_records (
                    record_id TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    storage_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value BLOB,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    checksum TEXT
                )
            """)
            
            # Crear índices
            await self.db_connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_key ON data_records(key)
            """)
            await self.db_connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_type ON data_records(data_type)
            """)
            await self.db_connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON data_records(created_at)
            """)
            
            await self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error al inicializar base de datos: {e}")
            raise
    
    async def _load_existing_data(self):
        """Cargar datos existentes."""
        try:
            # Cargar desde base de datos
            cursor = await self.db_connection.execute("""
                SELECT * FROM data_records WHERE storage_type = 'cache'
            """)
            
            rows = await cursor.fetchall()
            for row in rows:
                record = DataRecord(
                    record_id=row[0],
                    data_type=DataType(row[1]),
                    storage_type=StorageType(row[2]),
                    key=row[3],
                    value=self._deserialize_value(row[4], row[1]),
                    metadata=json.loads(row[5]) if row[5] else {},
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    expires_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    size_bytes=row[9],
                    checksum=row[10]
                )
                
                self.cache_storage[record.key] = record
            
            self.stats["cache_records"] = len(self.cache_storage)
            self.stats["database_records"] = len(rows)
            
        except Exception as e:
            logger.error(f"Error al cargar datos existentes: {e}")
    
    async def store(
        self,
        key: str,
        value: Any,
        data_type: DataType = DataType.TEXT,
        storage_type: StorageType = StorageType.MEMORY,
        metadata: Dict[str, Any] = None,
        expires_in: Optional[timedelta] = None
    ) -> str:
        """Almacenar datos."""
        try:
            record_id = str(uuid.uuid4())
            now = datetime.now()
            expires_at = now + expires_in if expires_in else None
            
            # Calcular tamaño y checksum
            serialized_value = self._serialize_value(value, data_type)
            size_bytes = len(serialized_value)
            checksum = self._calculate_checksum(serialized_value)
            
            # Crear registro
            record = DataRecord(
                record_id=record_id,
                data_type=data_type,
                storage_type=storage_type,
                key=key,
                value=value,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                size_bytes=size_bytes,
                checksum=checksum
            )
            
            # Almacenar según el tipo
            if storage_type == StorageType.MEMORY:
                await self._store_in_memory(record, serialized_value)
            elif storage_type == StorageType.CACHE:
                await self._store_in_cache(record, serialized_value)
            elif storage_type == StorageType.DATABASE:
                await self._store_in_database(record, serialized_value)
            elif storage_type == StorageType.FILE:
                await self._store_in_file(record, serialized_value)
            
            self.stats["operations_count"] += 1
            
            logger.debug(f"Datos almacenados: {key} ({storage_type.value})")
            return record_id
            
        except Exception as e:
            logger.error(f"Error al almacenar datos: {e}")
            raise
    
    async def retrieve(self, key: str, storage_type: Optional[StorageType] = None) -> Optional[Any]:
        """Recuperar datos."""
        try:
            # Buscar en memoria primero
            if storage_type is None or storage_type == StorageType.MEMORY:
                if key in self.memory_storage:
                    record = self.memory_storage[key]
                    if not self._is_expired(record):
                        return record.value
            
            # Buscar en cache
            if storage_type is None or storage_type == StorageType.CACHE:
                if key in self.cache_storage:
                    record = self.cache_storage[key]
                    if not self._is_expired(record):
                        return record.value
            
            # Buscar en base de datos
            if storage_type is None or storage_type == StorageType.DATABASE:
                record = await self._retrieve_from_database(key)
                if record and not self._is_expired(record):
                    return record.value
            
            # Buscar en archivo
            if storage_type is None or storage_type == StorageType.FILE:
                record = await self._retrieve_from_file(key)
                if record and not self._is_expired(record):
                    return record.value
            
            return None
            
        except Exception as e:
            logger.error(f"Error al recuperar datos: {e}")
            return None
    
    async def delete(self, key: str, storage_type: Optional[StorageType] = None) -> bool:
        """Eliminar datos."""
        try:
            deleted = False
            
            # Eliminar de memoria
            if storage_type is None or storage_type == StorageType.MEMORY:
                if key in self.memory_storage:
                    del self.memory_storage[key]
                    self.stats["memory_records"] -= 1
                    deleted = True
            
            # Eliminar de cache
            if storage_type is None or storage_type == StorageType.CACHE:
                if key in self.cache_storage:
                    del self.cache_storage[key]
                    self.stats["cache_records"] -= 1
                    deleted = True
            
            # Eliminar de base de datos
            if storage_type is None or storage_type == StorageType.DATABASE:
                if await self._delete_from_database(key):
                    deleted = True
            
            # Eliminar archivo
            if storage_type is None or storage_type == StorageType.FILE:
                if await self._delete_from_file(key):
                    deleted = True
            
            if deleted:
                self.stats["operations_count"] += 1
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error al eliminar datos: {e}")
            return False
    
    async def exists(self, key: str, storage_type: Optional[StorageType] = None) -> bool:
        """Verificar si existe un dato."""
        try:
            # Verificar en memoria
            if storage_type is None or storage_type == StorageType.MEMORY:
                if key in self.memory_storage:
                    return not self._is_expired(self.memory_storage[key])
            
            # Verificar en cache
            if storage_type is None or storage_type == StorageType.CACHE:
                if key in self.cache_storage:
                    return not self._is_expired(self.cache_storage[key])
            
            # Verificar en base de datos
            if storage_type is None or storage_type == StorageType.DATABASE:
                record = await self._retrieve_from_database(key)
                if record and not self._is_expired(record):
                    return True
            
            # Verificar archivo
            if storage_type is None or storage_type == StorageType.FILE:
                record = await self._retrieve_from_file(key)
                if record and not self._is_expired(record):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error al verificar existencia de datos: {e}")
            return False
    
    async def list_keys(
        self,
        pattern: Optional[str] = None,
        storage_type: Optional[StorageType] = None,
        limit: int = 100
    ) -> List[str]:
        """Listar claves."""
        try:
            keys = set()
            
            # Listar de memoria
            if storage_type is None or storage_type == StorageType.MEMORY:
                for key in self.memory_storage.keys():
                    if not pattern or pattern in key:
                        keys.add(key)
            
            # Listar de cache
            if storage_type is None or storage_type == StorageType.CACHE:
                for key in self.cache_storage.keys():
                    if not pattern or pattern in key:
                        keys.add(key)
            
            # Listar de base de datos
            if storage_type is None or storage_type == StorageType.DATABASE:
                db_keys = await self._list_database_keys(pattern)
                keys.update(db_keys)
            
            # Listar archivos
            if storage_type is None or storage_type == StorageType.FILE:
                file_keys = await self._list_file_keys(pattern)
                keys.update(file_keys)
            
            return list(keys)[:limit]
            
        except Exception as e:
            logger.error(f"Error al listar claves: {e}")
            return []
    
    async def search(
        self,
        query: str,
        data_type: Optional[DataType] = None,
        storage_type: Optional[StorageType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Buscar datos."""
        try:
            results = []
            
            # Buscar en memoria
            if storage_type is None or storage_type == StorageType.MEMORY:
                for key, record in self.memory_storage.items():
                    if self._matches_search(record, query, data_type):
                        results.append({
                            "key": key,
                            "value": record.value,
                            "data_type": record.data_type.value,
                            "storage_type": record.storage_type.value,
                            "created_at": record.created_at.isoformat(),
                            "metadata": record.metadata
                        })
            
            # Buscar en cache
            if storage_type is None or storage_type == StorageType.CACHE:
                for key, record in self.cache_storage.items():
                    if self._matches_search(record, query, data_type):
                        results.append({
                            "key": key,
                            "value": record.value,
                            "data_type": record.data_type.value,
                            "storage_type": record.storage_type.value,
                            "created_at": record.created_at.isoformat(),
                            "metadata": record.metadata
                        })
            
            # Buscar en base de datos
            if storage_type is None or storage_type == StorageType.DATABASE:
                db_results = await self._search_database(query, data_type)
                results.extend(db_results)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error al buscar datos: {e}")
            return []
    
    # Métodos de almacenamiento específicos
    async def _store_in_memory(self, record: DataRecord, serialized_value: bytes):
        """Almacenar en memoria."""
        self.memory_storage[record.key] = record
        self.stats["memory_records"] += 1
        self.stats["total_size_bytes"] += record.size_bytes
        
        # Limitar registros en memoria
        if len(self.memory_storage) > self.max_memory_records:
            # Eliminar el más antiguo
            oldest_key = min(
                self.memory_storage.keys(),
                key=lambda k: self.memory_storage[k].created_at
            )
            del self.memory_storage[oldest_key]
            self.stats["memory_records"] -= 1
    
    async def _store_in_cache(self, record: DataRecord, serialized_value: bytes):
        """Almacenar en cache."""
        self.cache_storage[record.key] = record
        self.stats["cache_records"] += 1
        
        # También almacenar en base de datos para persistencia
        await self._store_in_database(record, serialized_value)
        
        # Limitar registros en cache
        if len(self.cache_storage) > self.max_cache_records:
            # Eliminar el más antiguo
            oldest_key = min(
                self.cache_storage.keys(),
                key=lambda k: self.cache_storage[k].created_at
            )
            del self.cache_storage[oldest_key]
            self.stats["cache_records"] -= 1
    
    async def _store_in_database(self, record: DataRecord, serialized_value: bytes):
        """Almacenar en base de datos."""
        try:
            await self.db_connection.execute("""
                INSERT OR REPLACE INTO data_records 
                (record_id, data_type, storage_type, key, value, metadata, 
                 created_at, updated_at, expires_at, size_bytes, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.record_id,
                record.data_type.value,
                record.storage_type.value,
                record.key,
                serialized_value,
                json.dumps(record.metadata),
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.expires_at.isoformat() if record.expires_at else None,
                record.size_bytes,
                record.checksum
            ))
            
            await self.db_connection.commit()
            self.stats["database_records"] += 1
            
        except Exception as e:
            logger.error(f"Error al almacenar en base de datos: {e}")
            raise
    
    async def _store_in_file(self, record: DataRecord, serialized_value: bytes):
        """Almacenar en archivo."""
        try:
            file_path = self.data_directory / "files" / f"{record.key}.dat"
            file_path.parent.mkdir(exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(serialized_value)
            
            # Almacenar metadatos
            metadata_path = self.data_directory / "files" / f"{record.key}.meta"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "record_id": record.record_id,
                    "data_type": record.data_type.value,
                    "storage_type": record.storage_type.value,
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat(),
                    "expires_at": record.expires_at.isoformat() if record.expires_at else None,
                    "size_bytes": record.size_bytes,
                    "checksum": record.checksum,
                    "metadata": record.metadata
                }, f)
            
        except Exception as e:
            logger.error(f"Error al almacenar en archivo: {e}")
            raise
    
    async def _retrieve_from_database(self, key: str) -> Optional[DataRecord]:
        """Recuperar de base de datos."""
        try:
            cursor = await self.db_connection.execute("""
                SELECT * FROM data_records WHERE key = ?
            """, (key,))
            
            row = await cursor.fetchone()
            if row:
                return DataRecord(
                    record_id=row[0],
                    data_type=DataType(row[1]),
                    storage_type=StorageType(row[2]),
                    key=row[3],
                    value=self._deserialize_value(row[4], row[1]),
                    metadata=json.loads(row[5]) if row[5] else {},
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    expires_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    size_bytes=row[9],
                    checksum=row[10]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error al recuperar de base de datos: {e}")
            return None
    
    async def _retrieve_from_file(self, key: str) -> Optional[DataRecord]:
        """Recuperar de archivo."""
        try:
            file_path = self.data_directory / "files" / f"{key}.dat"
            metadata_path = self.data_directory / "files" / f"{key}.meta"
            
            if not file_path.exists() or not metadata_path.exists():
                return None
            
            # Leer metadatos
            with open(metadata_path, 'r') as f:
                metadata_data = json.load(f)
            
            # Leer datos
            with open(file_path, 'rb') as f:
                serialized_value = f.read()
            
            return DataRecord(
                record_id=metadata_data["record_id"],
                data_type=DataType(metadata_data["data_type"]),
                storage_type=StorageType(metadata_data["storage_type"]),
                key=key,
                value=self._deserialize_value(serialized_value, metadata_data["data_type"]),
                metadata=metadata_data["metadata"],
                created_at=datetime.fromisoformat(metadata_data["created_at"]),
                updated_at=datetime.fromisoformat(metadata_data["updated_at"]),
                expires_at=datetime.fromisoformat(metadata_data["expires_at"]) if metadata_data["expires_at"] else None,
                size_bytes=metadata_data["size_bytes"],
                checksum=metadata_data["checksum"]
            )
            
        except Exception as e:
            logger.error(f"Error al recuperar de archivo: {e}")
            return None
    
    async def _delete_from_database(self, key: str) -> bool:
        """Eliminar de base de datos."""
        try:
            cursor = await self.db_connection.execute("""
                DELETE FROM data_records WHERE key = ?
            """, (key,))
            
            await self.db_connection.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error al eliminar de base de datos: {e}")
            return False
    
    async def _delete_from_file(self, key: str) -> bool:
        """Eliminar archivo."""
        try:
            file_path = self.data_directory / "files" / f"{key}.dat"
            metadata_path = self.data_directory / "files" / f"{key}.meta"
            
            deleted = False
            if file_path.exists():
                file_path.unlink()
                deleted = True
            
            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error al eliminar archivo: {e}")
            return False
    
    async def _list_database_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Listar claves de base de datos."""
        try:
            if pattern:
                cursor = await self.db_connection.execute("""
                    SELECT key FROM data_records WHERE key LIKE ?
                """, (f"%{pattern}%",))
            else:
                cursor = await self.db_connection.execute("""
                    SELECT key FROM data_records
                """)
            
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
            
        except Exception as e:
            logger.error(f"Error al listar claves de base de datos: {e}")
            return []
    
    async def _list_file_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Listar claves de archivos."""
        try:
            files_dir = self.data_directory / "files"
            if not files_dir.exists():
                return []
            
            keys = []
            for file_path in files_dir.glob("*.dat"):
                key = file_path.stem
                if not pattern or pattern in key:
                    keys.append(key)
            
            return keys
            
        except Exception as e:
            logger.error(f"Error al listar claves de archivos: {e}")
            return []
    
    async def _search_database(self, query: str, data_type: Optional[DataType] = None) -> List[Dict[str, Any]]:
        """Buscar en base de datos."""
        try:
            sql = "SELECT * FROM data_records WHERE key LIKE ?"
            params = [f"%{query}%"]
            
            if data_type:
                sql += " AND data_type = ?"
                params.append(data_type.value)
            
            cursor = await self.db_connection.execute(sql, params)
            rows = await cursor.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "key": row[3],
                    "value": self._deserialize_value(row[4], row[1]),
                    "data_type": row[1],
                    "storage_type": row[2],
                    "created_at": row[6],
                    "metadata": json.loads(row[5]) if row[5] else {}
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error al buscar en base de datos: {e}")
            return []
    
    # Métodos auxiliares
    def _serialize_value(self, value: Any, data_type: DataType) -> bytes:
        """Serializar valor."""
        try:
            if data_type == DataType.TEXT:
                return str(value).encode('utf-8')
            elif data_type == DataType.JSON:
                return json.dumps(value).encode('utf-8')
            elif data_type == DataType.BINARY:
                return pickle.dumps(value)
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Error al serializar valor: {e}")
            return b""
    
    def _deserialize_value(self, serialized_value: bytes, data_type: str) -> Any:
        """Deserializar valor."""
        try:
            if data_type == DataType.TEXT.value:
                return serialized_value.decode('utf-8')
            elif data_type == DataType.JSON.value:
                return json.loads(serialized_value.decode('utf-8'))
            elif data_type == DataType.BINARY.value:
                return pickle.loads(serialized_value)
            else:
                return pickle.loads(serialized_value)
        except Exception as e:
            logger.error(f"Error al deserializar valor: {e}")
            return None
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calcular checksum."""
        return hashlib.md5(data).hexdigest()
    
    def _is_expired(self, record: DataRecord) -> bool:
        """Verificar si un registro ha expirado."""
        if record.expires_at is None:
            return False
        return datetime.now() > record.expires_at
    
    def _matches_search(self, record: DataRecord, query: str, data_type: Optional[DataType]) -> bool:
        """Verificar si un registro coincide con la búsqueda."""
        if data_type and record.data_type != data_type:
            return False
        
        # Buscar en la clave
        if query.lower() in record.key.lower():
            return True
        
        # Buscar en el valor si es texto
        if record.data_type == DataType.TEXT and isinstance(record.value, str):
            if query.lower() in record.value.lower():
                return True
        
        return False
    
    async def _save_memory_data(self):
        """Guardar datos en memoria."""
        try:
            for key, record in self.memory_storage.items():
                serialized_value = self._serialize_value(record.value, record.data_type)
                await self._store_in_database(record, serialized_value)
        except Exception as e:
            logger.error(f"Error al guardar datos en memoria: {e}")
    
    async def _periodic_cleanup(self):
        """Limpieza periódica."""
        while True:
            try:
                await self._cleanup_expired_records()
                await asyncio.sleep(3600)  # Cada hora
            except Exception as e:
                logger.error(f"Error en limpieza periódica: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_expired_records(self):
        """Limpiar registros expirados."""
        try:
            now = datetime.now()
            
            # Limpiar memoria
            expired_keys = []
            for key, record in self.memory_storage.items():
                if self._is_expired(record):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_storage[key]
                self.stats["memory_records"] -= 1
            
            # Limpiar cache
            expired_keys = []
            for key, record in self.cache_storage.items():
                if self._is_expired(record):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache_storage[key]
                self.stats["cache_records"] -= 1
            
            # Limpiar base de datos
            await self.db_connection.execute("""
                DELETE FROM data_records WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (now.isoformat(),))
            
            await self.db_connection.commit()
            
            if expired_keys:
                logger.info(f"Limpiados {len(expired_keys)} registros expirados")
                
        except Exception as e:
            logger.error(f"Error en limpieza de registros expirados: {e}")
    
    async def get_data_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de datos."""
        return {
            **self.stats,
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "memory_usage_mb": self.stats["total_size_bytes"] / (1024 * 1024),
            "compression_enabled": self.compression_enabled,
            "encryption_enabled": self.encryption_enabled,
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del gestor de datos."""
        try:
            return {
                "status": "healthy",
                "total_records": self.stats["total_records"],
                "memory_records": self.stats["memory_records"],
                "cache_records": self.stats["cache_records"],
                "database_records": self.stats["database_records"],
                "total_size_mb": self.stats["total_size_bytes"] / (1024 * 1024),
                "operations_count": self.stats["operations_count"],
                "database_connected": self.db_connection is not None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check del gestor de datos: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




