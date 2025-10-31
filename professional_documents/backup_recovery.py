"""
Sistema de Backup y Recuperación Avanzado para el Sistema de Documentos Profesionales

Este módulo implementa un sistema completo de backup y recuperación con
encriptación, compresión, versionado, y recuperación granular.
"""

import asyncio
import json
import os
import shutil
import tarfile
import zipfile
import gzip
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import dataclass, asdict
from enum import Enum
import aiofiles
import redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Tipos de backup"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """Estados de backup"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CompressionType(Enum):
    """Tipos de compresión"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    ZIP = "zip"

@dataclass
class BackupConfig:
    """Configuración de backup"""
    id: str
    name: str
    description: str
    backup_type: BackupType
    compression: CompressionType
    encryption: bool
    retention_days: int
    schedule: Optional[str] = None
    enabled: bool = True
    include_database: bool = True
    include_files: bool = True
    include_config: bool = True
    include_logs: bool = False
    max_file_size: int = 1024 * 1024 * 1024  # 1GB
    parallel_jobs: int = 4

@dataclass
class BackupJob:
    """Trabajo de backup"""
    id: str
    config_id: str
    status: BackupStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    progress: float = 0.0
    total_size: int = 0
    compressed_size: int = 0
    files_processed: int = 0
    files_total: int = 0
    error_message: Optional[str] = None
    backup_path: Optional[str] = None
    checksum: Optional[str] = None

@dataclass
class RestoreJob:
    """Trabajo de restauración"""
    id: str
    backup_id: str
    target_path: str
    status: BackupStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    progress: float = 0.0
    files_restored: int = 0
    files_total: int = 0
    error_message: Optional[str] = None

class EncryptionManager:
    """Gestor de encriptación para backups"""
    
    def __init__(self):
        self.encryption_key = self._get_or_create_encryption_key()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Obtener o crear clave de encriptación"""
        try:
            key_path = os.getenv("BACKUP_ENCRYPTION_KEY_PATH", "backup_encryption.key")
            
            if os.path.exists(key_path):
                with open(key_path, "rb") as key_file:
                    return key_file.read()
            else:
                # Crear nueva clave
                key = Fernet.generate_key()
                os.makedirs(os.path.dirname(key_path), exist_ok=True)
                with open(key_path, "wb") as key_file:
                    key_file.write(key)
                return key
        except Exception as e:
            logger.error(f"Error getting encryption key: {e}")
            raise
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encriptar datos"""
        try:
            fernet = Fernet(self.encryption_key)
            return fernet.encrypt(data)
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Desencriptar datos"""
        try:
            fernet = Fernet(self.encryption_key)
            return fernet.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise

class CompressionManager:
    """Gestor de compresión para backups"""
    
    @staticmethod
    def compress_data(data: bytes, compression_type: CompressionType) -> bytes:
        """Comprimir datos"""
        try:
            if compression_type == CompressionType.NONE:
                return data
            elif compression_type == CompressionType.GZIP:
                return gzip.compress(data)
            elif compression_type == CompressionType.BZIP2:
                import bz2
                return bz2.compress(data)
            elif compression_type == CompressionType.LZMA:
                import lzma
                return lzma.compress(data)
            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")
        except Exception as e:
            logger.error(f"Error compressing data: {e}")
            raise
    
    @staticmethod
    def decompress_data(compressed_data: bytes, compression_type: CompressionType) -> bytes:
        """Descomprimir datos"""
        try:
            if compression_type == CompressionType.NONE:
                return compressed_data
            elif compression_type == CompressionType.GZIP:
                return gzip.decompress(compressed_data)
            elif compression_type == CompressionType.BZIP2:
                import bz2
                return bz2.decompress(compressed_data)
            elif compression_type == CompressionType.LZMA:
                import lzma
                return lzma.decompress(compressed_data)
            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")
        except Exception as e:
            logger.error(f"Error decompressing data: {e}")
            raise

class DatabaseBackupManager:
    """Gestor de backup de base de datos"""
    
    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
    
    async def create_database_backup(self, backup_path: str) -> Dict[str, Any]:
        """Crear backup de base de datos"""
        try:
            import subprocess
            
            # Crear directorio de backup
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Ejecutar pg_dump para PostgreSQL
            if "postgresql" in self.db_connection_string:
                cmd = [
                    "pg_dump",
                    "--verbose",
                    "--clean",
                    "--no-owner",
                    "--no-privileges",
                    "--format=custom",
                    "--file", backup_path,
                    self.db_connection_string
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"pg_dump failed: {result.stderr}")
            
            # Obtener información del backup
            backup_size = os.path.getsize(backup_path)
            
            return {
                "backup_path": backup_path,
                "size": backup_size,
                "format": "custom",
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            raise
    
    async def restore_database_backup(self, backup_path: str) -> Dict[str, Any]:
        """Restaurar backup de base de datos"""
        try:
            import subprocess
            
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Ejecutar pg_restore para PostgreSQL
            if "postgresql" in self.db_connection_string:
                cmd = [
                    "pg_restore",
                    "--verbose",
                    "--clean",
                    "--no-owner",
                    "--no-privileges",
                    "--dbname", self.db_connection_string,
                    backup_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"pg_restore failed: {result.stderr}")
            
            return {
                "status": "completed",
                "restored_from": backup_path
            }
            
        except Exception as e:
            logger.error(f"Error restoring database backup: {e}")
            raise

class FileBackupManager:
    """Gestor de backup de archivos"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    async def create_file_backup(self, backup_path: str, include_patterns: List[str] = None, exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """Crear backup de archivos"""
        try:
            import fnmatch
            
            include_patterns = include_patterns or ["*"]
            exclude_patterns = exclude_patterns or [".git", "__pycache__", "*.pyc", "*.log"]
            
            # Crear archivo tar
            with tarfile.open(backup_path, "w:gz") as tar:
                for root, dirs, files in os.walk(self.base_path):
                    # Filtrar directorios
                    dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pattern) for pattern in exclude_patterns)]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, self.base_path)
                        
                        # Verificar patrones de inclusión y exclusión
                        if any(fnmatch.fnmatch(file, pattern) for pattern in include_patterns):
                            if not any(fnmatch.fnmatch(file, pattern) for pattern in exclude_patterns):
                                tar.add(file_path, arcname=relative_path)
            
            # Obtener información del backup
            backup_size = os.path.getsize(backup_path)
            
            return {
                "backup_path": backup_path,
                "size": backup_size,
                "format": "tar.gz",
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error creating file backup: {e}")
            raise
    
    async def restore_file_backup(self, backup_path: str, target_path: str) -> Dict[str, Any]:
        """Restaurar backup de archivos"""
        try:
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Crear directorio objetivo
            os.makedirs(target_path, exist_ok=True)
            
            # Extraer archivo tar
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(target_path)
            
            return {
                "status": "completed",
                "restored_to": target_path,
                "restored_from": backup_path
            }
            
        except Exception as e:
            logger.error(f"Error restoring file backup: {e}")
            raise

class BackupManager:
    """Gestor principal de backups"""
    
    def __init__(self, redis_client: redis.Redis, db_connection_string: str, base_path: str):
        self.redis = redis_client
        self.db_backup_manager = DatabaseBackupManager(db_connection_string)
        self.file_backup_manager = FileBackupManager(base_path)
        self.encryption_manager = EncryptionManager()
        self.compression_manager = CompressionManager()
        self.backup_configs = {}
        self.backup_jobs = {}
        self.restore_jobs = {}
        self._load_backup_configs()
    
    def _load_backup_configs(self):
        """Cargar configuraciones de backup"""
        try:
            # Configuraciones por defecto
            default_configs = {
                "daily_full": BackupConfig(
                    id="daily_full",
                    name="Daily Full Backup",
                    description="Full backup every day at 2 AM",
                    backup_type=BackupType.FULL,
                    compression=CompressionType.GZIP,
                    encryption=True,
                    retention_days=30,
                    schedule="0 2 * * *"
                ),
                "hourly_incremental": BackupConfig(
                    id="hourly_incremental",
                    name="Hourly Incremental Backup",
                    description="Incremental backup every hour",
                    backup_type=BackupType.INCREMENTAL,
                    compression=CompressionType.GZIP,
                    encryption=True,
                    retention_days=7,
                    schedule="0 * * * *"
                ),
                "weekly_snapshot": BackupConfig(
                    id="weekly_snapshot",
                    name="Weekly Snapshot",
                    description="Weekly snapshot backup",
                    backup_type=BackupType.SNAPSHOT,
                    compression=CompressionType.LZMA,
                    encryption=True,
                    retention_days=365,
                    schedule="0 3 * * 0"
                )
            }
            
            self.backup_configs = default_configs
            
        except Exception as e:
            logger.error(f"Error loading backup configs: {e}")
    
    async def create_backup(self, config_id: str, manual: bool = False) -> str:
        """Crear backup"""
        try:
            config = self.backup_configs.get(config_id)
            if not config:
                raise ValueError(f"Backup config {config_id} not found")
            
            # Crear trabajo de backup
            job_id = f"backup_{config_id}_{int(time.time())}"
            job = BackupJob(
                id=job_id,
                config_id=config_id,
                status=BackupStatus.PENDING,
                start_time=datetime.now()
            )
            
            self.backup_jobs[job_id] = job
            
            # Ejecutar backup en background
            asyncio.create_task(self._execute_backup(job, config))
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    async def _execute_backup(self, job: BackupJob, config: BackupConfig):
        """Ejecutar backup"""
        try:
            job.status = BackupStatus.RUNNING
            job.start_time = datetime.now()
            
            # Crear directorio de backup
            backup_dir = f"backups/{config.id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(backup_dir, exist_ok=True)
            
            backup_files = []
            total_size = 0
            
            # Backup de base de datos
            if config.include_database:
                job.progress = 10
                db_backup_path = os.path.join(backup_dir, "database.backup")
                db_result = await self.db_backup_manager.create_database_backup(db_backup_path)
                backup_files.append(db_result)
                total_size += db_result["size"]
            
            # Backup de archivos
            if config.include_files:
                job.progress = 30
                files_backup_path = os.path.join(backup_dir, "files.tar.gz")
                files_result = await self.file_backup_manager.create_file_backup(files_backup_path)
                backup_files.append(files_result)
                total_size += files_result["size"]
            
            # Backup de configuración
            if config.include_config:
                job.progress = 50
                config_backup_path = os.path.join(backup_dir, "config.json")
                await self._backup_configuration(config_backup_path)
                config_size = os.path.getsize(config_backup_path)
                backup_files.append({"backup_path": config_backup_path, "size": config_size})
                total_size += config_size
            
            # Backup de logs
            if config.include_logs:
                job.progress = 70
                logs_backup_path = os.path.join(backup_dir, "logs.tar.gz")
                await self._backup_logs(logs_backup_path)
                logs_size = os.path.getsize(logs_backup_path)
                backup_files.append({"backup_path": logs_backup_path, "size": logs_size})
                total_size += logs_size
            
            # Comprimir y encriptar backup completo
            job.progress = 80
            final_backup_path = os.path.join(backup_dir, f"{config.id}_backup.tar")
            await self._create_final_backup(backup_files, final_backup_path, config)
            
            # Calcular checksum
            job.progress = 90
            checksum = await self._calculate_checksum(final_backup_path)
            
            # Completar trabajo
            job.progress = 100
            job.status = BackupStatus.COMPLETED
            job.end_time = datetime.now()
            job.total_size = total_size
            job.compressed_size = os.path.getsize(final_backup_path)
            job.backup_path = final_backup_path
            job.checksum = checksum
            
            # Guardar información del backup
            await self._save_backup_info(job, config)
            
            logger.info(f"Backup completed: {job.id}")
            
        except Exception as e:
            job.status = BackupStatus.FAILED
            job.end_time = datetime.now()
            job.error_message = str(e)
            logger.error(f"Backup failed: {job.id} - {e}")
    
    async def _backup_configuration(self, config_path: str):
        """Backup de configuración"""
        try:
            config_data = {
                "backup_configs": {k: asdict(v) for k, v in self.backup_configs.items()},
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            async with aiofiles.open(config_path, 'w') as f:
                await f.write(json.dumps(config_data, indent=2))
                
        except Exception as e:
            logger.error(f"Error backing up configuration: {e}")
            raise
    
    async def _backup_logs(self, logs_path: str):
        """Backup de logs"""
        try:
            log_dirs = ["logs", "var/log"]
            log_files = []
            
            for log_dir in log_dirs:
                if os.path.exists(log_dir):
                    for root, dirs, files in os.walk(log_dir):
                        for file in files:
                            if file.endswith('.log'):
                                log_files.append(os.path.join(root, file))
            
            if log_files:
                with tarfile.open(logs_path, "w:gz") as tar:
                    for log_file in log_files:
                        tar.add(log_file, arcname=os.path.relpath(log_file, "."))
            else:
                # Crear archivo vacío si no hay logs
                with open(logs_path, 'w') as f:
                    f.write("")
                    
        except Exception as e:
            logger.error(f"Error backing up logs: {e}")
            raise
    
    async def _create_final_backup(self, backup_files: List[Dict[str, Any]], final_path: str, config: BackupConfig):
        """Crear backup final comprimido y encriptado"""
        try:
            with tarfile.open(final_path, "w") as tar:
                for backup_file in backup_files:
                    file_path = backup_file["backup_path"]
                    if os.path.exists(file_path):
                        tar.add(file_path, arcname=os.path.basename(file_path))
            
            # Aplicar compresión adicional si es necesario
            if config.compression != CompressionType.NONE:
                compressed_path = f"{final_path}.{config.compression.value}"
                with open(final_path, 'rb') as f_in:
                    data = f_in.read()
                    compressed_data = self.compression_manager.compress_data(data, config.compression)
                    with open(compressed_path, 'wb') as f_out:
                        f_out.write(compressed_data)
                os.remove(final_path)
                final_path = compressed_path
            
            # Aplicar encriptación si es necesario
            if config.encryption:
                encrypted_path = f"{final_path}.enc"
                with open(final_path, 'rb') as f_in:
                    data = f_in.read()
                    encrypted_data = self.encryption_manager.encrypt_data(data)
                    with open(encrypted_path, 'wb') as f_out:
                        f_out.write(encrypted_data)
                os.remove(final_path)
                final_path = encrypted_path
            
        except Exception as e:
            logger.error(f"Error creating final backup: {e}")
            raise
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calcular checksum del archivo"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            raise
    
    async def _save_backup_info(self, job: BackupJob, config: BackupConfig):
        """Guardar información del backup"""
        try:
            backup_info = {
                "job_id": job.id,
                "config_id": job.config_id,
                "backup_path": job.backup_path,
                "size": job.total_size,
                "compressed_size": job.compressed_size,
                "checksum": job.checksum,
                "created_at": job.start_time.isoformat(),
                "completed_at": job.end_time.isoformat(),
                "config": asdict(config)
            }
            
            # Guardar en Redis
            await self.redis.hset("backups", job.id, json.dumps(backup_info))
            
            # Guardar en archivo
            backup_info_path = f"{job.backup_path}.info"
            async with aiofiles.open(backup_info_path, 'w') as f:
                await f.write(json.dumps(backup_info, indent=2))
                
        except Exception as e:
            logger.error(f"Error saving backup info: {e}")
            raise
    
    async def restore_backup(self, backup_id: str, target_path: str = None) -> str:
        """Restaurar backup"""
        try:
            # Obtener información del backup
            backup_info = await self.redis.hget("backups", backup_id)
            if not backup_info:
                raise ValueError(f"Backup {backup_id} not found")
            
            backup_data = json.loads(backup_info)
            backup_path = backup_data["backup_path"]
            
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Crear trabajo de restauración
            restore_job_id = f"restore_{backup_id}_{int(time.time())}"
            restore_job = RestoreJob(
                id=restore_job_id,
                backup_id=backup_id,
                target_path=target_path or "restore",
                status=BackupStatus.PENDING,
                start_time=datetime.now()
            )
            
            self.restore_jobs[restore_job_id] = restore_job
            
            # Ejecutar restauración en background
            asyncio.create_task(self._execute_restore(restore_job, backup_data))
            
            return restore_job_id
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            raise
    
    async def _execute_restore(self, job: RestoreJob, backup_data: Dict[str, Any]):
        """Ejecutar restauración"""
        try:
            job.status = BackupStatus.RUNNING
            job.start_time = datetime.now()
            
            backup_path = backup_data["backup_path"]
            config = backup_data["config"]
            
            # Crear directorio temporal para extracción
            temp_dir = f"temp_restore_{job.id}"
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # Desencriptar si es necesario
                if config.get("encryption", False):
                    job.progress = 10
                    decrypted_path = f"{backup_path}.dec"
                    with open(backup_path, 'rb') as f_in:
                        encrypted_data = f_in.read()
                        decrypted_data = self.encryption_manager.decrypt_data(encrypted_data)
                        with open(decrypted_path, 'wb') as f_out:
                            f_out.write(decrypted_data)
                    backup_path = decrypted_path
                
                # Descomprimir si es necesario
                if config.get("compression") != "none":
                    job.progress = 20
                    decompressed_path = f"{backup_path}.decomp"
                    with open(backup_path, 'rb') as f_in:
                        compressed_data = f_in.read()
                        decompressed_data = self.compression_manager.decompress_data(
                            compressed_data, 
                            CompressionType(config["compression"])
                        )
                        with open(decompressed_path, 'wb') as f_out:
                            f_out.write(decompressed_data)
                    backup_path = decompressed_path
                
                # Extraer archivo tar
                job.progress = 30
                with tarfile.open(backup_path, "r") as tar:
                    tar.extractall(temp_dir)
                
                # Restaurar base de datos
                job.progress = 50
                db_backup_path = os.path.join(temp_dir, "database.backup")
                if os.path.exists(db_backup_path):
                    await self.db_backup_manager.restore_database_backup(db_backup_path)
                
                # Restaurar archivos
                job.progress = 70
                files_backup_path = os.path.join(temp_dir, "files.tar.gz")
                if os.path.exists(files_backup_path):
                    await self.file_backup_manager.restore_file_backup(files_backup_path, job.target_path)
                
                # Restaurar configuración
                job.progress = 90
                config_backup_path = os.path.join(temp_dir, "config.json")
                if os.path.exists(config_backup_path):
                    await self._restore_configuration(config_backup_path)
                
                # Completar trabajo
                job.progress = 100
                job.status = BackupStatus.COMPLETED
                job.end_time = datetime.now()
                
                logger.info(f"Restore completed: {job.id}")
                
            finally:
                # Limpiar archivos temporales
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                if os.path.exists(f"{backup_path}.dec"):
                    os.remove(f"{backup_path}.dec")
                if os.path.exists(f"{backup_path}.decomp"):
                    os.remove(f"{backup_path}.decomp")
            
        except Exception as e:
            job.status = BackupStatus.FAILED
            job.end_time = datetime.now()
            job.error_message = str(e)
            logger.error(f"Restore failed: {job.id} - {e}")
    
    async def _restore_configuration(self, config_path: str):
        """Restaurar configuración"""
        try:
            async with aiofiles.open(config_path, 'r') as f:
                config_data = json.loads(await f.read())
            
            # Restaurar configuraciones de backup
            if "backup_configs" in config_data:
                for config_id, config_dict in config_data["backup_configs"].items():
                    config = BackupConfig(**config_dict)
                    self.backup_configs[config_id] = config
            
            logger.info("Configuration restored successfully")
            
        except Exception as e:
            logger.error(f"Error restoring configuration: {e}")
            raise
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """Listar backups disponibles"""
        try:
            backups = []
            backup_data = await self.redis.hgetall("backups")
            
            for backup_id, backup_info in backup_data.items():
                backup_dict = json.loads(backup_info)
                backups.append(backup_dict)
            
            # Ordenar por fecha de creación
            backups.sort(key=lambda x: x["created_at"], reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Eliminar backup"""
        try:
            # Obtener información del backup
            backup_info = await self.redis.hget("backups", backup_id)
            if not backup_info:
                return False
            
            backup_data = json.loads(backup_info)
            backup_path = backup_data["backup_path"]
            
            # Eliminar archivo de backup
            if os.path.exists(backup_path):
                os.remove(backup_path)
            
            # Eliminar archivo de información
            info_path = f"{backup_path}.info"
            if os.path.exists(info_path):
                os.remove(info_path)
            
            # Eliminar de Redis
            await self.redis.hdel("backups", backup_id)
            
            logger.info(f"Backup deleted: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting backup: {e}")
            return False
    
    async def cleanup_old_backups(self):
        """Limpiar backups antiguos"""
        try:
            backups = await self.list_backups()
            current_time = datetime.now()
            
            for backup in backups:
                created_at = datetime.fromisoformat(backup["created_at"])
                config = backup["config"]
                retention_days = config.get("retention_days", 30)
                
                if (current_time - created_at).days > retention_days:
                    await self.delete_backup(backup["job_id"])
                    logger.info(f"Cleaned up old backup: {backup['job_id']}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    async def get_backup_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado del backup"""
        try:
            if job_id in self.backup_jobs:
                job = self.backup_jobs[job_id]
                return asdict(job)
            elif job_id in self.restore_jobs:
                job = self.restore_jobs[job_id]
                return asdict(job)
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting backup status: {e}")
            return None
    
    async def schedule_backups(self):
        """Programar backups automáticos"""
        try:
            import schedule
            
            for config_id, config in self.backup_configs.items():
                if config.schedule and config.enabled:
                    # Programar backup según cron
                    schedule.every().day.at("02:00").do(
                        lambda cid=config_id: asyncio.create_task(self.create_backup(cid))
                    )
            
            # Ejecutar scheduler
            while True:
                schedule.run_pending()
                await asyncio.sleep(60)  # Verificar cada minuto
                
        except Exception as e:
            logger.error(f"Error scheduling backups: {e}")

# Funciones de utilidad
async def create_backup_config(
    config_id: str,
    name: str,
    description: str,
    backup_type: BackupType,
    compression: CompressionType = CompressionType.GZIP,
    encryption: bool = True,
    retention_days: int = 30,
    schedule: str = None
) -> BackupConfig:
    """Crear configuración de backup"""
    return BackupConfig(
        id=config_id,
        name=name,
        description=description,
        backup_type=backup_type,
        compression=compression,
        encryption=encryption,
        retention_days=retention_days,
        schedule=schedule
    )

async def verify_backup_integrity(backup_path: str, expected_checksum: str) -> bool:
    """Verificar integridad del backup"""
    try:
        actual_checksum = hashlib.sha256()
        with open(backup_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                actual_checksum.update(chunk)
        
        return actual_checksum.hexdigest() == expected_checksum
    except Exception as e:
        logger.error(f"Error verifying backup integrity: {e}")
        return False

# Configuración de backup por defecto
DEFAULT_BACKUP_CONFIG = {
    "backup_directory": "backups",
    "temp_directory": "temp_backups",
    "max_backup_size": 1024 * 1024 * 1024 * 10,  # 10GB
    "compression_level": 6,
    "encryption_algorithm": "AES-256",
    "retention_policies": {
        "daily": 30,
        "weekly": 12,
        "monthly": 12,
        "yearly": 5
    },
    "notification_channels": ["email", "slack"],
    "backup_verification": True,
    "incremental_backup": True,
    "parallel_compression": True
}



























