"""
Sistema de backups automáticos
"""

import logging
import shutil
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import zipfile

from ..config import get_settings
from ..db.base import get_db_session
from ..services.storage_service import StorageService

logger = logging.getLogger(__name__)


class BackupService:
    """Servicio de backups"""
    
    def __init__(self):
        self.settings = get_settings()
        self.storage = StorageService()
        self.backup_dir = Path(self.settings.storage_path) / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        backup_type: str = "full",  # full, identities_only, content_only
        include_database: bool = True
    ) -> str:
        """
        Crea un backup
        
        Args:
            backup_type: Tipo de backup
            include_database: Si incluir base de datos
            
        Returns:
            Path del archivo de backup
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"backup_{backup_type}_{timestamp}.zip"
        backup_path = self.backup_dir / backup_filename
        
        logger.info(f"Creando backup: {backup_filename}")
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Backup de base de datos
            if include_database:
                db_path = Path(self.settings.database_url.replace("sqlite:///", ""))
                if db_path.exists():
                    zipf.write(db_path, f"database/{db_path.name}")
            
            # Backup de identidades
            if backup_type in ["full", "identities_only"]:
                self._backup_identities(zipf)
            
            # Backup de contenido generado
            if backup_type in ["full", "content_only"]:
                self._backup_content(zipf)
            
            # Backup de configuración
            if backup_type == "full":
                self._backup_config(zipf)
            
            # Metadata del backup
            metadata = {
                "backup_type": backup_type,
                "created_at": datetime.utcnow().isoformat(),
                "include_database": include_database
            }
            zipf.writestr("backup_metadata.json", json.dumps(metadata, indent=2))
        
        logger.info(f"Backup creado: {backup_path}")
        return str(backup_path)
    
    def _backup_identities(self, zipf: zipfile.ZipFile):
        """Backup de identidades"""
        with get_db_session() as db:
            from ..db.models import IdentityProfileModel
            identities = db.query(IdentityProfileModel).all()
            
            identities_data = []
            for identity in identities:
                identities_data.append({
                    "id": identity.id,
                    "username": identity.username,
                    "display_name": identity.display_name,
                    "bio": identity.bio,
                    "knowledge_base": identity.knowledge_base,
                    "created_at": identity.created_at.isoformat()
                })
            
            zipf.writestr("identities.json", json.dumps(identities_data, indent=2, default=str))
    
    def _backup_content(self, zipf: zipfile.ZipFile):
        """Backup de contenido generado"""
        with get_db_session() as db:
            from ..db.models import GeneratedContentModel
            content = db.query(GeneratedContentModel).all()
            
            content_data = []
            for item in content:
                content_data.append({
                    "id": item.id,
                    "identity_profile_id": item.identity_profile_id,
                    "platform": item.platform,
                    "content_type": item.content_type,
                    "content": item.content,
                    "title": item.title,
                    "hashtags": item.hashtags,
                    "generated_at": item.generated_at.isoformat()
                })
            
            zipf.writestr("content.json", json.dumps(content_data, indent=2, default=str))
    
    def _backup_config(self, zipf: zipfile.ZipFile):
        """Backup de configuración"""
        config_data = {
            "settings": {
                "max_videos_per_profile": self.settings.max_videos_per_profile,
                "max_posts_per_profile": self.settings.max_posts_per_profile,
                "content_temperature": self.settings.content_temperature
            }
        }
        zipf.writestr("config.json", json.dumps(config_data, indent=2))
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """Lista backups disponibles"""
        backups = []
        for backup_file in self.backup_dir.glob("backup_*.zip"):
            stat = backup_file.stat()
            backups.append({
                "filename": backup_file.name,
                "path": str(backup_file),
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        # Ordenar por fecha (más recientes primero)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        return backups
    
    def restore_backup(self, backup_path: str) -> bool:
        """
        Restaura un backup
        
        Args:
            backup_path: Path del archivo de backup
            
        Returns:
            True si se restauró exitosamente
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            logger.error(f"Backup no encontrado: {backup_path}")
            return False
        
        logger.info(f"Restaurando backup: {backup_path}")
        
        try:
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                # Leer metadata
                metadata_str = zipf.read("backup_metadata.json").decode()
                metadata = json.loads(metadata_str)
                
                # Restaurar base de datos si está incluida
                if metadata.get("include_database"):
                    db_path = Path(self.settings.database_url.replace("sqlite:///", ""))
                    if "database" in zipf.namelist():
                        # Hacer backup del DB actual
                        if db_path.exists():
                            backup_current = db_path.with_suffix(f".backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.db")
                            shutil.copy2(db_path, backup_current)
                        
                        # Restaurar
                        db_data = zipf.read([f for f in zipf.namelist() if f.startswith("database/")][0])
                        with open(db_path, 'wb') as f:
                            f.write(db_data)
                
                logger.info("Backup restaurado exitosamente")
                return True
                
        except Exception as e:
            logger.error(f"Error restaurando backup: {e}", exc_info=True)
            return False
    
    def cleanup_old_backups(self, days: int = 30):
        """Elimina backups antiguos"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted = 0
        
        for backup_file in self.backup_dir.glob("backup_*.zip"):
            file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
            if file_time < cutoff_date:
                backup_file.unlink()
                deleted += 1
                logger.info(f"Backup antiguo eliminado: {backup_file.name}")
        
        logger.info(f"Eliminados {deleted} backups antiguos")
        return deleted


# Singleton global
_backup_service: Optional[BackupService] = None


def get_backup_service() -> BackupService:
    """Obtiene instancia singleton del servicio de backups"""
    global _backup_service
    if _backup_service is None:
        _backup_service = BackupService()
    return _backup_service
