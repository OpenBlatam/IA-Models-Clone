"""
Servicio de versionado de identidades
"""

import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, Integer

from ..db.base import Base, get_db_session
from ..services.storage_service import StorageService

logger = logging.getLogger(__name__)


class IdentityVersionModel(Base):
    """Modelo de versión de identidad"""
    __tablename__ = "identity_versions"
    
    id = Column(String(64), primary_key=True, index=True)
    identity_profile_id = Column(String(64), ForeignKey("identity_profiles.id"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    snapshot = Column(JSON, nullable=False)  # Snapshot completo de la identidad
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    
    __table_args__ = (
        {"sqlite_autoincrement": True},
    )


class VersioningService:
    """Servicio para versionado de identidades"""
    
    def __init__(self):
        self.storage = StorageService()
        self._init_table()
    
    def _init_table(self):
        """Inicializa tabla de versiones"""
        from ..db.base import init_db
        init_db()
    
    def create_version(
        self,
        identity_id: str,
        notes: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> str:
        """
        Crea una nueva versión de identidad
        
        Args:
            identity_id: ID de la identidad
            notes: Notas sobre la versión
            created_by: Usuario que crea la versión
            
        Returns:
            ID de la versión
        """
        identity = self.storage.get_identity(identity_id)
        if not identity:
            raise ValueError(f"Identidad no encontrada: {identity_id}")
        
        # Obtener siguiente número de versión
        with get_db_session() as db:
            max_version = db.query(
                IdentityVersionModel.version_number
            ).filter_by(
                identity_profile_id=identity_id
            ).order_by(
                IdentityVersionModel.version_number.desc()
            ).first()
            
            next_version = (max_version[0] + 1) if max_version else 1
            
            # Crear versión
            version_id = str(uuid.uuid4())
            version = IdentityVersionModel(
                id=version_id,
                identity_profile_id=identity_id,
                version_number=next_version,
                snapshot=identity.model_dump(),
                notes=notes,
                created_by=created_by,
                created_at=datetime.utcnow()
            )
            
            db.add(version)
            db.commit()
            
            logger.info(f"Versión {next_version} creada para identidad {identity_id}")
            return version_id
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene una versión específica"""
        with get_db_session() as db:
            version = db.query(IdentityVersionModel).filter_by(id=version_id).first()
            if not version:
                return None
            
            return {
                "version_id": version.id,
                "identity_id": version.identity_profile_id,
                "version_number": version.version_number,
                "snapshot": version.snapshot,
                "created_at": version.created_at.isoformat(),
                "created_by": version.created_by,
                "notes": version.notes
            }
    
    def list_versions(self, identity_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Lista versiones de una identidad"""
        with get_db_session() as db:
            versions = db.query(IdentityVersionModel).filter_by(
                identity_profile_id=identity_id
            ).order_by(
                IdentityVersionModel.version_number.desc()
            ).limit(limit).all()
            
            return [
                {
                    "version_id": v.id,
                    "version_number": v.version_number,
                    "created_at": v.created_at.isoformat(),
                    "created_by": v.created_by,
                    "notes": v.notes
                }
                for v in versions
            ]
    
    def restore_version(self, version_id: str) -> str:
        """
        Restaura una versión de identidad
        
        Args:
            version_id: ID de la versión a restaurar
            
        Returns:
            ID de la nueva versión creada (backup antes de restaurar)
        """
        version_data = self.get_version(version_id)
        if not version_data:
            raise ValueError(f"Versión no encontrada: {version_id}")
        
        identity_id = version_data["identity_id"]
        
        # Crear backup de la versión actual antes de restaurar
        current_backup_id = self.create_version(
            identity_id,
            notes="Backup antes de restaurar versión"
        )
        
        # Restaurar desde snapshot
        snapshot = version_data["snapshot"]
        from ..core.models import IdentityProfile, ContentAnalysis
        
        # Reconstruir identidad desde snapshot
        # (simplificado, en producción necesitarías reconstruir todos los objetos)
        restored_identity = IdentityProfile(**snapshot)
        
        # Guardar identidad restaurada
        self.storage.save_identity(restored_identity)
        
        logger.info(f"Versión {version_id} restaurada para identidad {identity_id}")
        return current_backup_id




