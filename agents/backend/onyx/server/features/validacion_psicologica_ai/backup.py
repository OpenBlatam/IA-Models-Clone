"""
Sistema de Backup y Recuperación
=================================
Backup automático y recuperación de datos
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime, timedelta
import structlog
import json
import gzip
from pathlib import Path

from .models import PsychologicalValidation, SocialMediaConnection

logger = structlog.get_logger()


class BackupManager:
    """Gestor de backups"""
    
    def __init__(self, backup_dir: str = "./backups"):
        """
        Inicializar gestor
        
        Args:
            backup_dir: Directorio para backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info("BackupManager initialized", backup_dir=str(self.backup_dir))
    
    def create_backup(
        self,
        validations: List[PsychologicalValidation],
        connections: List[SocialMediaConnection],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crear backup
        
        Args:
            validations: Lista de validaciones
            connections: Lista de conexiones
            metadata: Metadatos adicionales
            
        Returns:
            Información del backup creado
        """
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        backup_file = self.backup_dir / f"{backup_id}.json.gz"
        
        backup_data = {
            "backup_id": backup_id,
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0",
            "validations": [v.to_dict() for v in validations],
            "connections": [c.to_dict() for c in connections],
            "metadata": metadata or {},
            "counts": {
                "validations": len(validations),
                "connections": len(connections)
            }
        }
        
        # Comprimir y guardar
        json_data = json.dumps(backup_data, default=str)
        compressed = gzip.compress(json_data.encode('utf-8'))
        
        with open(backup_file, 'wb') as f:
            f.write(compressed)
        
        logger.info(
            "Backup created",
            backup_id=backup_id,
            validations=len(validations),
            connections=len(connections)
        )
        
        return {
            "backup_id": backup_id,
            "file_path": str(backup_file),
            "created_at": backup_data["created_at"],
            "size_bytes": len(compressed),
            "counts": backup_data["counts"]
        }
    
    def restore_backup(
        self,
        backup_id: str
    ) -> Dict[str, Any]:
        """
        Restaurar desde backup
        
        Args:
            backup_id: ID del backup
            
        Returns:
            Datos restaurados
        """
        backup_file = self.backup_dir / f"{backup_id}.json.gz"
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup {backup_id} not found")
        
        # Leer y descomprimir
        with open(backup_file, 'rb') as f:
            compressed = f.read()
        
        json_data = gzip.decompress(compressed).decode('utf-8')
        backup_data = json.loads(json_data)
        
        logger.info(
            "Backup restored",
            backup_id=backup_id,
            validations=backup_data["counts"]["validations"],
            connections=backup_data["counts"]["connections"]
        )
        
        return backup_data
    
    def list_backups(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Listar backups disponibles
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de backups
        """
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("backup_*.json.gz"), reverse=True):
            try:
                # Leer metadata sin descomprimir todo
                with open(backup_file, 'rb') as f:
                    compressed = f.read()
                
                json_data = gzip.decompress(compressed).decode('utf-8')
                backup_data = json.loads(json_data)
                
                backups.append({
                    "backup_id": backup_data["backup_id"],
                    "created_at": backup_data["created_at"],
                    "size_bytes": len(compressed),
                    "counts": backup_data["counts"]
                })
            except Exception as e:
                logger.error("Error reading backup", file=str(backup_file), error=str(e))
        
        return backups[:limit]
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Eliminar backup
        
        Args:
            backup_id: ID del backup
            
        Returns:
            True si se eliminó exitosamente
        """
        backup_file = self.backup_dir / f"{backup_id}.json.gz"
        
        if backup_file.exists():
            backup_file.unlink()
            logger.info("Backup deleted", backup_id=backup_id)
            return True
        
        return False
    
    def cleanup_old_backups(self, days: int = 30) -> int:
        """
        Limpiar backups antiguos
        
        Args:
            days: Días de antigüedad para eliminar
            
        Returns:
            Número de backups eliminados
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted = 0
        
        for backup_file in self.backup_dir.glob("backup_*.json.gz"):
            try:
                # Extraer fecha del nombre del archivo
                file_date_str = backup_file.stem.replace("backup_", "").replace(".json", "")
                file_date = datetime.strptime(file_date_str, "%Y%m%d_%H%M%S")
                
                if file_date < cutoff_date:
                    backup_file.unlink()
                    deleted += 1
            except Exception as e:
                logger.error("Error cleaning backup", file=str(backup_file), error=str(e))
        
        logger.info("Old backups cleaned", deleted=deleted, days=days)
        return deleted


# Instancia global del gestor de backups
backup_manager = BackupManager()




