"""
Backup Manager - Sistema de backup y restore
=============================================
"""

import logging
import shutil
import zipfile
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Gestiona backups y restauración del sistema.
    """
    
    def __init__(self, backup_dir: str = "data/backups"):
        """
        Inicializar gestor de backups.
        
        Args:
            backup_dir: Directorio para backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        include_papers: bool = True,
        include_models: bool = True,
        include_data: bool = True,
        backup_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Crea un backup del sistema.
        
        Args:
            include_papers: Incluir papers
            include_models: Incluir modelos
            include_data: Incluir otros datos
            backup_name: Nombre del backup (opcional)
            
        Returns:
            Información del backup creado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = backup_name or f"backup_{timestamp}"
        backup_path = self.backup_dir / f"{backup_name}.zip"
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Backup de papers
                if include_papers:
                    papers_dir = Path("data/papers")
                    if papers_dir.exists():
                        for paper_file in papers_dir.rglob("*"):
                            if paper_file.is_file():
                                arcname = paper_file.relative_to(Path("data"))
                                zipf.write(paper_file, arcname)
                
                # Backup de modelos
                if include_models:
                    models_dir = Path("data/models")
                    if models_dir.exists():
                        for model_file in models_dir.rglob("*"):
                            if model_file.is_file():
                                arcname = model_file.relative_to(Path("data"))
                                zipf.write(model_file, arcname)
                
                # Backup de otros datos
                if include_data:
                    data_dirs = ["data/embeddings", "data/vector_db", "data/cache"]
                    for data_dir in data_dirs:
                        data_path = Path(data_dir)
                        if data_path.exists():
                            for data_file in data_path.rglob("*"):
                                if data_file.is_file():
                                    arcname = data_file.relative_to(Path("data"))
                                    zipf.write(data_file, arcname)
                
                # Metadata del backup
                metadata = {
                    "backup_name": backup_name,
                    "created_at": datetime.now().isoformat(),
                    "includes": {
                        "papers": include_papers,
                        "models": include_models,
                        "data": include_data
                    }
                }
                
                zipf.writestr("backup_metadata.json", json.dumps(metadata, indent=2))
            
            backup_size = backup_path.stat().st_size
            
            backup_info = {
                "backup_id": backup_name,
                "backup_path": str(backup_path),
                "size_bytes": backup_size,
                "size_mb": round(backup_size / (1024 * 1024), 2),
                "created_at": datetime.now().isoformat(),
                "includes": {
                    "papers": include_papers,
                    "models": include_models,
                    "data": include_data
                }
            }
            
            logger.info(f"Backup creado: {backup_name} ({backup_info['size_mb']} MB)")
            
            return backup_info
            
        except Exception as e:
            logger.error(f"Error creando backup: {e}")
            raise
    
    def restore_backup(
        self,
        backup_id: str,
        restore_papers: bool = True,
        restore_models: bool = True,
        restore_data: bool = True
    ) -> Dict[str, Any]:
        """
        Restaura un backup.
        
        Args:
            backup_id: ID del backup
            restore_papers: Restaurar papers
            restore_models: Restaurar modelos
            restore_data: Restaurar otros datos
            
        Returns:
            Información de la restauración
        """
        backup_path = self.backup_dir / f"{backup_id}.zip"
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup no encontrado: {backup_id}")
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # Leer metadata
                if "backup_metadata.json" in zipf.namelist():
                    metadata = json.loads(zipf.read("backup_metadata.json"))
                else:
                    metadata = {}
                
                # Restaurar archivos
                restored_files = []
                
                for file_info in zipf.infolist():
                    file_path = Path(file_info.filename)
                    
                    # Determinar si restaurar
                    should_restore = False
                    if restore_papers and "papers" in str(file_path):
                        should_restore = True
                    elif restore_models and "models" in str(file_path):
                        should_restore = True
                    elif restore_data and any(d in str(file_path) for d in ["embeddings", "vector_db", "cache"]):
                        should_restore = True
                    
                    if should_restore:
                        # Crear directorio si no existe
                        target_path = Path("data") / file_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Extraer archivo
                        zipf.extract(file_info, Path("data"))
                        restored_files.append(str(target_path))
            
            logger.info(f"Backup restaurado: {backup_id} ({len(restored_files)} archivos)")
            
            return {
                "backup_id": backup_id,
                "restored_files": len(restored_files),
                "restored_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error restaurando backup: {e}")
            raise
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """Lista backups disponibles"""
        backups = []
        
        for backup_file in self.backup_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    if "backup_metadata.json" in zipf.namelist():
                        metadata = json.loads(zipf.read("backup_metadata.json"))
                    else:
                        metadata = {
                            "backup_name": backup_file.stem,
                            "created_at": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat()
                        }
                
                backups.append({
                    "backup_id": backup_file.stem,
                    "size_mb": round(backup_file.stat().st_size / (1024 * 1024), 2),
                    "created_at": metadata.get("created_at", ""),
                    "includes": metadata.get("includes", {})
                })
            except Exception as e:
                logger.warning(f"Error procesando backup {backup_file}: {e}")
                continue
        
        # Ordenar por fecha (más recientes primero)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        
        return backups
    
    def delete_backup(self, backup_id: str) -> bool:
        """Elimina un backup"""
        backup_path = self.backup_dir / f"{backup_id}.zip"
        
        if backup_path.exists():
            backup_path.unlink()
            logger.info(f"Backup eliminado: {backup_id}")
            return True
        
        return False




