"""
Backup Service - Servicio de Backup
===================================

Servicio para crear y restaurar backups.
"""

import logging
import json
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import zipfile

logger = logging.getLogger(__name__)


class BackupService:
    """Servicio para backups"""
    
    def __init__(self, backup_path: str = "data/backups"):
        """
        Inicializar servicio de backup
        
        Args:
            backup_path: Ruta donde guardar backups
        """
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backup Service inicializado en {backup_path}")
    
    def create_backup(
        self,
        manager: Any,
        include_media: bool = True
    ) -> str:
        """
        Crear backup completo
        
        Args:
            manager: Instancia de CommunityManager
            include_media: Incluir archivos multimedia
            
        Returns:
            Ruta del archivo de backup
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_path / f"backup_{timestamp}.zip"
        
        try:
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Backup de posts
                posts = manager.scheduler.get_all_posts()
                posts_data = {
                    "posts": [p for p in posts],
                    "backup_date": datetime.now().isoformat()
                }
                zipf.writestr("posts.json", json.dumps(posts_data, indent=2, ensure_ascii=False))
                
                # Backup de memes metadata
                memes_data = {
                    "memes": list(manager.meme_manager.memes.values()),
                    "backup_date": datetime.now().isoformat()
                }
                zipf.writestr("memes.json", json.dumps(memes_data, indent=2, ensure_ascii=False))
                
                # Backup de templates
                if hasattr(manager, 'template_manager'):
                    templates_data = {
                        "templates": list(manager.template_manager.templates.values()),
                        "backup_date": datetime.now().isoformat()
                    }
                    zipf.writestr("templates.json", json.dumps(templates_data, indent=2, ensure_ascii=False))
                
                # Backup de conexiones (sin credenciales)
                connections_data = {
                    "connections": [
                        {
                            "platform": platform,
                            "connected": conn.get("connected", False)
                        }
                        for platform, conn in manager.social_connector.connections.items()
                    ],
                    "backup_date": datetime.now().isoformat()
                }
                zipf.writestr("connections.json", json.dumps(connections_data, indent=2, ensure_ascii=False))
                
                # Backup de media si se solicita
                if include_media:
                    meme_storage = Path(manager.meme_manager.storage_path)
                    if meme_storage.exists():
                        for file_path in meme_storage.rglob("*"):
                            if file_path.is_file():
                                zipf.write(file_path, f"media/{file_path.relative_to(meme_storage)}")
            
            logger.info(f"Backup creado: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Error creando backup: {e}")
            raise
    
    def restore_backup(
        self,
        backup_file: str,
        manager: Any
    ) -> Dict[str, Any]:
        """
        Restaurar desde backup
        
        Args:
            backup_file: Ruta del archivo de backup
            manager: Instancia de CommunityManager
            
        Returns:
            Dict con resultados de restauración
        """
        results = {
            "posts_restored": 0,
            "memes_restored": 0,
            "templates_restored": 0,
            "errors": []
        }
        
        try:
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                # Restaurar posts
                if "posts.json" in zipf.namelist():
                    posts_data = json.loads(zipf.read("posts.json"))
                    for post in posts_data.get("posts", []):
                        try:
                            manager.scheduler.add_post(post)
                            results["posts_restored"] += 1
                        except Exception as e:
                            results["errors"].append(f"Error restaurando post {post.get('id')}: {e}")
                
                # Restaurar memes
                if "memes.json" in zipf.namelist():
                    memes_data = json.loads(zipf.read("memes.json"))
                    for meme in memes_data.get("memes", []):
                        try:
                            manager.meme_manager.memes[meme["id"]] = meme
                            results["memes_restored"] += 1
                        except Exception as e:
                            results["errors"].append(f"Error restaurando meme {meme.get('id')}: {e}")
                
                # Restaurar templates
                if "templates.json" in zipf.namelist() and hasattr(manager, 'template_manager'):
                    templates_data = json.loads(zipf.read("templates.json"))
                    for template in templates_data.get("templates", []):
                        try:
                            manager.template_manager.templates[template["id"]] = template
                            results["templates_restored"] += 1
                        except Exception as e:
                            results["errors"].append(f"Error restaurando template {template.get('id')}: {e}")
                
                # Restaurar media
                for file_info in zipf.filelist:
                    if file_info.filename.startswith("media/"):
                        dest_path = Path(manager.meme_manager.storage_path) / file_info.filename[6:]
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        with zipf.open(file_info) as source, open(dest_path, 'wb') as dest:
                            dest.write(source.read())
            
            logger.info(f"Backup restaurado: {backup_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error restaurando backup: {e}")
            results["errors"].append(str(e))
            return results
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        Listar backups disponibles
        
        Returns:
            Lista de backups con metadata
        """
        backups = []
        
        for backup_file in self.backup_path.glob("backup_*.zip"):
            try:
                stat = backup_file.stat()
                backups.append({
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                logger.error(f"Error obteniendo info de backup {backup_file}: {e}")
        
        # Ordenar por fecha (más recientes primero)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        
        return backups
    
    def delete_backup(self, backup_file: str) -> bool:
        """
        Eliminar un backup
        
        Args:
            backup_file: Nombre del archivo o ruta completa
            
        Returns:
            True si se eliminó exitosamente
        """
        backup_path = Path(backup_file)
        if not backup_path.is_absolute():
            backup_path = self.backup_path / backup_file
        
        if backup_path.exists() and backup_path.suffix == ".zip":
            backup_path.unlink()
            logger.info(f"Backup eliminado: {backup_path}")
            return True
        
        return False
    
    def get_backup_info(self, backup_file: str) -> Optional[Dict[str, Any]]:
        """
        Obtener información detallada de un backup
        
        Args:
            backup_file: Nombre del archivo o ruta completa
            
        Returns:
            Dict con información del backup o None
        """
        backup_path = Path(backup_file)
        if not backup_path.is_absolute():
            backup_path = self.backup_path / backup_file
        
        if not backup_path.exists():
            return None
        
        try:
            stat = backup_path.stat()
            info = {
                "filename": backup_path.name,
                "path": str(backup_path),
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                info["files"] = zipf.namelist()
                info["file_count"] = len(zipf.namelist())
                
                try:
                    if "posts.json" in zipf.namelist():
                        posts_data = json.loads(zipf.read("posts.json"))
                        info["posts_count"] = len(posts_data.get("posts", []))
                    
                    if "memes.json" in zipf.namelist():
                        memes_data = json.loads(zipf.read("memes.json"))
                        info["memes_count"] = len(memes_data.get("memes", []))
                    
                    if "templates.json" in zipf.namelist():
                        templates_data = json.loads(zipf.read("templates.json"))
                        info["templates_count"] = len(templates_data.get("templates", []))
                except Exception as e:
                    logger.warning(f"Error leyendo contenido del backup: {e}")
            
            return info
            
        except Exception as e:
            logger.error(f"Error obteniendo info del backup: {e}")
            return None
    
    def verify_backup(self, backup_file: str) -> Dict[str, Any]:
        """
        Verificar integridad de un backup
        
        Args:
            backup_file: Nombre del archivo o ruta completa
            
        Returns:
            Dict con resultados de verificación
        """
        backup_path = Path(backup_file)
        if not backup_path.is_absolute():
            backup_path = self.backup_path / backup_file
        
        result = {
            "valid": False,
            "errors": [],
            "warnings": []
        }
        
        if not backup_path.exists():
            result["errors"].append("Archivo no encontrado")
            return result
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.testzip()
                
                required_files = ["posts.json"]
                for required in required_files:
                    if required not in zipf.namelist():
                        result["warnings"].append(f"Archivo requerido no encontrado: {required}")
                
                result["valid"] = True
                result["file_count"] = len(zipf.namelist())
                
        except zipfile.BadZipFile:
            result["errors"].append("Archivo ZIP corrupto")
        except Exception as e:
            result["errors"].append(f"Error verificando backup: {e}")
        
        return result
    
    def cleanup_old_backups(self, keep_last: int = 10) -> int:
        """
        Limpiar backups antiguos, manteniendo solo los N más recientes
        
        Args:
            keep_last: Número de backups a mantener
            
        Returns:
            Número de backups eliminados
        """
        backups = self.list_backups()
        
        if len(backups) <= keep_last:
            return 0
        
        to_delete = backups[keep_last:]
        deleted = 0
        
        for backup in to_delete:
            if self.delete_backup(backup["filename"]):
                deleted += 1
        
        logger.info(f"Limpiados {deleted} backups antiguos (manteniendo {keep_last} más recientes)")
        return deleted

