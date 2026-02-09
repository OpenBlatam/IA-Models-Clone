"""
Maintenance Utilities
=====================

Utilidades para mantenimiento del sistema.
"""

import logging
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MaintenanceManager:
    """
    Gestor de mantenimiento.
    
    Gestiona tareas de mantenimiento del sistema.
    """
    
    def __init__(self):
        """Inicializar gestor de mantenimiento."""
        self.maintenance_history: List[Dict[str, Any]] = []
    
    def cleanup_old_logs(
        self,
        log_directory: str = "logs",
        days_to_keep: int = 30
    ) -> Dict[str, Any]:
        """
        Limpiar logs antiguos.
        
        Args:
            log_directory: Directorio de logs
            days_to_keep: Días a mantener
            
        Returns:
            Información de limpieza
        """
        log_path = Path(log_directory)
        if not log_path.exists():
            return {"deleted": 0, "freed_space": 0}
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        freed_space = 0
        
        for log_file in log_path.glob("*.log"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    file_size = log_file.stat().st_size
                    log_file.unlink()
                    deleted_count += 1
                    freed_space += file_size
            except Exception as e:
                logger.warning(f"Error deleting log file {log_file}: {e}")
        
        result = {
            "deleted": deleted_count,
            "freed_space": freed_space,
            "freed_space_mb": freed_space / (1024 * 1024)
        }
        
        self._record_maintenance("cleanup_logs", result)
        logger.info(f"Cleaned up {deleted_count} old log files, freed {result['freed_space_mb']:.2f} MB")
        
        return result
    
    def cleanup_cache(
        self,
        cache_directory: str = "cache",
        max_size_mb: int = 1000
    ) -> Dict[str, Any]:
        """
        Limpiar caché.
        
        Args:
            cache_directory: Directorio de caché
            max_size_mb: Tamaño máximo en MB
            
        Returns:
            Información de limpieza
        """
        cache_path = Path(cache_directory)
        if not cache_path.exists():
            return {"deleted": 0, "freed_space": 0}
        
        # Calcular tamaño total
        total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if total_size <= max_size_bytes:
            return {"deleted": 0, "freed_space": 0, "current_size": total_size}
        
        # Eliminar archivos más antiguos hasta alcanzar el límite
        files = []
        for file in cache_path.rglob("*"):
            if file.is_file():
                files.append((file, file.stat().st_mtime, file.stat().st_size))
        
        # Ordenar por fecha (más antiguos primero)
        files.sort(key=lambda x: x[1])
        
        deleted_count = 0
        freed_space = 0
        current_size = total_size
        
        for file, _, size in files:
            if current_size <= max_size_bytes:
                break
            
            try:
                file.unlink()
                deleted_count += 1
                freed_space += size
                current_size -= size
            except Exception as e:
                logger.warning(f"Error deleting cache file {file}: {e}")
        
        result = {
            "deleted": deleted_count,
            "freed_space": freed_space,
            "freed_space_mb": freed_space / (1024 * 1024),
            "current_size": current_size,
            "current_size_mb": current_size / (1024 * 1024)
        }
        
        self._record_maintenance("cleanup_cache", result)
        logger.info(f"Cleaned up cache: {deleted_count} files, freed {result['freed_space_mb']:.2f} MB")
        
        return result
    
    def cleanup_temp_files(
        self,
        temp_directory: str = "temp",
        days_to_keep: int = 1
    ) -> Dict[str, Any]:
        """
        Limpiar archivos temporales.
        
        Args:
            temp_directory: Directorio temporal
            days_to_keep: Días a mantener
            
        Returns:
            Información de limpieza
        """
        temp_path = Path(temp_directory)
        if not temp_path.exists():
            return {"deleted": 0, "freed_space": 0}
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        freed_space = 0
        
        for file in temp_path.rglob("*"):
            if file.is_file():
                try:
                    file_time = datetime.fromtimestamp(file.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_size = file.stat().st_size
                        file.unlink()
                        deleted_count += 1
                        freed_space += file_size
                except Exception as e:
                    logger.warning(f"Error deleting temp file {file}: {e}")
        
        result = {
            "deleted": deleted_count,
            "freed_space": freed_space,
            "freed_space_mb": freed_space / (1024 * 1024)
        }
        
        self._record_maintenance("cleanup_temp", result)
        logger.info(f"Cleaned up {deleted_count} temp files, freed {result['freed_space_mb']:.2f} MB")
        
        return result
    
    def optimize_database(
        self,
        database_path: str = "data/database.db"
    ) -> Dict[str, Any]:
        """
        Optimizar base de datos (si existe).
        
        Args:
            database_path: Ruta de la base de datos
            
        Returns:
            Información de optimización
        """
        db_path = Path(database_path)
        if not db_path.exists():
            return {"optimized": False, "message": "Database not found"}
        
        try:
            import sqlite3
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # VACUUM para optimizar
            cursor.execute("VACUUM")
            
            # ANALYZE para actualizar estadísticas
            cursor.execute("ANALYZE")
            
            conn.commit()
            conn.close()
            
            result = {
                "optimized": True,
                "message": "Database optimized successfully"
            }
            
            self._record_maintenance("optimize_database", result)
            logger.info("Database optimized successfully")
            
            return result
        
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return {
                "optimized": False,
                "message": f"Error: {str(e)}"
            }
    
    def run_full_maintenance(self) -> Dict[str, Any]:
        """
        Ejecutar mantenimiento completo.
        
        Returns:
            Resumen de mantenimiento
        """
        logger.info("Running full maintenance...")
        
        results = {
            "logs": self.cleanup_old_logs(),
            "cache": self.cleanup_cache(),
            "temp": self.cleanup_temp_files(),
            "database": self.optimize_database()
        }
        
        total_freed = (
            results["logs"]["freed_space_mb"] +
            results["cache"]["freed_space_mb"] +
            results["temp"]["freed_space_mb"]
        )
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "total_freed_mb": total_freed
        }
        
        self._record_maintenance("full_maintenance", summary)
        logger.info(f"Full maintenance completed, freed {total_freed:.2f} MB total")
        
        return summary
    
    def _record_maintenance(self, task: str, result: Dict[str, Any]) -> None:
        """Registrar tarea de mantenimiento."""
        self.maintenance_history.append({
            "task": task,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_maintenance_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de mantenimiento."""
        return self.maintenance_history[-limit:]


# Instancia global
_maintenance_manager: Optional[MaintenanceManager] = None


def get_maintenance_manager() -> MaintenanceManager:
    """Obtener instancia global del gestor de mantenimiento."""
    global _maintenance_manager
    if _maintenance_manager is None:
        _maintenance_manager = MaintenanceManager()
    return _maintenance_manager






