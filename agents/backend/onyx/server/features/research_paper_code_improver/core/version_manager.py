"""
Version Manager - Sistema de versionado de mejoras
===================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class VersionManager:
    """
    Gestiona versiones de mejoras de código.
    """
    
    def __init__(self, versions_dir: str = "data/versions"):
        """
        Inicializar gestor de versiones.
        
        Args:
            versions_dir: Directorio para almacenar versiones
        """
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
    
    def create_version(
        self,
        file_path: str,
        code: str,
        improvement_data: Dict[str, Any],
        version_label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Crea una nueva versión de código mejorado.
        
        Args:
            file_path: Ruta del archivo
            code: Código mejorado
            improvement_data: Datos de la mejora
            version_label: Etiqueta de versión (opcional)
            
        Returns:
            Información de la versión creada
        """
        # Generar hash del código
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
        
        # Generar ID de versión
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{file_path.replace('/', '_')}_{timestamp}_{code_hash}"
        
        version_info = {
            "version_id": version_id,
            "file_path": file_path,
            "code_hash": code_hash,
            "version_label": version_label or f"v{timestamp}",
            "created_at": datetime.now().isoformat(),
            "improvement_data": improvement_data,
            "code": code
        }
        
        # Guardar versión
        version_file = self.versions_dir / f"{version_id}.json"
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Versión creada: {version_id}")
        
        return version_info
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene una versión específica.
        
        Args:
            version_id: ID de la versión
            
        Returns:
            Información de la versión o None
        """
        version_file = self.versions_dir / f"{version_id}.json"
        
        if not version_file.exists():
            return None
        
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error cargando versión: {e}")
            return None
    
    def list_versions(
        self,
        file_path: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Lista versiones.
        
        Args:
            file_path: Filtrar por ruta de archivo (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de versiones
        """
        versions = []
        
        for version_file in self.versions_dir.glob("*.json"):
            try:
                with open(version_file, "r", encoding="utf-8") as f:
                    version = json.load(f)
                
                if not file_path or version.get("file_path") == file_path:
                    # No incluir código completo en lista
                    version_info = {
                        "version_id": version["version_id"],
                        "file_path": version["file_path"],
                        "code_hash": version["code_hash"],
                        "version_label": version["version_label"],
                        "created_at": version["created_at"]
                    }
                    versions.append(version_info)
            except Exception as e:
                logger.warning(f"Error procesando {version_file}: {e}")
                continue
        
        # Ordenar por fecha (más recientes primero)
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        
        return versions[:limit]
    
    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str
    ) -> Dict[str, Any]:
        """
        Compara dos versiones.
        
        Args:
            version_id_1: ID de la primera versión
            version_id_2: ID de la segunda versión
            
        Returns:
            Comparación de versiones
        """
        version_1 = self.get_version(version_id_1)
        version_2 = self.get_version(version_id_2)
        
        if not version_1 or not version_2:
            return {
                "error": "Una o ambas versiones no encontradas"
            }
        
        code_1 = version_1.get("code", "")
        code_2 = version_2.get("code", "")
        
        # Calcular diferencias básicas
        lines_1 = code_1.split("\n")
        lines_2 = code_2.split("\n")
        
        return {
            "version_1": {
                "version_id": version_id_1,
                "lines": len(lines_1),
                "created_at": version_1["created_at"]
            },
            "version_2": {
                "version_id": version_id_2,
                "lines": len(lines_2),
                "created_at": version_2["created_at"]
            },
            "differences": {
                "lines_added": max(0, len(lines_2) - len(lines_1)),
                "lines_removed": max(0, len(lines_1) - len(lines_2)),
                "lines_changed": abs(len(lines_1) - len(lines_2))
            },
            "same_hash": version_1["code_hash"] == version_2["code_hash"]
        }
    
    def get_latest_version(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene la última versión de un archivo.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Última versión o None
        """
        versions = self.list_versions(file_path=file_path, limit=1)
        
        if versions:
            return self.get_version(versions[0]["version_id"])
        
        return None




