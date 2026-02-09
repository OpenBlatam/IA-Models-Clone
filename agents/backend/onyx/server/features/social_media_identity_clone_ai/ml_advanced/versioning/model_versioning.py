"""
Versionado avanzado de modelos
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import torch
import hashlib

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """Gestor de versionado de modelos"""
    
    def __init__(self, version_dir: str = "./models/versions"):
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.version_dir / "versions.json"
        self._load_versions()
    
    def _load_versions(self):
        """Carga versiones desde archivo"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                self.versions = json.load(f)
        else:
            self.versions = {"models": []}
    
    def _save_versions(self):
        """Guarda versiones a archivo"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def create_version(
        self,
        model: torch.nn.Module,
        model_name: str,
        metadata: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Crea nueva versión de modelo
        
        Args:
            model: Modelo a versionar
            model_name: Nombre del modelo
            metadata: Metadatos (accuracy, loss, etc.)
            tags: Tags para categorización
            
        Returns:
            Version ID
        """
        # Generar hash del modelo
        model_hash = self._compute_model_hash(model)
        
        # Crear versión
        version_id = f"{model_name}_v{len(self.versions['models']) + 1}"
        timestamp = datetime.utcnow().isoformat()
        
        version_info = {
            "version_id": version_id,
            "model_name": model_name,
            "model_hash": model_hash,
            "timestamp": timestamp,
            "metadata": metadata,
            "tags": tags or [],
            "is_active": False,
            "is_production": False
        }
        
        # Guardar modelo
        model_path = self.version_dir / f"{version_id}.pt"
        torch.save(model.state_dict(), model_path)
        version_info["model_path"] = str(model_path)
        
        self.versions["models"].append(version_info)
        self._save_versions()
        
        logger.info(f"Versión creada: {version_id}")
        return version_id
    
    def _compute_model_hash(self, model: torch.nn.Module) -> str:
        """Calcula hash del modelo"""
        model_str = str(model.state_dict())
        return hashlib.sha256(model_str.encode()).hexdigest()[:16]
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de versión"""
        for version in self.versions["models"]:
            if version["version_id"] == version_id:
                return version
        return None
    
    def list_versions(
        self,
        model_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Lista versiones"""
        versions = self.versions["models"]
        
        if model_name:
            versions = [v for v in versions if v["model_name"] == model_name]
        
        if tags:
            versions = [v for v in versions if any(tag in v.get("tags", []) for tag in tags)]
        
        return sorted(versions, key=lambda x: x["timestamp"], reverse=True)
    
    def set_production(self, version_id: str):
        """Marca versión como producción"""
        for version in self.versions["models"]:
            if version["version_id"] == version_id:
                # Desactivar otras versiones de producción
                for v in self.versions["models"]:
                    if v["model_name"] == version["model_name"]:
                        v["is_production"] = False
                
                version["is_production"] = True
                version["is_active"] = True
                break
        
        self._save_versions()
        logger.info(f"Versión {version_id} marcada como producción")
    
    def compare_versions(
        self,
        version_id1: str,
        version_id2: str
    ) -> Dict[str, Any]:
        """Compara dos versiones"""
        v1 = self.get_version(version_id1)
        v2 = self.get_version(version_id2)
        
        if not v1 or not v2:
            return {"error": "Versión no encontrada"}
        
        comparison = {
            "version1": version_id1,
            "version2": version_id2,
            "metadata_diff": {},
            "same_model": v1["model_hash"] == v2["model_hash"]
        }
        
        # Comparar metadata
        for key in set(v1["metadata"].keys()) | set(v2["metadata"].keys()):
            val1 = v1["metadata"].get(key)
            val2 = v2["metadata"].get(key)
            if val1 != val2:
                comparison["metadata_diff"][key] = {
                    "v1": val1,
                    "v2": val2
                }
        
        return comparison




