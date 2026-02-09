"""
Model Registry Service - Registro y versionado de modelos
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Etapas del modelo"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelRegistryService:
    """Servicio para registro de modelos"""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.versions: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        description: str,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Registrar nuevo modelo"""
        
        model_id = f"model_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        model = {
            "model_id": model_id,
            "name": model_name,
            "type": model_type,
            "description": description,
            "tags": tags or [],
            "stage": ModelStage.DEVELOPMENT.value,
            "created_at": datetime.now().isoformat(),
            "versions": []
        }
        
        self.models[model_id] = model
        
        return model
    
    def create_version(
        self,
        model_id: str,
        version_name: str,
        checkpoint_path: str,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Crear nueva versión del modelo"""
        
        model = self.models.get(model_id)
        
        if not model:
            raise ValueError(f"Modelo {model_id} no encontrado")
        
        version_number = len(model["versions"]) + 1
        
        version = {
            "version_id": f"{model_id}_v{version_number}",
            "version_number": version_number,
            "version_name": version_name,
            "checkpoint_path": checkpoint_path,
            "metrics": metrics or {},
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "stage": ModelStage.DEVELOPMENT.value
        }
        
        model["versions"].append(version["version_id"])
        
        if model_id not in self.versions:
            self.versions[model_id] = []
        
        self.versions[model_id].append(version)
        
        return version
    
    def promote_version(
        self,
        model_id: str,
        version_id: str,
        target_stage: str
    ) -> Dict[str, Any]:
        """Promover versión a nueva etapa"""
        
        versions = self.versions.get(model_id, [])
        version = next((v for v in versions if v["version_id"] == version_id), None)
        
        if not version:
            raise ValueError(f"Versión {version_id} no encontrada")
        
        old_stage = version["stage"]
        version["stage"] = target_stage
        version["promoted_at"] = datetime.now().isoformat()
        
        return {
            "version_id": version_id,
            "old_stage": old_stage,
            "new_stage": target_stage,
            "promoted_at": version["promoted_at"]
        }
    
    def get_latest_version(
        self,
        model_id: str,
        stage: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Obtener última versión"""
        
        versions = self.versions.get(model_id, [])
        
        if stage:
            versions = [v for v in versions if v["stage"] == stage]
        
        if not versions:
            return None
        
        return max(versions, key=lambda v: v["version_number"])
    
    def compare_versions(
        self,
        model_id: str,
        version_ids: List[str]
    ) -> Dict[str, Any]:
        """Comparar versiones"""
        
        versions = self.versions.get(model_id, [])
        selected_versions = [v for v in versions if v["version_id"] in version_ids]
        
        comparison = {
            "model_id": model_id,
            "versions": selected_versions,
            "compared_at": datetime.now().isoformat(),
            "note": "En producción, esto compararía métricas y rendimiento"
        }
        
        return comparison
    
    def search_models(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Buscar modelos"""
        
        results = []
        
        for model_id, model in self.models.items():
            match = True
            
            if query and query.lower() not in model["name"].lower() and query.lower() not in model.get("description", "").lower():
                match = False
            
            if tags and not all(tag in model.get("tags", []) for tag in tags):
                match = False
            
            if stage and model["stage"] != stage:
                match = False
            
            if match:
                results.append(model)
        
        return results




