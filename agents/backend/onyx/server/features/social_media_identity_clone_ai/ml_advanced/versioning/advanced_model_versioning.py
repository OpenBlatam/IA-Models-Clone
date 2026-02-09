"""
Versionado avanzado de modelos con hash, metadata y gestión completa

Mejoras:
- Hash SHA256 para integridad
- Metadata completa
- Comparación de versiones
- Gestión de producción
- Tags y labels
- Rollback automático
"""

import logging
import torch
import hashlib
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import pickle

from ..core.base_service import BaseService
from ..core.exceptions import ModelLoadingError, ValidationError

logger = logging.getLogger(__name__)


class AdvancedModelVersioning(BaseService):
    """
    Sistema avanzado de versionado de modelos
    
    Features:
    - Hash SHA256 para integridad
    - Metadata completa
    - Comparación de versiones
    - Gestión de producción
    - Tags y labels
    - Rollback automático
    """
    
    def __init__(self, models_dir: str = "./models"):
        super().__init__()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.models_dir / "versions.json"
        self._load_versions()
    
    def _load_versions(self):
        """Carga información de versiones"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    self.versions = json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando versiones: {e}")
                self.versions = {}
        else:
            self.versions = {}
    
    def _save_versions(self):
        """Guarda información de versiones"""
        try:
            with open(self.versions_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando versiones: {e}")
    
    def save_model_version(
        self,
        model: torch.nn.Module,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        is_production: bool = False
    ) -> Dict[str, Any]:
        """
        Guarda versión del modelo con hash y metadata
        
        Args:
            model: Modelo a guardar
            version: Versión del modelo (ej: "v1.0.0")
            metadata: Metadata adicional
            tags: Tags para el modelo
            is_production: Si es versión de producción
            
        Returns:
            Información de la versión guardada
        """
        try:
            # Crear directorio de versión
            version_dir = self.models_dir / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar modelo
            model_path = version_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            
            # Calcular hash
            with open(model_path, 'rb') as f:
                model_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Metadata completa
            version_metadata = {
                "version": version,
                "model_hash": model_hash,
                "model_path": str(model_path),
                "created_at": datetime.now().isoformat(),
                "tags": tags or [],
                "is_production": is_production,
                "metadata": metadata or {},
                "model_info": {
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
                }
            }
            
            # Guardar metadata
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(version_metadata, f, indent=2)
            
            # Actualizar registro de versiones
            self.versions[version] = version_metadata
            self._save_versions()
            
            # Si es producción, actualizar producción
            if is_production:
                self.set_production_version(version)
            
            self.logger.info(f"Modelo versión {version} guardado (hash: {model_hash[:8]}...)")
            
            return version_metadata
            
        except Exception as e:
            self.logger.error(f"Error guardando versión: {e}", exc_info=True)
            raise ModelLoadingError(
                f"Error guardando versión: {str(e)}",
                error_code="VERSION_SAVE_ERROR"
            ) from e
    
    def load_model_version(
        self,
        version: str,
        model_class: Optional[type] = None,
        verify_hash: bool = True
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Carga versión del modelo con verificación de hash
        
        Args:
            version: Versión a cargar
            model_class: Clase del modelo (opcional)
            verify_hash: Si verificar hash
            
        Returns:
            Tupla (modelo, metadata)
        """
        if version not in self.versions:
            raise ModelLoadingError(
                f"Versión {version} no encontrada",
                error_code="VERSION_NOT_FOUND"
            )
        
        try:
            version_info = self.versions[version]
            model_path = Path(version_info["model_path"])
            
            if not model_path.exists():
                raise ModelLoadingError(
                    f"Archivo de modelo no encontrado: {model_path}",
                    error_code="MODEL_FILE_NOT_FOUND"
                )
            
            # Verificar hash
            if verify_hash:
                with open(model_path, 'rb') as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()
                
                if current_hash != version_info["model_hash"]:
                    raise ModelLoadingError(
                        f"Hash no coincide para versión {version}. "
                        f"Esperado: {version_info['model_hash'][:8]}..., "
                        f"Obtenido: {current_hash[:8]}...",
                        error_code="HASH_MISMATCH"
                    )
            
            # Cargar modelo
            if model_class:
                model = model_class()
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            else:
                # Cargar solo state dict
                state_dict = torch.load(model_path, map_location='cpu')
                model = None
            
            # Cargar metadata
            metadata_path = model_path.parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = version_info
            
            self.logger.info(f"Modelo versión {version} cargado exitosamente")
            
            return model, metadata
            
        except ModelLoadingError:
            raise
        except Exception as e:
            self.logger.error(f"Error cargando versión: {e}", exc_info=True)
            raise ModelLoadingError(
                f"Error cargando versión: {str(e)}",
                error_code="VERSION_LOAD_ERROR"
            ) from e
    
    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compara dos versiones del modelo
        
        Args:
            version1: Primera versión
            version2: Segunda versión
            
        Returns:
            Comparación de versiones
        """
        if version1 not in self.versions:
            raise ValidationError(f"Versión {version1} no encontrada")
        if version2 not in self.versions:
            raise ValidationError(f"Versión {version2} no encontrada")
        
        v1_info = self.versions[version1]
        v2_info = self.versions[version2]
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "same_hash": v1_info["model_hash"] == v2_info["model_hash"],
            "model_info": {
                "v1": v1_info.get("model_info", {}),
                "v2": v2_info.get("model_info", {})
            },
            "metadata_diff": self._diff_metadata(
                v1_info.get("metadata", {}),
                v2_info.get("metadata", {})
            ),
            "created_at": {
                "v1": v1_info.get("created_at"),
                "v2": v2_info.get("created_at")
            }
        }
        
        return comparison
    
    def _diff_metadata(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcula diferencia entre metadata"""
        diff = {
            "added": {},
            "removed": {},
            "changed": {}
        }
        
        all_keys = set(metadata1.keys()) | set(metadata2.keys())
        
        for key in all_keys:
            if key not in metadata1:
                diff["added"][key] = metadata2[key]
            elif key not in metadata2:
                diff["removed"][key] = metadata1[key]
            elif metadata1[key] != metadata2[key]:
                diff["changed"][key] = {
                    "old": metadata1[key],
                    "new": metadata2[key]
                }
        
        return diff
    
    def set_production_version(self, version: str):
        """
        Establece versión de producción
        
        Args:
            version: Versión a establecer como producción
        """
        if version not in self.versions:
            raise ValidationError(f"Versión {version} no encontrada")
        
        # Desmarcar otras versiones de producción
        for v in self.versions:
            self.versions[v]["is_production"] = False
        
        # Marcar nueva versión
        self.versions[version]["is_production"] = True
        self.versions[version]["production_set_at"] = datetime.now().isoformat()
        
        self._save_versions()
        self.logger.info(f"Versión {version} establecida como producción")
    
    def get_production_version(self) -> Optional[str]:
        """Obtiene versión de producción actual"""
        for version, info in self.versions.items():
            if info.get("is_production", False):
                return version
        return None
    
    def list_versions(
        self,
        tags: Optional[List[str]] = None,
        production_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Lista versiones con filtros
        
        Args:
            tags: Filtrar por tags
            production_only: Solo versiones de producción
            
        Returns:
            Lista de versiones
        """
        versions = []
        
        for version, info in self.versions.items():
            # Filtrar producción
            if production_only and not info.get("is_production", False):
                continue
            
            # Filtrar tags
            if tags:
                version_tags = info.get("tags", [])
                if not any(tag in version_tags for tag in tags):
                    continue
            
            versions.append({
                "version": version,
                "created_at": info.get("created_at"),
                "tags": info.get("tags", []),
                "is_production": info.get("is_production", False),
                "model_hash": info.get("model_hash", "")[:8] + "...",
                "model_info": info.get("model_info", {})
            })
        
        # Ordenar por fecha
        versions.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
        
        return versions
    
    def rollback_to_version(self, version: str) -> Dict[str, Any]:
        """
        Hace rollback a una versión anterior
        
        Args:
            version: Versión a la que hacer rollback
            
        Returns:
            Información del rollback
        """
        if version not in self.versions:
            raise ValidationError(f"Versión {version} no encontrada")
        
        current_production = self.get_production_version()
        
        # Establecer nueva versión de producción
        self.set_production_version(version)
        
        return {
            "success": True,
            "rolled_back_to": version,
            "previous_production": current_production,
            "rolled_back_at": datetime.now().isoformat()
        }




