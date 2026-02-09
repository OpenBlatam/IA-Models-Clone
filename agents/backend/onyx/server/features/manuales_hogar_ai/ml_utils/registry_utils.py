"""
Registry Utils - Utilidades de Registro de Modelos
===================================================

Utilidades para registro y versionado de modelos.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
from pathlib import Path
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata de modelo."""
    name: str
    version: str
    architecture: str
    num_parameters: int
    created_at: str
    description: Optional[str] = None
    tags: List[str] = None
    metrics: Dict[str, float] = None
    hyperparameters: Dict[str, Any] = None
    file_path: Optional[str] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Inicialización post-construcción."""
        if self.tags is None:
            self.tags = []
        if self.metrics is None:
            self.metrics = {}
        if self.hyperparameters is None:
            self.hyperparameters = {}


class ModelRegistry:
    """
    Registro de modelos con versionado.
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        """
        Inicializar registro.
        
        Args:
            registry_path: Ruta del registro
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
        self.models: Dict[str, List[ModelMetadata]] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Cargar metadata existente."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for name, versions in data.items():
                        self.models[name] = [
                            ModelMetadata(**meta) for meta in versions
                        ]
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")
    
    def _save_metadata(self):
        """Guardar metadata."""
        data = {}
        for name, versions in self.models.items():
            data[name] = [asdict(meta) for meta in versions]
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_checksum(self, file_path: Path) -> str:
        """
        Calcular checksum de archivo.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Checksum
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def register(
        self,
        model: nn.Module,
        name: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        save_model: bool = True
    ) -> ModelMetadata:
        """
        Registrar modelo.
        
        Args:
            model: Modelo a registrar
            name: Nombre del modelo
            version: Versión (opcional, auto-incrementa)
            description: Descripción (opcional)
            tags: Tags (opcional)
            metrics: Métricas (opcional)
            hyperparameters: Hiperparámetros (opcional)
            save_model: Guardar modelo (opcional)
            
        Returns:
            Metadata del modelo
        """
        # Determinar versión
        if name not in self.models:
            self.models[name] = []
        
        if version is None:
            version = f"v{len(self.models[name]) + 1}"
        
        # Contar parámetros
        num_params = sum(p.numel() for p in model.parameters())
        
        # Obtener arquitectura
        architecture = model.__class__.__name__
        
        # Guardar modelo si es necesario
        file_path = None
        checksum = None
        if save_model:
            model_dir = self.registry_path / name / version
            model_dir.mkdir(parents=True, exist_ok=True)
            file_path = str(model_dir / "model.pt")
            torch.save(model.state_dict(), file_path)
            checksum = self._compute_checksum(Path(file_path))
        
        # Crear metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            architecture=architecture,
            num_parameters=num_params,
            created_at=datetime.now().isoformat(),
            description=description,
            tags=tags or [],
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            file_path=file_path,
            checksum=checksum
        )
        
        # Agregar al registro
        self.models[name].append(metadata)
        self._save_metadata()
        
        logger.info(f"Registered model: {name} {version}")
        return metadata
    
    def get_model(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """
        Obtener metadata de modelo.
        
        Args:
            name: Nombre del modelo
            version: Versión (opcional, retorna la más reciente)
            
        Returns:
            Metadata del modelo
        """
        if name not in self.models:
            return None
        
        if version is None:
            # Retornar versión más reciente
            return self.models[name][-1] if self.models[name] else None
        
        # Buscar versión específica
        for meta in self.models[name]:
            if meta.version == version:
                return meta
        
        return None
    
    def list_models(self) -> List[str]:
        """
        Listar nombres de modelos.
        
        Returns:
            Lista de nombres
        """
        return list(self.models.keys())
    
    def list_versions(self, name: str) -> List[str]:
        """
        Listar versiones de un modelo.
        
        Args:
            name: Nombre del modelo
            
        Returns:
            Lista de versiones
        """
        if name not in self.models:
            return []
        return [meta.version for meta in self.models[name]]
    
    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        model_class: Optional[type] = None
    ) -> Optional[nn.Module]:
        """
        Cargar modelo desde registro.
        
        Args:
            name: Nombre del modelo
            version: Versión (opcional)
            model_class: Clase del modelo (opcional)
            
        Returns:
            Modelo cargado
        """
        metadata = self.get_model(name, version)
        if metadata is None or metadata.file_path is None:
            return None
        
        if model_class is None:
            # Intentar cargar solo state dict
            state_dict = torch.load(metadata.file_path)
            logger.warning("No model class provided, returning state dict")
            return state_dict
        
        # Cargar modelo completo
        model = model_class()
        model.load_state_dict(torch.load(metadata.file_path))
        return model
    
    def compare_models(
        self,
        name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Comparar dos versiones de un modelo.
        
        Args:
            name: Nombre del modelo
            version1: Primera versión
            version2: Segunda versión
            
        Returns:
            Comparación
        """
        meta1 = self.get_model(name, version1)
        meta2 = self.get_model(name, version2)
        
        if meta1 is None or meta2 is None:
            return {}
        
        comparison = {
            'parameters': {
                'v1': meta1.num_parameters,
                'v2': meta2.num_parameters,
                'diff': meta2.num_parameters - meta1.num_parameters
            },
            'metrics': {
                'v1': meta1.metrics,
                'v2': meta2.metrics
            },
            'created_at': {
                'v1': meta1.created_at,
                'v2': meta2.created_at
            }
        }
        
        return comparison




