"""
Serialization Generator - Generador de utilidades de serialización
===================================================================

Genera utilidades para serialización avanzada:
- Model checkpointing avanzado
- State dict management
- Model versioning
- Backup y restore
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SerializationGenerator:
    """Generador de utilidades de serialización"""
    
    def __init__(self):
        """Inicializa el generador de serialización"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de serialización.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        serialization_dir = utils_dir / "serialization"
        serialization_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_checkpoint_manager(serialization_dir, keywords, project_info)
        self._generate_model_serializer(serialization_dir, keywords, project_info)
        self._generate_serialization_init(serialization_dir, keywords)
    
    def _generate_serialization_init(
        self,
        serialization_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de serialización"""
        
        init_content = '''"""
Serialization Utilities Module
===============================

Utilidades para serialización avanzada de modelos y checkpoints.
"""

from .checkpoint_manager import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
)
from .model_serializer import (
    ModelSerializer,
    serialize_model,
    deserialize_model,
    get_model_info,
)

__all__ = [
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "list_checkpoints",
    "ModelSerializer",
    "serialize_model",
    "deserialize_model",
    "get_model_info",
]
'''
        
        (serialization_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_checkpoint_manager(
        self,
        serialization_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera gestor de checkpoints"""
        
        checkpoint_content = '''"""
Checkpoint Manager - Gestor de checkpoints
===========================================

Sistema avanzado para gestionar checkpoints de modelos.
"""

import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Gestor avanzado de checkpoints.
    
    Permite guardar, cargar y gestionar múltiples checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 10,
        keep_best: bool = True,
    ):
        """
        Inicializa el gestor.
        
        Args:
            checkpoint_dir: Directorio donde guardar checkpoints
            max_checkpoints: Número máximo de checkpoints a mantener
            keep_best: Si mantener siempre el mejor checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.metadata = self._load_metadata()
        self.best_metric = float('-inf')
        self.best_checkpoint = None
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Carga metadata de checkpoints"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """Guarda metadata"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando metadata: {e}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Guarda un checkpoint.
        
        Args:
            model: Modelo a guardar
            optimizer: Optimizador (opcional)
            scheduler: Scheduler (opcional)
            epoch: Época actual
            step: Step actual
            metrics: Métricas (opcional)
            is_best: Si es el mejor checkpoint
            additional_info: Información adicional (opcional)
        
        Returns:
            Ruta al checkpoint guardado
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if metrics:
            checkpoint["metrics"] = metrics
            
            # Actualizar mejor métrica
            if "val_loss" in metrics:
                val_loss = metrics["val_loss"]
                if val_loss < -self.best_metric or self.best_checkpoint is None:
                    self.best_metric = -val_loss
                    self.best_checkpoint = checkpoint_path
        
        if additional_info:
            checkpoint["additional_info"] = additional_info
        
        # Guardar checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Actualizar metadata
        self.metadata[checkpoint_name] = {
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "timestamp": checkpoint["timestamp"],
            "is_best": is_best,
        }
        
        self._save_metadata()
        
        # Guardar mejor checkpoint por separado
        if is_best or (self.keep_best and checkpoint_path == self.best_checkpoint):
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy(checkpoint_path, best_path)
            logger.info(f"Mejor modelo guardado en {best_path}")
        
        # Limpiar checkpoints antiguos
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint guardado en {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        """
        Carga un checkpoint.
        
        Args:
            checkpoint_path: Ruta al checkpoint (opcional)
            model: Modelo donde cargar (opcional)
            optimizer: Optimizador donde cargar (opcional)
            scheduler: Scheduler donde cargar (opcional)
            load_best: Si cargar el mejor checkpoint
        
        Returns:
            Diccionario con información del checkpoint
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        elif checkpoint_path is None:
            # Cargar último checkpoint
            checkpoints = self.list_checkpoints()
            if not checkpoints:
                raise ValueError("No hay checkpoints disponibles")
            checkpoint_path = checkpoints[-1]["path"]
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if model:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Estado del modelo cargado")
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Estado del optimizador cargado")
        
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Estado del scheduler cargado")
        
        logger.info(f"Checkpoint cargado desde {checkpoint_path}")
        return checkpoint
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Lista todos los checkpoints disponibles.
        
        Returns:
            Lista de checkpoints con información
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pt"):
            if checkpoint_file.name in self.metadata:
                info = self.metadata[checkpoint_file.name].copy()
                info["path"] = checkpoint_file
                checkpoints.append(info)
        
        # Ordenar por epoch y step
        checkpoints.sort(key=lambda x: (x["epoch"], x["step"]))
        
        return checkpoints
    
    def _cleanup_old_checkpoints(self) -> None:
        """Elimina checkpoints antiguos si excede el máximo"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Mantener el mejor
        checkpoints_to_remove = checkpoints[:-self.max_checkpoints]
        
        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = checkpoint_info["path"]
            if checkpoint_path.exists() and not checkpoint_info.get("is_best", False):
                checkpoint_path.unlink()
                logger.info(f"Checkpoint antiguo eliminado: {checkpoint_path}")


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    epoch: int = 0,
    **kwargs,
) -> Path:
    """
    Función helper para guardar checkpoint.
    
    Args:
        checkpoint_dir: Directorio de checkpoints
        model: Modelo a guardar
        epoch: Época
        **kwargs: Argumentos adicionales
    
    Returns:
        Ruta al checkpoint
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.save_checkpoint(model, epoch=epoch, **kwargs)


def load_checkpoint(
    checkpoint_dir: Path,
    model: Optional[torch.nn.Module] = None,
    load_best: bool = False,
) -> Dict[str, Any]:
    """
    Función helper para cargar checkpoint.
    
    Args:
        checkpoint_dir: Directorio de checkpoints
        model: Modelo donde cargar (opcional)
        load_best: Si cargar el mejor
    
    Returns:
        Información del checkpoint
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.load_checkpoint(model=model, load_best=load_best)


def list_checkpoints(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    """
    Función helper para listar checkpoints.
    
    Args:
        checkpoint_dir: Directorio de checkpoints
    
    Returns:
        Lista de checkpoints
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.list_checkpoints()
'''
        
        (serialization_dir / "checkpoint_manager.py").write_text(checkpoint_content, encoding="utf-8")
    
    def _generate_model_serializer(
        self,
        serialization_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera serializador de modelos"""
        
        serializer_content = '''"""
Model Serializer - Serializador de modelos
==========================================

Utilidades para serializar y deserializar modelos con metadata.
"""

import torch
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelSerializer:
    """
    Serializador de modelos con metadata.
    
    Permite guardar modelos con información completa.
    """
    
    def __init__(self, models_dir: Path):
        """
        Inicializa el serializador.
        
        Args:
            models_dir: Directorio donde guardar modelos
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def serialize_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        include_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Path:
        """
        Serializa un modelo con metadata.
        
        Args:
            model: Modelo a serializar
            model_name: Nombre del modelo
            metadata: Metadata adicional (opcional)
            include_optimizer: Si incluir optimizador
            optimizer: Optimizador (opcional)
        
        Returns:
            Ruta al modelo serializado
        """
        model_path = self.models_dir / f"{model_name}.pt"
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        # Guardar modelo
        save_dict = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
        }
        
        if include_optimizer and optimizer:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(save_dict, model_path)
        
        # Calcular hash del modelo
        model_hash = self._calculate_model_hash(model_path)
        
        # Guardar metadata
        model_metadata = {
            "model_name": model_name,
            "model_path": str(model_path),
            "model_hash": model_hash,
            "model_class": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=2)
        
        logger.info(f"Modelo serializado en {model_path}")
        logger.info(f"Metadata guardada en {metadata_path}")
        
        return model_path
    
    def deserialize_model(
        self,
        model_name: str,
        model_class: Optional[torch.nn.Module] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Deserializa un modelo.
        
        Args:
            model_name: Nombre del modelo
            model_class: Clase del modelo (opcional)
            device: Dispositivo donde cargar
        
        Returns:
            Diccionario con modelo y metadata
        """
        model_path = self.models_dir / f"{model_name}.pt"
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        # Cargar modelo
        checkpoint = torch.load(model_path, map_location=device)
        
        # Cargar metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
        # Verificar hash
        current_hash = self._calculate_model_hash(model_path)
        if "model_hash" in metadata and current_hash != metadata["model_hash"]:
            logger.warning("Hash del modelo no coincide. El modelo puede estar corrupto.")
        
        result = {
            "model_state_dict": checkpoint["model_state_dict"],
            "metadata": metadata,
        }
        
        if "optimizer_state_dict" in checkpoint:
            result["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
        
        if model_class:
            model = model_class()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            result["model"] = model
        
        logger.info(f"Modelo deserializado desde {model_path}")
        return result
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Obtiene información de un modelo.
        
        Args:
            model_name: Nombre del modelo
        
        Returns:
            Diccionario con información
        """
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata no encontrada: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """
        Calcula hash de un modelo.
        
        Args:
            model_path: Ruta al modelo
        
        Returns:
            Hash del modelo
        """
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


def serialize_model(
    models_dir: Path,
    model: torch.nn.Module,
    model_name: str,
    **kwargs,
) -> Path:
    """
    Función helper para serializar modelo.
    
    Args:
        models_dir: Directorio de modelos
        model: Modelo a serializar
        model_name: Nombre del modelo
        **kwargs: Argumentos adicionales
    
    Returns:
        Ruta al modelo
    """
    serializer = ModelSerializer(models_dir)
    return serializer.serialize_model(model, model_name, **kwargs)


def deserialize_model(
    models_dir: Path,
    model_name: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Función helper para deserializar modelo.
    
    Args:
        models_dir: Directorio de modelos
        model_name: Nombre del modelo
        **kwargs: Argumentos adicionales
    
    Returns:
        Diccionario con modelo y metadata
    """
    serializer = ModelSerializer(models_dir)
    return serializer.deserialize_model(model_name, **kwargs)


def get_model_info(models_dir: Path, model_name: str) -> Dict[str, Any]:
    """
    Función helper para obtener información de modelo.
    
    Args:
        models_dir: Directorio de modelos
        model_name: Nombre del modelo
    
    Returns:
        Diccionario con información
    """
    serializer = ModelSerializer(models_dir)
    return serializer.get_model_info(model_name)
'''
        
        (serialization_dir / "model_serializer.py").write_text(serializer_content, encoding="utf-8")

