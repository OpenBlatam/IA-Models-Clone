"""
Data Generator - Generador de data loaders y datasets
======================================================

Genera módulos especializados para manejo de datos:
- Custom datasets
- Data loaders optimizados
- Data transformations
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generador de utilidades de datos"""
    
    def __init__(self):
        """Inicializa el generador de datos"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de datos.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar módulos de datos
        self._generate_dataset(utils_dir, keywords, project_info)
        self._generate_dataloader(utils_dir, keywords, project_info)
        self._generate_transforms(utils_dir, keywords, project_info)
        self._generate_data_init(utils_dir, keywords)
    
    def _generate_data_init(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de datos"""
        
        init_content = '''"""
Data Utilities Module
=====================

Utilidades para carga y procesamiento de datos.
"""

from .dataset import CustomDataset
from .dataloader import create_dataloader
from .transforms import get_transforms

__all__ = [
    "CustomDataset",
    "create_dataloader",
    "get_transforms",
]
'''
        
        data_dir = utils_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_dataset(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera clase CustomDataset"""
        
        dataset_content = '''"""
Custom Dataset - Dataset personalizado
========================================

Implementa torch.utils.data.Dataset para tu caso de uso específico.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Any, List
import logging

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """
    Dataset personalizado.
    
    Extiende torch.utils.data.Dataset para tu caso de uso específico.
    """
    
    def __init__(
        self,
        data: List[Any],
        labels: Optional[List[Any]] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Inicializa el dataset.
        
        Args:
            data: Datos del dataset
            labels: Etiquetas (opcional)
            transform: Transformaciones a aplicar (opcional)
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        
        if labels is not None and len(data) != len(labels):
            raise ValueError("Data y labels deben tener la misma longitud")
    
    def __len__(self) -> int:
        """Retorna el tamaño del dataset"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Any:
        """
        Obtiene un item del dataset.
        
        Args:
            idx: Índice del item
            
        Returns:
            Item procesado
        """
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
            label = self.labels[idx]
            return sample, label
        
        return sample
    
    def get_statistics(self) -> dict:
        """Obtiene estadísticas del dataset"""
        return {
            "size": len(self.data),
            "has_labels": self.labels is not None,
            "transform": self.transform is not None,
        }
'''
        
        data_dir = utils_dir / "data"
        (data_dir / "dataset.py").write_text(dataset_content, encoding="utf-8")
    
    def _generate_dataloader(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades para DataLoader"""
        
        dataloader_content = '''"""
DataLoader Utilities - Utilidades para DataLoader
==================================================

Implementa DataLoader eficiente con mejoras de performance.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
) -> DataLoader:
    """
    Crea un DataLoader optimizado.
    
    Args:
        dataset: Dataset a usar
        batch_size: Tamaño del batch
        shuffle: Si mezclar los datos
        num_workers: Número de workers para carga paralela
        pin_memory: Si usar pinned memory (mejor para GPU)
        drop_last: Si descartar último batch incompleto
        distributed: Si usar DistributedSampler
        
    Returns:
        DataLoader configurado
    """
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=16 if num_workers > 0 else None,  # Prefetch ultra máximo para velocidad
        pin_memory_device="cuda" if torch.cuda.is_available() else "cpu",  # Optimización adicional
        multiprocessing_context="spawn" if num_workers > 0 else None,  # Más rápido que fork
        generator=torch.Generator() if torch.cuda.is_available() else None,  # Generator para reproducibilidad rápida
    )
    
    logger.info(
        f"DataLoader creado: batch_size={batch_size}, "
        f"num_workers={num_workers}, shuffle={shuffle}"
    )
    
    return dataloader
'''
        
        data_dir = utils_dir / "data"
        (data_dir / "dataloader.py").write_text(dataloader_content, encoding="utf-8")
    
    def _generate_transforms(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades para transformaciones"""
        
        transforms_content = '''"""
Data Transforms - Transformaciones de datos
=============================================

Utilidades para transformar datos antes del entrenamiento.
"""

from typing import Callable, Optional
import torch
import torchvision.transforms as transforms

def get_transforms(
    transform_type: str = "none",
    **kwargs,
) -> Optional[Callable]:
    """
    Obtiene transformaciones según el tipo.
    
    Args:
        transform_type: Tipo de transformación (none, normalize, augment)
        **kwargs: Argumentos adicionales
        
    Returns:
        Función de transformación o None
    """
    if transform_type == "none":
        return None
    
    elif transform_type == "normalize":
        mean = kwargs.get("mean", [0.5, 0.5, 0.5])
        std = kwargs.get("std", [0.5, 0.5, 0.5])
        return transforms.Normalize(mean=mean, std=std)
    
    elif transform_type == "augment":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    
    else:
        raise ValueError(f"Transform type {transform_type} no soportado")
'''
        
        data_dir = utils_dir / "data"
        (data_dir / "transforms.py").write_text(transforms_content, encoding="utf-8")

