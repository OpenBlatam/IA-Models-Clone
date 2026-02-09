"""
Route Dataset
=============

Dataset y DataLoader para datos de enrutamiento.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .preprocessing import RoutePreprocessor


@dataclass
class RouteSample:
    """Muestra de datos de ruta."""
    features: np.ndarray
    target: np.ndarray
    metadata: Dict[str, Any]


class RouteDataset(Dataset):
    """
    Dataset para entrenamiento de modelos de enrutamiento.
    """
    
    def __init__(
        self,
        features: List[np.ndarray],
        targets: List[np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        preprocessor: Optional[RoutePreprocessor] = None
    ):
        """
        Inicializar dataset.
        
        Args:
            features: Lista de arrays de features
            targets: Lista de arrays de targets
            metadata: Metadata adicional (opcional)
            preprocessor: Preprocesador (opcional)
        """
        self.features = features
        self.targets = targets
        self.metadata = metadata or [{}] * len(features)
        self.preprocessor = preprocessor
        
        # Validar longitudes
        assert len(features) == len(targets), "Features y targets deben tener la misma longitud"
        assert len(features) == len(self.metadata), "Metadata debe tener la misma longitud"
    
    def __len__(self) -> int:
        """Tamaño del dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Obtener muestra.
        
        Args:
            idx: Índice
            
        Returns:
            (features, target, metadata)
        """
        features = self.features[idx].copy()
        target = self.targets[idx].copy()
        metadata = self.metadata[idx].copy()
        
        # Preprocesar si hay preprocesador
        if self.preprocessor:
            features = self.preprocessor.transform_features(features)
            target = self.preprocessor.transform_target(target)
        
        # Convertir a tensores
        features_tensor = torch.FloatTensor(features)
        target_tensor = torch.FloatTensor(target)
        
        return features_tensor, target_tensor, metadata
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple['RouteDataset', 'RouteDataset', 'RouteDataset']:
        """
        Dividir dataset en train/val/test.
        
        Args:
            train_ratio: Proporción de entrenamiento
            val_ratio: Proporción de validación
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        total = len(self)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_features = self.features[:train_size]
        train_targets = self.targets[:train_size]
        train_metadata = self.metadata[:train_size]
        
        val_features = self.features[train_size:train_size + val_size]
        val_targets = self.targets[train_size:train_size + val_size]
        val_metadata = self.metadata[train_size:train_size + val_size]
        
        test_features = self.features[train_size + val_size:]
        test_targets = self.targets[train_size + val_size:]
        test_metadata = self.metadata[train_size + val_size:]
        
        train_dataset = RouteDataset(train_features, train_targets, train_metadata, self.preprocessor)
        val_dataset = RouteDataset(val_features, val_targets, val_metadata, self.preprocessor)
        test_dataset = RouteDataset(test_features, test_targets, test_metadata, self.preprocessor)
        
        return train_dataset, val_dataset, test_dataset


class RouteDataLoader:
    """
    Wrapper para DataLoader con configuración predefinida.
    """
    
    @staticmethod
    def create(
        dataset: RouteDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False
    ) -> DataLoader:
        """
        Crear DataLoader.
        
        Args:
            dataset: Dataset
            batch_size: Tamaño de batch
            shuffle: Mezclar datos
            num_workers: Número de workers
            pin_memory: Pin memory para GPU
            drop_last: Descartar último batch incompleto
            
        Returns:
            DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    
    @staticmethod
    def create_train_val_loaders(
        train_dataset: RouteDataset,
        val_dataset: RouteDataset,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Crear DataLoaders de entrenamiento y validación.
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            batch_size: Tamaño de batch
            num_workers: Número de workers
            pin_memory: Pin memory para GPU
            
        Returns:
            (train_loader, val_loader)
        """
        train_loader = RouteDataLoader.create(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = RouteDataLoader.create(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader




