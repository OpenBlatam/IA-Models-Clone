"""
Data Utilities - Utilidades para datos
=======================================

Funciones de utilidad para procesamiento y manejo de datos.
Sigue mejores prácticas de PyTorch.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

logger = logging.getLogger(__name__)


class TensorDataset(Dataset):
    """Dataset simple para tensores"""
    
    def __init__(self, *tensors):
        """
        Args:
            *tensors: Tensores que deben tener la misma primera dimensión
        """
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
    
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def __len__(self):
        return self.tensors[0].size(0)


def create_data_splits(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Dividir dataset en train/val/test.
    
    Args:
        dataset: Dataset completo
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
        random_seed: Semilla para reproducibilidad
    
    Returns:
        Tupla con (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    logger.info(
        f"Dataset split: Train={len(train_dataset)}, "
        f"Val={len(val_dataset)}, Test={len(test_dataset)}"
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = False
) -> DataLoader:
    """
    Crear DataLoader optimizado.
    
    Args:
        dataset: Dataset
        batch_size: Tamaño de batch
        shuffle: Si True, mezclar datos
        num_workers: Número de workers para carga de datos
        pin_memory: Pin memory para transferencia más rápida a GPU
        drop_last: Si True, descartar último batch incompleto
        persistent_workers: Si True, mantener workers vivos entre epochs
    
    Returns:
        DataLoader configurado
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    
    return dataloader


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    dim: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalizar tensor.
    
    Args:
        tensor: Tensor a normalizar
        mean: Media (si None, se calcula)
        std: Desviación estándar (si None, se calcula)
        dim: Dimensión para calcular estadísticas
    
    Returns:
        Tupla con (tensor_normalizado, mean, std)
    """
    if mean is None:
        mean = tensor.mean(dim=dim, keepdim=True)
    if std is None:
        std = tensor.std(dim=dim, keepdim=True) + 1e-8
    
    normalized = (tensor - mean) / std
    return normalized, mean, std


def one_hot_encode(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Codificar labels en one-hot.
    
    Args:
        labels: Tensor de labels (shape: [N])
        num_classes: Número de clases
    
    Returns:
        Tensor one-hot (shape: [N, num_classes])
    """
    return torch.zeros(labels.size(0), num_classes).scatter_(
        1, labels.unsqueeze(1), 1
    )


def balance_dataset(
    dataset: Dataset,
    labels: torch.Tensor,
    method: str = "undersample"
) -> Subset:
    """
    Balancear dataset.
    
    Args:
        dataset: Dataset
        labels: Labels del dataset
        method: Método ('undersample' o 'oversample')
    
    Returns:
        Dataset balanceado
    """
    unique_labels, counts = torch.unique(labels, return_counts=True)
    min_count = counts.min().item()
    
    indices = []
    for label in unique_labels:
        label_indices = (labels == label).nonzero(as_tuple=True)[0]
        
        if method == "undersample":
            # Submuestrear clases mayoritarias
            selected = torch.randperm(len(label_indices))[:min_count]
            indices.extend(label_indices[selected].tolist())
        elif method == "oversample":
            # Sobremuestrear clases minoritarias
            num_samples = min_count
            if len(label_indices) < num_samples:
                # Repetir muestras
                repeats = num_samples // len(label_indices)
                remainder = num_samples % len(label_indices)
                selected = torch.cat([
                    label_indices.repeat(repeats),
                    label_indices[:remainder]
                ])
            else:
                selected = label_indices[:num_samples]
            indices.extend(selected.tolist())
    
    return Subset(dataset, indices)


def get_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Calcular pesos de clases para balancear pérdida.
    
    Args:
        labels: Tensor de labels
    
    Returns:
        Tensor de pesos (uno por clase)
    """
    unique_labels, counts = torch.unique(labels, return_counts=True)
    total = len(labels)
    num_classes = len(unique_labels)
    
    weights = torch.zeros(num_classes)
    for i, label in enumerate(unique_labels):
        weights[i] = total / (num_classes * counts[i].item())
    
    return weights


def collate_fn_pad(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function que hace padding a secuencias de diferente longitud.
    
    Args:
        batch: Lista de tuplas (inputs, targets, ...)
    
    Returns:
        Tupla de tensores con padding
    """
    # Separar inputs y targets
    inputs = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch])
    
    # Hacer padding de inputs
    max_len = max(inp.size(0) for inp in inputs)
    padded_inputs = []
    
    for inp in inputs:
        pad_size = max_len - inp.size(0)
        if pad_size > 0:
            padded = torch.nn.functional.pad(inp, (0, pad_size), value=0)
        else:
            padded = inp
        padded_inputs.append(padded)
    
    inputs_tensor = torch.stack(padded_inputs)
    
    return inputs_tensor, targets




