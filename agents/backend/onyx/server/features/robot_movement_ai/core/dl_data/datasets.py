"""
Datasets
========

Datasets personalizados para datos de robots.
"""

import logging
from typing import Optional, Callable, Tuple
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Dataset = None

logger = logging.getLogger(__name__)


class RobotDataset(Dataset):
    """
    Dataset para datos de robot.
    
    Dataset básico para pares input-target.
    """
    
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Inicializar dataset.
        
        Args:
            inputs: Datos de entrada [N, input_size]
            targets: Datos objetivo [N, output_size]
            transform: Transformación para inputs (opcional)
            target_transform: Transformación para targets (opcional)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for datasets")
        
        if len(inputs) != len(targets):
            raise ValueError("Inputs and targets must have the same length")
        
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        """Tamaño del dataset."""
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtener item.
        
        Args:
            idx: Índice
            
        Returns:
            Tupla (input, target)
        """
        input_data = self.inputs[idx]
        target = self.targets[idx]
        
        if self.transform:
            input_data = self.transform(input_data)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return input_data, target


class RobotSequenceDataset(Dataset):
    """
    Dataset para secuencias temporales de robot.
    
    Útil para modelos LSTM/RNN.
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 10,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Inicializar dataset de secuencias.
        
        Args:
            sequences: Secuencias de datos [N, seq_len, features]
            targets: Objetivos [N, output_size]
            sequence_length: Longitud de secuencia
            transform: Transformación para secuencias (opcional)
            target_transform: Transformación para targets (opcional)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for datasets")
        
        if len(sequences) != len(targets):
            raise ValueError("Sequences and targets must have the same length")
        
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        """Tamaño del dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtener item.
        
        Args:
            idx: Índice
            
        Returns:
            Tupla (sequence, target)
        """
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return sequence, target




