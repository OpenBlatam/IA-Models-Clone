"""
PyTorch Dataset for Robot Movement Data
=======================================

Modular dataset classes for loading and preprocessing robot movement data.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import json

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """
    Dataset para trayectorias de robot.
    
    Carga y preprocesa datos de trayectorias para entrenamiento.
    """
    
    def __init__(
        self,
        data_path: str,
        trajectory_length: int = 100,
        trajectory_dim: int = 3,
        normalize: bool = True,
        augment: bool = False,
        cache_data: bool = True
    ):
        """
        Inicializar dataset.
        
        Args:
            data_path: Ruta a archivo de datos (JSON, NPY, o directorio)
            trajectory_length: Longitud de trayectorias
            trajectory_dim: Dimensión de cada punto (3 para x,y,z)
            normalize: Normalizar datos
            augment: Aplicar data augmentation
            cache_data: Cachear datos en memoria
        """
        self.data_path = Path(data_path)
        self.trajectory_length = trajectory_length
        self.trajectory_dim = trajectory_dim
        self.normalize = normalize
        self.augment = augment
        self.cache_data = cache_data
        
        # Cargar datos
        self.data = self._load_data()
        self.mean = None
        self.std = None
        
        if self.normalize:
            self._compute_statistics()
            self._normalize_data()
        
        logger.info(f"Loaded dataset with {len(self.data)} trajectories")
    
    def _load_data(self) -> List[np.ndarray]:
        """Cargar datos desde archivo."""
        if self.data_path.is_file():
            if self.data_path.suffix == '.json':
                return self._load_json()
            elif self.data_path.suffix == '.npy':
                return self._load_npy()
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        elif self.data_path.is_dir():
            return self._load_directory()
        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
    
    def _load_json(self) -> List[np.ndarray]:
        """Cargar desde JSON."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        trajectories = []
        for item in data:
            if 'trajectory' in item:
                traj = np.array(item['trajectory'], dtype=np.float32)
                if traj.shape[-1] == self.trajectory_dim:
                    trajectories.append(traj)
        
        return trajectories
    
    def _load_npy(self) -> List[np.ndarray]:
        """Cargar desde NPY."""
        data = np.load(self.data_path, allow_pickle=True)
        if isinstance(data, np.ndarray):
            if len(data.shape) == 3:  # [num_trajectories, length, dim]
                return [data[i] for i in range(len(data))]
            elif len(data.shape) == 2:  # [length, dim]
                return [data]
        return []
    
    def _load_directory(self) -> List[np.ndarray]:
        """Cargar desde directorio de archivos."""
        trajectories = []
        for file_path in self.data_path.glob('*.npy'):
            traj = np.load(file_path)
            if traj.shape[-1] == self.trajectory_dim:
                trajectories.append(traj)
        return trajectories
    
    def _compute_statistics(self):
        """Calcular estadísticas para normalización."""
        all_data = np.concatenate(self.data, axis=0)
        self.mean = np.mean(all_data, axis=0, keepdims=True)
        self.std = np.std(all_data, axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)  # Evitar división por cero
    
    def _normalize_data(self):
        """Normalizar datos."""
        if self.mean is None or self.std is None:
            self._compute_statistics()
        
        self.data = [
            (traj - self.mean) / self.std
            for traj in self.data
        ]
    
    def _augment_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Aplicar data augmentation."""
        # Ruido gaussiano
        noise = np.random.normal(0, 0.01, trajectory.shape).astype(np.float32)
        trajectory = trajectory + noise
        
        # Escalado aleatorio
        scale = np.random.uniform(0.95, 1.05)
        trajectory = trajectory * scale
        
        return trajectory
    
    def __len__(self) -> int:
        """Tamaño del dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Obtener item del dataset.
        
        Args:
            idx: Índice del item
            
        Returns:
            Diccionario con 'trajectory' y metadata
        """
        trajectory = self.data[idx].copy()
        
        # Asegurar longitud correcta
        if len(trajectory) > self.trajectory_length:
            # Truncar
            start_idx = np.random.randint(0, len(trajectory) - self.trajectory_length + 1)
            trajectory = trajectory[start_idx:start_idx + self.trajectory_length]
        elif len(trajectory) < self.trajectory_length:
            # Padding
            pad_length = self.trajectory_length - len(trajectory)
            pad = np.tile(trajectory[-1:], (pad_length, 1))
            trajectory = np.concatenate([trajectory, pad], axis=0)
        
        # Data augmentation
        if self.augment:
            trajectory = self._augment_trajectory(trajectory)
        
        # Convertir a tensor
        trajectory_tensor = torch.from_numpy(trajectory).float()
        
        return {
            'trajectory': trajectory_tensor,
            'length': len(trajectory),
            'index': idx
        }
    
    def denormalize(self, trajectory: np.ndarray) -> np.ndarray:
        """Desnormalizar trayectoria."""
        if self.mean is None or self.std is None:
            return trajectory
        return trajectory * self.std + self.mean


class CommandDataset(Dataset):
    """
    Dataset para comandos de lenguaje natural.
    
    Usado para entrenar modelos de NLP para interpretación de comandos.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
        cache_data: bool = True
    ):
        """
        Inicializar dataset de comandos.
        
        Args:
            data_path: Ruta a archivo JSON con comandos
            tokenizer: Tokenizer de Transformers
            max_length: Longitud máxima de secuencia
            cache_data: Cachear datos en memoria
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_data = cache_data
        
        self.data = self._load_data()
        logger.info(f"Loaded command dataset with {len(self.data)} examples")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Cargar datos de comandos."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self) -> int:
        """Tamaño del dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Obtener item del dataset.
        
        Args:
            idx: Índice del item
            
        Returns:
            Diccionario con 'input_ids', 'attention_mask', 'labels'
        """
        item = self.data[idx]
        command = item.get('command', '')
        intent = item.get('intent', '')
        parameters = item.get('parameters', {})
        
        # Tokenizar comando
        encoding = self.tokenizer(
            command,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenizar intención como label
        intent_encoding = self.tokenizer(
            intent,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': intent_encoding['input_ids'].squeeze(0),
            'parameters': parameters,
            'command': command
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Crear DataLoader con configuración optimizada.
    
    Args:
        dataset: Dataset de PyTorch
        batch_size: Tamaño del batch
        shuffle: Mezclar datos
        num_workers: Número de workers para carga
        pin_memory: Pin memory para transferencia GPU más rápida
        drop_last: Eliminar último batch incompleto
        
    Returns:
        DataLoader configurado
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )













