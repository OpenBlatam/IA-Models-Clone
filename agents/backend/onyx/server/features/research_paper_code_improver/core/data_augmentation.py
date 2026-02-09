"""
Data Augmentation Manager - Gestor de aumentación de datos
===========================================================
"""

import logging
import torch
import torchvision.transforms as transforms
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuración de aumentación"""
    rotation: float = 15.0
    translation: Tuple[float, float] = (0.1, 0.1)
    scale: Tuple[float, float] = (0.9, 1.1)
    shear: float = 10.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    horizontal_flip: bool = True
    vertical_flip: bool = False
    random_crop: bool = True
    color_jitter: bool = True


class DataAugmentationManager:
    """Gestor de aumentación de datos"""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.transforms: Dict[str, Any] = {}
    
    def create_image_transforms(
        self,
        augment: bool = True,
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ) -> transforms.Compose:
        """Crea transformaciones para imágenes"""
        transform_list = []
        
        if augment:
            # Aumentación
            if self.config.random_crop:
                transform_list.append(transforms.RandomResizedCrop(224))
            else:
                transform_list.append(transforms.Resize(256))
                transform_list.append(transforms.RandomCrop(224))
            
            if self.config.horizontal_flip:
                transform_list.append(transforms.RandomHorizontalFlip())
            
            if self.config.vertical_flip:
                transform_list.append(transforms.RandomVerticalFlip())
            
            if self.config.color_jitter:
                transform_list.append(transforms.ColorJitter(
                    brightness=self.config.brightness,
                    contrast=self.config.contrast,
                    saturation=self.config.saturation,
                    hue=self.config.hue
                ))
            
            # Rotación y transformaciones afines
            transform_list.append(transforms.RandomAffine(
                degrees=self.config.rotation,
                translate=self.config.translation,
                scale=self.config.scale,
                shear=self.config.shear
            ))
        else:
            # Solo resize y crop para validación
            transform_list.append(transforms.Resize(256))
            transform_list.append(transforms.CenterCrop(224))
        
        # Convertir a tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalizar
        if normalize:
            mean = mean or [0.485, 0.456, 0.406]
            std = std or [0.229, 0.224, 0.225]
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)
    
    def create_text_augmentation(
        self,
        synonym_replacement: bool = True,
        random_deletion: bool = True,
        random_swap: bool = True,
        random_insertion: bool = True
    ) -> Callable:
        """Crea aumentación para texto"""
        def augment_text(text: str) -> str:
            words = text.split()
            
            if synonym_replacement and len(words) > 0:
                # Simplificado - en producción usar WordNet o similar
                pass
            
            if random_deletion and len(words) > 1:
                # Eliminar palabra aleatoria con probabilidad
                if np.random.random() < 0.1:
                    idx = np.random.randint(len(words))
                    words.pop(idx)
            
            if random_swap and len(words) > 1:
                # Intercambiar palabras adyacentes
                if np.random.random() < 0.1:
                    idx = np.random.randint(len(words) - 1)
                    words[idx], words[idx + 1] = words[idx + 1], words[idx]
            
            return " ".join(words)
        
        return augment_text
    
    def create_mixup(
        self,
        alpha: float = 0.2
    ) -> Callable:
        """Crea función de Mixup"""
        def mixup_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            inputs = batch.get("inputs") or batch.get("input_ids")
            labels = batch.get("labels")
            
            if inputs is None or labels is None:
                return batch
            
            batch_size = inputs.size(0)
            indices = torch.randperm(batch_size)
            
            lam = np.random.beta(alpha, alpha)
            
            mixed_inputs = lam * inputs + (1 - lam) * inputs[indices]
            mixed_labels = lam * labels + (1 - lam) * labels[indices]
            
            batch["inputs"] = mixed_inputs
            batch["labels"] = mixed_labels
            
            return batch
        
        return mixup_batch
    
    def create_cutmix(
        self,
        alpha: float = 1.0
    ) -> Callable:
        """Crea función de CutMix"""
        def cutmix_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            inputs = batch.get("inputs") or batch.get("input_ids")
            labels = batch.get("labels")
            
            if inputs is None or labels is None:
                return batch
            
            batch_size = inputs.size(0)
            indices = torch.randperm(batch_size)
            
            lam = np.random.beta(alpha, alpha)
            
            # Para imágenes: cortar y pegar región
            # Para texto: simplificado
            mixed_inputs = inputs.clone()
            mixed_labels = lam * labels + (1 - lam) * labels[indices]
            
            batch["inputs"] = mixed_inputs
            batch["labels"] = mixed_labels
            
            return batch
        
        return cutmix_batch
    
    def register_transform(self, name: str, transform: Callable):
        """Registra una transformación personalizada"""
        self.transforms[name] = transform
        logger.info(f"Transformación {name} registrada")
    
    def apply_transform(self, name: str, data: Any) -> Any:
        """Aplica una transformación"""
        if name not in self.transforms:
            raise ValueError(f"Transformación {name} no encontrada")
        return self.transforms[name](data)

