"""
Data Augmentation System - Sistema de aumentación de datos
==========================================================
Aumentación de datos para texto e imágenes
"""

import logging
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Callable
import random
import numpy as np

try:
    from PIL import Image
    from torchvision import transforms
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextAugmentation:
    """Aumentación de datos para texto"""
    
    @staticmethod
    def random_deletion(text: str, p: float = 0.1) -> str:
        """Elimina palabras aleatoriamente"""
        words = text.split()
        if len(words) == 1:
            return text
        
        num_deletions = max(1, int(len(words) * p))
        indices_to_delete = random.sample(range(len(words)), num_deletions)
        
        new_words = [word for i, word in enumerate(words) if i not in indices_to_delete]
        return " ".join(new_words)
    
    @staticmethod
    def random_swap(text: str, n: int = 1) -> str:
        """Intercambia palabras aleatoriamente"""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return " ".join(words)
    
    @staticmethod
    def random_insertion(text: str, n: int = 1) -> str:
        """Inserta palabras aleatoriamente"""
        words = text.split()
        if len(words) == 0:
            return text
        
        for _ in range(n):
            random_word = random.choice(words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_word)
        
        return " ".join(words)
    
    @staticmethod
    def synonym_replacement(text: str, n: int = 1) -> str:
        """Reemplaza palabras con sinónimos (simplificado)"""
        # En producción, usar biblioteca de sinónimos
        words = text.split()
        if len(words) == 0:
            return text
        
        # Simulación simple
        return text  # Implementar con biblioteca de sinónimos


class ImageAugmentation:
    """Aumentación de datos para imágenes"""
    
    def __init__(self):
        if not VISION_AVAILABLE:
            logger.warning("Vision libraries not available")
    
    def get_transforms(
        self,
        train: bool = True,
        resize: tuple = (224, 224),
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225)
    ) -> transforms.Compose:
        """Obtiene transformaciones"""
        if not VISION_AVAILABLE:
            return None
        
        if train:
            return transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.1)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    
    def augment_image(self, image: Image.Image, n_augmentations: int = 1) -> List[Image.Image]:
        """Aumenta una imagen"""
        if not VISION_AVAILABLE:
            return [image]
        
        augmented = []
        transform = self.get_transforms(train=True)
        
        for _ in range(n_augmentations):
            # Aplicar transformaciones
            img_tensor = transform(image)
            # Convertir de vuelta a PIL (simplificado)
            augmented.append(image)  # En producción, convertir tensor a PIL
        
        return augmented


class AugmentedDataset(Dataset):
    """Dataset con aumentación"""
    
    def __init__(
        self,
        base_dataset: Dataset,
        augmentation_fn: Optional[Callable] = None,
        augmentation_prob: float = 0.5
    ):
        self.base_dataset = base_dataset
        self.augmentation_fn = augmentation_fn
        self.augmentation_prob = augmentation_prob
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        # Aplicar aumentación con probabilidad
        if self.augmentation_fn and random.random() < self.augmentation_prob:
            item = self.augmentation_fn(item)
        
        return item


class DataAugmentationManager:
    """Gestor de aumentación de datos"""
    
    def __init__(self):
        self.text_aug = TextAugmentation()
        self.image_aug = ImageAugmentation()
    
    def augment_text(self, text: str, method: str = "random") -> str:
        """Aumenta texto"""
        if method == "deletion":
            return self.text_aug.random_deletion(text)
        elif method == "swap":
            return self.text_aug.random_swap(text)
        elif method == "insertion":
            return self.text_aug.random_insertion(text)
        elif method == "synonym":
            return self.text_aug.synonym_replacement(text)
        else:
            # Aplicar múltiples métodos aleatoriamente
            methods = ["deletion", "swap", "insertion"]
            method = random.choice(methods)
            return self.augment_text(text, method)
    
    def augment_image(self, image: Image.Image, n: int = 1) -> List[Image.Image]:
        """Aumenta imagen"""
        return self.image_aug.augment_image(image, n)
    
    def create_augmented_dataset(
        self,
        dataset: Dataset,
        augmentation_fn: Callable,
        prob: float = 0.5
    ) -> AugmentedDataset:
        """Crea dataset aumentado"""
        return AugmentedDataset(dataset, augmentation_fn, prob)




