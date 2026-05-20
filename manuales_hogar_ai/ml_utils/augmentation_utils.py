"""
Augmentation Utils - Utilidades de Data Augmentation
====================================================

Utilidades para data augmentation de texto e imágenes.
"""

import logging
import random
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Intentar importar bibliotecas opcionales
try:
    from PIL import Image, ImageEnhance, ImageFilter
    _has_pil = True
except ImportError:
    _has_pil = False

try:
    import torchvision.transforms as transforms
    _has_torchvision = True
except ImportError:
    _has_torchvision = False


class TextAugmenter:
    """
    Augmentador de texto con múltiples técnicas.
    """
    
    def __init__(
        self,
        synonym_replacement: bool = True,
        random_insertion: bool = True,
        random_deletion: bool = True,
        random_swap: bool = True,
        synonym_dict: Optional[Dict[str, List[str]]] = None
    ):
        """
        Inicializar augmentador de texto.
        
        Args:
            synonym_replacement: Reemplazar palabras con sinónimos
            random_insertion: Insertar palabras aleatorias
            random_deletion: Eliminar palabras aleatorias
            random_swap: Intercambiar palabras
            synonym_dict: Diccionario de sinónimos
        """
        self.synonym_replacement = synonym_replacement
        self.random_insertion = random_insertion
        self.random_deletion = random_deletion
        self.random_swap = random_swap
        self.synonym_dict = synonym_dict or {}
    
    def augment(
        self,
        text: str,
        num_augmentations: int = 1,
        alpha: float = 0.1
    ) -> List[str]:
        """
        Generar versiones aumentadas del texto.
        
        Args:
            text: Texto original
            num_augmentations: Número de aumentaciones
            alpha: Proporción de palabras a modificar
            
        Returns:
            Lista de textos aumentados
        """
        words = text.split()
        num_words = len(words)
        num_changes = max(1, int(alpha * num_words))
        
        augmented_texts = []
        
        for _ in range(num_augmentations):
            augmented_words = words.copy()
            
            # Aplicar técnicas aleatorias
            if self.synonym_replacement and random.random() < 0.3:
                augmented_words = self._synonym_replace(augmented_words, num_changes)
            
            if self.random_insertion and random.random() < 0.3:
                augmented_words = self._random_insert(augmented_words, num_changes)
            
            if self.random_deletion and random.random() < 0.3:
                augmented_words = self._random_delete(augmented_words, num_changes)
            
            if self.random_swap and random.random() < 0.3:
                augmented_words = self._random_swap(augmented_words, num_changes)
            
            augmented_texts.append(' '.join(augmented_words))
        
        return augmented_texts
    
    def _synonym_replace(self, words: List[str], num_words: int) -> List[str]:
        """Reemplazar palabras con sinónimos."""
        words_copy = words.copy()
        indices = random.sample(range(len(words)), min(num_words, len(words)))
        
        for idx in indices:
            word = words_copy[idx].lower()
            if word in self.synonym_dict and self.synonym_dict[word]:
                words_copy[idx] = random.choice(self.synonym_dict[word])
        
        return words_copy
    
    def _random_insert(self, words: List[str], num_words: int) -> List[str]:
        """Insertar palabras aleatorias."""
        words_copy = words.copy()
        for _ in range(num_words):
            random_word = random.choice(words)
            random_idx = random.randint(0, len(words_copy))
            words_copy.insert(random_idx, random_word)
        return words_copy
    
    def _random_delete(self, words: List[str], num_words: int) -> List[str]:
        """Eliminar palabras aleatorias."""
        if len(words) <= 1:
            return words
        
        words_copy = words.copy()
        indices = random.sample(range(len(words_copy)), min(num_words, len(words_copy) - 1))
        indices.sort(reverse=True)
        
        for idx in indices:
            words_copy.pop(idx)
        
        return words_copy
    
    def _random_swap(self, words: List[str], num_words: int) -> List[str]:
        """Intercambiar palabras aleatorias."""
        words_copy = words.copy()
        for _ in range(num_words):
            if len(words_copy) < 2:
                break
            idx1, idx2 = random.sample(range(len(words_copy)), 2)
            words_copy[idx1], words_copy[idx2] = words_copy[idx2], words_copy[idx1]
        return words_copy


class ImageAugmenter:
    """
    Augmentador de imágenes con transformaciones.
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-15, 15),
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        blur: bool = True,
        noise: bool = True
    ):
        """
        Inicializar augmentador de imágenes.
        
        Args:
            rotation_range: Rango de rotación en grados
            brightness_range: Rango de brillo
            contrast_range: Rango de contraste
            saturation_range: Rango de saturación
            horizontal_flip: Volteo horizontal
            vertical_flip: Volteo vertical
            blur: Aplicar blur
            noise: Aplicar ruido
        """
        if not _has_pil:
            raise ImportError("PIL/Pillow is required for image augmentation")
        
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.blur = blur
        self.noise = noise
    
    def augment(
        self,
        image: Image.Image,
        num_augmentations: int = 1
    ) -> List[Image.Image]:
        """
        Generar versiones aumentadas de la imagen.
        
        Args:
            image: Imagen original
            num_augmentations: Número de aumentaciones
            
        Returns:
            Lista de imágenes aumentadas
        """
        augmented_images = []
        
        for _ in range(num_augmentations):
            aug_image = image.copy()
            
            # Rotación
            if self.rotation_range:
                angle = random.uniform(*self.rotation_range)
                aug_image = aug_image.rotate(angle, expand=True)
            
            # Volteos
            if self.horizontal_flip and random.random() < 0.5:
                aug_image = aug_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            if self.vertical_flip and random.random() < 0.5:
                aug_image = aug_image.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Ajustes de color
            if self.brightness_range:
                factor = random.uniform(*self.brightness_range)
                enhancer = ImageEnhance.Brightness(aug_image)
                aug_image = enhancer.enhance(factor)
            
            if self.contrast_range:
                factor = random.uniform(*self.contrast_range)
                enhancer = ImageEnhance.Contrast(aug_image)
                aug_image = enhancer.enhance(factor)
            
            if self.saturation_range:
                factor = random.uniform(*self.saturation_range)
                enhancer = ImageEnhance.Color(aug_image)
                aug_image = enhancer.enhance(factor)
            
            # Blur
            if self.blur and random.random() < 0.3:
                aug_image = aug_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
            
            # Ruido
            if self.noise and random.random() < 0.3:
                aug_image = self._add_noise(aug_image)
            
            augmented_images.append(aug_image)
        
        return augmented_images
    
    def _add_noise(self, image: Image.Image, noise_factor: float = 0.1) -> Image.Image:
        """Agregar ruido gaussiano a la imagen."""
        img_array = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, noise_factor * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)


class TorchAugmenter:
    """
    Augmentador usando torchvision transforms.
    """
    
    def __init__(
        self,
        use_color_jitter: bool = True,
        use_random_affine: bool = True,
        use_random_crop: bool = True,
        use_normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Inicializar augmentador con torchvision.
        
        Args:
            use_color_jitter: Usar color jitter
            use_random_affine: Usar transformaciones afines
            use_random_crop: Usar random crop
            use_normalize: Normalizar imágenes
            mean: Media para normalización
            std: Desviación estándar para normalización
        """
        if not _has_torchvision:
            raise ImportError("torchvision is required for TorchAugmenter")
        
        transform_list = []
        
        if use_color_jitter:
            transform_list.append(transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ))
        
        if use_random_affine:
            transform_list.append(transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ))
        
        if use_random_crop:
            transform_list.append(transforms.RandomCrop(224, padding=4))
        
        transform_list.append(transforms.ToTensor())
        
        if use_normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        self.transform = transforms.Compose(transform_list)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Aplicar transformaciones.
        
        Args:
            image: Imagen PIL
            
        Returns:
            Tensor transformado
        """
        return self.transform(image)


class MixUpAugmenter:
    """
    Augmentador usando técnica MixUp.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Inicializar MixUp augmenter.
        
        Args:
            alpha: Parámetro de distribución Beta
        """
        self.alpha = alpha
    
    def mixup(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Aplicar MixUp.
        
        Args:
            x: Features
            y: Labels
            
        Returns:
            Tupla (x_mixed, y_mixed, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, (y_a, y_b), lam


class CutMixAugmenter:
    """
    Augmentador usando técnica CutMix.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Inicializar CutMix augmenter.
        
        Args:
            alpha: Parámetro de distribución Beta
        """
        self.alpha = alpha
    
    def cutmix(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Aplicar CutMix.
        
        Args:
            x: Features (imágenes)
            y: Labels
            
        Returns:
            Tupla (x_mixed, y_mixed, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Obtener dimensiones
        _, _, h, w = x.size()
        
        # Calcular región de corte
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Posición aleatoria
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Aplicar CutMix
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Ajustar lambda según área real
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        y_a, y_b = y, y[index]
        
        return x, (y_a, y_b), lam




