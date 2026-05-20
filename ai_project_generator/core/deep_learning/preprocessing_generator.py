"""
Preprocessing Generator - Generador de utilidades de preprocesamiento
======================================================================

Genera utilidades para preprocesamiento avanzado:
- Text preprocessing
- Image preprocessing
- Data normalization
- Feature engineering
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PreprocessingGenerator:
    """Generador de utilidades de preprocesamiento"""
    
    def __init__(self):
        """Inicializa el generador de preprocesamiento"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de preprocesamiento.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessing_dir = utils_dir / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_text_preprocessor(preprocessing_dir, keywords, project_info)
        self._generate_image_preprocessor(preprocessing_dir, keywords, project_info)
        self._generate_data_normalizer(preprocessing_dir, keywords, project_info)
        self._generate_preprocessing_init(preprocessing_dir, keywords)
    
    def _generate_preprocessing_init(
        self,
        preprocessing_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de preprocesamiento"""
        
        init_content = '''"""
Preprocessing Utilities Module
================================

Utilidades para preprocesamiento avanzado de datos.
"""

from .text_preprocessor import (
    TextPreprocessor,
    preprocess_text,
    tokenize_text,
    normalize_text,
)
from .image_preprocessor import (
    ImagePreprocessor,
    preprocess_image,
    resize_image,
    normalize_image,
)
from .data_normalizer import (
    DataNormalizer,
    normalize_data,
    standardize_data,
    min_max_scale,
)

__all__ = [
    "TextPreprocessor",
    "preprocess_text",
    "tokenize_text",
    "normalize_text",
    "ImagePreprocessor",
    "preprocess_image",
    "resize_image",
    "normalize_image",
    "DataNormalizer",
    "normalize_data",
    "standardize_data",
    "min_max_scale",
]
'''
        
        (preprocessing_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_text_preprocessor(
        self,
        preprocessing_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera preprocesador de texto"""
        
        text_preprocessor_content = '''"""
Text Preprocessor - Preprocesador de texto
===========================================

Utilidades para preprocesamiento avanzado de texto.
"""

import re
from typing import List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Preprocesador de texto avanzado.
    
    Proporciona funciones para limpiar, normalizar y tokenizar texto.
    """
    
    def __init__(self):
        """Inicializa el preprocesador"""
        pass
    
    def preprocess_text(
        self,
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_whitespace: bool = True,
        remove_stopwords: bool = False,
        custom_filters: Optional[List[Callable]] = None,
    ) -> str:
        """
        Preprocesa texto con múltiples opciones.
        
        Args:
            text: Texto a preprocesar
            lowercase: Si convertir a minúsculas
            remove_punctuation: Si remover puntuación
            remove_numbers: Si remover números
            remove_whitespace: Si normalizar espacios en blanco
            remove_stopwords: Si remover stopwords (requiere nltk)
            custom_filters: Filtros personalizados adicionales
        
        Returns:
            Texto preprocesado
        """
        if not isinstance(text, str):
            raise ValueError(f"Texto debe ser string, recibido: {type(text)}")
        
        processed = text
        
        if lowercase:
            processed = processed.lower()
        
        if remove_punctuation:
            processed = re.sub(r'[^\\w\\s]', '', processed)
        
        if remove_numbers:
            processed = re.sub(r'\\d+', '', processed)
        
        if remove_whitespace:
            processed = re.sub(r'\\s+', ' ', processed).strip()
        
        if remove_stopwords:
            try:
                import nltk
                from nltk.corpus import stopwords
                try:
                    stop_words = set(stopwords.words('english'))
                except LookupError:
                    nltk.download('stopwords')
                    stop_words = set(stopwords.words('english'))
                
                words = processed.split()
                processed = ' '.join([w for w in words if w not in stop_words])
            except ImportError:
                logger.warning("NLTK no disponible, omitiendo remoción de stopwords")
        
        if custom_filters:
            for filter_func in custom_filters:
                processed = filter_func(processed)
        
        return processed
    
    def tokenize_text(
        self,
        text: str,
        method: str = "whitespace",
        max_length: Optional[int] = None,
    ) -> List[str]:
        """
        Tokeniza texto.
        
        Args:
            text: Texto a tokenizar
            method: Método de tokenización (whitespace, word, subword)
            max_length: Longitud máxima de tokens (opcional)
        
        Returns:
            Lista de tokens
        """
        if method == "whitespace":
            tokens = text.split()
        elif method == "word":
            tokens = re.findall(r'\\b\\w+\\b', text)
        elif method == "subword":
            # Tokenización básica por caracteres
            tokens = list(text)
        else:
            raise ValueError(f"Método de tokenización no soportado: {method}")
        
        if max_length:
            tokens = tokens[:max_length]
        
        return tokens
    
    def normalize_text(
        self,
        text: str,
    ) -> str:
        """
        Normaliza texto (unicode, acentos, etc.).
        
        Args:
            text: Texto a normalizar
        
        Returns:
            Texto normalizado
        """
        # Normalizar unicode
        try:
            import unicodedata
            text = unicodedata.normalize('NFKD', text)
        except ImportError:
            pass
        
        # Remover acentos (opcional)
        text = re.sub(r'[àáâãäå]', 'a', text)
        text = re.sub(r'[èéêë]', 'e', text)
        text = re.sub(r'[ìíîï]', 'i', text)
        text = re.sub(r'[òóôõö]', 'o', text)
        text = re.sub(r'[ùúûü]', 'u', text)
        text = re.sub(r'[ñ]', 'n', text)
        text = re.sub(r'[ç]', 'c', text)
        
        return text


def preprocess_text(
    text: str,
    **kwargs,
) -> str:
    """
    Función helper para preprocesar texto.
    
    Args:
        text: Texto a preprocesar
        **kwargs: Argumentos adicionales
    
    Returns:
        Texto preprocesado
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_text(text, **kwargs)


def tokenize_text(
    text: str,
    **kwargs,
) -> List[str]:
    """
    Función helper para tokenizar texto.
    
    Args:
        text: Texto a tokenizar
        **kwargs: Argumentos adicionales
    
    Returns:
        Lista de tokens
    """
    preprocessor = TextPreprocessor()
    return preprocessor.tokenize_text(text, **kwargs)


def normalize_text(
    text: str,
) -> str:
    """
    Función helper para normalizar texto.
    
    Args:
        text: Texto a normalizar
    
    Returns:
        Texto normalizado
    """
    preprocessor = TextPreprocessor()
    return preprocessor.normalize_text(text)
'''
        
        (preprocessing_dir / "text_preprocessor.py").write_text(text_preprocessor_content, encoding="utf-8")
    
    def _generate_image_preprocessor(
        self,
        preprocessing_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera preprocesador de imágenes"""
        
        image_preprocessor_content = '''"""
Image Preprocessor - Preprocesador de imágenes
===============================================

Utilidades para preprocesamiento avanzado de imágenes.
"""

import logging
from typing import Tuple, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocesador de imágenes avanzado.
    
    Proporciona funciones para redimensionar, normalizar y transformar imágenes.
    """
    
    def __init__(self):
        """Inicializa el preprocesador"""
        pass
    
    def preprocess_image(
        self,
        image,
        size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Preprocesa imagen.
        
        Args:
            image: Imagen a preprocesar (PIL, numpy, o tensor)
            size: Tamaño objetivo (width, height) (opcional)
            normalize: Si normalizar
            mean: Media para normalización (opcional)
            std: Desviación estándar para normalización (opcional)
        
        Returns:
            Imagen preprocesada
        """
        # Convertir a PIL si es necesario
        if PIL_AVAILABLE and isinstance(image, Image.Image):
            pil_image = image
        elif NUMPY_AVAILABLE and isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif TORCH_AVAILABLE and isinstance(image, torch.Tensor):
            # Convertir tensor a PIL
            if image.dim() == 3:
                image_np = image.permute(1, 2, 0).numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
            else:
                raise ValueError(f"Tensor debe ser 3D, recibido: {image.dim()}D")
        else:
            raise ValueError(f"Tipo de imagen no soportado: {type(image)}")
        
        # Redimensionar si es necesario
        if size:
            pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
        
        # Convertir a tensor si torch está disponible
        if TORCH_AVAILABLE:
            transform_list = [transforms.ToTensor()]
            
            if normalize:
                if mean is None:
                    mean = (0.485, 0.456, 0.406)  # ImageNet mean
                if std is None:
                    std = (0.229, 0.224, 0.225)  # ImageNet std
                transform_list.append(transforms.Normalize(mean=mean, std=std))
            
            transform = transforms.Compose(transform_list)
            return transform(pil_image)
        else:
            # Retornar PIL si torch no está disponible
            return pil_image
    
    def resize_image(
        self,
        image,
        size: Tuple[int, int],
        resample: int = Image.Resampling.LANCZOS if PIL_AVAILABLE else None,
    ):
        """
        Redimensiona imagen.
        
        Args:
            image: Imagen a redimensionar
            size: Tamaño objetivo (width, height)
            resample: Método de resampling (opcional)
        
        Returns:
            Imagen redimensionada
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL no disponible. Instala con: pip install pillow")
        
        if isinstance(image, Image.Image):
            return image.resize(size, resample)
        elif NUMPY_AVAILABLE and isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            return pil_image.resize(size, resample)
        else:
            raise ValueError(f"Tipo de imagen no soportado: {type(image)}")
    
    def normalize_image(
        self,
        image,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Normaliza imagen.
        
        Args:
            image: Imagen a normalizar
            mean: Media para normalización (opcional)
            std: Desviación estándar para normalización (opcional)
        
        Returns:
            Imagen normalizada
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch no disponible. Instala con: pip install torch")
        
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Imagen debe ser Tensor, recibido: {type(image)}")
        
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)
        
        normalize_transform = transforms.Normalize(mean=mean, std=std)
        return normalize_transform(image)


def preprocess_image(
    image,
    **kwargs,
):
    """
    Función helper para preprocesar imagen.
    
    Args:
        image: Imagen a preprocesar
        **kwargs: Argumentos adicionales
    
    Returns:
        Imagen preprocesada
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.preprocess_image(image, **kwargs)


def resize_image(
    image,
    size: Tuple[int, int],
    **kwargs,
):
    """
    Función helper para redimensionar imagen.
    
    Args:
        image: Imagen a redimensionar
        size: Tamaño objetivo
        **kwargs: Argumentos adicionales
    
    Returns:
        Imagen redimensionada
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.resize_image(image, size, **kwargs)


def normalize_image(
    image,
    **kwargs,
):
    """
    Función helper para normalizar imagen.
    
    Args:
        image: Imagen a normalizar
        **kwargs: Argumentos adicionales
    
    Returns:
        Imagen normalizada
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.normalize_image(image, **kwargs)
'''
        
        (preprocessing_dir / "image_preprocessor.py").write_text(image_preprocessor_content, encoding="utf-8")
    
    def _generate_data_normalizer(
        self,
        preprocessing_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera normalizador de datos"""
        
        normalizer_content = '''"""
Data Normalizer - Normalizador de datos
========================================

Utilidades para normalización y estandarización de datos.
"""

import logging
from typing import Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Normalizador de datos.
    
    Proporciona funciones para normalizar y estandarizar datos.
    """
    
    def __init__(self):
        """Inicializa el normalizador"""
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
    
    def normalize_data(
        self,
        data,
        method: str = "standard",
        fit: bool = True,
    ):
        """
        Normaliza datos.
        
        Args:
            data: Datos a normalizar
            method: Método de normalización (standard, min_max)
            fit: Si calcular estadísticas de los datos
        
        Returns:
            Datos normalizados
        """
        if method == "standard":
            return self.standardize_data(data, fit=fit)
        elif method == "min_max":
            return self.min_max_scale(data, fit=fit)
        else:
            raise ValueError(f"Método de normalización no soportado: {method}")
    
    def standardize_data(
        self,
        data,
        fit: bool = True,
    ):
        """
        Estandariza datos (z-score normalization).
        
        Args:
            data: Datos a estandarizar
            fit: Si calcular media y desviación estándar
        
        Returns:
            Datos estandarizados
        """
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            if fit or self.mean is None:
                self.mean = data.mean()
                self.std = data.std()
            
            return (data - self.mean) / (self.std + 1e-8)
        
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            if fit or self.mean is None:
                self.mean = np.mean(data)
                self.std = np.std(data)
            
            return (data - self.mean) / (self.std + 1e-8)
        
        else:
            raise ValueError(f"Tipo de datos no soportado: {type(data)}")
    
    def min_max_scale(
        self,
        data,
        fit: bool = True,
        feature_range: tuple = (0, 1),
    ):
        """
        Escala datos usando min-max normalization.
        
        Args:
            data: Datos a escalar
            fit: Si calcular min y max
            feature_range: Rango objetivo (min, max)
        
        Returns:
            Datos escalados
        """
        min_val, max_val = feature_range
        
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            if fit or self.min is None:
                self.min = data.min()
                self.max = data.max()
            
            scaled = (data - self.min) / (self.max - self.min + 1e-8)
            return scaled * (max_val - min_val) + min_val
        
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            if fit or self.min is None:
                self.min = np.min(data)
                self.max = np.max(data)
            
            scaled = (data - self.min) / (self.max - self.min + 1e-8)
            return scaled * (max_val - min_val) + min_val
        
        else:
            raise ValueError(f"Tipo de datos no soportado: {type(data)}")


def normalize_data(
    data,
    **kwargs,
):
    """
    Función helper para normalizar datos.
    
    Args:
        data: Datos a normalizar
        **kwargs: Argumentos adicionales
    
    Returns:
        Datos normalizados
    """
    normalizer = DataNormalizer()
    return normalizer.normalize_data(data, **kwargs)


def standardize_data(
    data,
    **kwargs,
):
    """
    Función helper para estandarizar datos.
    
    Args:
        data: Datos a estandarizar
        **kwargs: Argumentos adicionales
    
    Returns:
        Datos estandarizados
    """
    normalizer = DataNormalizer()
    return normalizer.standardize_data(data, **kwargs)


def min_max_scale(
    data,
    **kwargs,
):
    """
    Función helper para escalar datos.
    
    Args:
        data: Datos a escalar
        **kwargs: Argumentos adicionales
    
    Returns:
        Datos escalados
    """
    normalizer = DataNormalizer()
    return normalizer.min_max_scale(data, **kwargs)
'''
        
        (preprocessing_dir / "data_normalizer.py").write_text(normalizer_content, encoding="utf-8")

