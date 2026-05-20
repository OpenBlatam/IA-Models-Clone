"""
Sistema de análisis avanzado de textura con ML
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
import numpy as np
try:
    from scipy import ndimage
    from scipy.ndimage import label
except ImportError:
    # Fallback si scipy no está disponible
    ndimage = None
    label = None


@dataclass
class TextureFeature:
    """Característica de textura"""
    feature_name: str
    value: float
    unit: str = ""
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "feature_name": self.feature_name,
            "value": self.value,
            "unit": self.unit,
            "description": self.description
        }


@dataclass
class MLTextureAnalysis:
    """Análisis de textura con ML"""
    id: str
    user_id: str
    image_id: str
    features: List[TextureFeature]
    texture_score: float  # 0.0 to 100.0
    pore_density: float
    wrinkle_density: float
    smoothness_score: float
    uniformity_score: float
    ml_model_version: str
    confidence: float
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_id": self.image_id,
            "features": [f.to_dict() for f in self.features],
            "texture_score": self.texture_score,
            "pore_density": self.pore_density,
            "wrinkle_density": self.wrinkle_density,
            "smoothness_score": self.smoothness_score,
            "uniformity_score": self.uniformity_score,
            "ml_model_version": self.ml_model_version,
            "confidence": self.confidence,
            "created_at": self.created_at
        }


class AdvancedTextureML:
    """Sistema de análisis avanzado de textura con ML"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.analyses: Dict[str, List[MLTextureAnalysis]] = {}
        self.model_version = "1.0.0"
    
    def analyze_texture(self, user_id: str, image_id: str, image_data: np.ndarray) -> MLTextureAnalysis:
        """Analiza textura usando técnicas avanzadas de ML"""
        
        # Extraer características usando técnicas avanzadas
        features = self._extract_texture_features(image_data)
        
        # Calcular métricas
        pore_density = self._calculate_pore_density(image_data)
        wrinkle_density = self._calculate_wrinkle_density(image_data)
        smoothness_score = self._calculate_smoothness(image_data)
        uniformity_score = self._calculate_uniformity(image_data)
        
        # Calcular score de textura general
        texture_score = self._calculate_texture_score(
            pore_density, wrinkle_density, smoothness_score, uniformity_score
        )
        
        # Calcular confianza del modelo
        confidence = self._calculate_confidence(image_data)
        
        analysis = MLTextureAnalysis(
            id=str(uuid.uuid4()),
            user_id=user_id,
            image_id=image_id,
            features=features,
            texture_score=texture_score,
            pore_density=pore_density,
            wrinkle_density=wrinkle_density,
            smoothness_score=smoothness_score,
            uniformity_score=uniformity_score,
            ml_model_version=self.model_version,
            confidence=confidence
        )
        
        if user_id not in self.analyses:
            self.analyses[user_id] = []
        self.analyses[user_id].append(analysis)
        
        return analysis
    
    def _extract_texture_features(self, image: np.ndarray) -> List[TextureFeature]:
        """Extrae características de textura"""
        features = []
        
        # Contraste
        contrast = np.std(image)
        features.append(TextureFeature(
            feature_name="contrast",
            value=float(contrast),
            unit="std",
            description="Contraste de la textura"
        ))
        
        # Energía (uniformidad)
        energy = np.sum(image ** 2) / (image.shape[0] * image.shape[1])
        features.append(TextureFeature(
            feature_name="energy",
            value=float(energy),
            unit="normalized",
            description="Energía de la textura"
        ))
        
        # Entropía
        hist, _ = np.histogram(image.flatten(), bins=256)
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features.append(TextureFeature(
            feature_name="entropy",
            value=float(entropy),
            unit="bits",
            description="Entropía de la textura"
        ))
        
        # Homogeneidad
        homogeneity = self._calculate_homogeneity(image)
        features.append(TextureFeature(
            feature_name="homogeneity",
            value=float(homogeneity),
            unit="normalized",
            description="Homogeneidad de la textura"
        ))
        
        return features
    
    def _calculate_homogeneity(self, image: np.ndarray) -> float:
        """Calcula homogeneidad"""
        # Simplificado: variación local
        if ndimage is None:
            # Fallback sin scipy
            local_variance = np.var(image.astype(float))
        else:
            kernel = np.ones((3, 3)) / 9
            local_mean = ndimage.convolve(image.astype(float), kernel)
            local_variance = np.mean((image.astype(float) - local_mean) ** 2)
        homogeneity = 1.0 / (1.0 + local_variance / 1000.0)
        return float(homogeneity)
    
    def _calculate_pore_density(self, image: np.ndarray) -> float:
        """Calcula densidad de poros"""
        # Detección de poros usando detección de círculos
        gray = image if len(image.shape) == 2 else np.mean(image, axis=2)
        
        if ndimage is None or label is None:
            # Fallback sin scipy
            threshold = np.mean(gray) - np.std(gray)
            binary = gray < threshold
            pore_count = np.sum(binary) // 20  # Aproximación
        else:
            # Aplicar threshold adaptativo
            blurred = ndimage.gaussian_filter(gray, sigma=2)
            threshold = np.mean(blurred) - np.std(blurred)
            binary = blurred < threshold
            
            # Contar regiones pequeñas (poros)
            labeled, num_features = label(binary)
            
            # Filtrar por tamaño (poros típicamente pequeños)
            pore_count = 0
            for i in range(1, num_features + 1):
                size = np.sum(labeled == i)
                if 5 < size < 50:  # Tamaño típico de poros
                    pore_count += 1
        
        # Normalizar por área
        area = image.shape[0] * image.shape[1]
        density = (pore_count / area) * 10000  # Poros por cm² (aproximado)
        
        return float(density)
    
    def _calculate_wrinkle_density(self, image: np.ndarray) -> float:
        """Calcula densidad de arrugas"""
        gray = image if len(image.shape) == 2 else np.mean(image, axis=2)
        
        if ndimage is None:
            # Fallback sin scipy - usar gradiente simple
            edges = np.abs(np.diff(gray, axis=0)) + np.abs(np.diff(gray, axis=1))
            edges = np.pad(edges, ((0, 1), (0, 1)), mode='edge')
        else:
            # Detectar bordes (arrugas)
            sobel_x = ndimage.sobel(gray, axis=1)
            sobel_y = ndimage.sobel(gray, axis=0)
            edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        
        # Threshold para arrugas
        threshold = np.percentile(edges, 90)
        wrinkles = edges > threshold
        
        # Calcular densidad
        wrinkle_pixels = np.sum(wrinkles)
        total_pixels = image.shape[0] * image.shape[1]
        density = (wrinkle_pixels / total_pixels) * 100
        
        return float(density)
    
    def _calculate_smoothness(self, image: np.ndarray) -> float:
        """Calcula suavidad"""
        gray = image if len(image.shape) == 2 else np.mean(image, axis=2)
        
        if ndimage is None:
            # Fallback sin scipy
            local_variance = np.var(gray.astype(float))
        else:
            # Calcular variación local
            kernel = np.ones((5, 5)) / 25
            local_mean = ndimage.convolve(gray.astype(float), kernel)
            local_variance = np.mean((gray.astype(float) - local_mean) ** 2)
        
        # Suavidad inversa a varianza
        smoothness = 100.0 / (1.0 + local_variance / 100.0)
        
        return float(min(100.0, smoothness))
    
    def _calculate_uniformity(self, image: np.ndarray) -> float:
        """Calcula uniformidad"""
        gray = image if len(image.shape) == 2 else np.mean(image, axis=2)
        
        # Calcular desviación estándar global
        std = np.std(gray)
        mean = np.mean(gray)
        
        # Uniformidad inversa a coeficiente de variación
        cv = std / (mean + 1e-10)
        uniformity = 100.0 / (1.0 + cv)
        
        return float(min(100.0, uniformity))
    
    def _calculate_texture_score(self, pore_density: float, wrinkle_density: float,
                                 smoothness: float, uniformity: float) -> float:
        """Calcula score de textura general"""
        # Normalizar métricas
        normalized_pores = max(0, 100 - pore_density * 2)
        normalized_wrinkles = max(0, 100 - wrinkle_density * 2)
        
        # Ponderar métricas
        score = (
            normalized_pores * 0.25 +
            normalized_wrinkles * 0.25 +
            smoothness * 0.25 +
            uniformity * 0.25
        )
        
        return float(max(0.0, min(100.0, score)))
    
    def _calculate_confidence(self, image: np.ndarray) -> float:
        """Calcula confianza del modelo"""
        # Factores que afectan confianza
        factors = []
        
        # Resolución
        resolution = image.shape[0] * image.shape[1]
        if resolution > 1000000:
            factors.append(1.0)
        elif resolution > 500000:
            factors.append(0.8)
        else:
            factors.append(0.6)
        
        # Contraste
        contrast = np.std(image)
        if contrast > 30:
            factors.append(1.0)
        elif contrast > 15:
            factors.append(0.8)
        else:
            factors.append(0.6)
        
        # Confianza promedio
        confidence = np.mean(factors)
        return float(confidence)
