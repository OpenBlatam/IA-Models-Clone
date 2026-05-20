"""
Módulo para calcular métricas de calidad de la piel
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class SkinQualityScore:
    """Puntuación de calidad de la piel"""
    overall_score: float  # 0-100
    texture_score: float  # 0-100
    hydration_score: float  # 0-100
    elasticity_score: float  # 0-100
    pigmentation_score: float  # 0-100
    pore_size_score: float  # 0-100
    wrinkles_score: float  # 0-100
    redness_score: float  # 0-100
    dark_spots_score: float  # 0-100
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "overall_score": round(self.overall_score, 2),
            "texture_score": round(self.texture_score, 2),
            "hydration_score": round(self.hydration_score, 2),
            "elasticity_score": round(self.elasticity_score, 2),
            "pigmentation_score": round(self.pigmentation_score, 2),
            "pore_size_score": round(self.pore_size_score, 2),
            "wrinkles_score": round(self.wrinkles_score, 2),
            "redness_score": round(self.redness_score, 2),
            "dark_spots_score": round(self.dark_spots_score, 2),
        }


class SkinQualityMetrics:
    """Calcula métricas de calidad de la piel a partir de imágenes"""
    
    def __init__(self):
        """Inicializa el calculador de métricas"""
        pass
    
    def analyze_texture(self, image: np.ndarray) -> float:
        """
        Analiza la textura de la piel
        Retorna un score de 0-100 (100 = textura perfecta)
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Calcular varianza de Laplacian (medida de nitidez/textura)
        from scipy import ndimage
        laplacian_var = ndimage.laplace(gray).var()
        
        # Normalizar a 0-100 (valores típicos: 0-1000)
        texture_score = min(100, (laplacian_var / 10) * 10)
        
        return max(0, texture_score)
    
    def analyze_hydration(self, image: np.ndarray) -> float:
        """
        Analiza el nivel de hidratación de la piel
        Retorna un score de 0-100 (100 = muy hidratada)
        """
        # La hidratación se refleja en el brillo y uniformidad del color
        if len(image.shape) == 3:
            # Analizar canal de luminosidad
            hsv = self._rgb_to_hsv(image)
            value_channel = hsv[:, :, 2]
            
            # Piel hidratada tiene más uniformidad en el brillo
            std_dev = np.std(value_channel)
            mean_value = np.mean(value_channel)
            
            # Menor desviación = más hidratada
            hydration_score = 100 - min(100, std_dev * 2)
            
            # Ajustar según el brillo promedio
            if mean_value > 0.6:
                hydration_score += 10
            
            return max(0, min(100, hydration_score))
        
        return 50.0  # Valor por defecto
    
    def analyze_elasticity(self, image: np.ndarray) -> float:
        """
        Analiza la elasticidad de la piel (basado en textura y firmeza)
        Retorna un score de 0-100 (100 = muy elástica)
        """
        # La elasticidad se relaciona con la suavidad y uniformidad
        texture_score = self.analyze_texture(image)
        
        # Piel elástica tiene textura más suave
        elasticity_score = texture_score * 0.8
        
        # Ajustar según análisis de líneas finas
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
            edges = self._detect_edges(gray)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Menos bordes = más elástica
            elasticity_score += (1 - edge_density) * 20
        
        return max(0, min(100, elasticity_score))
    
    def analyze_pigmentation(self, image: np.ndarray) -> float:
        """
        Analiza la uniformidad de la pigmentación
        Retorna un score de 0-100 (100 = pigmentación uniforme)
        """
        if len(image.shape) == 3:
            # Analizar variación de color
            std_r = np.std(image[:, :, 0])
            std_g = np.std(image[:, :, 1])
            std_b = np.std(image[:, :, 2])
            
            avg_std = (std_r + std_g + std_b) / 3
            
            # Menor variación = más uniforme
            pigmentation_score = 100 - min(100, avg_std * 0.5)
            
            return max(0, pigmentation_score)
        
        return 50.0
    
    def analyze_pore_size(self, image: np.ndarray) -> float:
        """
        Analiza el tamaño de los poros
        Retorna un score de 0-100 (100 = poros muy pequeños)
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Detectar poros como regiones oscuras pequeñas
        from scipy import ndimage
        
        # Aplicar filtro gaussiano para suavizar
        blurred = ndimage.gaussian_filter(gray, sigma=1)
        
        # Detectar regiones oscuras (poros)
        threshold = np.mean(blurred) - np.std(blurred)
        dark_regions = blurred < threshold
        
        # Contar y medir tamaño de poros
        labeled, num_features = ndimage.label(dark_regions)
        
        if num_features > 0:
            sizes = ndimage.sum(dark_regions, labeled, range(1, num_features + 1))
            avg_pore_size = np.mean(sizes) if len(sizes) > 0 else 0
            
            # Normalizar (poros pequeños = score alto)
            pore_score = 100 - min(100, avg_pore_size * 10)
        else:
            pore_score = 100  # Sin poros visibles
        
        return max(0, pore_score)
    
    def analyze_wrinkles(self, image: np.ndarray) -> float:
        """
        Analiza la presencia de arrugas
        Retorna un score de 0-100 (100 = sin arrugas)
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Detectar líneas (arrugas)
        edges = self._detect_edges(gray)
        
        # Filtrar líneas largas (arrugas)
        from scipy import ndimage
        
        # Aplicar transformada de Hough para detectar líneas
        # Simplificado: contar densidad de bordes
        edge_density = np.sum(edges > 0) / edges.size
        
        # Menos arrugas = score más alto
        wrinkles_score = 100 - min(100, edge_density * 500)
        
        return max(0, wrinkles_score)
    
    def analyze_redness(self, image: np.ndarray) -> float:
        """
        Analiza el enrojecimiento de la piel
        Retorna un score de 0-100 (100 = sin enrojecimiento)
        """
        if len(image.shape) == 3:
            # Analizar canal rojo vs otros canales
            r_channel = image[:, :, 0].astype(float)
            g_channel = image[:, :, 1].astype(float)
            b_channel = image[:, :, 2].astype(float)
            
            # Calcular ratio rojo
            total = r_channel + g_channel + b_channel
            total = np.where(total == 0, 1, total)  # Evitar división por cero
            
            red_ratio = r_channel / total
            avg_red_ratio = np.mean(red_ratio)
            
            # Ratio normal de piel: ~0.3-0.4
            # Ratio alto (>0.5) indica enrojecimiento
            if avg_red_ratio > 0.5:
                redness_score = 100 - ((avg_red_ratio - 0.5) * 200)
            else:
                redness_score = 100
            
            return max(0, min(100, redness_score))
        
        return 50.0
    
    def analyze_dark_spots(self, image: np.ndarray) -> float:
        """
        Analiza la presencia de manchas oscuras
        Retorna un score de 0-100 (100 = sin manchas)
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Detectar regiones oscuras (manchas)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Manchas son regiones significativamente más oscuras
        threshold = mean_intensity - 2 * std_intensity
        dark_spots = gray < threshold
        
        # Calcular área de manchas
        spot_area = np.sum(dark_spots) / dark_spots.size
        
        # Menos manchas = score más alto
        dark_spots_score = 100 - min(100, spot_area * 500)
        
        return max(0, dark_spots_score)
    
    def calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """
        Calcula el score general de calidad de piel
        """
        weights = {
            "texture": 0.15,
            "hydration": 0.20,
            "elasticity": 0.15,
            "pigmentation": 0.15,
            "pore_size": 0.10,
            "wrinkles": 0.10,
            "redness": 0.10,
            "dark_spots": 0.05,
        }
        
        overall = sum(scores.get(key, 50) * weight 
                     for key, weight in weights.items())
        
        return round(overall, 2)
    
    def analyze_complete(self, image: np.ndarray) -> SkinQualityScore:
        """
        Realiza análisis completo de calidad de piel
        """
        scores = {
            "texture": self.analyze_texture(image),
            "hydration": self.analyze_hydration(image),
            "elasticity": self.analyze_elasticity(image),
            "pigmentation": self.analyze_pigmentation(image),
            "pore_size": self.analyze_pore_size(image),
            "wrinkles": self.analyze_wrinkles(image),
            "redness": self.analyze_redness(image),
            "dark_spots": self.analyze_dark_spots(image),
        }
        
        overall_score = self.calculate_overall_score(scores)
        
        return SkinQualityScore(
            overall_score=overall_score,
            texture_score=scores["texture"],
            hydration_score=scores["hydration"],
            elasticity_score=scores["elasticity"],
            pigmentation_score=scores["pigmentation"],
            pore_size_score=scores["pore_size"],
            wrinkles_score=scores["wrinkles"],
            redness_score=scores["redness"],
            dark_spots_score=scores["dark_spots"],
        )
    
    def _rgb_to_hsv(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convierte RGB a HSV"""
        rgb = rgb_image.astype(float) / 255.0
        
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = np.where(max_val == 0, 0, delta / max_val)
        
        # Hue
        h = np.zeros_like(max_val)
        
        mask = delta != 0
        h[mask & (max_val == r)] = ((g[mask & (max_val == r)] - b[mask & (max_val == r)]) / delta[mask & (max_val == r)]) % 6
        h[mask & (max_val == g)] = ((b[mask & (max_val == g)] - r[mask & (max_val == g)]) / delta[mask & (max_val == g)]) + 2
        h[mask & (max_val == b)] = ((r[mask & (max_val == b)] - g[mask & (max_val == b)]) / delta[mask & (max_val == b)]) + 4
        
        h = h / 6.0
        
        hsv = np.stack([h, s, v], axis=2)
        return hsv
    
    def _detect_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """Detecta bordes usando Sobel"""
        from scipy import ndimage
        
        sobel_x = ndimage.sobel(gray_image, axis=1)
        sobel_y = ndimage.sobel(gray_image, axis=0)
        edges = np.hypot(sobel_x, sobel_y)
        
        return edges






