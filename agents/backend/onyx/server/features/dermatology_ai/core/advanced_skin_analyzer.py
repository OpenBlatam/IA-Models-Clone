"""
Analizador avanzado de piel con técnicas mejoradas
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import ndimage, signal
import cv2

from .skin_quality_metrics import SkinQualityMetrics, SkinQualityScore
from .skin_conditions_detector import SkinConditionsDetector
from ..utils.logger import logger
from ..utils.exceptions import AnalysisError


class AdvancedSkinAnalyzer(SkinQualityMetrics):
    """Analizador avanzado con técnicas mejoradas de análisis"""
    
    def __init__(self):
        """Inicializa el analizador avanzado"""
        super().__init__()
        self.conditions_detector = SkinConditionsDetector()
    
    def analyze_texture_advanced(self, image: np.ndarray) -> Dict[str, float]:
        """
        Análisis avanzado de textura usando múltiples técnicas
        
        Returns:
            Diccionario con métricas detalladas de textura
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 1. Análisis de varianza de Laplacian (nitidez)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Análisis de textura usando GLCM (Gray-Level Co-occurrence Matrix)
        # Simplificado: usar estadísticas de segundo orden
        from scipy import ndimage
        
        # Gradientes
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        gradient_magnitude = np.hypot(grad_x, grad_y)
        
        # Uniformidad de textura (menor variación = más uniforme)
        texture_uniformity = 100 - min(100, np.std(gradient_magnitude) / 10)
        
        # Suavidad (menor variación de gradientes = más suave)
        smoothness = 100 - min(100, np.mean(gradient_magnitude) / 5)
        
        # 3. Análisis de frecuencia (FFT)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Energía en diferentes bandas de frecuencia
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        
        # Baja frecuencia (textura suave)
        low_freq_mask = np.zeros((h, w))
        cv2.circle(low_freq_mask, (center_w, center_h), min(h, w) // 4, 1, -1)
        low_freq_energy = np.sum(magnitude_spectrum * low_freq_mask)
        
        # Alta frecuencia (textura rugosa)
        high_freq_mask = 1 - low_freq_mask
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
        
        # Ratio de suavidad
        smoothness_ratio = low_freq_energy / (low_freq_energy + high_freq_energy + 1e-10)
        smoothness_score = smoothness_ratio * 100
        
        # Score combinado
        texture_score = (
            min(100, laplacian_var / 10) * 0.3 +
            texture_uniformity * 0.3 +
            smoothness * 0.2 +
            smoothness_score * 0.2
        )
        
        return {
            "texture_score": max(0, min(100, texture_score)),
            "sharpness": min(100, laplacian_var / 10),
            "uniformity": texture_uniformity,
            "smoothness": smoothness,
            "frequency_analysis": smoothness_score
        }
    
    def analyze_hydration_advanced(self, image: np.ndarray) -> Dict[str, float]:
        """
        Análisis avanzado de hidratación
        
        Returns:
            Diccionario con métricas detalladas de hidratación
        """
        if len(image.shape) != 3:
            return {"hydration_score": 50.0, "moisture_level": 50.0, "oil_level": 50.0}
        
        # Convertir a LAB para mejor análisis de luminosidad
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(float) / 255.0
        
        # 1. Análisis de brillo (piel hidratada tiene más brillo)
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)
        
        # 2. Análisis de uniformidad (piel hidratada es más uniforme)
        uniformity = 100 - min(100, std_brightness * 200)
        
        # 3. Análisis de reflectividad (usando análisis de gradientes)
        grad_x = ndimage.sobel(l_channel, axis=1)
        grad_y = ndimage.sobel(l_channel, axis=0)
        gradient_magnitude = np.hypot(grad_x, grad_y)
        
        # Piel hidratada tiene menos variación de brillo
        reflectivity_score = 100 - min(100, np.std(gradient_magnitude) * 50)
        
        # 4. Análisis de color (piel hidratada tiene tonos más vivos)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(float) / 255.0
        mean_saturation = np.mean(saturation)
        
        # Score combinado
        hydration_score = (
            mean_brightness * 100 * 0.3 +
            uniformity * 0.3 +
            reflectivity_score * 0.2 +
            mean_saturation * 100 * 0.2
        )
        
        # Nivel de humedad (0-100)
        moisture_level = min(100, mean_brightness * 100 + uniformity * 0.5)
        
        # Nivel de grasa (análisis de brillo excesivo)
        oil_level = min(100, (mean_brightness - 0.7) * 200) if mean_brightness > 0.7 else 0
        
        return {
            "hydration_score": max(0, min(100, hydration_score)),
            "moisture_level": max(0, min(100, moisture_level)),
            "oil_level": max(0, min(100, oil_level)),
            "brightness": mean_brightness * 100,
            "uniformity": uniformity
        }
    
    def analyze_pores_advanced(self, image: np.ndarray) -> Dict[str, float]:
        """
        Análisis avanzado de poros
        
        Returns:
            Diccionario con métricas detalladas de poros
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 1. Detección de poros usando análisis de morfología
        # Aplicar filtro gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar regiones oscuras (poros)
        threshold = np.mean(blurred) - 0.5 * np.std(blurred)
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Operaciones morfológicas para limpiar
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos de poros
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {
                "pore_size_score": 100.0,
                "pore_count": 0,
                "pore_density": 0.0,
                "avg_pore_size": 0.0
            }
        
        # Calcular áreas de poros
        pore_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 2]
        
        if not pore_areas:
            return {
                "pore_size_score": 100.0,
                "pore_count": 0,
                "pore_density": 0.0,
                "avg_pore_size": 0.0
            }
        
        avg_pore_size = np.mean(pore_areas)
        pore_count = len(pore_areas)
        pore_density = (np.sum(pore_areas) / gray.size) * 100
        
        # Score: poros más pequeños y menos densos = mejor
        # Normalizar según tamaño de imagen
        normalized_size = avg_pore_size / (gray.shape[0] * gray.shape[1] / 10000)
        pore_size_score = 100 - min(100, normalized_size * 10)
        
        # Penalizar alta densidad
        density_penalty = min(50, pore_density * 2)
        pore_size_score = max(0, pore_size_score - density_penalty)
        
        return {
            "pore_size_score": max(0, min(100, pore_size_score)),
            "pore_count": pore_count,
            "pore_density": round(pore_density, 2),
            "avg_pore_size": round(avg_pore_size, 2)
        }
    
    def analyze_wrinkles_advanced(self, image: np.ndarray) -> Dict[str, float]:
        """
        Análisis avanzado de arrugas
        
        Returns:
            Diccionario con métricas detalladas de arrugas
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 1. Detección de líneas usando HoughLinesP
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                              minLineLength=30, maxLineGap=10)
        
        line_count = len(lines) if lines is not None else 0
        
        # 2. Análisis de densidad de bordes
        edge_density = np.sum(edges > 0) / edges.size
        
        # 3. Análisis de profundidad de arrugas (usando análisis de gradientes)
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        gradient_magnitude = np.hypot(grad_x, grad_y)
        
        # Arrugas profundas tienen gradientes altos
        deep_wrinkles = np.sum(gradient_magnitude > np.percentile(gradient_magnitude, 90))
        deep_wrinkle_ratio = deep_wrinkles / gradient_magnitude.size
        
        # 4. Análisis de dirección de arrugas (arrugas tienen dirección preferencial)
        # Simplificado: usar variación de dirección de gradientes
        gradient_direction = np.arctan2(grad_y, grad_x)
        direction_variance = np.std(gradient_direction)
        
        # Score combinado
        # Menos líneas, menos densidad de bordes, menos arrugas profundas = mejor
        wrinkles_score = (
            (1 - min(1, line_count / 100)) * 100 * 0.3 +
            (1 - min(1, edge_density * 10)) * 100 * 0.3 +
            (1 - min(1, deep_wrinkle_ratio * 20)) * 100 * 0.2 +
            (1 - min(1, direction_variance / np.pi)) * 100 * 0.2
        )
        
        return {
            "wrinkles_score": max(0, min(100, wrinkles_score)),
            "line_count": line_count,
            "edge_density": round(edge_density * 100, 2),
            "deep_wrinkle_ratio": round(deep_wrinkle_ratio * 100, 2),
            "wrinkle_severity": "mild" if wrinkles_score > 70 else 
                                "moderate" if wrinkles_score > 40 else "severe"
        }
    
    def analyze_complete_advanced(self, image: np.ndarray) -> Dict:
        """
        Análisis completo avanzado con todas las métricas mejoradas
        
        Returns:
            Diccionario con análisis completo
        """
        try:
            logger.info("Iniciando análisis avanzado de piel")
            
            # Análisis básicos mejorados
            texture_analysis = self.analyze_texture_advanced(image)
            hydration_analysis = self.analyze_hydration_advanced(image)
            pores_analysis = self.analyze_pores_advanced(image)
            wrinkles_analysis = self.analyze_wrinkles_advanced(image)
            
            # Análisis básicos (para compatibilidad)
            pigmentation_score = self.analyze_pigmentation(image)
            elasticity_score = self.analyze_elasticity(image)
            redness_score = self.analyze_redness(image)
            dark_spots_score = self.analyze_dark_spots(image)
            
            # Calcular score general mejorado
            scores = {
                "texture": texture_analysis["texture_score"],
                "hydration": hydration_analysis["hydration_score"],
                "elasticity": elasticity_score,
                "pigmentation": pigmentation_score,
                "pore_size": pores_analysis["pore_size_score"],
                "wrinkles": wrinkles_analysis["wrinkles_score"],
                "redness": redness_score,
                "dark_spots": dark_spots_score,
            }
            
            overall_score = self.calculate_overall_score(scores)
            
            # Detectar condiciones
            conditions = self.conditions_detector.detect_all_conditions(image)
            
            # Compilar resultados
            result = {
                "quality_scores": {
                    "overall_score": round(overall_score, 2),
                    "texture_score": round(texture_analysis["texture_score"], 2),
                    "hydration_score": round(hydration_analysis["hydration_score"], 2),
                    "elasticity_score": round(elasticity_score, 2),
                    "pigmentation_score": round(pigmentation_score, 2),
                    "pore_size_score": round(pores_analysis["pore_size_score"], 2),
                    "wrinkles_score": round(wrinkles_analysis["wrinkles_score"], 2),
                    "redness_score": round(redness_score, 2),
                    "dark_spots_score": round(dark_spots_score, 2),
                },
                "detailed_metrics": {
                    "texture": texture_analysis,
                    "hydration": hydration_analysis,
                    "pores": pores_analysis,
                    "wrinkles": wrinkles_analysis,
                },
                "conditions": [
                    {
                        "name": cond.name,
                        "confidence": round(cond.confidence, 2),
                        "severity": cond.severity,
                        "description": cond.description,
                        "affected_area_percentage": round(cond.affected_area_percentage, 2)
                    }
                    for cond in conditions
                ]
            }
            
            logger.info(f"Análisis completado - Score general: {overall_score:.2f}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error en análisis avanzado: {str(e)}", exc_info=True)
            raise AnalysisError(f"Error en análisis: {str(e)}")






