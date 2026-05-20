"""
Detector de condiciones de la piel
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SkinCondition:
    """Condición detectada en la piel"""
    name: str
    confidence: float  # 0-1
    severity: str  # "mild", "moderate", "severe"
    description: str
    affected_area_percentage: float  # 0-100


class SkinConditionsDetector:
    """Detecta condiciones específicas de la piel"""
    
    def __init__(self):
        """Inicializa el detector"""
        self.conditions_map = {
            "acne": {
                "description": "Presencia de acné o granos",
                "severity_levels": {
                    "mild": (0.1, 0.3),
                    "moderate": (0.3, 0.6),
                    "severe": (0.6, 1.0)
                }
            },
            "rosacea": {
                "description": "Enrojecimiento crónico (rosácea)",
                "severity_levels": {
                    "mild": (0.1, 0.3),
                    "moderate": (0.3, 0.6),
                    "severe": (0.6, 1.0)
                }
            },
            "eczema": {
                "description": "Eczema o dermatitis",
                "severity_levels": {
                    "mild": (0.1, 0.3),
                    "moderate": (0.3, 0.6),
                    "severe": (0.6, 1.0)
                }
            },
            "hyperpigmentation": {
                "description": "Hiperpigmentación o manchas oscuras",
                "severity_levels": {
                    "mild": (0.1, 0.3),
                    "moderate": (0.3, 0.6),
                    "severe": (0.6, 1.0)
                }
            },
            "dryness": {
                "description": "Piel seca o deshidratada",
                "severity_levels": {
                    "mild": (0.1, 0.3),
                    "moderate": (0.3, 0.6),
                    "severe": (0.6, 1.0)
                }
            },
            "sensitivity": {
                "description": "Piel sensible o irritada",
                "severity_levels": {
                    "mild": (0.1, 0.3),
                    "moderate": (0.3, 0.6),
                    "severe": (0.6, 1.0)
                }
            }
        }
    
    def detect_acne(self, image: np.ndarray) -> Optional[SkinCondition]:
        """Detecta presencia de acné"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Detectar protuberancias (granos) como regiones elevadas
        from scipy import ndimage
        
        # Aplicar filtro para detectar protuberancias
        blurred = ndimage.gaussian_filter(gray, sigma=2)
        laplacian = ndimage.laplace(blurred)
        
        # Detectar regiones con alta curvatura (granos)
        threshold = np.std(laplacian)
        bumps = laplacian > threshold
        
        affected_area = np.sum(bumps) / bumps.size
        
        if affected_area > 0.05:  # Más del 5% del área
            confidence = min(1.0, affected_area * 5)
            severity = self._determine_severity(affected_area, "acne")
            
            return SkinCondition(
                name="acne",
                confidence=confidence,
                severity=severity,
                description=self.conditions_map["acne"]["description"],
                affected_area_percentage=affected_area * 100
            )
        
        return None
    
    def detect_rosacea(self, image: np.ndarray) -> Optional[SkinCondition]:
        """Detecta rosácea (enrojecimiento crónico)"""
        if len(image.shape) == 3:
            r_channel = image[:, :, 0].astype(float)
            g_channel = image[:, :, 1].astype(float)
            b_channel = image[:, :, 2].astype(float)
            
            total = r_channel + g_channel + b_channel
            total = np.where(total == 0, 1, total)
            
            red_ratio = r_channel / total
            avg_red_ratio = np.mean(red_ratio)
            
            # Rosácea tiene ratio rojo muy alto (>0.55) y distribuido
            if avg_red_ratio > 0.55:
                # Verificar distribución (rosácea es más uniforme que acné)
                std_red = np.std(red_ratio)
                
                if std_red < 0.1:  # Distribución uniforme
                    confidence = min(1.0, (avg_red_ratio - 0.55) * 10)
                    affected_area = min(100, (avg_red_ratio - 0.5) * 200)
                    severity = self._determine_severity(avg_red_ratio - 0.5, "rosacea")
                    
                    return SkinCondition(
                        name="rosacea",
                        confidence=confidence,
                        severity=severity,
                        description=self.conditions_map["rosacea"]["description"],
                        affected_area_percentage=affected_area
                    )
        
        return None
    
    def detect_hyperpigmentation(self, image: np.ndarray) -> Optional[SkinCondition]:
        """Detecta hiperpigmentación"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Manchas oscuras son regiones significativamente más oscuras
        threshold = mean_intensity - 1.5 * std_intensity
        dark_spots = gray < threshold
        
        affected_area = np.sum(dark_spots) / dark_spots.size
        
        if affected_area > 0.1:  # Más del 10% del área
            confidence = min(1.0, affected_area * 3)
            severity = self._determine_severity(affected_area, "hyperpigmentation")
            
            return SkinCondition(
                name="hyperpigmentation",
                confidence=confidence,
                severity=severity,
                description=self.conditions_map["hyperpigmentation"]["description"],
                affected_area_percentage=affected_area * 100
            )
        
        return None
    
    def detect_dryness(self, image: np.ndarray) -> Optional[SkinCondition]:
        """Detecta sequedad de la piel"""
        if len(image.shape) == 3:
            hsv = self._rgb_to_hsv(image)
            value_channel = hsv[:, :, 2]
            
            # Piel seca tiene más variación y menos brillo
            std_dev = np.std(value_channel)
            mean_value = np.mean(value_channel)
            
            # Alta variación + bajo brillo = sequedad
            if std_dev > 0.15 and mean_value < 0.5:
                confidence = min(1.0, (0.15 - mean_value) * 5 + (std_dev - 0.15) * 2)
                severity = self._determine_severity(1 - mean_value, "dryness")
                
                return SkinCondition(
                    name="dryness",
                    confidence=confidence,
                    severity=severity,
                    description=self.conditions_map["dryness"]["description"],
                    affected_area_percentage=min(100, (1 - mean_value) * 100)
                )
        
        return None
    
    def detect_all_conditions(self, image: np.ndarray) -> List[SkinCondition]:
        """Detecta todas las condiciones posibles"""
        conditions = []
        
        # Detectar cada condición
        acne = self.detect_acne(image)
        if acne:
            conditions.append(acne)
        
        rosacea = self.detect_rosacea(image)
        if rosacea:
            conditions.append(rosacea)
        
        hyperpigmentation = self.detect_hyperpigmentation(image)
        if hyperpigmentation:
            conditions.append(hyperpigmentation)
        
        dryness = self.detect_dryness(image)
        if dryness:
            conditions.append(dryness)
        
        return conditions
    
    def _determine_severity(self, value: float, condition_type: str) -> str:
        """Determina la severidad basada en el valor"""
        if condition_type in self.conditions_map:
            levels = self.conditions_map[condition_type]["severity_levels"]
            
            if value >= levels["severe"][0]:
                return "severe"
            elif value >= levels["moderate"][0]:
                return "moderate"
            else:
                return "mild"
        
        return "mild"
    
    def _rgb_to_hsv(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convierte RGB a HSV"""
        rgb = rgb_image.astype(float) / 255.0
        
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val
        
        v = max_val
        s = np.where(max_val == 0, 0, delta / max_val)
        
        h = np.zeros_like(max_val)
        mask = delta != 0
        h[mask & (max_val == r)] = ((g[mask & (max_val == r)] - b[mask & (max_val == r)]) / delta[mask & (max_val == r)]) % 6
        h[mask & (max_val == g)] = ((b[mask & (max_val == g)] - r[mask & (max_val == g)]) / delta[mask & (max_val == g)]) + 2
        h[mask & (max_val == b)] = ((r[mask & (max_val == b)] - g[mask & (max_val == b)]) / delta[mask & (max_val == b)]) + 4
        
        h = h / 6.0
        
        return np.stack([h, s, v], axis=2)






