"""
Analizador de múltiples áreas del cuerpo
"""

from typing import Dict, List, Optional
from enum import Enum
import numpy as np
import cv2


class BodyArea(str, Enum):
    """Áreas del cuerpo"""
    FACE = "face"
    FOREHEAD = "forehead"
    CHEEKS = "cheeks"
    NOSE = "nose"
    CHIN = "chin"
    NECK = "neck"
    CHEST = "chest"
    ARMS = "arms"
    HANDS = "hands"
    LEGS = "legs"
    BACK = "back"


class BodyAreaAnalyzer:
    """Analiza diferentes áreas del cuerpo"""
    
    def __init__(self):
        """Inicializa el analizador de áreas"""
        self.area_characteristics = {
            BodyArea.FACE: {
                "sensitivity": "high",
                "common_concerns": ["acne", "wrinkles", "pigmentation"],
                "recommended_products": ["cleanser", "moisturizer", "sunscreen", "serum"]
            },
            BodyArea.FOREHEAD: {
                "sensitivity": "medium",
                "common_concerns": ["oiliness", "acne", "wrinkles"],
                "recommended_products": ["cleanser", "toner", "moisturizer"]
            },
            BodyArea.CHEEKS: {
                "sensitivity": "high",
                "common_concerns": ["dryness", "redness", "pigmentation"],
                "recommended_products": ["moisturizer", "serum", "sunscreen"]
            },
            BodyArea.NOSE: {
                "sensitivity": "medium",
                "common_concerns": ["large_pores", "oiliness", "blackheads"],
                "recommended_products": ["toner", "exfoliant", "cleanser"]
            },
            BodyArea.CHIN: {
                "sensitivity": "medium",
                "common_concerns": ["acne", "oiliness"],
                "recommended_products": ["cleanser", "treatment", "toner"]
            },
            BodyArea.NECK: {
                "sensitivity": "high",
                "common_concerns": ["aging", "dryness", "wrinkles"],
                "recommended_products": ["moisturizer", "serum", "sunscreen"]
            },
            BodyArea.CHEST: {
                "sensitivity": "medium",
                "common_concerns": ["acne", "pigmentation", "dryness"],
                "recommended_products": ["cleanser", "moisturizer", "sunscreen"]
            },
            BodyArea.ARMS: {
                "sensitivity": "low",
                "common_concerns": ["dryness", "keratosis", "pigmentation"],
                "recommended_products": ["moisturizer", "exfoliant", "sunscreen"]
            },
            BodyArea.HANDS: {
                "sensitivity": "medium",
                "common_concerns": ["aging", "dryness", "spots"],
                "recommended_products": ["moisturizer", "sunscreen", "treatment"]
            },
            BodyArea.LEGS: {
                "sensitivity": "low",
                "common_concerns": ["dryness", "keratosis", "cellulite"],
                "recommended_products": ["moisturizer", "exfoliant"]
            },
            BodyArea.BACK: {
                "sensitivity": "medium",
                "common_concerns": ["acne", "dryness"],
                "recommended_products": ["cleanser", "treatment", "moisturizer"]
            }
        }
    
    def detect_body_area(self, image: np.ndarray, 
                        user_input: Optional[BodyArea] = None) -> BodyArea:
        """
        Detecta o valida el área del cuerpo en la imagen
        
        Args:
            image: Imagen a analizar
            user_input: Área especificada por el usuario (opcional)
            
        Returns:
            Área detectada o especificada
        """
        # Si el usuario especifica, usar esa
        if user_input:
            return user_input
        
        # Intentar detectar automáticamente (simplificado)
        # En producción, usar modelos ML más avanzados
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Heurísticas simples basadas en proporciones
        if aspect_ratio > 0.7 and aspect_ratio < 1.3:
            # Imagen cuadrada o casi cuadrada - probablemente cara
            return BodyArea.FACE
        elif aspect_ratio > 1.5:
            # Imagen horizontal - podría ser pecho, espalda, brazos
            return BodyArea.CHEST
        else:
            # Imagen vertical - podría ser cara completa, cuello
            return BodyArea.FACE
    
    def analyze_area(self, image: np.ndarray, area: BodyArea,
                    analysis_result: Dict) -> Dict:
        """
        Analiza un área específica del cuerpo
        
        Args:
            image: Imagen del área
            area: Área del cuerpo
            analysis_result: Resultado del análisis general
            
        Returns:
            Análisis específico del área
        """
        characteristics = self.area_characteristics.get(area, {})
        
        # Ajustar scores según características del área
        quality_scores = analysis_result.get("quality_scores", {}).copy()
        
        # Áreas más sensibles pueden tener scores más bajos normalmente
        if characteristics.get("sensitivity") == "high":
            # Ajustar ligeramente hacia abajo (más estricto)
            for key in quality_scores:
                if key != "overall_score":
                    quality_scores[key] = max(0, quality_scores[key] - 5)
        
        # Recalcular score general
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
        
        overall_score = sum(
            quality_scores.get(key, 50) * weight 
            for key, weight in weights.items()
        )
        quality_scores["overall_score"] = round(overall_score, 2)
        
        # Recomendaciones específicas del área
        area_recommendations = {
            "area": area.value,
            "sensitivity": characteristics.get("sensitivity", "medium"),
            "common_concerns": characteristics.get("common_concerns", []),
            "recommended_product_categories": characteristics.get("recommended_products", []),
            "area_specific_tips": self._get_area_tips(area)
        }
        
        return {
            "area": area.value,
            "quality_scores": quality_scores,
            "conditions": analysis_result.get("conditions", []),
            "recommendations": area_recommendations
        }
    
    def analyze_multiple_areas(self, area_images: Dict[BodyArea, np.ndarray],
                             analyzer) -> Dict:
        """
        Analiza múltiples áreas del cuerpo
        
        Args:
            area_images: Diccionario de área -> imagen
            analyzer: Instancia de SkinAnalyzer
            
        Returns:
            Análisis completo de todas las áreas
        """
        results = {}
        
        for area, image in area_images.items():
            # Analizar cada área
            analysis = analyzer.analyze_image(image)
            area_result = self.analyze_area(image, area, analysis)
            results[area.value] = area_result
        
        # Análisis comparativo
        comparison = self._compare_areas(results)
        
        return {
            "areas_analyzed": list(area_images.keys()),
            "results": results,
            "comparison": comparison,
            "overall_assessment": self._overall_assessment(results)
        }
    
    def _compare_areas(self, results: Dict) -> Dict:
        """Compara diferentes áreas"""
        if len(results) < 2:
            return {}
        
        scores = {
            area: result["quality_scores"].get("overall_score", 0)
            for area, result in results.items()
        }
        
        best_area = max(scores.items(), key=lambda x: x[1])
        worst_area = min(scores.items(), key=lambda x: x[1])
        
        return {
            "best_area": {
                "area": best_area[0],
                "score": best_area[1]
            },
            "worst_area": {
                "area": worst_area[0],
                "score": worst_area[1]
            },
            "score_range": max(scores.values()) - min(scores.values()),
            "average_score": sum(scores.values()) / len(scores)
        }
    
    def _overall_assessment(self, results: Dict) -> Dict:
        """Genera evaluación general de todas las áreas"""
        all_scores = []
        all_conditions = []
        
        for result in results.values():
            scores = result.get("quality_scores", {})
            all_scores.append(scores.get("overall_score", 0))
            all_conditions.extend(result.get("conditions", []))
        
        # Condiciones más comunes
        condition_counts = {}
        for condition in all_conditions:
            name = condition.get("name", "unknown")
            condition_counts[name] = condition_counts.get(name, 0) + 1
        
        most_common = sorted(
            condition_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            "average_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "areas_analyzed": len(results),
            "most_common_conditions": [{"name": name, "frequency": count} for name, count in most_common],
            "recommendation": self._generate_overall_recommendation(all_scores, most_common)
        }
    
    def _generate_overall_recommendation(self, scores: List[float],
                                        common_conditions: List) -> str:
        """Genera recomendación general"""
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score < 50:
            return "Se recomienda consulta dermatológica. Múltiples áreas necesitan atención."
        elif avg_score < 70:
            return "Algunas áreas necesitan mejoras. Enfócate en rutinas específicas por área."
        else:
            return "Tu piel está en buen estado general. Mantén tu rutina actual."
    
    def _get_area_tips(self, area: BodyArea) -> List[str]:
        """Obtiene tips específicos para un área"""
        tips_map = {
            BodyArea.FACE: [
                "Limpia tu cara dos veces al día",
                "Usa protector solar todos los días",
                "Sé gentil al aplicar productos"
            ],
            BodyArea.FOREHEAD: [
                "Puede ser más grasa que otras áreas",
                "Usa productos oil-free si es necesario",
                "No te olvides del protector solar"
            ],
            BodyArea.CHEEKS: [
                "Área más sensible, usa productos suaves",
                "Hidrata bien esta área",
                "Evita frotar demasiado"
            ],
            BodyArea.NOSE: [
                "Área con más poros, exfolia regularmente",
                "Usa productos para minimizar poros",
                "Limpia bien para evitar puntos negros"
            ],
            BodyArea.NECK: [
                "Extiende tu rutina facial al cuello",
                "Área propensa a envejecimiento",
                "No olvides el protector solar"
            ],
            BodyArea.CHEST: [
                "Puede ser más seca que la cara",
                "Usa productos hidratantes",
                "Protege del sol"
            ]
        }
        
        return tips_map.get(area, [
            "Mantén el área limpia",
            "Hidrata regularmente",
            "Protege del sol"
        ])






