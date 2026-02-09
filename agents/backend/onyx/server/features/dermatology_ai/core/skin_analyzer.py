"""
Analizador principal de piel
"""

from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image
import io
import time

from .skin_quality_metrics import SkinQualityMetrics, SkinQualityScore
from .skin_conditions_detector import SkinConditionsDetector, SkinCondition
from .advanced_skin_analyzer import AdvancedSkinAnalyzer
from ..utils.logger import logger
from ..utils.exceptions import AnalysisError
from ..utils.cache import analysis_cache


class SkinAnalyzer:
    """Analizador principal de piel que combina métricas y detección de condiciones"""
    
    def __init__(self, use_advanced: bool = True, use_cache: bool = True):
        """
        Inicializa el analizador
        
        Args:
            use_advanced: Usar análisis avanzado (mejor precisión, más lento)
            use_cache: Usar cache para mejorar rendimiento
        """
        self.use_advanced = use_advanced
        self.use_cache = use_cache
        
        if use_advanced:
            self.quality_metrics = AdvancedSkinAnalyzer()
        else:
            self.quality_metrics = SkinQualityMetrics()
        
        self.conditions_detector = SkinConditionsDetector()
    
    def analyze_image(self, image: Union[np.ndarray, Image.Image, bytes, str], 
                     use_cache: Optional[bool] = None) -> Dict:
        """
        Analiza una imagen de piel
        
        Args:
            image: Imagen como numpy array, PIL Image, bytes, o path
            use_cache: Si usar cache (None = usar configuración por defecto)
            
        Returns:
            Diccionario con análisis completo
        """
        start_time = time.time()
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        try:
            # Convertir imagen a numpy array
            img_array = self._load_image(image)
            
            # Generar clave de cache
            cache_key = None
            if use_cache:
                import hashlib
                cache_key = hashlib.md5(img_array.tobytes()).hexdigest()
                cached_result = analysis_cache.get(cache_key)
                if cached_result:
                    logger.debug("Resultado obtenido del cache")
                    return cached_result
            
            # Analizar calidad
            if self.use_advanced and hasattr(self.quality_metrics, 'analyze_complete_advanced'):
                logger.debug("Usando análisis avanzado")
                advanced_result = self.quality_metrics.analyze_complete_advanced(img_array)
                quality_scores = SkinQualityScore(**advanced_result["quality_scores"])
                conditions = advanced_result.get("conditions", [])
            else:
                quality_scores = self.quality_metrics.analyze_complete(img_array)
                conditions = self.conditions_detector.detect_all_conditions(img_array)
            
            # Compilar resultados
            result = {
                "quality_scores": quality_scores.to_dict() if hasattr(quality_scores, 'to_dict') else quality_scores,
                "conditions": [
                    {
                        "name": cond.name if hasattr(cond, 'name') else cond["name"],
                        "confidence": round(cond.confidence if hasattr(cond, 'confidence') else cond["confidence"], 2),
                        "severity": cond.severity if hasattr(cond, 'severity') else cond["severity"],
                        "description": cond.description if hasattr(cond, 'description') else cond.get("description", ""),
                        "affected_area_percentage": round(cond.affected_area_percentage if hasattr(cond, 'affected_area_percentage') else cond.get("affected_area_percentage", 0), 2)
                    }
                    for cond in conditions
                ],
                "skin_type": self._determine_skin_type(quality_scores, conditions),
                "recommendations_priority": self._get_priority_areas(quality_scores, conditions)
            }
            
            # Agregar métricas detalladas si están disponibles
            if self.use_advanced and hasattr(self.quality_metrics, 'analyze_complete_advanced'):
                advanced_result = self.quality_metrics.analyze_complete_advanced(img_array)
                if "detailed_metrics" in advanced_result:
                    result["detailed_metrics"] = advanced_result["detailed_metrics"]
            
            # Cachear resultado
            if use_cache and cache_key:
                analysis_cache.set(cache_key, result)
            
            duration = time.time() - start_time
            logger.log_analysis("image", duration, True)
            
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            logger.log_analysis("image", duration, False)
            logger.error(f"Error analizando imagen: {str(e)}", exc_info=True)
            raise AnalysisError(f"Error en análisis de imagen: {str(e)}")
    
    def analyze_video(self, video_frames: List[np.ndarray]) -> Dict:
        """
        Analiza un video (secuencia de frames)
        
        Args:
            video_frames: Lista de frames como numpy arrays
            
        Returns:
            Diccionario con análisis agregado
        """
        if not video_frames:
            raise ValueError("La lista de frames está vacía")
        
        # Analizar cada frame
        frame_results = []
        for frame in video_frames:
            result = self.analyze_image(frame)
            frame_results.append(result)
        
        # Agregar resultados
        aggregated = self._aggregate_results(frame_results)
        
        return aggregated
    
    def _load_image(self, image: Union[np.ndarray, Image.Image, bytes, str]) -> np.ndarray:
        """Carga y convierte imagen a numpy array"""
        if isinstance(image, np.ndarray):
            return image
        
        if isinstance(image, Image.Image):
            return np.array(image)
        
        if isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
            return np.array(img)
        
        if isinstance(image, str):
            # Asumir que es un path
            img = Image.open(image)
            return np.array(img)
        
        raise ValueError(f"Tipo de imagen no soportado: {type(image)}")
    
    def _determine_skin_type(self, quality_scores: Union[SkinQualityScore, Dict], 
                            conditions: List) -> str:
        """Determina el tipo de piel basado en análisis"""
        # Convertir a dict si es necesario
        if hasattr(quality_scores, 'to_dict'):
            scores_dict = quality_scores.to_dict()
        elif isinstance(quality_scores, dict):
            scores_dict = quality_scores
        else:
            scores_dict = {}
        
        # Extraer scores
        hydration = scores_dict.get("hydration_score", 50)
        pore_size = scores_dict.get("pore_size_score", 50)
        texture = scores_dict.get("texture_score", 50)
        redness = scores_dict.get("redness_score", 50)
        
        # Lógica para determinar tipo de piel
        condition_names = []
        for c in conditions:
            if hasattr(c, 'name'):
                condition_names.append(c.name)
            elif isinstance(c, dict):
                condition_names.append(c.get("name", ""))
        
        if "dryness" in condition_names:
            return "dry"
        
        if "rosacea" in condition_names or "sensitivity" in condition_names:
            return "sensitive"
        
        if hydration < 50:
            return "dry"
        
        if hydration > 70 and pore_size < 50:
            return "oily"
        
        if texture > 70 and hydration > 60:
            return "normal"
        
        return "combination"
    
    def _get_priority_areas(self, quality_scores: Union[SkinQualityScore, Dict],
                           conditions: List) -> List[str]:
        """Obtiene áreas prioritarias para mejorar"""
        priorities = []
        
        # Convertir a dict si es necesario
        if hasattr(quality_scores, 'to_dict'):
            scores_dict = quality_scores.to_dict()
        elif isinstance(quality_scores, dict):
            scores_dict = quality_scores
        else:
            scores_dict = {}
        
        # Basado en scores bajos
        if scores_dict.get("hydration_score", 50) < 50:
            priorities.append("hydration")
        
        if scores_dict.get("texture_score", 50) < 50:
            priorities.append("texture")
        
        if scores_dict.get("pigmentation_score", 50) < 50:
            priorities.append("pigmentation")
        
        if scores_dict.get("wrinkles_score", 50) < 50:
            priorities.append("anti_aging")
        
        if scores_dict.get("pore_size_score", 50) < 50:
            priorities.append("pore_care")
        
        # Basado en condiciones detectadas
        for condition in conditions:
            severity = None
            name = None
            
            if hasattr(condition, 'severity'):
                severity = condition.severity
                name = condition.name
            elif isinstance(condition, dict):
                severity = condition.get("severity")
                name = condition.get("name")
            
            if severity in ["moderate", "severe"] and name:
                priorities.append(name)
        
        return priorities[:5]  # Top 5 prioridades
    
    def _aggregate_results(self, frame_results: List[Dict]) -> Dict:
        """Agrega resultados de múltiples frames"""
        if not frame_results:
            return {}
        
        # Promediar scores de calidad
        quality_scores_list = [r["quality_scores"] for r in frame_results]
        
        aggregated_scores = {}
        for key in quality_scores_list[0].keys():
            values = [s[key] for s in quality_scores_list]
            aggregated_scores[key] = round(sum(values) / len(values), 2)
        
        # Agregar condiciones (usar las más frecuentes)
        all_conditions = []
        for result in frame_results:
            all_conditions.extend(result["conditions"])
        
        # Contar frecuencia de condiciones
        condition_counts = {}
        for cond in all_conditions:
            name = cond["name"]
            if name not in condition_counts:
                condition_counts[name] = {
                    "count": 0,
                    "total_confidence": 0,
                    "severities": []
                }
            condition_counts[name]["count"] += 1
            condition_counts[name]["total_confidence"] += cond["confidence"]
            condition_counts[name]["severities"].append(cond["severity"])
        
        # Crear condiciones agregadas
        aggregated_conditions = []
        for name, data in condition_counts.items():
            avg_confidence = data["total_confidence"] / data["count"]
            most_common_severity = max(set(data["severities"]), 
                                     key=data["severities"].count)
            
            aggregated_conditions.append({
                "name": name,
                "confidence": round(avg_confidence, 2),
                "severity": most_common_severity,
                "frequency": round(data["count"] / len(frame_results), 2)
            })
        
        # Determinar tipo de piel más común
        skin_types = [r["skin_type"] for r in frame_results]
        most_common_skin_type = max(set(skin_types), key=skin_types.count)
        
        # Prioridades más frecuentes
        all_priorities = []
        for r in frame_results:
            all_priorities.extend(r["recommendations_priority"])
        
        priority_counts = {}
        for p in all_priorities:
            priority_counts[p] = priority_counts.get(p, 0) + 1
        
        top_priorities = sorted(priority_counts.items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:5]
        
        return {
            "quality_scores": aggregated_scores,
            "conditions": aggregated_conditions,
            "skin_type": most_common_skin_type,
            "recommendations_priority": [p[0] for p in top_priorities],
            "analysis_frames": len(frame_results)
        }

