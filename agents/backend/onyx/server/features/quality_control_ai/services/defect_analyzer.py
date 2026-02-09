"""
Analizador de Defectos - Análisis detallado de defectos
"""

import logging
from typing import Dict, List, Optional
import numpy as np
from collections import Counter

from ..core.defect_classifier import Defect, DefectType
from ..utils.detection_utils import DetectionUtils

logger = logging.getLogger(__name__)

COVERAGE_HIGH_THRESHOLD = 10.0
SEVERE_DEFECTS_COUNT_THRESHOLD = 3
MULTIPLE_DEFECTS_COUNT_THRESHOLD = 10
LARGE_AREA_THRESHOLD = 50000
SEVERE_DEFECTS_ALERT_THRESHOLD = 5
SMALL_DEFECT_AREA = 500
MEDIUM_DEFECT_AREA = 2000


class DefectAnalyzer:
    """
    Analizador para análisis detallado de defectos detectados
    """
    
    def __init__(self):
        """Inicializar analizador de defectos"""
        self.detection_utils = DetectionUtils()
        logger.info("Defect Analyzer initialized")
    
    def analyze_defects(self, defects: List[Defect], image_size: tuple) -> Dict:
        """
        Analizar defectos y generar estadísticas
        
        Args:
            defects: Lista de defectos
            image_size: Tamaño de la imagen (width, height)
            
        Returns:
            Diccionario con análisis detallado
        """
        if not defects:
            return {
                "total_defects": 0,
                "status": "no_defects",
                "analysis": {}
            }
        
        # Estadísticas básicas
        total_defects = len(defects)
        total_area = sum(d.area for d in defects)
        image_area = image_size[0] * image_size[1]
        coverage_percentage = (total_area / image_area * 100) if image_area > 0 else 0
        
        # Análisis por tipo
        type_analysis = self._analyze_by_type(defects)
        
        # Análisis por severidad
        severity_analysis = self._analyze_by_severity(defects)
        
        # Análisis espacial
        spatial_analysis = self._analyze_spatial_distribution(defects, image_size)
        
        # Análisis de tamaño
        size_analysis = self._analyze_sizes(defects)
        
        # Determinar estado
        status = self._determine_status(defects, coverage_percentage)
        
        return {
            "total_defects": total_defects,
            "total_area": total_area,
            "coverage_percentage": round(coverage_percentage, 2),
            "status": status,
            "type_analysis": type_analysis,
            "severity_analysis": severity_analysis,
            "spatial_analysis": spatial_analysis,
            "size_analysis": size_analysis,
            "recommendations": self._generate_recommendations(defects, type_analysis, severity_analysis)
        }
    
    def _analyze_by_type(self, defects: List[Defect]) -> Dict:
        """Analizar defectos por tipo"""
        type_counts = Counter(d.defect_type for d in defects)
        type_areas = {}
        type_confidences = {}
        
        for defect in defects:
            defect_type = defect.defect_type
            if defect_type not in type_areas:
                type_areas[defect_type] = []
                type_confidences[defect_type] = []
            
            type_areas[defect_type].append(defect.area)
            type_confidences[defect_type].append(defect.confidence)
        
        analysis = {}
        for defect_type, count in type_counts.items():
            analysis[defect_type.value] = {
                "count": count,
                "percentage": round(count / len(defects) * 100, 2),
                "total_area": sum(type_areas[defect_type]),
                "average_area": round(np.mean(type_areas[defect_type]), 2),
                "average_confidence": round(np.mean(type_confidences[defect_type]), 2),
                "max_confidence": round(max(type_confidences[defect_type]), 2)
            }
        
        return analysis
    
    def _analyze_by_severity(self, defects: List[Defect]) -> Dict:
        """Analizar defectos por severidad"""
        severity_counts = Counter(d.severity for d in defects)
        severity_areas = {}
        
        for defect in defects:
            severity = defect.severity
            if severity not in severity_areas:
                severity_areas[severity] = []
            severity_areas[severity].append(defect.area)
        
        analysis = {}
        for severity, count in severity_counts.items():
            analysis[severity] = {
                "count": count,
                "percentage": round(count / len(defects) * 100, 2),
                "total_area": sum(severity_areas[severity]),
                "average_area": round(np.mean(severity_areas[severity]), 2)
            }
        
        return analysis
    
    def _analyze_spatial_distribution(self, defects: List[Defect], image_size: tuple) -> Dict:
        """Analizar distribución espacial de defectos"""
        if not defects:
            return {}
        
        width, height = image_size
        
        # Calcular centroides
        centroids = []
        for defect in defects:
            x, y, w, h = defect.location
            center_x = x + w // 2
            center_y = y + h // 2
            centroids.append((center_x, center_y))
        
        # Análisis por cuadrantes
        quadrant_counts = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
        for cx, cy in centroids:
            if cx < width / 2 and cy < height / 2:
                quadrant_counts["top_left"] += 1
            elif cx >= width / 2 and cy < height / 2:
                quadrant_counts["top_right"] += 1
            elif cx < width / 2 and cy >= height / 2:
                quadrant_counts["bottom_left"] += 1
            else:
                quadrant_counts["bottom_right"] += 1
        
        # Calcular densidad
        center_x = np.mean([c[0] for c in centroids])
        center_y = np.mean([c[1] for c in centroids])
        
        # Calcular dispersión
        if len(centroids) > 1:
            std_x = np.std([c[0] for c in centroids])
            std_y = np.std([c[1] for c in centroids])
            dispersion = np.sqrt(std_x ** 2 + std_y ** 2)
        else:
            dispersion = 0.0
        
        return {
            "quadrant_distribution": quadrant_counts,
            "center_of_mass": (round(center_x, 2), round(center_y, 2)),
            "dispersion": round(dispersion, 2),
            "is_clustered": dispersion < (width + height) / 4
        }
    
    def _analyze_sizes(self, defects: List[Defect]) -> Dict:
        """Analizar tamaños de defectos"""
        if not defects:
            return {}
        
        areas = [d.area for d in defects]
        
        return {
            "min_area": int(min(areas)),
            "max_area": int(max(areas)),
            "mean_area": round(np.mean(areas), 2),
            "median_area": round(np.median(areas), 2),
            "std_area": round(np.std(areas), 2),
            "small_defects": sum(1 for a in areas if a < SMALL_DEFECT_AREA),
            "medium_defects": sum(1 for a in areas if SMALL_DEFECT_AREA <= a < MEDIUM_DEFECT_AREA),
            "large_defects": sum(1 for a in areas if a >= MEDIUM_DEFECT_AREA)
        }
    
    def _determine_status(self, defects: List[Defect], coverage_percentage: float) -> str:
        """Determinar estado basado en defectos"""
        critical_defects = [d for d in defects if d.severity == "critical"]
        severe_defects = [d for d in defects if d.severity == "severe"]
        
        if critical_defects:
            return "critical"
        elif severe_defects and len(severe_defects) > SEVERE_DEFECTS_COUNT_THRESHOLD:
            return "severe"
        elif coverage_percentage > COVERAGE_HIGH_THRESHOLD:
            return "high_coverage"
        elif len(defects) > MULTIPLE_DEFECTS_COUNT_THRESHOLD:
            return "multiple_defects"
        else:
            return "acceptable"
    
    def _generate_recommendations(
        self,
        defects: List[Defect],
        type_analysis: Dict,
        severity_analysis: Dict
    ) -> List[str]:
        """Generar recomendaciones basadas en análisis"""
        recommendations = []
        
        # Verificar defectos críticos
        if severity_analysis.get("critical", {}).get("count", 0) > 0:
            recommendations.append("Rechazar producto - defectos críticos detectados")
        
        if type_analysis:
            most_common_type = max(type_analysis.items(), key=lambda x: x[1]["count"])
            if most_common_type[1]["count"] > SEVERE_DEFECTS_COUNT_THRESHOLD:
                recommendations.append(
                    f"Revisar proceso de producción - múltiples defectos de tipo '{most_common_type[0]}' detectados"
                )
        
        total_area = sum(d.area for d in defects)
        if total_area > LARGE_AREA_THRESHOLD:
            recommendations.append("Revisar calidad general - área significativa afectada por defectos")
        
        if severity_analysis.get("severe", {}).get("count", 0) > SEVERE_DEFECTS_ALERT_THRESHOLD:
            recommendations.append("Revisar estándares de calidad - múltiples defectos severos")
        
        if not recommendations:
            recommendations.append("Producto dentro de estándares de calidad aceptables")
        
        return recommendations
    
    def compare_with_reference(
        self,
        current_defects: List[Defect],
        reference_defects: List[Defect]
    ) -> Dict:
        """
        Comparar defectos actuales con referencia
        
        Args:
            current_defects: Defectos actuales
            reference_defects: Defectos de referencia
            
        Returns:
            Diccionario con comparación
        """
        current_analysis = self.analyze_defects(current_defects, (1000, 1000))
        reference_analysis = self.analyze_defects(reference_defects, (1000, 1000))
        
        return {
            "current": current_analysis,
            "reference": reference_analysis,
            "difference": {
                "defect_count_diff": len(current_defects) - len(reference_defects),
                "improvement": len(current_defects) < len(reference_defects)
            }
        }






