"""
Sistema de análisis de fotos con diferentes formatos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class FormatAnalysis:
    """Análisis con formato específico"""
    format_type: str  # "jpg", "png", "heic", "raw", "webp"
    image_url: str
    file_size: int  # bytes
    compression_ratio: Optional[float] = None
    analysis_quality: str  # "poor", "fair", "good", "excellent"
    detectable_features: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.detectable_features is None:
            self.detectable_features = []
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "format_type": self.format_type,
            "image_url": self.image_url,
            "file_size": self.file_size,
            "compression_ratio": self.compression_ratio,
            "analysis_quality": self.analysis_quality,
            "detectable_features": self.detectable_features,
            "recommendations": self.recommendations
        }


@dataclass
class FormatReport:
    """Reporte de análisis de formato"""
    id: str
    user_id: str
    analyses: List[FormatAnalysis]
    optimal_format: str
    format_comparison: Dict
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "analyses": [a.to_dict() for a in self.analyses],
            "optimal_format": self.optimal_format,
            "format_comparison": self.format_comparison,
            "created_at": self.created_at
        }


class FormatAnalysisSystem:
    """Sistema de análisis con diferentes formatos"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[FormatReport]] = {}  # user_id -> [reports]
    
    def analyze_with_format(self, user_id: str, format_data: List[Dict]) -> FormatReport:
        """Analiza con diferentes formatos"""
        analyses = []
        
        for fmt_data in format_data:
            format_type = fmt_data.get("format_type", "jpg")
            image_url = fmt_data.get("image_url", "")
            file_size = fmt_data.get("file_size", 0)
            
            # Determinar calidad según formato
            analysis_quality, compression_ratio = self._evaluate_format(format_type, file_size)
            
            # Características detectables
            detectable_features = self._get_detectable_features(format_type)
            
            # Recomendaciones
            recommendations = self._generate_format_recommendations(format_type, analysis_quality)
            
            analysis = FormatAnalysis(
                format_type=format_type,
                image_url=image_url,
                file_size=file_size,
                compression_ratio=compression_ratio,
                analysis_quality=analysis_quality,
                detectable_features=detectable_features,
                recommendations=recommendations
            )
            analyses.append(analysis)
        
        # Determinar formato óptimo
        optimal_format = "png"  # Por defecto
        best_analysis = max(analyses, key=lambda a: self._get_quality_score(a.analysis_quality))
        if best_analysis:
            optimal_format = best_analysis.format_type
        
        # Comparación de formatos
        format_comparison = {
            "best_quality": best_analysis.analysis_quality if best_analysis else "fair",
            "recommended_format": "png",
            "note": "Formatos sin pérdida (PNG, RAW) ofrecen mejor calidad para análisis"
        }
        
        report = FormatReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            analyses=analyses,
            optimal_format=optimal_format,
            format_comparison=format_comparison
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def _evaluate_format(self, format_type: str, file_size: int) -> tuple:
        """Evalúa formato y retorna calidad y ratio de compresión"""
        quality_map = {
            "raw": ("excellent", None),
            "png": ("excellent", None),
            "heic": ("good", 0.5),
            "webp": ("good", 0.6),
            "jpg": ("fair", 0.7)
        }
        
        quality, compression = quality_map.get(format_type.lower(), ("fair", 0.7))
        
        # Ajustar según tamaño de archivo
        if file_size < 100000:  # < 100KB
            quality = "poor"
        elif file_size > 5000000:  # > 5MB
            if quality == "fair":
                quality = "good"
        
        return quality, compression
    
    def _get_detectable_features(self, format_type: str) -> List[str]:
        """Obtiene características detectables según formato"""
        features = {
            "raw": ["Todas las características", "Máximo detalle", "Sin pérdida de información"],
            "png": ["Todas las características", "Alto detalle", "Sin compresión"],
            "heic": ["Todas las características", "Buen detalle", "Compresión eficiente"],
            "webp": ["Mayoría de características", "Buen detalle", "Compresión moderada"],
            "jpg": ["Características básicas", "Detalle limitado", "Compresión con pérdida"]
        }
        return features.get(format_type.lower(), features["jpg"])
    
    def _generate_format_recommendations(self, format_type: str, quality: str) -> List[str]:
        """Genera recomendaciones según formato"""
        recommendations = []
        
        if quality in ["poor", "fair"]:
            recommendations.append("Usa formato PNG o RAW para mejor análisis")
        
        if format_type.lower() == "jpg":
            recommendations.append("JPG tiene compresión con pérdida. Considera PNG para análisis detallado")
        
        if format_type.lower() in ["raw", "png"]:
            recommendations.append("Excelente formato para análisis preciso")
        
        return recommendations
    
    def _get_quality_score(self, quality: str) -> int:
        """Obtiene score numérico de calidad"""
        scores = {"poor": 1, "fair": 2, "good": 3, "excellent": 4}
        return scores.get(quality, 2)
    
    def get_user_reports(self, user_id: str) -> List[FormatReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])






