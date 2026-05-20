"""
Dermatology AI - Sistema de Análisis de Piel y Recomendaciones de Skincare
==========================================================================

Sistema completo de análisis de piel con IA que incluye:
- Análisis avanzado de imágenes y videos
- Recomendaciones personalizadas de skincare
- Seguimiento de historial de análisis
- Generación de reportes y visualizaciones
"""

__version__ = "5.5.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for skin analysis and skincare recommendations"

from .core.skin_analyzer import SkinAnalyzer
from .core.advanced_skin_analyzer import AdvancedSkinAnalyzer
from .services.skincare_recommender import SkincareRecommender
from .services.image_processor import ImageProcessor
from .services.video_processor import VideoProcessor
from .services.history_tracker import HistoryTracker, AnalysisRecord
from .services.report_generator import ReportGenerator
from .services.visualization import VisualizationGenerator
from .utils.logger import logger, DermatologyLogger
from .utils.cache import AnalysisCache, analysis_cache
from .utils.exceptions import (
    DermatologyAIException,
    ImageProcessingError,
    VideoProcessingError,
    AnalysisError,
    ValidationError
)

__all__ = [
    "SkinAnalyzer",
    "AdvancedSkinAnalyzer",
    "SkincareRecommender",
    "ImageProcessor",
    "VideoProcessor",
    "HistoryTracker",
    "AnalysisRecord",
    "ReportGenerator",
    "VisualizationGenerator",
    "logger",
    "DermatologyLogger",
    "AnalysisCache",
    "analysis_cache",
    "DermatologyAIException",
    "ImageProcessingError",
    "VideoProcessingError",
    "AnalysisError",
    "ValidationError",
]

