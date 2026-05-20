"""
Service Registry - Configuración centralizada de servicios
Registra todos los servicios en el contenedor de inyección de dependencias
"""

import logging
from typing import Any

from ..core.dependency_injection import get_container, register_service
from ..services.spotify_service import SpotifyService
from ..core.music_analyzer import MusicAnalyzer
from ..services.music_coach import MusicCoach
from ..services.comparison_service import ComparisonService
from ..services.export_service import ExportService
from ..services.history_service import HistoryService
from ..services.analytics_service import analytics_service
from ..services.webhook_service import WebhookService
from ..services.favorites_service import FavoritesService
from ..services.tagging_service import TaggingService
from ..services.auth_service import AuthService
from ..services.playlist_service import PlaylistService
from ..services.intelligent_recommender import IntelligentRecommender
from ..services.dashboard_service import DashboardService
from ..services.notification_service import NotificationService
from ..services.trends_analyzer import TrendsAnalyzer
from ..services.collaboration_analyzer import CollaborationAnalyzer
from ..services.alert_service import AlertService
from ..services.temporal_analyzer import TemporalAnalyzer
from ..services.quality_analyzer import QualityAnalyzer
from ..services.contextual_recommender import ContextualRecommender
from ..services.playlist_analyzer import PlaylistAnalyzer
from ..services.artist_comparator import ArtistComparator
from ..services.discovery_service import DiscoveryService
from ..services.cover_remix_analyzer import CoverRemixAnalyzer
from ..services.instrumentation_analyzer import InstrumentationAnalyzer
from ..services.trend_predictor import TrendPredictor

logger = logging.getLogger(__name__)


def register_all_services():
    """
    Registra todos los servicios en el contenedor de DI
    """
    container = get_container()
    
    services = [
        ("spotify_service", SpotifyService),
        ("music_analyzer", MusicAnalyzer),
        ("music_coach", MusicCoach),
        ("comparison_service", ComparisonService),
        ("export_service", ExportService),
        ("history_service", HistoryService),
        ("webhook_service", WebhookService),
        ("favorites_service", FavoritesService),
        ("tagging_service", TaggingService),
        ("auth_service", AuthService),
        ("playlist_service", PlaylistService),
        ("intelligent_recommender", IntelligentRecommender),
        ("dashboard_service", DashboardService),
        ("notification_service", NotificationService),
        ("trends_analyzer", TrendsAnalyzer),
        ("collaboration_analyzer", CollaborationAnalyzer),
        ("alert_service", AlertService),
        ("temporal_analyzer", TemporalAnalyzer),
        ("quality_analyzer", QualityAnalyzer),
        ("contextual_recommender", ContextualRecommender),
        ("playlist_analyzer", PlaylistAnalyzer),
        ("artist_comparator", ArtistComparator),
        ("discovery_service", DiscoveryService),
        ("cover_remix_analyzer", CoverRemixAnalyzer),
        ("instrumentation_analyzer", InstrumentationAnalyzer),
        ("trend_predictor", TrendPredictor),
    ]
    
    for service_name, service_class in services:
        try:
            register_service(service_name, service_class, singleton=True)
            logger.debug(f"Registered service: {service_name}")
        except Exception as e:
            logger.error(f"Failed to register service {service_name}: {e}")
    
    container.register_instance("analytics_service", analytics_service)
    
    try:
        from ..services.lyrics_analyzer import LyricsAnalyzer
        from ..services.melodic_pattern_analyzer import MelodicPatternAnalyzer
        from ..services.dynamics_analyzer import DynamicsAnalyzer
        from ..services.market_analyzer import MarketAnalyzer
        from ..services.audio_file_analyzer import AudioFileAnalyzer
        from ..services.benchmark_service import BenchmarkService
        from ..services.advanced_harmonic_analyzer import AdvancedHarmonicAnalyzer
        from ..services.performance_metrics import PerformanceMetrics
        from ..services.advanced_structure_analyzer import AdvancedStructureAnalyzer
        from ..services.enhanced_recommender import EnhancedRecommender
        from ..services.success_predictor import SuccessPredictor
        from ..services.advanced_collaboration_analyzer import AdvancedCollaborationAnalyzer
        from ..services.advanced_rhythmic_analyzer import AdvancedRhythmicAnalyzer
        from ..services.data_visualization import DataVisualization
        from ..services.advanced_report_generator import AdvancedReportGenerator
        from ..services.realtime_audio_analyzer import RealtimeAudioAnalyzer
        from ..services.deep_learning_service import DeepLearningService
        
        optional_services = [
            ("lyrics_analyzer", LyricsAnalyzer),
            ("melodic_pattern_analyzer", MelodicPatternAnalyzer),
            ("dynamics_analyzer", DynamicsAnalyzer),
            ("market_analyzer", MarketAnalyzer),
            ("audio_file_analyzer", AudioFileAnalyzer),
            ("benchmark_service", BenchmarkService),
            ("advanced_harmonic_analyzer", AdvancedHarmonicAnalyzer),
            ("performance_metrics", PerformanceMetrics),
            ("advanced_structure_analyzer", AdvancedStructureAnalyzer),
            ("enhanced_recommender", EnhancedRecommender),
            ("success_predictor", SuccessPredictor),
            ("advanced_collaboration_analyzer", AdvancedCollaborationAnalyzer),
            ("advanced_rhythmic_analyzer", AdvancedRhythmicAnalyzer),
            ("data_visualization", DataVisualization),
            ("advanced_report_generator", AdvancedReportGenerator),
            ("realtime_audio_analyzer", RealtimeAudioAnalyzer),
            ("deep_learning_service", DeepLearningService),
        ]
        
        for service_name, service_class in optional_services:
            try:
                register_service(service_name, service_class, singleton=True)
                logger.debug(f"Registered optional service: {service_name}")
            except Exception as e:
                logger.warning(f"Failed to register optional service {service_name}: {e}")
    except ImportError as e:
        logger.warning(f"Some optional services could not be imported: {e}")
    
    try:
        from ..services.emotion_analyzer import EmotionAnalyzer
        from ..services.genre_detector import GenreDetector
        
        register_service("emotion_analyzer", EmotionAnalyzer, singleton=True)
        register_service("genre_detector", GenreDetector, singleton=True)
        logger.debug("Registered ML services: emotion_analyzer, genre_detector")
    except ImportError as e:
        logger.warning(f"ML services could not be imported: {e}")
    
    logger.info("All services registered successfully")


def get_service(service_name: str) -> Any:
    """
    Helper function para obtener un servicio del contenedor
    """
    from ..core.dependency_injection import get_service as _get_service
    return _get_service(service_name)







