"""
API endpoints para análisis de piel y recomendaciones de skincare

⚠️ DEPRECATED: This file is deprecated and kept only for backward compatibility.
Please use the modular router from `dermatology_api_modular.py` instead.

This monolithic file contains 5939+ lines and should not be used for new development.
All new endpoints should be added to the appropriate router in the `routers/` directory.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import numpy as np
from PIL import Image
import io
import time
import json

from ..core.skin_analyzer import SkinAnalyzer
from ..services.image_processor import ImageProcessor
from ..services.video_processor import VideoProcessor
from ..services.skincare_recommender import SkincareRecommender
from ..services.history_tracker import HistoryTracker
from ..services.report_generator import ReportGenerator
from ..services.visualization import VisualizationGenerator
from ..services.database import DatabaseManager
from ..services.analytics import AnalyticsEngine
from ..services.alert_system import AlertSystem
from ..services.product_database import ProductDatabase, ProductCategory
from ..services.body_area_analyzer import BodyAreaAnalyzer, BodyArea
from ..services.export_manager import ExportManager
from ..services.webhook_manager import WebhookManager, WebhookEvent
from ..services.auth_manager import AuthManager
from ..services.backup_manager import BackupManager
from ..services.notification_service import NotificationService, NotificationType, NotificationPriority
from ..services.metrics_dashboard import MetricsDashboard
from ..services.report_templates import ReportTemplateEngine, ReportTemplate
from ..services.async_queue import AsyncQueue, TaskPriority
from ..services.integration_service import IntegrationService, IntegrationType
from ..services.event_system import EventSystem, EventType
from ..services.advanced_search import AdvancedSearchEngine, SearchOperator
from ..services.enhanced_export import EnhancedExportManager
from ..services.collaboration_service import CollaborationService, SharePermission
from ..services.tagging_system import TaggingSystem
from ..services.model_versioning import ModelVersioning, ModelStatus
from ..services.realtime_metrics import RealtimeMetrics
from ..services.health_monitor import HealthMonitor, HealthStatus
from ..services.business_metrics import BusinessMetrics
from ..services.batch_processor import BatchProcessor, BatchStatus
from ..services.api_documentation import APIDocumentation
from ..services.security_enhancer import SecurityEnhancer
from ..services.intelligent_recommender import IntelligentRecommender
from ..services.feedback_system import FeedbackSystem, FeedbackType
from ..services.ab_testing import ABTestingSystem, Variant
from ..services.personalization_engine import PersonalizationEngine
from ..services.gamification import GamificationSystem
from ..services.challenge_system import ChallengeSystem
from ..services.social_features import SocialFeatures
from ..services.trend_predictor import TrendPredictor
from ..services.advanced_comparison import AdvancedComparison
from ..services.enhanced_ml import EnhancedMLSystem, MLModelType
from ..services.iot_integration import IoTIntegration, DeviceType
from ..services.push_notifications import PushNotificationService, PushPlatform
from ..services.advanced_reporting import AdvancedReporting, ReportConfig, ReportFormat
from ..services.image_analysis_advanced import AdvancedImageAnalysis
from ..services.ml_recommender import MLRecommender
from ..services.advanced_monitoring import AdvancedMonitoring
from ..services.condition_predictor import ConditionPredictor
from ..services.video_analysis_advanced import AdvancedVideoAnalysis
from ..services.learning_system import LearningSystem
from ..services.progress_analyzer import ProgressAnalyzer
from ..services.smart_recommender import SmartRecommender
from ..services.intelligent_alerts import IntelligentAlertSystem, AlertSeverity
from ..services.predictive_analytics import PredictiveAnalytics
from ..services.routine_comparator import RoutineComparator
from ..services.product_tracker import ProductTracker
from ..services.smart_reminders import SmartReminderSystem, ReminderType
from ..services.market_trends import MarketTrendsAnalyzer
from ..services.skin_goals import SkinGoalsManager, GoalStatus
from ..services.skin_journal import SkinJournal
from ..services.expert_consultation import ExpertConsultationSystem, ConsultationStatus
from ..services.ingredient_analyzer import IngredientAnalyzer
from ..services.custom_recipes import CustomRecipesManager
from ..services.product_comparison import ProductComparisonSystem
from ..services.reviews_ratings import ReviewsRatingsSystem
from ..services.before_after_analysis import BeforeAfterAnalysis
from ..services.budget_tracker import BudgetTracker
from ..services.community_features import CommunityFeatures
from ..services.wearable_integration import WearableIntegration
from ..services.weather_climate_analysis import WeatherClimateAnalysis
from ..services.enhanced_notifications import EnhancedNotificationSystem, NotificationType
from ..services.ai_photo_analysis import AIPhotoAnalysisSystem
from ..services.seasonal_recommendations import SeasonalRecommendationsSystem, Season
from ..services.allergy_tracker import AllergyTracker
from ..services.advanced_texture_analysis import AdvancedTextureAnalysis
from ..services.product_needs_predictor import ProductNeedsPredictor
from ..services.habit_analyzer import HabitAnalyzer
from ..services.personalized_coaching import PersonalizedCoaching
from ..services.medical_treatment_tracker import MedicalTreatmentTracker
from ..services.visual_progress_tracker import VisualProgressTracker
from ..services.pharmacy_integration import PharmacyIntegration
from ..services.product_reminder_system import ProductReminderSystem
from ..services.ingredient_conflict_checker import IngredientConflictChecker
from ..services.comparative_analysis import ComparativeAnalysisSystem
from ..services.budget_based_recommendations import BudgetBasedRecommendations
from ..services.historical_photo_analysis import HistoricalPhotoAnalysis
from ..services.product_trend_analyzer import ProductTrendAnalyzer
from ..services.age_analysis import AgeAnalysisSystem
from ..services.age_based_recommendations import AgeBasedRecommendations
from ..services.multi_condition_analyzer import MultiConditionAnalyzer
from ..services.successful_routines import SuccessfulRoutinesSystem
from ..services.multi_angle_analysis import MultiAngleAnalysis
from ..services.lifestyle_recommendations import LifestyleRecommendations
from ..services.medical_device_integration import MedicalDeviceIntegration
from ..services.benchmark_analysis import BenchmarkAnalysis
from ..services.lighting_analysis import LightingAnalysisSystem
from ..services.genetic_recommendations import GeneticRecommendations
from ..services.professional_treatment_tracker import ProfessionalTreatmentTracker
from ..services.ai_progress_analysis import AIProgressAnalysis
from ..services.climate_condition_analysis import ClimateAnalysisSystem
from ..services.time_based_recommendations import TimeBasedRecommendations
from ..services.side_effect_tracker import SideEffectTracker
from ..services.advanced_texture_ml import AdvancedTextureML
from ..services.skin_state_analysis import SkinStateAnalysisSystem
from ..services.fitness_based_recommendations import FitnessBasedRecommendations
from ..services.supplement_tracker import SupplementTracker
from ..services.temporal_comparison import TemporalComparisonSystem
from ..services.resolution_analysis import ResolutionAnalysisSystem
from ..services.monthly_budget_recommendations import MonthlyBudgetRecommendations
from ..services.sleep_habit_tracker import SleepHabitTracker
from ..services.future_prediction import FuturePredictionSystem
from ..services.format_analysis import FormatAnalysisSystem
from ..services.water_type_recommendations import WaterTypeRecommendations
from ..services.stress_tracker import StressTracker
from ..services.intelligent_alerts import IntelligentAlerts
from ..services.device_analysis import DeviceAnalysisSystem
from ..services.medication_recommendations import MedicationRecommendations
from ..services.hormonal_tracker import HormonalTracker
from ..services.advanced_ml_analysis import AdvancedMLAnalysis
from ..services.environmental_tracker import EnvironmentalTracker
from ..services.routine_optimizer import RoutineOptimizer
from ..services.skin_concern_tracker import SkinConcernTracker
from ..services.product_effectiveness_tracker import ProductEffectivenessTracker
from ..services.natural_lighting_analysis import NaturalLightingAnalysis
from ..services.ethnic_skin_recommendations import EthnicSkinRecommendations
from ..services.seasonal_changes_tracker import SeasonalChangesTracker
from ..services.anonymous_comparison import AnonymousComparison
from ..services.distance_analysis import DistanceAnalysisSystem
from ..services.occupation_recommendations import OccupationRecommendations
from ..services.diet_tracker import DietTracker
from ..services.plateau_detection import PlateauDetection as PlateauDetectionSystem
from ..services.trend_prediction import TrendPredictionSystem
from ..services.local_weather_recommendations import LocalWeatherRecommendations
from ..services.custom_routine_tracker import CustomRoutineTracker
from ..services.product_compatibility import ProductCompatibility
from ..services.progress_visualization import ProgressVisualizationSystem
from ..services.budget_recommendations import BudgetRecommendations
from ..services.side_effect_tracker import SideEffectTracker
from ..services.advanced_texture_ml import AdvancedTextureML
from ..utils.logger import logger
from ..utils.rate_limiter import RateLimiter
from ..utils.endpoint_rate_limiter import EndpointRateLimiter
from ..utils.intelligent_cache import IntelligentCache
from ..utils.advanced_validator import AdvancedImageValidator
import cv2
from ..utils.exceptions import (
    ImageProcessingError, VideoProcessingError, 
    AnalysisError, ValidationError
)
import hashlib
from fastapi import Header, Depends

router = APIRouter(prefix="/dermatology", tags=["dermatology"])

# Inicializar componentes
skin_analyzer = SkinAnalyzer(use_advanced=True, use_cache=True)
image_processor = ImageProcessor()
video_processor = VideoProcessor()
skincare_recommender = SkincareRecommender()
history_tracker = HistoryTracker()
report_generator = ReportGenerator()
visualization_generator = VisualizationGenerator()
db_manager = DatabaseManager()
analytics_engine = AnalyticsEngine(db_manager)
alert_system = AlertSystem()
product_database = ProductDatabase()
body_area_analyzer = BodyAreaAnalyzer()
export_manager = ExportManager()
webhook_manager = WebhookManager()
auth_manager = AuthManager()
backup_manager = BackupManager()
notification_service = NotificationService()
advanced_validator = AdvancedImageValidator()
metrics_dashboard = MetricsDashboard(db_manager, analytics_engine)
template_engine = ReportTemplateEngine()
async_queue = AsyncQueue(max_workers=5)
integration_service = IntegrationService()
event_system = EventSystem()
search_engine = AdvancedSearchEngine()
enhanced_export = EnhancedExportManager()
collaboration_service = CollaborationService()
tagging_system = TaggingSystem()
model_versioning = ModelVersioning()
realtime_metrics = RealtimeMetrics()
health_monitor = HealthMonitor()
business_metrics = BusinessMetrics()
batch_processor = BatchProcessor(max_workers=5)
api_documentation = APIDocumentation()
security_enhancer = SecurityEnhancer()
intelligent_recommender = IntelligentRecommender()
feedback_system = FeedbackSystem()
ab_testing = ABTestingSystem()
personalization_engine = PersonalizationEngine()
gamification = GamificationSystem()
challenge_system = ChallengeSystem()
social_features = SocialFeatures()
trend_predictor = TrendPredictor()
advanced_comparison = AdvancedComparison()
enhanced_ml = EnhancedMLSystem()
iot_integration = IoTIntegration()
push_notifications = PushNotificationService()
advanced_reporting = AdvancedReporting()
image_analysis_advanced = AdvancedImageAnalysis()
ml_recommender = MLRecommender()
advanced_monitoring = AdvancedMonitoring()
condition_predictor = ConditionPredictor()
video_analysis_advanced = AdvancedVideoAnalysis()
learning_system = LearningSystem()
progress_analyzer = ProgressAnalyzer()
smart_recommender = SmartRecommender()
intelligent_alerts = IntelligentAlertSystem()
predictive_analytics = PredictiveAnalytics()
routine_comparator = RoutineComparator()
product_tracker = ProductTracker()
smart_reminders = SmartReminderSystem()
market_trends = MarketTrendsAnalyzer()
skin_goals = SkinGoalsManager()
skin_journal = SkinJournal()
expert_consultation = ExpertConsultationSystem()
ingredient_analyzer = IngredientAnalyzer()
custom_recipes = CustomRecipesManager()
product_comparison = ProductComparisonSystem()
reviews_ratings = ReviewsRatingsSystem()
before_after_analysis = BeforeAfterAnalysis()
budget_tracker = BudgetTracker()
community_features = CommunityFeatures()
wearable_integration = WearableIntegration()
weather_climate = WeatherClimateAnalysis()
enhanced_notifications = EnhancedNotificationSystem()
ai_photo_analysis = AIPhotoAnalysisSystem()
seasonal_recommendations = SeasonalRecommendationsSystem()
allergy_tracker = AllergyTracker()
advanced_texture_analysis = AdvancedTextureAnalysis()
product_needs_predictor = ProductNeedsPredictor()
habit_analyzer = HabitAnalyzer()
personalized_coaching = PersonalizedCoaching()
medical_treatment_tracker = MedicalTreatmentTracker()
visual_progress_tracker = VisualProgressTracker()
pharmacy_integration = PharmacyIntegration()
product_reminder_system = ProductReminderSystem()
ingredient_conflict_checker = IngredientConflictChecker()
comparative_analysis = ComparativeAnalysisSystem()
budget_recommendations = BudgetBasedRecommendations()
historical_photo_analysis = HistoricalPhotoAnalysis()
product_trend_analyzer = ProductTrendAnalyzer()
age_analysis = AgeAnalysisSystem()
age_recommendations = AgeBasedRecommendations()
multi_condition_analyzer = MultiConditionAnalyzer()
successful_routines = SuccessfulRoutinesSystem()
multi_angle_analysis = MultiAngleAnalysis()
lifestyle_recommendations = LifestyleRecommendations()
medical_device_integration = MedicalDeviceIntegration()
benchmark_analysis = BenchmarkAnalysis()
lighting_analysis = LightingAnalysisSystem()
genetic_recommendations = GeneticRecommendations()
professional_treatment_tracker = ProfessionalTreatmentTracker()
ai_progress_analysis = AIProgressAnalysis()
climate_analysis = ClimateAnalysisSystem()
time_based_recommendations = TimeBasedRecommendations()
side_effect_tracker = SideEffectTracker()
advanced_texture_ml = AdvancedTextureML()
skin_state_analysis = SkinStateAnalysisSystem()
fitness_recommendations = FitnessBasedRecommendations()
supplement_tracker = SupplementTracker()
temporal_comparison = TemporalComparisonSystem()
resolution_analysis = ResolutionAnalysisSystem()
monthly_budget_recommendations = MonthlyBudgetRecommendations()
sleep_habit_tracker = SleepHabitTracker()
future_prediction = FuturePredictionSystem()
format_analysis = FormatAnalysisSystem()
water_recommendations = WaterTypeRecommendations()
stress_tracker = StressTracker()
intelligent_alerts = IntelligentAlerts()
device_analysis = DeviceAnalysisSystem()
medication_recommendations = MedicationRecommendations()
hormonal_tracker = HormonalTracker()
advanced_ml_analysis = AdvancedMLAnalysis()
environmental_tracker = EnvironmentalTracker()
routine_optimizer = RoutineOptimizer()
skin_concern_tracker = SkinConcernTracker()
product_effectiveness_tracker = ProductEffectivenessTracker()
natural_lighting_analysis = NaturalLightingAnalysis()
ethnic_skin_recommendations = EthnicSkinRecommendations()
seasonal_changes_tracker = SeasonalChangesTracker()
anonymous_comparison = AnonymousComparison()
distance_analysis = DistanceAnalysisSystem()
occupation_recommendations = OccupationRecommendations()
diet_tracker = DietTracker()
plateau_detection = PlateauDetectionSystem()
trend_prediction = TrendPredictionSystem()
local_weather_recommendations = LocalWeatherRecommendations()
custom_routine_tracker = CustomRoutineTracker()
product_compatibility = ProductCompatibility()
progress_visualization = ProgressVisualizationSystem()
budget_recommendations = BudgetRecommendations()
side_effect_tracker = SideEffectTracker()
advanced_texture_ml = AdvancedTextureML()
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
endpoint_rate_limiter = EndpointRateLimiter()
intelligent_cache = IntelligentCache(default_ttl=3600, max_size=1000)

# Configurar límites por endpoint
endpoint_rate_limiter.set_limit("/analyze-image", max_requests=50, window_seconds=60)
endpoint_rate_limiter.set_limit("/analyze-video", max_requests=20, window_seconds=60)
endpoint_rate_limiter.set_limit("/get-recommendations", max_requests=100, window_seconds=60)


@router.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(..., description="Imagen de piel para analizar"),
    enhance: bool = Form(True, description="Mejorar imagen antes de análisis"),
    use_advanced: bool = Form(True, description="Usar análisis avanzado"),
    use_cache: bool = Form(True, description="Usar cache para mejorar rendimiento")
):
    """
    Analiza una imagen de piel y proporciona métricas de calidad
    
    - **file**: Imagen (JPG, PNG) de la piel
    - **enhance**: Si se debe mejorar la imagen antes del análisis
    - **use_advanced**: Usar análisis avanzado (mejor precisión)
    - **use_cache**: Usar cache para mejorar rendimiento
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recibida solicitud de análisis de imagen: {file.filename}")
        
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise ValidationError("El archivo debe ser una imagen (JPG, PNG, etc.)")
        
        # Leer imagen
        image_bytes = await file.read()
        logger.debug(f"Imagen leída: {len(image_bytes)} bytes")
        
        # Validación avanzada
        is_valid_advanced, validation_info = advanced_validator.validate_image_comprehensive(
            image_bytes, filename=file.filename
        )
        
        if not is_valid_advanced:
            errors = validation_info.get("errors", [])
            raise ValidationError(f"Validación fallida: {'; '.join(errors)}")
        
        # Procesar imagen
        processed_image, is_valid, message = image_processor.process_for_analysis(image_bytes)
        
        if not is_valid:
            raise ImageProcessingError(message)
        
        # Mejorar si se solicita
        if enhance:
            logger.debug("Mejorando imagen")
            processed_image = image_processor.enhance_for_analysis(processed_image)
        
        # Configurar analizador
        if use_advanced != skin_analyzer.use_advanced:
            skin_analyzer.use_advanced = use_advanced
        
        # Analizar
        logger.debug("Iniciando análisis de piel")
        analysis_result = skin_analyzer.analyze_image(processed_image, use_cache=use_cache)
        analysis_result["analysis_type"] = "image"
        
        # Generar hash de imagen para tracking
        image_hash = hashlib.md5(image_bytes).hexdigest()
        
        # Guardar en historial (archivo y base de datos)
        try:
            record_id = history_tracker.save_analysis(
                analysis_result,
                image_hash=image_hash,
                metadata={"enhanced": enhance, "advanced": use_advanced}
            )
            analysis_result["record_id"] = record_id
            
            # Guardar también en base de datos
            db_manager.save_analysis(
                record_id,
                user_id=None,  # Se puede pasar user_id si está disponible
                analysis_result=analysis_result,
                image_hash=image_hash,
                metadata={"enhanced": enhance, "advanced": use_advanced}
            )
            
            # Verificar alertas
            alerts = alert_system.check_analysis_alerts(analysis_result)
            if alerts:
                analysis_result["alerts"] = [
                    {
                        "level": alert.level.value,
                        "title": alert.title,
                        "message": alert.message
                    }
                    for alert in alerts
                ]
            
            # Disparar webhook si hay análisis completado
            try:
                await webhook_manager.trigger_webhook(
                    WebhookEvent.ANALYSIS_COMPLETED,
                    {"analysis_id": record_id, "result": analysis_result}
                )
            except Exception as e:
                logger.warning(f"Error disparando webhook: {str(e)}")
            
            # Enviar notificación (si hay user_id)
            try:
                user_id = None  # Se puede obtener de token o parámetro
                if user_id:
                    notification_service.send_analysis_complete_notification(
                        user_id, analysis_result
                    )
            except Exception as e:
                logger.warning(f"Error enviando notificación: {str(e)}")
        except Exception as e:
            logger.warning(f"Error guardando en historial: {str(e)}")
        
        duration = time.time() - start_time
        logger.log_api_request("/analyze-image", "POST", 200, duration)
        
        return JSONResponse(content={
            "success": True,
            "analysis": analysis_result,
            "message": message,
            "processing_time": round(duration, 2),
            "settings": {
                "enhanced": enhance,
                "advanced_analysis": use_advanced,
                "cached": use_cache
            }
        })
    
    except ValidationError as e:
        duration = time.time() - start_time
        logger.log_api_request("/analyze-image", "POST", 400, duration)
        logger.warning(f"Error de validación: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except (ImageProcessingError, AnalysisError) as e:
        duration = time.time() - start_time
        logger.log_api_request("/analyze-image", "POST", 400, duration)
        logger.error(f"Error de procesamiento: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except HTTPException:
        raise
    
    except Exception as e:
        duration = time.time() - start_time
        logger.log_api_request("/analyze-image", "POST", 500, duration)
        logger.error(f"Error inesperado analizando imagen: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analizando imagen: {str(e)}"
        )


@router.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(..., description="Video de piel para analizar"),
    max_frames: int = Form(30, description="Máximo número de frames a procesar")
):
    """
    Analiza un video de piel y proporciona análisis agregado
    
    - **file**: Video (MP4, AVI) de la piel
    - **max_frames**: Máximo número de frames a procesar
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recibida solicitud de análisis de video: {file.filename}")
        
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith('video/'):
            raise ValidationError("El archivo debe ser un video (MP4, AVI, etc.)")
        
        # Leer video
        video_bytes = await file.read()
        logger.debug(f"Video leído: {len(video_bytes)} bytes")
        
        # Validar video
        is_valid, message = video_processor.validate_video(video_bytes)
        if not is_valid:
            raise VideoProcessingError(message)
        
        # Extraer frames
        video_processor.max_frames = max_frames
        logger.debug(f"Extrayendo hasta {max_frames} frames")
        frames = video_processor.extract_frames(video_bytes)
        
        if not frames:
            raise VideoProcessingError("No se pudieron extraer frames del video")
        
        logger.debug(f"Frames extraídos: {len(frames)}")
        
        # Analizar frames
        analysis_result = skin_analyzer.analyze_video(frames)
        
        duration = time.time() - start_time
        logger.log_api_request("/analyze-video", "POST", 200, duration)
        
        return JSONResponse(content={
            "success": True,
            "analysis": analysis_result,
            "frames_analyzed": len(frames),
            "processing_time": round(duration, 2),
            "message": f"Análisis completado con {len(frames)} frames"
        })
    
    except (ValidationError, VideoProcessingError) as e:
        duration = time.time() - start_time
        logger.log_api_request("/analyze-video", "POST", 400, duration)
        logger.warning(f"Error de validación/procesamiento: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except HTTPException:
        raise
    
    except Exception as e:
        duration = time.time() - start_time
        logger.log_api_request("/analyze-video", "POST", 500, duration)
        logger.error(f"Error inesperado analizando video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analizando video: {str(e)}"
        )


@router.post("/get-recommendations")
async def get_recommendations(
    file: UploadFile = File(..., description="Imagen de piel para análisis"),
    include_routine: bool = Form(True, description="Incluir rutina completa")
):
    """
    Obtiene recomendaciones de skincare basadas en análisis de piel
    
    - **file**: Imagen de piel
    - **include_routine**: Si incluir rutina completa de skincare
    """
    try:
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser una imagen"
            )
        
        # Leer y procesar imagen
        image_bytes = await file.read()
        processed_image, is_valid, message = image_processor.process_for_analysis(image_bytes)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Analizar
        analysis_result = skin_analyzer.analyze_image(processed_image)
        
        # Generar recomendaciones
        recommendations = skincare_recommender.generate_recommendations(analysis_result)
        
        response = {
            "success": True,
            "analysis": analysis_result,
            "recommendations": recommendations if include_routine else {
                "specific_recommendations": recommendations["specific_recommendations"],
                "tips": recommendations["tips"]
            }
        }
        
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando recomendaciones: {str(e)}"
        )


@router.post("/analyze-from-analysis")
async def get_recommendations_from_analysis(
    analysis_data: dict
):
    """
    Obtiene recomendaciones basadas en un análisis previo
    
    - **analysis_data**: Datos de análisis previo (quality_scores, conditions, etc.)
    """
    try:
        # Validar estructura de datos
        required_fields = ["quality_scores", "conditions", "skin_type"]
        for field in required_fields:
            if field not in analysis_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Campo requerido faltante: {field}"
                )
        
        # Generar recomendaciones
        recommendations = skincare_recommender.generate_recommendations(analysis_data)
        
        return JSONResponse(content={
            "success": True,
            "recommendations": recommendations
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando recomendaciones: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Verifica el estado del servicio"""
    return JSONResponse(content={
        "status": "healthy",
        "service": "dermatology_ai",
        "version": "5.2.0"
    })


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = Query(50, description="Límite de registros")):
    """
    Obtiene historial de análisis de un usuario
    
    - **user_id**: ID del usuario
    - **limit**: Límite de registros a retornar
    """
    try:
        history = history_tracker.get_user_history(user_id, limit=limit)
        
        return JSONResponse(content={
            "success": True,
            "user_id": user_id,
            "count": len(history),
            "history": [record.to_dict() for record in history]
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo historial: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo historial: {str(e)}"
        )


@router.get("/history/compare/{record_id1}/{record_id2}")
async def compare_analyses(record_id1: str, record_id2: str):
    """
    Compara dos análisis
    
    - **record_id1**: ID del primer análisis
    - **record_id2**: ID del segundo análisis
    """
    try:
        comparison = history_tracker.compare_analyses(record_id1, record_id2)
        
        return JSONResponse(content={
            "success": True,
            "comparison": comparison
        })
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error comparando análisis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error comparando análisis: {str(e)}"
        )


@router.get("/history/timeline/{user_id}")
async def get_timeline(user_id: str, metric: str = Query("overall_score", description="Métrica a trackear")):
    """
    Obtiene línea de tiempo de progreso
    
    - **user_id**: ID del usuario
    - **metric**: Métrica a visualizar
    """
    try:
        timeline = history_tracker.get_progress_timeline(user_id, metric=metric)
        
        return JSONResponse(content={
            "success": True,
            "user_id": user_id,
            "metric": metric,
            "timeline": timeline
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo timeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo timeline: {str(e)}"
        )


@router.post("/report/json")
async def generate_json_report(analysis_result: dict):
    """
    Genera reporte en formato JSON
    
    - **analysis_result**: Resultado del análisis
    """
    try:
        recommendations = skincare_recommender.generate_recommendations(analysis_result)
        report = report_generator.generate_json_report(
            analysis_result,
            recommendations=recommendations
        )
        
        return JSONResponse(
            content=json.loads(report),
            media_type="application/json"
        )
    
    except Exception as e:
        logger.error(f"Error generando reporte JSON: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generando reporte: {str(e)}"
        )


@router.post("/report/pdf")
async def generate_pdf_report(analysis_result: dict):
    """
    Genera reporte en formato PDF
    
    - **analysis_result**: Resultado del análisis
    """
    try:
        recommendations = skincare_recommender.generate_recommendations(analysis_result)
        pdf_bytes = report_generator.generate_pdf_report(
            analysis_result,
            recommendations=recommendations
        )
        
        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=skin_analysis_report.pdf"}
        )
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="reportlab no está instalado. Instale con: pip install reportlab"
        )
    except Exception as e:
        logger.error(f"Error generando reporte PDF: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generando reporte PDF: {str(e)}"
        )


@router.post("/report/html")
async def generate_html_report(analysis_result: dict):
    """
    Genera reporte en formato HTML
    
    - **analysis_result**: Resultado del análisis
    """
    try:
        recommendations = skincare_recommender.generate_recommendations(analysis_result)
        html = report_generator.generate_html_report(
            analysis_result,
            recommendations=recommendations
        )
        
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html)
    
    except Exception as e:
        logger.error(f"Error generando reporte HTML: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generando reporte HTML: {str(e)}"
        )


@router.post("/visualization/radar")
async def generate_radar_chart(quality_scores: dict):
    """
    Genera gráfico radar de métricas
    
    - **quality_scores**: Diccionario con scores de calidad
    """
    try:
        img_base64 = visualization_generator.generate_radar_chart(quality_scores)
        
        return JSONResponse(content={
            "success": True,
            "image_base64": img_base64,
            "format": "png"
        })
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="matplotlib no está instalado. Instale con: pip install matplotlib"
        )
    except Exception as e:
        logger.error(f"Error generando gráfico: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generando gráfico: {str(e)}"
        )


@router.post("/visualization/timeline")
async def generate_timeline_chart(timeline_data: list):
    """
    Genera gráfico de línea de tiempo
    
    - **timeline_data**: Lista de puntos en el tiempo
    """
    try:
        img_base64 = visualization_generator.generate_timeline_chart(timeline_data)
        
        return JSONResponse(content={
            "success": True,
            "image_base64": img_base64,
            "format": "png"
        })
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="matplotlib no está instalado"
        )
    except Exception as e:
        logger.error(f"Error generando timeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generando timeline: {str(e)}"
        )


@router.post("/visualization/comparison")
async def generate_comparison_chart(comparison_data: dict):
    """
    Genera gráfico de comparación antes/después
    
    - **comparison_data**: Diccionario con scores_before y scores_after
    """
    try:
        scores_before = comparison_data.get("scores_before", {})
        scores_after = comparison_data.get("scores_after", {})
        
        img_base64 = visualization_generator.generate_comparison_chart(
            scores_before, scores_after
        )
        
        return JSONResponse(content={
            "success": True,
            "image_base64": img_base64,
            "format": "png"
        })
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="matplotlib no está instalado"
        )
    except Exception as e:
        logger.error(f"Error generando comparación: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generando comparación: {str(e)}"
        )


@router.get("/analytics/user/{user_id}")
async def get_user_analytics(
    user_id: str,
    days: int = Query(30, description="Días de historial a analizar")
):
    """
    Obtiene analytics e insights para un usuario
    
    - **user_id**: ID del usuario
    - **days**: Días de historial a analizar
    """
    try:
        insights = analytics_engine.get_user_insights(user_id, days=days)
        
        return JSONResponse(content={
            "success": True,
            "insights": insights
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo analytics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo analytics: {str(e)}"
        )


@router.get("/analytics/progress/{user_id}")
async def get_progress_report(
    user_id: str,
    start_date: Optional[str] = Query(None, description="Fecha de inicio (ISO format)"),
    end_date: Optional[str] = Query(None, description="Fecha de fin (ISO format)")
):
    """
    Obtiene reporte de progreso para un usuario
    
    - **user_id**: ID del usuario
    - **start_date**: Fecha de inicio (opcional)
    - **end_date**: Fecha de fin (opcional)
    """
    try:
        progress = analytics_engine.get_progress_report(
            user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return JSONResponse(content={
            "success": True,
            "progress": progress
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo reporte de progreso: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo reporte de progreso: {str(e)}"
        )


@router.get("/analytics/system")
async def get_system_analytics(
    days: int = Query(30, description="Días de historial a analizar")
):
    """
    Obtiene analytics del sistema completo
    
    - **days**: Días de historial a analizar
    """
    try:
        analytics = analytics_engine.get_system_analytics(days=days)
        
        return JSONResponse(content={
            "success": True,
            "analytics": analytics
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo analytics del sistema: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo analytics: {str(e)}"
        )


@router.get("/alerts/{user_id}")
async def get_user_alerts(
    user_id: str,
    unread_only: bool = Query(False, description="Solo alertas no leídas")
):
    """
    Obtiene alertas de un usuario
    
    - **user_id**: ID del usuario
    - **unread_only**: Solo alertas no leídas
    """
    try:
        alerts = alert_system.get_user_alerts(user_id, unread_only=unread_only)
        
        return JSONResponse(content={
            "success": True,
            "alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "acknowledged": alert.acknowledged,
                    "metadata": alert.metadata
                }
                for alert in alerts
            ]
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo alertas: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo alertas: {str(e)}"
        )


@router.get("/alerts/{user_id}/summary")
async def get_alert_summary(user_id: str):
    """
    Obtiene resumen de alertas de un usuario
    
    - **user_id**: ID del usuario
    """
    try:
        summary = alert_system.get_alert_summary(user_id)
        
        return JSONResponse(content={
            "success": True,
            "summary": summary
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo resumen de alertas: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo resumen: {str(e)}"
        )


@router.post("/alerts/{user_id}/acknowledge/{alert_id}")
async def acknowledge_alert(user_id: str, alert_id: str):
    """
    Marca una alerta como leída
    
    - **user_id**: ID del usuario
    - **alert_id**: ID de la alerta
    """
    try:
        alert_system.acknowledge_alert(user_id, alert_id)
        
        return JSONResponse(content={
            "success": True,
            "message": "Alerta marcada como leída"
        })
    
    except Exception as e:
        logger.error(f"Error marcando alerta: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error marcando alerta: {str(e)}"
        )


@router.get("/statistics/{user_id}")
async def get_user_statistics(user_id: str):
    """
    Obtiene estadísticas de un usuario
    
    - **user_id**: ID del usuario
    """
    try:
        stats = db_manager.get_statistics(user_id=user_id)
        
        return JSONResponse(content={
            "success": True,
            "statistics": stats
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estadísticas: {str(e)}"
        )


@router.get("/products/search")
async def search_products(
    category: Optional[str] = Query(None, description="Categoría del producto"),
    skin_type: Optional[str] = Query(None, description="Tipo de piel"),
    concern: Optional[str] = Query(None, description="Preocupación a tratar"),
    price_range: Optional[str] = Query(None, description="Rango de precio"),
    min_rating: float = Query(0.0, description="Rating mínimo"),
    limit: int = Query(10, description="Límite de resultados")
):
    """
    Busca productos de skincare
    
    - **category**: Categoría del producto
    - **skin_type**: Tipo de piel
    - **concern**: Preocupación a tratar
    - **price_range**: Rango de precio (budget, mid-range, premium)
    - **min_rating**: Rating mínimo (0-5)
    - **limit**: Límite de resultados
    """
    try:
        products = product_database.search_products(
            category=ProductCategory(category) if category else None,
            skin_type=skin_type if skin_type else None,
            concern=concern,
            price_range=price_range,
            min_rating=min_rating,
            limit=limit
        )
        
        return JSONResponse(content={
            "success": True,
            "count": len(products),
            "products": [product.to_dict() for product in products]
        })
    
    except Exception as e:
        logger.error(f"Error buscando productos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error buscando productos: {str(e)}"
        )


@router.get("/products/{product_id}")
async def get_product(product_id: str):
    """
    Obtiene un producto por ID
    
    - **product_id**: ID del producto
    """
    try:
        product = product_database.get_product(product_id)
        
        if not product:
            raise HTTPException(status_code=404, detail="Producto no encontrado")
        
        return JSONResponse(content={
            "success": True,
            "product": product.to_dict()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo producto: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo producto: {str(e)}"
        )


@router.post("/products/recommend")
async def recommend_products(analysis_result: dict):
    """
    Obtiene productos recomendados basados en análisis
    
    - **analysis_result**: Resultado del análisis
    """
    try:
        skin_type = analysis_result.get("skin_type", "normal")
        conditions = analysis_result.get("conditions", [])
        priorities = analysis_result.get("recommendations_priority", [])
        
        concerns = [c.get("name") for c in conditions]
        
        products = product_database.get_recommended_products(
            skin_type=skin_type,
            concerns=concerns,
            priorities=priorities,
            limit=5
        )
        
        return JSONResponse(content={
            "success": True,
            "products": [product.to_dict() for product in products]
        })
    
    except Exception as e:
        logger.error(f"Error recomendando productos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error recomendando productos: {str(e)}"
        )


@router.post("/products/compare")
async def compare_products(product_ids: List[str]):
    """
    Compara múltiples productos
    
    - **product_ids**: Lista de IDs de productos a comparar
    """
    try:
        comparison = product_database.compare_products(product_ids)
        
        return JSONResponse(content={
            "success": True,
            "comparison": comparison
        })
    
    except Exception as e:
        logger.error(f"Error comparando productos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error comparando productos: {str(e)}"
        )


@router.post("/analyze-body-area")
async def analyze_body_area(
    file: UploadFile = File(..., description="Imagen del área del cuerpo"),
    area: str = Form(..., description="Área del cuerpo (face, forehead, cheeks, etc.)")
):
    """
    Analiza un área específica del cuerpo
    
    - **file**: Imagen del área
    - **area**: Área del cuerpo a analizar
    """
    try:
        # Validar área
        try:
            body_area = BodyArea(area.lower())
        except ValueError:
            raise ValidationError(f"Área inválida: {area}")
        
        # Leer y procesar imagen
        image_bytes = await file.read()
        processed_image, is_valid, message = image_processor.process_for_analysis(image_bytes)
        
        if not is_valid:
            raise ImageProcessingError(message)
        
        # Analizar
        analysis_result = skin_analyzer.analyze_image(processed_image)
        
        # Análisis específico del área
        area_result = body_area_analyzer.analyze_area(
            processed_image,
            body_area,
            analysis_result
        )
        
        return JSONResponse(content={
            "success": True,
            "area_analysis": area_result
        })
    
    except (ValidationError, ImageProcessingError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analizando área: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analizando área: {str(e)}"
        )


@router.post("/export/history/csv")
async def export_history_csv(user_id: str):
    """
    Exporta historial a CSV
    
    - **user_id**: ID del usuario
    """
    try:
        history = history_tracker.get_user_history(user_id, limit=1000)
        
        if not history:
            raise HTTPException(status_code=404, detail="No hay historial para exportar")
        
        # Convertir a lista de diccionarios
        history_data = [record.to_dict() for record in history]
        
        csv_bytes = export_manager.export_to_csv(history_data)
        
        from fastapi.responses import Response
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=history_{user_id}.csv"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exportando historial: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error exportando historial: {str(e)}"
        )


@router.post("/export/history/excel")
async def export_history_excel(user_id: str):
    """
    Exporta historial a Excel
    
    - **user_id**: ID del usuario
    """
    try:
        history = history_tracker.get_user_history(user_id, limit=1000)
        
        if not history:
            raise HTTPException(status_code=404, detail="No hay historial para exportar")
        
        # Convertir a lista de diccionarios
        history_data = [record.to_dict() for record in history]
        
        excel_bytes = export_manager.export_history_to_excel(history_data)
        
        from fastapi.responses import Response
        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=history_{user_id}.xlsx"}
        )
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pandas no está instalado. Instale con: pip install pandas openpyxl"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exportando historial: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error exportando historial: {str(e)}"
        )


@router.post("/export/comparison/csv")
async def export_comparison_csv(comparison_data: dict):
    """
    Exporta comparación a CSV
    
    - **comparison_data**: Datos de comparación
    """
    try:
        csv_bytes = export_manager.export_comparison_to_csv(comparison_data)
        
        from fastapi.responses import Response
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=comparison.csv"}
        )
    
    except Exception as e:
        logger.error(f"Error exportando comparación: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error exportando comparación: {str(e)}"
        )


@router.post("/webhooks/register")
async def register_webhook(
    url: str = Form(..., description="URL del webhook"),
    events: List[str] = Form(..., description="Eventos a suscribir"),
    secret: Optional[str] = Form(None, description="Secreto para validación")
):
    """Registra un nuevo webhook"""
    try:
        webhook_events = [WebhookEvent(event) for event in events]
        webhook_id = webhook_manager.register_webhook(url=url, events=webhook_events, secret=secret)
        return JSONResponse(content={"success": True, "webhook_id": webhook_id})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registrando webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.delete("/webhooks/{webhook_id}")
async def unregister_webhook(webhook_id: str):
    """Elimina un webhook"""
    try:
        success = webhook_manager.unregister_webhook(webhook_id)
        if not success:
            raise HTTPException(status_code=404, detail="Webhook no encontrado")
        return JSONResponse(content={"success": True})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/webhooks")
async def list_webhooks():
    """Lista todos los webhooks"""
    try:
        webhooks = webhook_manager.list_webhooks()
        return JSONResponse(content={"success": True, "webhooks": [w.to_dict() for w in webhooks]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/webhooks/{webhook_id}/history")
async def get_webhook_history(webhook_id: str, limit: int = Query(100)):
    """Obtiene historial de un webhook"""
    try:
        history = webhook_manager.get_webhook_history(webhook_id, limit=limit)
        return JSONResponse(content={"success": True, "history": history})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/auth/register")
async def register_user(
    email: str = Form(...),
    password: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """Registra un nuevo usuario"""
    try:
        metadata_dict = json.loads(metadata) if metadata else None
        user = auth_manager.create_user(email, password, metadata_dict)
        return JSONResponse(content={"success": True, "user": user.to_dict()})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/auth/login")
async def login(email: str = Form(...), password: str = Form(...)):
    """Autentica un usuario"""
    try:
        token = auth_manager.authenticate(email, password)
        if not token:
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        return JSONResponse(content={"success": True, "token": token, "token_type": "Bearer"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/auth/me")
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Obtiene usuario actual"""
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Token no proporcionado")
        token = authorization.split(" ")[1]
        user_id = auth_manager.verify_token(token)
        if not user_id:
            raise HTTPException(status_code=401, detail="Token inválido")
        user = auth_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        return JSONResponse(content={"success": True, "user": user.to_dict()})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/backup/create")
async def create_backup(
    include_history: bool = Form(True),
    include_products: bool = Form(True),
    include_cache: bool = Form(False)
):
    """Crea un backup"""
    try:
        backup_file = backup_manager.create_backup(include_history, include_products, include_cache)
        return JSONResponse(content={"success": True, "backup_file": backup_file})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/backup/list")
async def list_backups():
    """Lista backups"""
    try:
        backups = backup_manager.list_backups()
        return JSONResponse(content={"success": True, "backups": backups})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/backup/restore")
async def restore_backup(
    backup_file: str = Form(...),
    restore_history: bool = Form(True),
    restore_products: bool = Form(True),
    restore_cache: bool = Form(False)
):
    """Restaura un backup"""
    try:
        success = backup_manager.restore_backup(backup_file, restore_history, restore_products, restore_cache)
        if not success:
            raise HTTPException(status_code=400, detail="Error restaurando")
        return JSONResponse(content={"success": True})
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.delete("/backup/{backup_file}")
async def delete_backup(backup_file: str):
    """Elimina un backup"""
    try:
        success = backup_manager.delete_backup(backup_file)
        if not success:
            raise HTTPException(status_code=404, detail="Backup no encontrado")
        return JSONResponse(content={"success": True})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/notifications/{user_id}")
async def get_notifications(
    user_id: str,
    unread_only: bool = Query(False),
    limit: int = Query(50)
):
    """Obtiene notificaciones de un usuario"""
    try:
        notifications = notification_service.get_user_notifications(
            user_id, unread_only=unread_only, limit=limit
        )
        return JSONResponse(content={
            "success": True,
            "notifications": [n.to_dict() for n in notifications]
        })
    except Exception as e:
        logger.error(f"Error obteniendo notificaciones: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/notifications/{user_id}/unread-count")
async def get_unread_count(user_id: str):
    """Obtiene cantidad de notificaciones no leídas"""
    try:
        count = notification_service.get_unread_count(user_id)
        return JSONResponse(content={"success": True, "unread_count": count})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/notifications/{user_id}/mark-read/{notification_id}")
async def mark_notification_read(user_id: str, notification_id: str):
    """Marca notificación como leída"""
    try:
        notification_service.mark_as_read(user_id, notification_id)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/validate/image")
async def validate_image_advanced(file: UploadFile = File(...)):
    """Validación avanzada de imagen"""
    try:
        image_bytes = await file.read()
        is_valid, info = advanced_validator.validate_image_comprehensive(
            image_bytes, filename=file.filename
        )
        return JSONResponse(content={
            "success": True,
            "valid": is_valid,
            "validation_info": info
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/validate/video")
async def validate_video_advanced(file: UploadFile = File(...)):
    """Validación avanzada de video"""
    try:
        video_bytes = await file.read()
        is_valid, info = advanced_validator.validate_video_comprehensive(
            video_bytes, filename=file.filename
        )
        return JSONResponse(content={
            "success": True,
            "valid": is_valid,
            "validation_info": info
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/dashboard/overview")
async def get_dashboard_overview(days: int = Query(7)):
    """Obtiene overview del dashboard"""
    try:
        overview = metrics_dashboard.get_system_overview(days=days)
        return JSONResponse(content={"success": True, "overview": overview})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/dashboard/performance")
async def get_performance_metrics():
    """Obtiene métricas de rendimiento"""
    try:
        metrics = metrics_dashboard.get_performance_metrics()
        return JSONResponse(content={"success": True, "metrics": metrics})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/dashboard/usage")
async def get_usage_statistics(days: int = Query(30)):
    """Obtiene estadísticas de uso"""
    try:
        stats = metrics_dashboard.get_usage_statistics(days=days)
        return JSONResponse(content={"success": True, "statistics": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/templates")
async def list_report_templates():
    """Lista todas las plantillas de reporte disponibles"""
    try:
        templates = template_engine.list_templates()
        return JSONResponse(content={"success": True, "templates": templates})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/report/with-template")
async def generate_report_with_template(
    template: str = Form(..., description="Nombre de la plantilla"),
    analysis_data: str = Form(..., description="Datos del análisis (JSON)")
):
    """Genera reporte con plantilla específica"""
    try:
        analysis_dict = json.loads(analysis_data)
        report_structure = template_engine.generate_report_structure(
            template, analysis_dict
        )
        return JSONResponse(content={"success": True, "report": report_structure})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/queue/task")
async def enqueue_task(
    task_type: str = Form(...),
    data: str = Form(...),
    priority: str = Form("normal")
):
    """Agrega tarea a la cola asíncrona"""
    try:
        task_data = json.loads(data)
        task_priority = TaskPriority(priority)
        task_id = await async_queue.enqueue(task_type, task_data, task_priority)
        return JSONResponse(content={"success": True, "task_id": task_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/queue/task/{task_id}")
async def get_task_status(task_id: str):
    """Obtiene estado de una tarea"""
    try:
        task = await async_queue.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Tarea no encontrada")
        return JSONResponse(content={"success": True, "task": task.to_dict()})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/queue/stats")
async def get_queue_stats():
    """Obtiene estadísticas de la cola"""
    try:
        stats = async_queue.get_queue_stats()
        return JSONResponse(content={"success": True, "stats": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/integrations")
async def list_integrations():
    """Lista todas las integraciones"""
    try:
        integrations = integration_service.list_integrations()
        return JSONResponse(content={"success": True, "integrations": integrations})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/events/history")
async def get_event_history(
    event_type: Optional[str] = Query(None),
    limit: int = Query(100)
):
    """Obtiene historial de eventos"""
    try:
        event_type_enum = EventType(event_type) if event_type else None
        events = event_system.get_event_history(event_type_enum, limit=limit)
        return JSONResponse(content={
            "success": True,
            "events": [e.to_dict() for e in events]
        })
    except ValueError:
        raise HTTPException(status_code=400, detail="Tipo de evento inválido")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/events/stats")
async def get_event_stats():
    """Obtiene estadísticas de eventos"""
    try:
        stats = event_system.get_event_stats()
        return JSONResponse(content={"success": True, "stats": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/search/analyses")
async def search_analyses(
    user_id: str = Form(...),
    filters: str = Form("[]"),
    sort: Optional[str] = Form(None),
    limit: Optional[int] = Form(None)
):
    """Búsqueda avanzada en análisis"""
    try:
        import json
        filter_list = json.loads(filters)
        
        # Obtener historial
        history = history_tracker.get_user_history(user_id, limit=1000)
        analyses = [h.to_dict() for h in history]
        
        # Construir filtros
        from ..services.advanced_search import SearchFilter
        search_filters = [
            SearchFilter(
                field=f["field"],
                operator=SearchOperator(f["operator"]),
                value=f["value"]
            )
            for f in filter_list
        ]
        
        # Ordenamiento
        from ..services.advanced_search import SortOption
        sort_option = None
        if sort:
            direction = "desc" if sort.startswith("-") else "asc"
            field = sort.lstrip("-")
            sort_option = SortOption(field=field, direction=direction)
        
        # Buscar
        results = search_engine.search_analyses(analyses, search_filters, sort_option, limit)
        
        return JSONResponse(content={"success": True, "results": results, "count": len(results)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/export/{format}")
async def export_data_enhanced(
    format: str,
    data: str = Form(...)
):
    """Exporta datos en formato específico"""
    try:
        import json
        data_dict = json.loads(data)
        
        if format == "json":
            content = enhanced_export.export_to_json(data_dict)
            media_type = "application/json"
        elif format == "csv":
            if isinstance(data_dict, list):
                content = enhanced_export.export_to_csv(data_dict)
            else:
                content = enhanced_export.export_to_csv([data_dict])
            media_type = "text/csv"
        elif format == "xml":
            content = enhanced_export.export_to_xml(data_dict)
            media_type = "application/xml"
        elif format == "markdown":
            content = enhanced_export.export_to_markdown(data_dict)
            media_type = "text/markdown"
        elif format == "yaml":
            content = enhanced_export.export_to_yaml(data_dict)
            media_type = "text/yaml"
        else:
            raise HTTPException(status_code=400, detail=f"Formato no soportado: {format}")
        
        from fastapi.responses import Response
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=export.{format}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/export/formats")
async def get_export_formats():
    """Obtiene formatos de exportación disponibles"""
    try:
        formats = enhanced_export.get_supported_formats()
        return JSONResponse(content={"success": True, "formats": formats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/share")
async def share_resource(
    resource_type: str = Form(...),
    resource_id: str = Form(...),
    owner_id: str = Form(...),
    shared_with: str = Form("[]"),
    permission: str = Form("view"),
    expires_at: Optional[str] = Form(None),
    public: bool = Form(False)
):
    """Comparte un recurso"""
    try:
        shared_list = json.loads(shared_with)
        share_perm = SharePermission(permission)
        share_id = collaboration_service.share_resource(
            resource_type, resource_id, owner_id, shared_list,
            share_perm, expires_at, public
        )
        return JSONResponse(content={"success": True, "share_id": share_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/share/{share_id}")
async def get_shared_resource(share_id: str):
    """Obtiene recurso compartido"""
    try:
        resource = collaboration_service.get_shared_resource(share_id)
        if not resource:
            raise HTTPException(status_code=404, detail="Recurso no encontrado")
        return JSONResponse(content={"success": True, "resource": resource.to_dict()})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.delete("/share/{share_id}")
async def revoke_share(share_id: str, owner_id: str = Query(...)):
    """Revoca un recurso compartido"""
    try:
        success = collaboration_service.revoke_share(share_id, owner_id)
        if not success:
            raise HTTPException(status_code=404, detail="Recurso no encontrado")
        return JSONResponse(content={"success": True})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/comments")
async def add_comment(
    resource_id: str = Form(...),
    user_id: str = Form(...),
    comment: str = Form(...)
):
    """Agrega un comentario"""
    try:
        comment_id = collaboration_service.add_comment(resource_id, user_id, comment)
        return JSONResponse(content={"success": True, "comment_id": comment_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/comments/{resource_id}")
async def get_comments(resource_id: str):
    """Obtiene comentarios de un recurso"""
    try:
        comments = collaboration_service.get_comments(resource_id)
        return JSONResponse(content={"success": True, "comments": comments})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/tags")
async def create_tag(
    name: str = Form(...),
    category: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """Crea un tag"""
    try:
        tag = tagging_system.create_tag(name, category, color, description)
        return JSONResponse(content={"success": True, "tag": tag.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/tags/resource")
async def tag_resource(
    resource_id: str = Form(...),
    tags: str = Form(...)
):
    """Etiqueta un recurso"""
    try:
        tag_list = json.loads(tags)
        tagging_system.tag_resource(resource_id, tag_list)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/tags/resource/{resource_id}")
async def get_resource_tags(resource_id: str):
    """Obtiene tags de un recurso"""
    try:
        tags = tagging_system.get_resource_tags(resource_id)
        return JSONResponse(content={"success": True, "tags": [t.to_dict() for t in tags]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/tags/stats")
async def get_tag_stats():
    """Obtiene estadísticas de tags"""
    try:
        stats = tagging_system.get_tag_stats()
        return JSONResponse(content={"success": True, "stats": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/models/version")
async def register_model_version(
    model_name: str = Form(...),
    version: str = Form(...),
    model_path: str = Form(...),
    status: str = Form("draft"),
    description: Optional[str] = Form(None)
):
    """Registra una versión de modelo"""
    try:
        model_status = ModelStatus(status)
        model_version = model_versioning.register_model_version(
            model_name, version, model_path, model_status, description
        )
        return JSONResponse(content={"success": True, "version": model_version.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/models/{model_name}/versions")
async def get_model_versions(model_name: str):
    """Obtiene versiones de un modelo"""
    try:
        versions = model_versioning.get_model_versions(model_name)
        return JSONResponse(content={"success": True, "versions": [v.to_dict() for v in versions]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Obtiene información de un modelo"""
    try:
        info = model_versioning.get_model_info(model_name)
        return JSONResponse(content={"success": True, "model": info})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/metrics/realtime")
async def get_realtime_metrics(
    metric_name: Optional[str] = Query(None),
    window_seconds: int = Query(300)
):
    """Obtiene métricas en tiempo real"""
    try:
        if metric_name:
            stats = realtime_metrics.get_statistics(metric_name, window_seconds)
            return JSONResponse(content={"success": True, "metric": metric_name, "statistics": stats})
        else:
            dashboard = realtime_metrics.get_dashboard_data()
            return JSONResponse(content={"success": True, "dashboard": dashboard})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/metrics/realtime/record")
async def record_realtime_metric(
    metric_name: str = Form(...),
    value: float = Form(...),
    tags: Optional[str] = Form(None)
):
    """Registra una métrica en tiempo real"""
    try:
        tags_dict = json.loads(tags) if tags else None
        realtime_metrics.record(metric_name, value, tags_dict)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/health/detailed")
async def get_detailed_health():
    """Obtiene health check detallado"""
    try:
        overall = health_monitor.get_overall_health()
        return JSONResponse(content={"success": True, "health": overall})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/health/check/{check_name}")
async def run_health_check(check_name: str):
    """Ejecuta un health check específico"""
    try:
        check = await health_monitor.run_check(check_name)
        return JSONResponse(content={"success": True, "check": check.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/business/kpis")
async def get_business_kpis():
    """Obtiene KPIs de negocio"""
    try:
        kpis = business_metrics.get_kpis()
        return JSONResponse(content={"success": True, "kpis": kpis})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/business/metrics")
async def record_business_metric(
    name: str = Form(...),
    value: float = Form(...),
    unit: str = Form(""),
    category: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Registra una métrica de negocio"""
    try:
        metadata_dict = json.loads(metadata) if metadata else None
        business_metrics.record_metric(name, value, unit, category, metadata_dict)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/business/metrics")
async def get_business_metrics(
    name: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    days: int = Query(30)
):
    """Obtiene métricas de negocio"""
    try:
        metrics = business_metrics.get_metrics(name, category, days)
        return JSONResponse(content={
            "success": True,
            "metrics": [m.to_dict() for m in metrics],
            "count": len(metrics)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/batch/process")
async def process_batch(
    items: str = Form(...),
    processor_type: str = Form(...)
):
    """Procesa un lote de items"""
    try:
        items_list = json.loads(items)
        
        # Definir procesador según tipo
        def processor(item):
            # Placeholder - implementar según tipo
            return {"processed": item}
        
        batch_id = await batch_processor.process_batch(items_list, processor)
        return JSONResponse(content={"success": True, "batch_id": batch_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    """Obtiene estado de un batch"""
    try:
        job = batch_processor.get_batch_status(batch_id)
        if not job:
            raise HTTPException(status_code=404, detail="Batch no encontrado")
        return JSONResponse(content={"success": True, "batch": job.to_dict()})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/docs")
async def get_api_documentation():
    """Obtiene documentación completa de la API"""
    try:
        docs = api_documentation.get_all_docs()
        return JSONResponse(content={"success": True, "documentation": docs})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/docs/openapi")
async def get_openapi_spec():
    """Obtiene especificación OpenAPI"""
    try:
        spec = api_documentation.generate_openapi_spec()
        return JSONResponse(content=spec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/security/report")
async def get_security_report():
    """Obtiene reporte de seguridad"""
    try:
        report = security_enhancer.get_security_report()
        return JSONResponse(content={"success": True, "report": report})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/security/validate")
async def validate_input_security(
    input_data: str = Form(...),
    input_type: str = Form("general")
):
    """Valida input de forma segura"""
    try:
        is_valid, error_message = security_enhancer.validate_input(input_data, input_type)
        return JSONResponse(content={
            "success": True,
            "valid": is_valid,
            "error": error_message
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/recommendations/intelligent")
async def get_intelligent_recommendations(
    user_id: str = Form(...),
    analysis_data: str = Form(...),
    history_data: Optional[str] = Form(None)
):
    """Obtiene recomendaciones inteligentes"""
    try:
        analysis_dict = json.loads(analysis_data)
        history_list = json.loads(history_data) if history_data else None
        
        recommendations = intelligent_recommender.generate_intelligent_recommendations(
            user_id, analysis_dict, history_list
        )
        
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/feedback")
async def submit_feedback(
    user_id: str = Form(...),
    feedback_type: str = Form(...),
    target_id: str = Form(...),
    rating: float = Form(...),
    comment: Optional[str] = Form(None),
    helpful: Optional[bool] = Form(None)
):
    """Envía feedback"""
    try:
        fb_type = FeedbackType(feedback_type)
        feedback_id = feedback_system.submit_feedback(
            user_id, fb_type, target_id, rating, comment, helpful
        )
        return JSONResponse(content={"success": True, "feedback_id": feedback_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/feedback/{target_id}")
async def get_target_feedback(target_id: str):
    """Obtiene feedback de un recurso"""
    try:
        feedbacks = feedback_system.get_target_feedback(target_id)
        stats = feedback_system.get_feedback_stats(target_id)
        
        return JSONResponse(content={
            "success": True,
            "feedbacks": [f.to_dict() for f in feedbacks],
            "statistics": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ab-test/create")
async def create_ab_test(
    name: str = Form(...),
    description: str = Form(...),
    variants: str = Form(...),
    traffic_split: Optional[str] = Form(None)
):
    """Crea un test A/B"""
    try:
        variants_list = json.loads(variants)
        split_dict = json.loads(traffic_split) if traffic_split else None
        
        test_id = ab_testing.create_test(name, description, variants_list, split_dict)
        return JSONResponse(content={"success": True, "test_id": test_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ab-test/{test_id}/assign")
async def assign_ab_variant(test_id: str, user_id: str = Query(...)):
    """Asigna variante A/B a usuario"""
    try:
        variant = ab_testing.assign_variant(test_id, user_id)
        return JSONResponse(content={"success": True, "variant": variant})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ab-test/{test_id}/results")
async def get_ab_test_results(test_id: str):
    """Obtiene resultados de test A/B"""
    try:
        results = ab_testing.get_test_results(test_id)
        test = ab_testing.get_test(test_id)
        
        return JSONResponse(content={
            "success": True,
            "test": test.to_dict() if test else None,
            "results": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/personalization/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Obtiene perfil de usuario"""
    try:
        profile = personalization_engine.get_or_create_profile(user_id)
        insights = personalization_engine.get_user_insights(user_id)
        
        return JSONResponse(content={
            "success": True,
            "profile": profile.to_dict(),
            "insights": insights
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/personalization/recommendations")
async def get_personalized_recommendations(
    user_id: str = Form(...),
    analysis_data: str = Form(...)
):
    """Obtiene recomendaciones personalizadas"""
    try:
        analysis_dict = json.loads(analysis_data)
        recommendations = personalization_engine.get_personalized_recommendations(
            user_id, analysis_dict
        )
        
        return JSONResponse(content={"success": True, "recommendations": recommendations})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/personalization/behavior")
async def record_behavior(
    user_id: str = Form(...),
    action: str = Form(...),
    data: str = Form(...)
):
    """Registra comportamiento del usuario"""
    try:
        data_dict = json.loads(data)
        personalization_engine.record_behavior(user_id, action, data_dict)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/gamification/stats/{user_id}")
async def get_gamification_stats(user_id: str):
    """Obtiene estadísticas de gamificación"""
    try:
        stats = gamification.get_user_stats(user_id)
        return JSONResponse(content={"success": True, "stats": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/gamification/achievements/{user_id}")
async def get_user_achievements(user_id: str):
    """Obtiene logros de un usuario"""
    try:
        achievements = gamification.get_user_achievements(user_id)
        return JSONResponse(content={
            "success": True,
            "achievements": [a.to_dict() for a in achievements]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/gamification/leaderboard")
async def get_leaderboard(limit: int = Query(100)):
    """Obtiene leaderboard"""
    try:
        leaderboard = gamification.get_leaderboard(limit=limit)
        return JSONResponse(content={"success": True, "leaderboard": leaderboard})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/challenges/available/{user_id}")
async def get_available_challenges(user_id: str):
    """Obtiene challenges disponibles"""
    try:
        challenges = challenge_system.get_available_challenges(user_id)
        return JSONResponse(content={
            "success": True,
            "challenges": [c.to_dict() for c in challenges]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/challenges/start")
async def start_challenge(
    user_id: str = Form(...),
    challenge_id: str = Form(...)
):
    """Inicia un challenge"""
    try:
        success = challenge_system.start_challenge(user_id, challenge_id)
        return JSONResponse(content={"success": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/challenges/user/{user_id}")
async def get_user_challenges(user_id: str):
    """Obtiene challenges de un usuario"""
    try:
        challenges = challenge_system.get_user_challenges(user_id)
        return JSONResponse(content={"success": True, "challenges": challenges})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/social/follow")
async def follow_user(
    user_id: str = Form(...),
    follow_user_id: str = Form(...)
):
    """Sigue a un usuario"""
    try:
        success = social_features.follow_user(user_id, follow_user_id)
        return JSONResponse(content={"success": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/social/post")
async def create_social_post(
    user_id: str = Form(...),
    content: str = Form(...),
    analysis_id: Optional[str] = Form(None)
):
    """Crea un post social"""
    try:
        post_id = social_features.create_post(user_id, content, analysis_id)
        return JSONResponse(content={"success": True, "post_id": post_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/social/feed/{user_id}")
async def get_social_feed(user_id: str, limit: int = Query(20)):
    """Obtiene feed social"""
    try:
        feed = social_features.get_feed(user_id, limit=limit)
        return JSONResponse(content={
            "success": True,
            "feed": [p.to_dict() for p in feed]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/social/stats/{user_id}")
async def get_social_stats(user_id: str):
    """Obtiene estadísticas sociales"""
    try:
        stats = social_features.get_user_stats(user_id)
        return JSONResponse(content={"success": True, "stats": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/trends/predict")
async def predict_trends(
    historical_data: str = Form(...),
    metrics: str = Form(...),
    timeframe: str = Form("30d")
):
    """Predice tendencias basadas en datos históricos"""
    try:
        data_list = json.loads(historical_data)
        metrics_list = json.loads(metrics)
        
        predictions = trend_predictor.predict_multiple_metrics(
            data_list, metrics_list, timeframe
        )
        
        return JSONResponse(content={
            "success": True,
            "predictions": [p.to_dict() for p in predictions]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/compare/advanced")
async def advanced_compare(
    analysis1_data: str = Form(...),
    analysis2_data: str = Form(...)
):
    """Comparación avanzada entre dos análisis"""
    try:
        analysis1 = json.loads(analysis1_data)
        analysis2 = json.loads(analysis2_data)
        
        comparison = advanced_comparison.compare_analyses(analysis1, analysis2)
        
        return JSONResponse(content={"success": True, "comparison": comparison})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/compare/multiple")
async def compare_multiple_analyses(analyses_data: str = Form(...)):
    """Compara múltiples análisis"""
    try:
        analyses_list = json.loads(analyses_data)
        
        comparison = advanced_comparison.compare_multiple(analyses_list)
        
        return JSONResponse(content={"success": True, "comparison": comparison})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ml/predict")
async def ml_predict(
    model_id: str = Form(...),
    features: str = Form(...)
):
    """Realiza predicción ML"""
    try:
        import numpy as np
        features_array = np.array(json.loads(features))
        
        prediction = enhanced_ml.predict(model_id, features_array)
        
        return JSONResponse(content={
            "success": True,
            "prediction": prediction.to_dict()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ml/stats")
async def get_ml_stats():
    """Obtiene estadísticas de ML"""
    try:
        stats = enhanced_ml.get_prediction_stats()
        return JSONResponse(content={"success": True, "stats": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/iot/register")
async def register_iot_device(
    user_id: str = Form(...),
    device_type: str = Form(...),
    name: str = Form(...),
    model: Optional[str] = Form(None),
    capabilities: Optional[str] = Form(None)
):
    """Registra un dispositivo IoT"""
    try:
        device_type_enum = DeviceType(device_type)
        capabilities_list = json.loads(capabilities) if capabilities else None
        device_id = iot_integration.register_device(
            user_id, device_type_enum, name, model, capabilities_list
        )
        return JSONResponse(content={"success": True, "device_id": device_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/iot/devices/{user_id}")
async def get_user_devices(user_id: str):
    """Obtiene dispositivos de un usuario"""
    try:
        devices = iot_integration.get_user_devices(user_id)
        return JSONResponse(content={
            "success": True,
            "devices": [d.to_dict() for d in devices]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/iot/data")
async def record_iot_data(
    device_id: str = Form(...),
    data: str = Form(...)
):
    """Registra datos de dispositivo IoT"""
    try:
        data_dict = json.loads(data)
        iot_integration.record_device_data(device_id, data_dict)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/push/register")
async def register_push_token(
    user_id: str = Form(...),
    platform: str = Form(...),
    token: str = Form(...)
):
    """Registra token de push"""
    try:
        platform_enum = PushPlatform(platform)
        push_notifications.register_token(user_id, platform_enum, token)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/push/send")
async def send_push_notification(
    user_id: str = Form(...),
    title: str = Form(...),
    body: str = Form(...),
    platform: Optional[str] = Form(None),
    data: Optional[str] = Form(None),
    priority: str = Form("normal")
):
    """Envía notificación push"""
    try:
        platform_enum = PushPlatform(platform) if platform else None
        data_dict = json.loads(data) if data else None
        notification_ids = push_notifications.send_push(
            user_id, title, body, platform_enum, data_dict, priority
        )
        return JSONResponse(content={
            "success": True,
            "notification_ids": notification_ids
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reports/advanced")
async def generate_advanced_report(
    analysis_data: str = Form(...),
    format: str = Form(...),
    config_data: Optional[str] = Form(None)
):
    """Genera reporte avanzado"""
    try:
        analysis_dict = json.loads(analysis_data)
        format_enum = ReportFormat(format)
        
        if config_data:
            config_dict = json.loads(config_data)
            config = ReportConfig(**config_dict)
        else:
            config = ReportConfig(format=format_enum)
        
        report_bytes = advanced_reporting.generate_advanced_report(analysis_dict, config)
        
        from fastapi.responses import Response
        media_types = {
            ReportFormat.PDF: "application/pdf",
            ReportFormat.HTML: "text/html",
            ReportFormat.JSON: "application/json",
            ReportFormat.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ReportFormat.CSV: "text/csv",
            ReportFormat.MARKDOWN: "text/markdown"
        }
        
        return Response(
            content=report_bytes,
            media_type=media_types.get(format_enum, "application/octet-stream"),
            headers={"Content-Disposition": f"attachment; filename=report.{format}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/reports/formats")
async def get_report_formats():
    """Obtiene formatos de reporte disponibles"""
    try:
        formats = advanced_reporting.get_supported_formats()
        return JSONResponse(content={"success": True, "formats": formats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/image-analysis/advanced")
async def advanced_image_analysis(
    file: UploadFile = File(...),
    analysis_type: str = Form("all")
):
    """Análisis avanzado de imagen"""
    try:
        image_bytes = await file.read()
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        results = {}
        
        if analysis_type in ["all", "texture"]:
            results["texture"] = image_analysis_advanced.analyze_texture_features(image)
        
        if analysis_type in ["all", "color"]:
            results["color"] = image_analysis_advanced.analyze_color_features(image)
        
        if analysis_type in ["all", "geometric"]:
            results["geometric"] = image_analysis_advanced.analyze_geometric_features(image)
        
        return JSONResponse(content={"success": True, "analysis": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ml-recommendations/generate")
async def generate_ml_recommendations(
    user_id: str = Form(...),
    analysis_features: str = Form(...),
    top_k: int = Form(10)
):
    """Genera recomendaciones basadas en ML"""
    try:
        import numpy as np
        features_array = np.array(json.loads(analysis_features))
        
        recommendations = ml_recommender.generate_ml_recommendations(
            user_id, features_array, top_k
        )
        
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/monitoring/metrics/{metric_name}")
async def get_monitoring_metric(
    metric_name: str,
    hours: int = Query(24)
):
    """Obtiene métrica de monitoreo"""
    try:
        stats = advanced_monitoring.get_metric_statistics(metric_name, hours)
        history = advanced_monitoring.get_metric_history(metric_name, hours)
        
        return JSONResponse(content={
            "success": True,
            "statistics": stats,
            "history": [m.to_dict() for m in history[-100:]]  # Últimos 100 puntos
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/monitoring/health")
async def get_monitoring_health():
    """Obtiene salud del sistema desde monitoreo"""
    try:
        health = advanced_monitoring.get_system_health()
        return JSONResponse(content={"success": True, "health": health})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/monitoring/alerts")
async def get_monitoring_alerts(limit: int = Query(100)):
    """Obtiene alertas de monitoreo"""
    try:
        alerts = advanced_monitoring.get_alerts(limit=limit)
        return JSONResponse(content={"success": True, "alerts": alerts})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/conditions/predict")
async def predict_conditions(
    analysis_data: str = Form(...),
    history_data: Optional[str] = Form(None)
):
    """Predice condiciones de piel"""
    try:
        analysis_dict = json.loads(analysis_data)
        history_list = json.loads(history_data) if history_data else None
        
        predictions = condition_predictor.predict_conditions(analysis_dict, history_list)
        
        return JSONResponse(content={
            "success": True,
            "predictions": [p.to_dict() for p in predictions]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/video-analysis/advanced")
async def advanced_video_analysis(
    frames_data: str = Form(...)
):
    """Análisis avanzado de video"""
    try:
        import numpy as np
        frames_list = json.loads(frames_data)
        
        # Convertir a numpy arrays (en producción vendrían como bytes)
        frames = [np.array(frame) for frame in frames_list]
        
        analysis = video_analysis_advanced.analyze_video_frames(frames)
        
        return JSONResponse(content={"success": True, "analysis": analysis})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/video-analysis/key-frames")
async def extract_key_frames(
    frames_data: str = Form(...),
    num_frames: int = Form(5)
):
    """Extrae frames clave del video"""
    try:
        import numpy as np
        frames_list = json.loads(frames_data)
        frames = [np.array(frame) for frame in frames_list]
        
        key_frame_indices = video_analysis_advanced.extract_key_frames(frames, num_frames)
        
        return JSONResponse(content={
            "success": True,
            "key_frame_indices": key_frame_indices
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/learning/learn")
async def learn_from_data(
    user_id: str = Form(...),
    data_points: str = Form(...)
):
    """Aprende de datos del usuario"""
    try:
        data_list = json.loads(data_points)
        
        insights = learning_system.learn_from_data(user_id, data_list)
        
        return JSONResponse(content={
            "success": True,
            "insights": [i.to_dict() for i in insights]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/learning/insights/{user_id}")
async def get_learning_insights(user_id: str):
    """Obtiene insights de aprendizaje"""
    try:
        insights = learning_system.get_user_insights(user_id)
        return JSONResponse(content={"success": True, "insights": insights})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/learning/predict/{user_id}")
async def predict_future_state(
    user_id: str,
    days_ahead: int = Query(30)
):
    """Predice estado futuro del usuario"""
    try:
        prediction = learning_system.predict_future_state(user_id, days_ahead)
        return JSONResponse(content={"success": True, "prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/progress/analyze")
async def analyze_progress(
    historical_data: str = Form(...),
    timeframe_days: int = Form(30)
):
    """Analiza progreso del usuario"""
    try:
        data_list = json.loads(historical_data)
        
        analysis = progress_analyzer.analyze_progress(data_list, timeframe_days)
        
        return JSONResponse(content={"success": True, "analysis": analysis})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/progress/timeline/{user_id}")
async def get_progress_timeline(
    user_id: str,
    metric_name: str = Query("overall_score")
):
    """Obtiene timeline de progreso"""
    try:
        # Obtener historial (placeholder - usar history_tracker real)
        history = history_tracker.get_user_history(user_id, limit=1000)
        history_dicts = [h.to_dict() for h in history]
        
        timeline = progress_analyzer.get_progress_timeline(history_dicts, metric_name)
        
        return JSONResponse(content={"success": True, "timeline": timeline})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/smart-recommendations/generate")
async def generate_smart_recommendations(
    user_id: str = Form(...),
    analysis_data: str = Form(...),
    user_profile: Optional[str] = Form(None),
    progress_data: Optional[str] = Form(None)
):
    """Genera recomendaciones inteligentes mejoradas"""
    try:
        analysis_dict = json.loads(analysis_data)
        profile_dict = json.loads(user_profile) if user_profile else None
        progress_dict = json.loads(progress_data) if progress_data else None
        
        recommendations = smart_recommender.generate_smart_recommendations(
            user_id, analysis_dict, profile_dict, progress_dict
        )
        
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/alerts/intelligent/check")
async def check_intelligent_alerts(
    user_id: str = Form(...),
    analysis_data: str = Form(...)
):
    """Verifica y crea alertas inteligentes"""
    try:
        analysis_dict = json.loads(analysis_data)
        
        alerts = intelligent_alerts.check_and_create_alerts(user_id, analysis_dict)
        
        return JSONResponse(content={
            "success": True,
            "alerts_created": len(alerts),
            "alerts": [a.to_dict() for a in alerts]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/alerts/intelligent/{user_id}")
async def get_intelligent_alerts(
    user_id: str,
    include_dismissed: bool = Query(False)
):
    """Obtiene alertas inteligentes de un usuario"""
    try:
        alerts = intelligent_alerts.get_user_alerts(user_id, include_dismissed)
        stats = intelligent_alerts.get_alert_statistics(user_id)
        
        return JSONResponse(content={
            "success": True,
            "alerts": [a.to_dict() for a in alerts],
            "statistics": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/alerts/intelligent/{user_id}/dismiss/{alert_id}")
async def dismiss_intelligent_alert(user_id: str, alert_id: str):
    """Descarta una alerta inteligente"""
    try:
        success = intelligent_alerts.dismiss_alert(user_id, alert_id)
        return JSONResponse(content={"success": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/predictive/insights")
async def generate_predictive_insights(
    historical_data: str = Form(...),
    metrics: Optional[str] = Form(None)
):
    """Genera insights predictivos"""
    try:
        data_list = json.loads(historical_data)
        metrics_list = json.loads(metrics) if metrics else None
        
        insights = predictive_analytics.generate_predictive_insights(data_list, metrics_list)
        risk_assessment = predictive_analytics.get_risk_assessment(insights)
        
        return JSONResponse(content={
            "success": True,
            "insights": [i.to_dict() for i in insights],
            "risk_assessment": risk_assessment
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/routines/create")
async def create_routine(
    user_id: str = Form(...),
    name: str = Form(...),
    products: str = Form(...),
    steps: str = Form(...),
    morning: bool = Form(True),
    evening: bool = Form(False)
):
    """Crea una nueva rutina de skincare"""
    try:
        products_list = json.loads(products)
        steps_list = json.loads(steps)
        
        routine = routine_comparator.create_routine(
            user_id, name, products_list, steps_list, morning, evening
        )
        
        return JSONResponse(content={"success": True, "routine": routine.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/routines/compare")
async def compare_routines(
    routine1_id: str = Form(...),
    routine2_id: str = Form(...)
):
    """Compara dos rutinas"""
    try:
        comparison = routine_comparator.compare_routines(routine1_id, routine2_id)
        return JSONResponse(content={"success": True, "comparison": comparison.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/routines/user/{user_id}")
async def get_user_routines(user_id: str):
    """Obtiene rutinas de un usuario"""
    try:
        routines = routine_comparator.get_user_routines(user_id)
        return JSONResponse(content={
            "success": True,
            "routines": [r.to_dict() for r in routines]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/products/track")
async def track_product_usage(
    user_id: str = Form(...),
    product_id: str = Form(...),
    product_name: str = Form(...),
    usage_date: str = Form(...),
    frequency: str = Form("daily"),
    effectiveness_rating: Optional[int] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Registra uso de producto"""
    try:
        usage = product_tracker.record_usage(
            user_id, product_id, product_name, usage_date,
            frequency, effectiveness_rating, notes
        )
        return JSONResponse(content={"success": True, "usage": usage.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/products/insights/{user_id}/{product_id}")
async def get_product_insights(user_id: str, product_id: str):
    """Obtiene insights de un producto"""
    try:
        insight = product_tracker.get_product_insights(user_id, product_id)
        if not insight:
            return JSONResponse(content={"success": False, "message": "No data found"})
        return JSONResponse(content={"success": True, "insight": insight.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/products/summary/{user_id}")
async def get_product_summary(user_id: str):
    """Obtiene resumen de productos del usuario"""
    try:
        summary = product_tracker.get_user_product_summary(user_id)
        return JSONResponse(content={"success": True, "summary": summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reminders/create")
async def create_reminder(
    user_id: str = Form(...),
    type: str = Form(...),
    title: str = Form(...),
    message: str = Form(...),
    scheduled_time: str = Form(...),
    frequency: str = Form("once")
):
    """Crea un recordatorio"""
    try:
        reminder_type = ReminderType(type)
        reminder = smart_reminders.create_reminder(
            user_id, reminder_type, title, message, scheduled_time, frequency
        )
        return JSONResponse(content={"success": True, "reminder": reminder.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reminders/routine")
async def create_routine_reminder(
    user_id: str = Form(...),
    routine_name: str = Form(...),
    time: str = Form(...),
    morning: bool = Form(True)
):
    """Crea recordatorio de rutina"""
    try:
        reminder = smart_reminders.create_routine_reminder(user_id, routine_name, time, morning)
        return JSONResponse(content={"success": True, "reminder": reminder.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/reminders/pending/{user_id}")
async def get_pending_reminders(user_id: str):
    """Obtiene recordatorios pendientes"""
    try:
        reminders = smart_reminders.get_pending_reminders(user_id)
        return JSONResponse(content={
            "success": True,
            "reminders": [r.to_dict() for r in reminders]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reminders/complete/{user_id}/{reminder_id}")
async def complete_reminder(user_id: str, reminder_id: str):
    """Marca recordatorio como completado"""
    try:
        success = smart_reminders.mark_completed(user_id, reminder_id)
        return JSONResponse(content={"success": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/market-trends/analyze")
async def analyze_market_trend(
    category: str = Form(...),
    historical_data: str = Form(...)
):
    """Analiza tendencia de mercado de una categoría"""
    try:
        data_list = json.loads(historical_data)
        trend = market_trends.analyze_category_trend(category, data_list)
        return JSONResponse(content={"success": True, "trend": trend.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/market-trends/compare")
async def compare_market_categories(
    categories: str = Form(...),
    historical_data: str = Form(...)
):
    """Compara múltiples categorías"""
    try:
        categories_list = json.loads(categories)
        data_list = json.loads(historical_data)
        
        trends = market_trends.compare_categories(categories_list, data_list)
        
        return JSONResponse(content={
            "success": True,
            "trends": {k: v.to_dict() for k, v in trends.items()}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/market-trends/recommend")
async def get_recommended_categories(
    historical_data: str = Form(...),
    user_preferences: Optional[str] = Form(None)
):
    """Obtiene categorías recomendadas"""
    try:
        data_list = json.loads(historical_data)
        prefs = json.loads(user_preferences) if user_preferences else {}
        
        categories = market_trends.get_recommended_categories(prefs, data_list)
        
        return JSONResponse(content={
            "success": True,
            "recommended_categories": categories
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/goals/create")
async def create_skin_goal(
    user_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    target_metric: str = Form(...),
    target_value: float = Form(...),
    current_value: float = Form(0.0),
    deadline: Optional[str] = Form(None)
):
    """Crea un nuevo objetivo de piel"""
    try:
        goal = skin_goals.create_goal(
            user_id, title, description, target_metric,
            target_value, current_value, deadline
        )
        return JSONResponse(content={"success": True, "goal": goal.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/goals/update/{user_id}/{goal_id}")
async def update_goal_progress(
    user_id: str,
    goal_id: str,
    current_value: float = Form(...)
):
    """Actualiza progreso de un objetivo"""
    try:
        success = skin_goals.update_goal_progress(user_id, goal_id, current_value)
        return JSONResponse(content={"success": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/goals/user/{user_id}")
async def get_user_goals(user_id: str, status: Optional[str] = Query(None)):
    """Obtiene objetivos del usuario"""
    try:
        goal_status = GoalStatus(status) if status else None
        goals = skin_goals.get_user_goals(user_id, goal_status)
        stats = skin_goals.get_goals_statistics(user_id)
        
        return JSONResponse(content={
            "success": True,
            "goals": [g.to_dict() for g in goals],
            "statistics": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/goals/complete/{user_id}/{goal_id}")
async def complete_goal(user_id: str, goal_id: str):
    """Marca objetivo como completado"""
    try:
        success = skin_goals.complete_goal(user_id, goal_id)
        return JSONResponse(content={"success": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/journal/entry")
async def create_journal_entry(
    user_id: str = Form(...),
    date: str = Form(...),
    mood: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    skin_condition: Optional[str] = Form(None),
    products_used: Optional[str] = Form(None),
    photos: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Crea una entrada en el diario"""
    try:
        products = json.loads(products_used) if products_used else None
        photos_list = json.loads(photos) if photos else None
        tags_list = json.loads(tags) if tags else None
        
        entry = skin_journal.create_entry(
            user_id, date, mood, notes, skin_condition,
            products, photos_list, tags_list
        )
        return JSONResponse(content={"success": True, "entry": entry.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/journal/user/{user_id}")
async def get_journal_entries(
    user_id: str,
    limit: int = Query(50),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Obtiene entradas del diario"""
    try:
        entries = skin_journal.get_user_entries(user_id, limit, start_date, end_date)
        stats = skin_journal.get_journal_statistics(user_id)
        
        return JSONResponse(content={
            "success": True,
            "entries": [e.to_dict() for e in entries],
            "statistics": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/journal/search/{user_id}")
async def search_journal_entries(user_id: str, query: str = Query(...)):
    """Busca entradas en el diario"""
    try:
        entries = skin_journal.search_entries(user_id, query)
        return JSONResponse(content={
            "success": True,
            "entries": [e.to_dict() for e in entries]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/experts/available")
async def get_available_experts():
    """Obtiene expertos disponibles"""
    try:
        experts = expert_consultation.get_available_experts()
        return JSONResponse(content={"success": True, "experts": experts})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/experts/consultation/request")
async def request_expert_consultation(
    user_id: str = Form(...),
    expert_id: str = Form(...),
    consultation_type: str = Form(...),
    scheduled_time: Optional[str] = Form(None)
):
    """Solicita consulta con experto"""
    try:
        consultation = expert_consultation.request_consultation(
            user_id, expert_id, consultation_type, scheduled_time
        )
        return JSONResponse(content={"success": True, "consultation": consultation.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/experts/consultations/{user_id}")
async def get_user_consultations(user_id: str, status: Optional[str] = Query(None)):
    """Obtiene consultas del usuario"""
    try:
        consultation_status = ConsultationStatus(status) if status else None
        consultations = expert_consultation.get_user_consultations(user_id, consultation_status)
        return JSONResponse(content={
            "success": True,
            "consultations": [c.to_dict() for c in consultations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ingredients/analyze")
async def analyze_product_ingredients(
    product_name: str = Form(...),
    ingredients: str = Form(...),
    user_skin_type: Optional[str] = Form(None)
):
    """Analiza ingredientes de un producto"""
    try:
        ingredients_list = json.loads(ingredients)
        analysis = ingredient_analyzer.analyze_product_ingredients(
            product_name, ingredients_list, user_skin_type
        )
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ingredients/info/{ingredient_name}")
async def get_ingredient_info(ingredient_name: str):
    """Obtiene información de un ingrediente"""
    try:
        ingredient = ingredient_analyzer.get_ingredient_info(ingredient_name)
        if not ingredient:
            return JSONResponse(content={"success": False, "message": "Ingrediente no encontrado"})
        return JSONResponse(content={"success": True, "ingredient": ingredient.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ingredients/compatibility/{ingredient_name}")
async def check_ingredient_compatibility(
    ingredient_name: str,
    skin_type: str = Query(...)
):
    """Verifica compatibilidad de ingrediente"""
    try:
        result = ingredient_analyzer.check_ingredient_compatibility(ingredient_name, skin_type)
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/recipes/create")
async def create_custom_recipe(
    user_id: str = Form(...),
    name: str = Form(...),
    description: str = Form(...),
    ingredients: str = Form(...),
    instructions: str = Form(...),
    skin_type_target: str = Form(...),
    benefits: str = Form(...),
    preparation_time: int = Form(10),
    shelf_life: str = Form("1 week")
):
    """Crea una receta personalizada"""
    try:
        ingredients_list = json.loads(ingredients)
        instructions_list = json.loads(instructions)
        skin_types = json.loads(skin_type_target)
        benefits_list = json.loads(benefits)
        
        recipe = custom_recipes.create_recipe(
            user_id, name, description, ingredients_list,
            instructions_list, skin_types, benefits_list,
            preparation_time, shelf_life
        )
        return JSONResponse(content={"success": True, "recipe": recipe.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/recipes/from-template")
async def create_recipe_from_template(
    user_id: str = Form(...),
    template_name: str = Form(...),
    modifications: Optional[str] = Form(None)
):
    """Crea receta desde plantilla"""
    try:
        mods = json.loads(modifications) if modifications else None
        recipe = custom_recipes.create_from_template(user_id, template_name, mods)
        return JSONResponse(content={"success": True, "recipe": recipe.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/recipes/user/{user_id}")
async def get_user_recipes(user_id: str):
    """Obtiene recetas del usuario"""
    try:
        recipes = custom_recipes.get_user_recipes(user_id)
        return JSONResponse(content={
            "success": True,
            "recipes": [r.to_dict() for r in recipes]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/recipes/templates")
async def get_recipe_templates():
    """Obtiene plantillas de recetas"""
    try:
        templates = custom_recipes.get_templates()
        return JSONResponse(content={"success": True, "templates": templates})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/recipes/recommend/{user_id}")
async def recommend_recipes(
    user_id: str,
    skin_type: str = Form(...),
    desired_benefits: str = Form(...)
):
    """Recomienda recetas personalizadas"""
    try:
        benefits_list = json.loads(desired_benefits)
        recipes = custom_recipes.recommend_recipes(user_id, skin_type, benefits_list)
        return JSONResponse(content={
            "success": True,
            "recipes": [r.to_dict() for r in recipes]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/products/compare")
async def compare_products(
    user_id: str = Form(...),
    products: str = Form(...),
    criteria: Optional[str] = Form(None)
):
    """Compara productos"""
    try:
        products_list = json.loads(products)
        criteria_list = json.loads(criteria) if criteria else None
        
        comparison = product_comparison.compare_products(user_id, products_list, criteria_list)
        return JSONResponse(content={"success": True, "comparison": comparison.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reviews/create")
async def create_review(
    user_id: str = Form(...),
    product_id: str = Form(...),
    product_name: str = Form(...),
    rating: int = Form(...),
    title: str = Form(...),
    comment: str = Form(...),
    pros: Optional[str] = Form(None),
    cons: Optional[str] = Form(None),
    verified_purchase: bool = Form(False)
):
    """Crea una reseña"""
    try:
        pros_list = json.loads(pros) if pros else None
        cons_list = json.loads(cons) if cons else None
        
        review = reviews_ratings.create_review(
            user_id, product_id, product_name, rating, title, comment,
            pros_list, cons_list, verified_purchase
        )
        return JSONResponse(content={"success": True, "review": review.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/reviews/product/{product_id}")
async def get_product_reviews(
    product_id: str,
    limit: int = Query(50),
    sort_by: str = Query("newest")
):
    """Obtiene reseñas de un producto"""
    try:
        reviews = reviews_ratings.get_product_reviews(product_id, limit, sort_by)
        rating = reviews_ratings.get_product_rating(product_id)
        
        return JSONResponse(content={
            "success": True,
            "reviews": [r.to_dict() for r in reviews],
            "rating": rating.to_dict() if rating else None
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/reviews/top-rated")
async def get_top_rated_products(limit: int = Query(10)):
    """Obtiene productos mejor calificados"""
    try:
        top_products = reviews_ratings.get_top_rated_products(limit)
        return JSONResponse(content={
            "success": True,
            "products": [p.to_dict() for p in top_products]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/before-after/create")
async def create_before_after_comparison(
    user_id: str = Form(...),
    before_image_url: str = Form(...),
    after_image_url: str = Form(...),
    before_analysis: str = Form(...),
    after_analysis: str = Form(...),
    time_interval_days: int = Form(...)
):
    """Crea comparación antes/después"""
    try:
        before_dict = json.loads(before_analysis)
        after_dict = json.loads(after_analysis)
        
        comparison = before_after_analysis.create_comparison(
            user_id, before_image_url, after_image_url,
            before_dict, after_dict, time_interval_days
        )
        return JSONResponse(content={"success": True, "comparison": comparison.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/before-after/user/{user_id}")
async def get_user_before_after_comparisons(user_id: str):
    """Obtiene comparaciones antes/después del usuario"""
    try:
        comparisons = before_after_analysis.get_user_comparisons(user_id)
        return JSONResponse(content={
            "success": True,
            "comparisons": [c.to_dict() for c in comparisons]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/budget/add-entry")
async def add_budget_entry(
    user_id: str = Form(...),
    product_name: str = Form(...),
    amount: float = Form(...),
    category: str = Form(...),
    purchase_date: str = Form(...),
    product_id: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega entrada de presupuesto"""
    try:
        entry = budget_tracker.add_entry(
            user_id, product_name, amount, category, purchase_date,
            product_id, notes
        )
        return JSONResponse(content={"success": True, "entry": entry.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/budget/summary/{user_id}")
async def get_budget_summary(user_id: str, period: str = Query("month")):
    """Obtiene resumen de presupuesto"""
    try:
        summary = budget_tracker.get_budget_summary(user_id, period)
        return JSONResponse(content={"success": True, "summary": summary.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/budget/set-limit")
async def set_budget_limit(
    user_id: str = Form(...),
    period: str = Form(...),
    limit: float = Form(...)
):
    """Establece límite de presupuesto"""
    try:
        budget_tracker.set_budget_limit(user_id, period, limit)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/community/post")
async def create_community_post(
    user_id: str = Form(...),
    username: str = Form(...),
    title: str = Form(...),
    content: str = Form(...),
    post_type: str = Form(...),
    tags: Optional[str] = Form(None)
):
    """Crea un post en la comunidad"""
    try:
        tags_list = json.loads(tags) if tags else None
        post = community_features.create_post(
            user_id, username, title, content, post_type, tags_list
        )
        return JSONResponse(content={"success": True, "post": post.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/community/posts")
async def get_community_posts(
    limit: int = Query(50),
    post_type: Optional[str] = Query(None),
    tags: Optional[str] = Query(None),
    sort_by: str = Query("newest")
):
    """Obtiene posts de la comunidad"""
    try:
        tags_list = json.loads(tags) if tags else None
        posts = community_features.get_posts(limit, post_type, tags_list, sort_by)
        return JSONResponse(content={
            "success": True,
            "posts": [p.to_dict() for p in posts]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/community/comment")
async def add_community_comment(
    post_id: str = Form(...),
    user_id: str = Form(...),
    username: str = Form(...),
    content: str = Form(...)
):
    """Agrega comentario a un post"""
    try:
        comment = community_features.add_comment(post_id, user_id, username, content)
        return JSONResponse(content={"success": True, "comment": comment.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/wearable/register")
async def register_wearable_device(
    user_id: str = Form(...),
    device_id: str = Form(...),
    device_type: str = Form(...),
    device_name: str = Form(...)
):
    """Registra un dispositivo wearable"""
    try:
        device = wearable_integration.register_device(
            user_id, device_id, device_type, device_name
        )
        return JSONResponse(content={"success": True, "device": device})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/wearable/sync")
async def sync_wearable_data(
    user_id: str = Form(...),
    device_id: str = Form(...),
    device_type: str = Form(...),
    metric_type: str = Form(...),
    value: float = Form(...),
    unit: str = Form(...),
    timestamp: str = Form(...)
):
    """Sincroniza datos de wearable"""
    try:
        data = wearable_integration.sync_data(
            user_id, device_id, device_type, metric_type, value, unit, timestamp
        )
        return JSONResponse(content={"success": True, "data": data.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/wearable/insights/{user_id}")
async def get_wearable_insights(user_id: str, metric_type: str = Query(...)):
    """Obtiene insights de datos de wearable"""
    try:
        insight = wearable_integration.generate_insights(user_id, metric_type)
        return JSONResponse(content={
            "success": True,
            "insight": insight.to_dict() if insight else None
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/weather/analyze")
async def analyze_weather_impact(
    location: str = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    uv_index: float = Form(...),
    wind_speed: float = Form(...),
    air_quality: Optional[float] = Form(None),
    user_skin_type: str = Form(...)
):
    """Analiza impacto del clima en la piel"""
    try:
        from ..services.weather_climate_analysis import WeatherData
        
        weather = WeatherData(
            location=location,
            temperature=temperature,
            humidity=humidity,
            uv_index=uv_index,
            wind_speed=wind_speed,
            air_quality=air_quality
        )
        
        recommendation = weather_climate.analyze_weather_impact(weather, user_skin_type)
        return JSONResponse(content={"success": True, "recommendation": recommendation.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/notifications/{user_id}")
async def get_user_notifications(
    user_id: str,
    unread_only: bool = Query(False),
    limit: int = Query(50)
):
    """Obtiene notificaciones del usuario"""
    try:
        notifications = enhanced_notifications.get_user_notifications(
            user_id, unread_only, limit
        )
        stats = enhanced_notifications.get_notification_stats(user_id)
        
        return JSONResponse(content={
            "success": True,
            "notifications": [n.to_dict() for n in notifications],
            "stats": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/notifications/mark-read/{user_id}/{notification_id}")
async def mark_notification_read(user_id: str, notification_id: str):
    """Marca notificación como leída"""
    try:
        success = enhanced_notifications.mark_as_read(user_id, notification_id)
        return JSONResponse(content={"success": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ai-photo/analyze")
async def analyze_photo_with_ai(
    user_id: str = Form(...),
    image_url: str = Form(...),
    analysis_type: str = Form(...)
):
    """Analiza foto con IA avanzada"""
    try:
        analysis = ai_photo_analysis.analyze_photo(user_id, image_url, analysis_type)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ai-photo/compare")
async def compare_ai_analyses(
    user_id: str = Form(...),
    analysis1_id: str = Form(...),
    analysis2_id: str = Form(...)
):
    """Compara dos análisis de IA"""
    try:
        comparison = ai_photo_analysis.compare_analyses(analysis1_id, analysis2_id, user_id)
        return JSONResponse(content={"success": True, "comparison": comparison})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/seasonal/recommendations")
async def get_seasonal_recommendations(
    season: str = Query(...),
    location: str = Query(...),
    user_skin_type: str = Query(...)
):
    """Obtiene recomendaciones estacionales"""
    try:
        season_enum = Season(season)
        recommendations = seasonal_recommendations.get_seasonal_recommendations(
            season_enum, location, user_skin_type
        )
        return JSONResponse(content={"success": True, "recommendations": recommendations.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/seasonal/current")
async def get_current_season_recommendations(
    location: str = Query(...),
    user_skin_type: str = Query(...)
):
    """Obtiene recomendaciones para estación actual"""
    try:
        current_season = seasonal_recommendations.get_current_season(location)
        recommendations = seasonal_recommendations.get_seasonal_recommendations(
            current_season, location, user_skin_type
        )
        return JSONResponse(content={
            "success": True,
            "current_season": current_season.value,
            "recommendations": recommendations.to_dict()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/allergies/record")
async def record_allergy(
    user_id: str = Form(...),
    allergen: str = Form(...),
    reaction_type: str = Form(...),
    severity: str = Form(...),
    occurred_date: str = Form(...),
    product_name: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Registra una reacción alérgica"""
    try:
        record = allergy_tracker.record_allergy(
            user_id, allergen, reaction_type, severity, occurred_date,
            product_name, notes
        )
        return JSONResponse(content={"success": True, "record": record.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/allergies/profile/{user_id}")
async def get_allergy_profile(user_id: str):
    """Obtiene perfil de alergias del usuario"""
    try:
        profile = allergy_tracker.get_allergy_profile(user_id)
        return JSONResponse(content={"success": True, "profile": profile.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/allergies/check-product")
async def check_product_safety(
    user_id: str = Form(...),
    product_ingredients: str = Form(...)
):
    """Verifica seguridad de producto"""
    try:
        ingredients_list = json.loads(product_ingredients)
        safety = allergy_tracker.check_product_safety(user_id, ingredients_list)
        return JSONResponse(content={"success": True, "safety": safety})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/texture/analyze")
async def analyze_texture_advanced(
    user_id: str = Form(...),
    image_url: str = Form(...)
):
    """Analiza textura avanzada"""
    try:
        analysis = advanced_texture_analysis.analyze_texture(user_id, image_url)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/texture/compare")
async def compare_textures(
    user_id: str = Form(...),
    analysis1_id: str = Form(...),
    analysis2_id: str = Form(...)
):
    """Compara dos análisis de textura"""
    try:
        comparison = advanced_texture_analysis.compare_textures(
            user_id, analysis1_id, analysis2_id
        )
        return JSONResponse(content={"success": True, "comparison": comparison})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/product-needs/predict")
async def predict_product_needs(
    user_id: str = Form(...),
    current_products: str = Form(...),
    skin_analysis: str = Form(...),
    usage_history: str = Form(...)
):
    """Predice necesidades de productos"""
    try:
        products_list = json.loads(current_products)
        analysis_dict = json.loads(skin_analysis)
        history_list = json.loads(usage_history)
        
        needs = product_needs_predictor.predict_needs(
            user_id, products_list, analysis_dict, history_list
        )
        return JSONResponse(content={
            "success": True,
            "needs": [n.to_dict() for n in needs]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/habits/record")
async def record_habit(
    user_id: str = Form(...),
    habit_type: str = Form(...),
    date: str = Form(...),
    completed: bool = Form(...),
    notes: Optional[str] = Form(None)
):
    """Registra un hábito"""
    try:
        habit_analyzer.record_habit(user_id, habit_type, date, completed, notes)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/habits/analyze/{user_id}")
async def analyze_habits(user_id: str, days: int = Query(30)):
    """Analiza hábitos del usuario"""
    try:
        analysis = habit_analyzer.analyze_habits(user_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/coaching/session")
async def create_coaching_session(
    user_id: str = Form(...),
    session_type: str = Form(...),
    user_data: str = Form(...)
):
    """Crea sesión de coaching"""
    try:
        data_dict = json.loads(user_data)
        session = personalized_coaching.create_coaching_session(
            user_id, session_type, data_dict
        )
        return JSONResponse(content={"success": True, "session": session.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/coaching/sessions/{user_id}")
async def get_coaching_sessions(user_id: str, session_type: Optional[str] = Query(None)):
    """Obtiene sesiones de coaching del usuario"""
    try:
        sessions = personalized_coaching.get_user_sessions(user_id, session_type)
        return JSONResponse(content={
            "success": True,
            "sessions": [s.to_dict() for s in sessions]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/medical-treatment/add")
async def add_medical_treatment(
    user_id: str = Form(...),
    treatment_name: str = Form(...),
    treatment_type: str = Form(...),
    start_date: str = Form(...),
    frequency: str = Form(...),
    doctor_name: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    dosage: Optional[str] = Form(None),
    instructions: Optional[str] = Form(None)
):
    """Agrega tratamiento médico"""
    try:
        instructions_list = json.loads(instructions) if instructions else None
        treatment = medical_treatment_tracker.add_treatment(
            user_id, treatment_name, treatment_type, start_date, frequency,
            doctor_name, end_date, dosage, instructions_list
        )
        return JSONResponse(content={"success": True, "treatment": treatment.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/medical-treatment/progress")
async def record_treatment_progress(
    treatment_id: str = Form(...),
    date: str = Form(...),
    adherence: bool = Form(...),
    notes: Optional[str] = Form(None),
    side_effects: Optional[str] = Form(None),
    effectiveness_rating: Optional[int] = Form(None)
):
    """Registra progreso de tratamiento"""
    try:
        side_effects_list = json.loads(side_effects) if side_effects else None
        progress = medical_treatment_tracker.record_progress(
            treatment_id, date, adherence, notes, side_effects_list, effectiveness_rating
        )
        return JSONResponse(content={"success": True, "progress": progress.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/medical-treatment/user/{user_id}")
async def get_user_treatments(user_id: str, status: Optional[str] = Query(None)):
    """Obtiene tratamientos del usuario"""
    try:
        treatments = medical_treatment_tracker.get_user_treatments(user_id, status)
        return JSONResponse(content={
            "success": True,
            "treatments": [t.to_dict() for t in treatments]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/medical-treatment/adherence/{treatment_id}")
async def get_treatment_adherence(treatment_id: str):
    """Obtiene tasa de adherencia de tratamiento"""
    try:
        adherence_rate = medical_treatment_tracker.get_adherence_rate(treatment_id)
        progress = medical_treatment_tracker.get_treatment_progress(treatment_id)
        
        return JSONResponse(content={
            "success": True,
            "adherence_rate": adherence_rate,
            "progress_records": len(progress)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/visual-progress/add")
async def add_visual_progress_entry(
    user_id: str = Form(...),
    image_url: str = Form(...),
    date: str = Form(...),
    analysis_data: str = Form(...),
    notes: Optional[str] = Form(None)
):
    """Agrega entrada de progreso visual"""
    try:
        analysis_dict = json.loads(analysis_data)
        entry = visual_progress_tracker.add_progress_entry(
            user_id, image_url, date, analysis_dict, notes
        )
        return JSONResponse(content={"success": True, "entry": entry.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/visual-progress/timeline/{user_id}")
async def get_progress_timeline(user_id: str):
    """Obtiene timeline de progreso visual"""
    try:
        timeline = visual_progress_tracker.get_progress_timeline(user_id)
        return JSONResponse(content={"success": True, "timeline": timeline.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/pharmacy/register")
async def register_pharmacy(
    name: str = Form(...),
    address: str = Form(...),
    phone: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    rating: Optional[float] = Form(None)
):
    """Registra una farmacia"""
    try:
        pharmacy = pharmacy_integration.register_pharmacy(
            name, address, phone, latitude, longitude, rating
        )
        return JSONResponse(content={"success": True, "pharmacy": pharmacy.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/pharmacy/nearby")
async def find_nearby_pharmacies(
    latitude: float = Query(...),
    longitude: float = Query(...),
    radius_km: float = Query(5.0)
):
    """Encuentra farmacias cercanas"""
    try:
        pharmacies = pharmacy_integration.find_nearby_pharmacies(latitude, longitude, radius_km)
        return JSONResponse(content={
            "success": True,
            "pharmacies": [p.to_dict() for p in pharmacies]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/pharmacy/product-availability")
async def check_product_availability(
    product_id: str = Query(...),
    product_name: str = Query(...),
    pharmacy_id: Optional[str] = Query(None),
    latitude: Optional[float] = Query(None),
    longitude: Optional[float] = Query(None)
):
    """Verifica disponibilidad de producto en farmacias"""
    try:
        availability = pharmacy_integration.check_product_availability(
            product_id, product_name, pharmacy_id, latitude, longitude
        )
        return JSONResponse(content={
            "success": True,
            "availability": [a.to_dict() for a in availability]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/product-reminders/add-inventory")
async def add_product_to_inventory(
    user_id: str = Form(...),
    product_id: str = Form(...),
    product_name: str = Form(...),
    purchase_date: str = Form(...),
    expiry_date: Optional[str] = Form(None),
    quantity: int = Form(1),
    usage_frequency: str = Form("daily")
):
    """Agrega producto al inventario"""
    try:
        product_reminder_system.add_product_to_inventory(
            user_id, product_id, product_name, purchase_date,
            expiry_date, quantity, usage_frequency
        )
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/product-reminders/{user_id}")
async def get_product_reminders(
    user_id: str,
    reminder_type: Optional[str] = Query(None),
    include_completed: bool = Query(False)
):
    """Obtiene recordatorios de productos"""
    try:
        reminders = product_reminder_system.get_user_reminders(
            user_id, reminder_type, include_completed
        )
        summary = product_reminder_system.get_inventory_summary(user_id)
        
        return JSONResponse(content={
            "success": True,
            "reminders": [r.to_dict() for r in reminders],
            "inventory_summary": summary
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ingredients/check-conflicts")
async def check_ingredient_conflicts(products: str = Form(...)):
    """Verifica conflictos de ingredientes entre productos"""
    try:
        products_list = json.loads(products)
        compatibility = ingredient_conflict_checker.check_product_compatibility(products_list)
        return JSONResponse(content={"success": True, "compatibility": compatibility.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/comparative/add-data")
async def add_user_data_for_comparison(
    user_id: str = Form(...),
    skin_analysis: str = Form(...),
    age: Optional[int] = Form(None),
    skin_type: Optional[str] = Form(None)
):
    """Agrega datos de usuario para comparación"""
    try:
        analysis_dict = json.loads(skin_analysis)
        comparative_analysis.add_user_data(user_id, analysis_dict, age, skin_type)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/comparative/compare/{user_id}")
async def compare_with_peers(user_id: str, peer_group: str = Query("general")):
    """Compara usuario con sus pares"""
    try:
        comparison = comparative_analysis.compare_with_peers(user_id, peer_group)
        return JSONResponse(content={"success": True, "comparison": comparison.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/budget/recommendations")
async def get_budget_recommendations(
    budget_limit: float = Query(...),
    skin_needs: str = Query(...),
    essential_only: bool = Query(False)
):
    """Obtiene recomendaciones basadas en presupuesto"""
    try:
        needs_list = json.loads(skin_needs)
        recommendations = budget_recommendations.get_budget_recommendations(
            budget_limit, needs_list, essential_only
        )
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/budget/optimize-routine")
async def optimize_routine_for_budget(
    current_routine: str = Form(...),
    budget_limit: float = Form(...)
):
    """Optimiza rutina para presupuesto"""
    try:
        routine_list = json.loads(current_routine)
        optimization = budget_recommendations.optimize_routine_for_budget(routine_list, budget_limit)
        return JSONResponse(content={"success": True, "optimization": optimization})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/historical-photos/add")
async def add_historical_photo(
    user_id: str = Form(...),
    image_url: str = Form(...),
    date: str = Form(...),
    analysis_data: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega foto histórica"""
    try:
        analysis_dict = json.loads(analysis_data) if analysis_data else None
        tags_list = json.loads(tags) if tags else None
        
        photo = historical_photo_analysis.add_photo(
            user_id, image_url, date, analysis_dict, tags_list, notes
        )
        return JSONResponse(content={"success": True, "photo": photo.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/historical-photos/timeline/{user_id}")
async def get_photo_timeline(
    user_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Obtiene timeline de fotos históricas"""
    try:
        timeline = historical_photo_analysis.get_photo_timeline(user_id, start_date, end_date)
        return JSONResponse(content={"success": True, "timeline": timeline.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/product-trends/add-data")
async def add_product_trend_data(
    product_id: str = Form(...),
    product_name: str = Form(...),
    category: str = Form(...),
    popularity: float = Form(...),
    satisfaction: float = Form(...),
    price: float = Form(...)
):
    """Agrega dato de tendencia de producto"""
    try:
        product_trend_analyzer.add_product_data(
            product_id, product_name, category, popularity, satisfaction, price
        )
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/product-trends/product/{product_id}")
async def get_product_trend(product_id: str):
    """Obtiene tendencia de un producto"""
    try:
        trend = product_trend_analyzer.analyze_product_trend(product_id)
        return JSONResponse(content={
            "success": True,
            "trend": trend.to_dict() if trend else None
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/product-trends/category/{category}")
async def get_category_trends(category: str):
    """Obtiene tendencias de una categoría"""
    try:
        category_trend = product_trend_analyzer.analyze_category_trends(category)
        return JSONResponse(content={
            "success": True,
            "trend": category_trend.to_dict() if category_trend else None
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/product-trends/trending")
async def get_trending_products(limit: int = Query(10)):
    """Obtiene productos en tendencia"""
    try:
        trending = product_trend_analyzer.get_trending_products(limit)
        return JSONResponse(content={
            "success": True,
            "trending_products": [t.to_dict() for t in trending]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/age/analyze")
async def analyze_age(
    user_id: str = Form(...),
    image_url: str = Form(...),
    chronological_age: int = Form(...),
    skin_analysis: str = Form(...)
):
    """Analiza edad aparente"""
    try:
        analysis_dict = json.loads(skin_analysis)
        analysis = age_analysis.analyze_age(user_id, image_url, chronological_age, analysis_dict)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/age/recommendations")
async def get_age_recommendations(age: int = Query(...), skin_type: str = Query(...)):
    """Obtiene recomendaciones basadas en edad"""
    try:
        recommendations = age_recommendations.get_age_recommendations(age, skin_type)
        return JSONResponse(content={"success": True, "recommendations": recommendations.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/conditions/multi-analyze")
async def analyze_multiple_conditions(
    user_id: str = Form(...),
    image_url: str = Form(...),
    analysis_data: str = Form(...)
):
    """Analiza múltiples condiciones"""
    try:
        analysis_dict = json.loads(analysis_data)
        report = multi_condition_analyzer.analyze_conditions(user_id, image_url, analysis_dict)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/successful-routines/add")
async def add_successful_routine(
    user_id: str = Form(...),
    routine_name: str = Form(...),
    products: str = Form(...),
    skin_type: str = Form(...),
    age_range: str = Form(...),
    improvement_percentage: float = Form(...),
    time_to_results_weeks: int = Form(...),
    user_rating: float = Form(...),
    verified: bool = Form(False)
):
    """Agrega una rutina exitosa"""
    try:
        products_list = json.loads(products)
        routine = successful_routines.add_successful_routine(
            user_id, routine_name, products_list, skin_type, age_range,
            improvement_percentage, time_to_results_weeks, user_rating, verified
        )
        return JSONResponse(content={"success": True, "routine": routine.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/successful-routines/match")
async def find_matching_routines(
    skin_type: str = Query(...),
    age_range: str = Query(...),
    concerns: str = Query(...),
    limit: int = Query(5)
):
    """Encuentra rutinas que coincidan"""
    try:
        concerns_list = json.loads(concerns)
        matches = successful_routines.find_matching_routines(
            skin_type, age_range, concerns_list, limit
        )
        return JSONResponse(content={
            "success": True,
            "matches": [m.to_dict() for m in matches]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/successful-routines/top")
async def get_top_routines(
    skin_type: Optional[str] = Query(None),
    age_range: Optional[str] = Query(None),
    limit: int = Query(10)
):
    """Obtiene rutinas top"""
    try:
        routines = successful_routines.get_top_routines(skin_type, age_range, limit)
        return JSONResponse(content={
            "success": True,
            "routines": [r.to_dict() for r in routines]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/multi-angle/analyze")
async def analyze_multiple_angles(
    user_id: str = Form(...),
    angle_images: str = Form(...)
):
    """Analiza múltiples ángulos"""
    try:
        images_dict = json.loads(angle_images)
        report = multi_angle_analysis.analyze_multiple_angles(user_id, images_dict)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/lifestyle/create-profile")
async def create_lifestyle_profile(
    user_id: str = Form(...),
    diet: str = Form(...),
    exercise_frequency: str = Form(...),
    sleep_hours: float = Form(...),
    stress_level: str = Form(...),
    sun_exposure: str = Form(...),
    smoking: bool = Form(False),
    alcohol_consumption: str = Form("none")
):
    """Crea perfil de estilo de vida"""
    try:
        profile = lifestyle_recommendations.create_profile(
            user_id, diet, exercise_frequency, sleep_hours,
            stress_level, sun_exposure, smoking, alcohol_consumption
        )
        return JSONResponse(content={"success": True, "profile": profile.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/lifestyle/recommendations/{user_id}")
async def get_lifestyle_recommendations(user_id: str):
    """Obtiene recomendaciones basadas en estilo de vida"""
    try:
        recommendations = lifestyle_recommendations.get_lifestyle_recommendations(user_id)
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/medical-device/register")
async def register_medical_device(
    user_id: str = Form(...),
    device_type: str = Form(...),
    device_name: str = Form(...),
    manufacturer: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    calibration_date: Optional[str] = Form(None)
):
    """Registra dispositivo médico"""
    try:
        device = medical_device_integration.register_device(
            user_id, device_type, device_name, manufacturer, model, calibration_date
        )
        return JSONResponse(content={"success": True, "device": device.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/medical-device/sync-reading")
async def sync_device_reading(
    device_id: str = Form(...),
    reading_type: str = Form(...),
    value: float = Form(...),
    unit: str = Form(...),
    timestamp: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """Sincroniza lectura de dispositivo"""
    try:
        metadata_dict = json.loads(metadata) if metadata else None
        reading = medical_device_integration.sync_device_reading(
            device_id, reading_type, value, unit, timestamp, metadata_dict
        )
        return JSONResponse(content={"success": True, "reading": reading.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/medical-device/analyze/{device_id}")
async def analyze_device_data(device_id: str, days: int = Query(30)):
    """Analiza datos de dispositivo"""
    try:
        analysis = medical_device_integration.analyze_device_data(device_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/benchmark/add-score")
async def add_benchmark_score(
    user_id: str = Form(...),
    metric_name: str = Form(...),
    value: float = Form(...)
):
    """Agrega score para benchmark"""
    try:
        benchmark_analysis.add_user_score(user_id, metric_name, value)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/benchmark/report/{user_id}")
async def get_benchmark_report(user_id: str):
    """Obtiene reporte de benchmarks"""
    try:
        report = benchmark_analysis.generate_benchmark_report(user_id)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/lighting/analyze")
async def analyze_with_lighting(
    user_id: str = Form(...),
    lighting_images: str = Form(...)
):
    """Analiza con diferentes tipos de iluminación"""
    try:
        images_dict = json.loads(lighting_images)
        report = lighting_analysis.analyze_with_lighting(user_id, images_dict)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/genetic/create-profile")
async def create_genetic_profile(
    user_id: str = Form(...),
    skin_type_genetic: str = Form(...),
    aging_tendency: str = Form(...),
    collagen_production: str = Form(...),
    melanin_production: str = Form(...),
    sensitivity_tendency: str = Form(...),
    hydration_capacity: str = Form(...)
):
    """Crea perfil genético"""
    try:
        profile = genetic_recommendations.create_genetic_profile(
            user_id, skin_type_genetic, aging_tendency, collagen_production,
            melanin_production, sensitivity_tendency, hydration_capacity
        )
        return JSONResponse(content={"success": True, "profile": profile.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/genetic/recommendations/{user_id}")
async def get_genetic_recommendations(user_id: str):
    """Obtiene recomendaciones basadas en genética"""
    try:
        recommendations = genetic_recommendations.get_genetic_recommendations(user_id)
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/professional-treatment/add")
async def add_professional_treatment(
    user_id: str = Form(...),
    treatment_name: str = Form(...),
    treatment_type: str = Form(...),
    provider_name: str = Form(...),
    treatment_date: str = Form(...),
    provider_license: Optional[str] = Form(None),
    cost: Optional[float] = Form(None),
    duration_minutes: Optional[int] = Form(None),
    before_photos: Optional[str] = Form(None),
    after_photos: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    follow_up_date: Optional[str] = Form(None)
):
    """Agrega tratamiento profesional"""
    try:
        before_list = json.loads(before_photos) if before_photos else None
        after_list = json.loads(after_photos) if after_photos else None
        
        treatment = professional_treatment_tracker.add_treatment(
            user_id, treatment_name, treatment_type, provider_name, treatment_date,
            provider_license, cost, duration_minutes, before_list, after_list,
            notes, follow_up_date
        )
        return JSONResponse(content={"success": True, "treatment": treatment.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/professional-treatment/user/{user_id}")
async def get_professional_treatments(user_id: str, treatment_type: Optional[str] = Query(None)):
    """Obtiene tratamientos profesionales del usuario"""
    try:
        treatments = professional_treatment_tracker.get_user_treatments(user_id, treatment_type)
        stats = professional_treatment_tracker.get_treatment_statistics(user_id)
        
        return JSONResponse(content={
            "success": True,
            "treatments": [t.to_dict() for t in treatments],
            "statistics": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ai-progress/add-data")
async def add_progress_data_point(
    user_id: str = Form(...),
    timestamp: str = Form(...),
    metrics: str = Form(...)
):
    """Agrega punto de datos para análisis de progreso"""
    try:
        metrics_dict = json.loads(metrics)
        ai_progress_analysis.add_data_point(user_id, timestamp, metrics_dict)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ai-progress/analyze/{user_id}")
async def analyze_ai_progress(user_id: str, days: int = Query(90)):
    """Analiza progreso con IA"""
    try:
        report = ai_progress_analysis.analyze_progress(user_id, days)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/climate/analyze")
async def analyze_with_climate(
    user_id: str = Form(...),
    climate_data: str = Form(...)
):
    """Analiza con diferentes condiciones climáticas"""
    try:
        climate_list = json.loads(climate_data)
        report = climate_analysis.analyze_with_climate(user_id, climate_list)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/time-based/generate-routine")
async def generate_time_based_routine(
    user_id: str = Form(...),
    skin_type: str = Form(...),
    concerns: str = Form(...)
):
    """Genera rutina basada en horarios"""
    try:
        concerns_list = json.loads(concerns) if concerns else []
        routine = time_based_recommendations.generate_time_based_routine(user_id, skin_type, concerns_list)
        return JSONResponse(content={"success": True, "routine": routine.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/time-based/routine/{user_id}")
async def get_time_based_routine(user_id: str):
    """Obtiene rutina basada en horarios"""
    try:
        routine = time_based_recommendations.get_time_based_routine(user_id)
        if not routine:
            raise HTTPException(status_code=404, detail="Rutina no encontrada")
        return JSONResponse(content={"success": True, "routine": routine.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/side-effect/report")
async def report_side_effect(
    user_id: str = Form(...),
    product_name: str = Form(...),
    effect_type: str = Form(...),
    severity: str = Form(...),
    description: str = Form(...),
    start_date: str = Form(...),
    action_taken: Optional[str] = Form(None)
):
    """Reporta efecto secundario"""
    try:
        effect = side_effect_tracker.report_side_effect(
            user_id, product_name, effect_type, severity, description, start_date, action_taken
        )
        return JSONResponse(content={"success": True, "effect": effect.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/side-effect/resolve")
async def resolve_side_effect(
    effect_id: str = Form(...),
    user_id: str = Form(...),
    end_date: str = Form(...)
):
    """Resuelve efecto secundario"""
    try:
        success = side_effect_tracker.resolve_side_effect(effect_id, user_id, end_date)
        if not success:
            raise HTTPException(status_code=404, detail="Efecto secundario no encontrado")
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/side-effect/report/{user_id}")
async def get_side_effect_report(user_id: str):
    """Obtiene reporte de efectos secundarios"""
    try:
        report = side_effect_tracker.generate_side_effect_report(user_id)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/texture-ml/analyze")
async def analyze_texture_ml(
    user_id: str = Form(...),
    image_url: str = Form(...)
):
    """Analiza textura con ML avanzado"""
    try:
        analysis = advanced_texture_ml.analyze_texture_ml(user_id, image_url)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/texture-ml/compare")
async def compare_texture_analyses(
    analysis_id_1: str = Query(...),
    analysis_id_2: str = Query(...)
):
    """Compara dos análisis de textura"""
    try:
        comparison = advanced_texture_ml.compare_texture_analyses(analysis_id_1, analysis_id_2)
        return JSONResponse(content={"success": True, "comparison": comparison})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/skin-state/analyze")
async def analyze_with_skin_state(
    user_id: str = Form(...),
    state_data: str = Form(...)
):
    """Analiza con diferentes estados de la piel"""
    try:
        state_list = json.loads(state_data)
        report = skin_state_analysis.analyze_with_state(user_id, state_list)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/fitness/create-profile")
async def create_fitness_profile(
    user_id: str = Form(...),
    workout_frequency: str = Form(...),
    workout_type: str = Form(...),
    sweat_level: str = Form(...),
    shower_frequency_post_workout: str = Form(...),
    outdoor_activities: bool = Form(...)
):
    """Crea perfil de actividad física"""
    try:
        workout_list = json.loads(workout_type) if workout_type else []
        profile = fitness_recommendations.create_fitness_profile(
            user_id, workout_frequency, workout_list, sweat_level,
            shower_frequency_post_workout, outdoor_activities
        )
        return JSONResponse(content={"success": True, "profile": profile.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/fitness/recommendations/{user_id}")
async def get_fitness_recommendations(user_id: str):
    """Obtiene recomendaciones basadas en actividad física"""
    try:
        recommendations = fitness_recommendations.get_fitness_recommendations(user_id)
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/supplement/add")
async def add_supplement(
    user_id: str = Form(...),
    supplement_name: str = Form(...),
    supplement_type: str = Form(...),
    dosage: str = Form(...),
    frequency: str = Form(...),
    start_date: str = Form(...),
    purpose: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega suplemento"""
    try:
        supplement = supplement_tracker.add_supplement(
            user_id, supplement_name, supplement_type, dosage,
            frequency, start_date, purpose, notes
        )
        return JSONResponse(content={"success": True, "supplement": supplement.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/supplement/deactivate")
async def deactivate_supplement(
    supplement_id: str = Form(...),
    user_id: str = Form(...),
    end_date: str = Form(...)
):
    """Desactiva suplemento"""
    try:
        success = supplement_tracker.deactivate_supplement(supplement_id, user_id, end_date)
        if not success:
            raise HTTPException(status_code=404, detail="Suplemento no encontrado")
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/supplement/report/{user_id}")
async def get_supplement_report(user_id: str):
    """Obtiene reporte de suplementos"""
    try:
        report = supplement_tracker.generate_supplement_report(user_id)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/temporal-comparison/add-data")
async def add_temporal_data_point(
    user_id: str = Form(...),
    timestamp: str = Form(...),
    metrics: str = Form(...)
):
    """Agrega punto de datos para comparación temporal"""
    try:
        metrics_dict = json.loads(metrics)
        temporal_comparison.add_data_point(user_id, timestamp, metrics_dict)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/temporal-comparison/generate")
async def generate_temporal_comparison(
    user_id: str = Form(...),
    baseline_date: str = Form(...),
    current_date: str = Form(...)
):
    """Genera comparación temporal"""
    try:
        report = temporal_comparison.generate_comparison(user_id, baseline_date, current_date)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/resolution/analyze")
async def analyze_with_resolution(
    user_id: str = Form(...),
    resolution_data: str = Form(...)
):
    """Analiza con diferentes resoluciones"""
    try:
        res_list = json.loads(resolution_data)
        report = resolution_analysis.analyze_with_resolution(user_id, res_list)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/budget/create-profile")
async def create_budget_profile(
    user_id: str = Form(...),
    monthly_budget: float = Form(...),
    currency: str = Form(...),
    priority_areas: str = Form(...),
    willingness_to_splurge: str = Form(...)
):
    """Crea perfil de presupuesto"""
    try:
        priority_list = json.loads(priority_areas) if priority_areas else []
        profile = monthly_budget_recommendations.create_budget_profile(
            user_id, monthly_budget, currency, priority_list, willingness_to_splurge
        )
        return JSONResponse(content={"success": True, "profile": profile.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/budget/routine/{user_id}")
async def get_budget_routine(user_id: str):
    """Obtiene rutina basada en presupuesto"""
    try:
        routine = monthly_budget_recommendations.generate_budget_routine(user_id)
        return JSONResponse(content={"success": True, "routine": routine.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/sleep/add-record")
async def add_sleep_record(
    user_id: str = Form(...),
    sleep_date: str = Form(...),
    bedtime: str = Form(...),
    wake_time: str = Form(...),
    sleep_duration_hours: float = Form(...),
    sleep_quality: str = Form(...),
    notes: Optional[str] = Form(None)
):
    """Agrega registro de sueño"""
    try:
        record = sleep_habit_tracker.add_sleep_record(
            user_id, sleep_date, bedtime, wake_time,
            sleep_duration_hours, sleep_quality, notes
        )
        return JSONResponse(content={"success": True, "record": record.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/sleep/analyze/{user_id}")
async def analyze_sleep_habits(user_id: str, days: int = Query(30)):
    """Analiza hábitos de sueño"""
    try:
        analysis = sleep_habit_tracker.analyze_sleep_habits(user_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/future-prediction/add-data")
async def add_future_prediction_data(
    user_id: str = Form(...),
    timestamp: str = Form(...),
    metrics: str = Form(...)
):
    """Agrega punto de datos para predicción futura"""
    try:
        metrics_dict = json.loads(metrics)
        future_prediction.add_data_point(user_id, timestamp, metrics_dict)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/future-prediction/generate/{user_id}")
async def generate_future_prediction(user_id: str, days: int = Query(90)):
    """Genera predicción futura"""
    try:
        report = future_prediction.generate_future_prediction(user_id, days)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/format/analyze")
async def analyze_with_format(
    user_id: str = Form(...),
    format_data: str = Form(...)
):
    """Analiza con diferentes formatos"""
    try:
        fmt_list = json.loads(format_data)
        report = format_analysis.analyze_with_format(user_id, fmt_list)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/water/create-profile")
async def create_water_profile(
    user_id: str = Form(...),
    water_type: str = Form(...),
    ph_level: Optional[float] = Form(None),
    mineral_content: str = Form("medium"),
    chlorine_present: bool = Form(False)
):
    """Crea perfil de tipo de agua"""
    try:
        profile = water_recommendations.create_water_profile(
            user_id, water_type, ph_level, mineral_content, chlorine_present
        )
        return JSONResponse(content={"success": True, "profile": profile.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/water/recommendations/{user_id}")
async def get_water_recommendations(user_id: str):
    """Obtiene recomendaciones basadas en tipo de agua"""
    try:
        recommendations = water_recommendations.get_water_recommendations(user_id)
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/stress/add-record")
async def add_stress_record(
    user_id: str = Form(...),
    stress_date: str = Form(...),
    stress_level: int = Form(...),
    stress_source: Optional[str] = Form(None),
    physical_symptoms: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega registro de estrés"""
    try:
        symptoms_list = json.loads(physical_symptoms) if physical_symptoms else None
        record = stress_tracker.add_stress_record(
            user_id, stress_date, stress_level, stress_source, symptoms_list, notes
        )
        return JSONResponse(content={"success": True, "record": record.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/stress/analyze/{user_id}")
async def analyze_stress(user_id: str, days: int = Query(30)):
    """Analiza estrés"""
    try:
        analysis = stress_tracker.analyze_stress(user_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/alerts/add-data")
async def add_alert_data_point(
    user_id: str = Form(...),
    timestamp: str = Form(...),
    metrics: str = Form(...)
):
    """Agrega punto de datos para alertas"""
    try:
        metrics_dict = json.loads(metrics)
        intelligent_alerts.add_data_point(user_id, timestamp, metrics_dict)
        alerts = intelligent_alerts.check_alerts(user_id)
        return JSONResponse(content={
            "success": True,
            "alerts_generated": len(alerts),
            "alerts": [a.to_dict() for a in alerts]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/alerts/{user_id}")
async def get_user_alerts(user_id: str, unread_only: bool = Query(False)):
    """Obtiene alertas del usuario"""
    try:
        alerts = intelligent_alerts.get_user_alerts(user_id, unread_only)
        return JSONResponse(content={
            "success": True,
            "alerts": [a.to_dict() for a in alerts]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/alerts/{user_id}/critical")
async def get_critical_alerts(user_id: str):
    """Obtiene alertas críticas"""
    try:
        alerts = intelligent_alerts.get_critical_alerts(user_id)
        return JSONResponse(content={
            "success": True,
            "alerts": [a.to_dict() for a in alerts]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/device/analyze")
async def analyze_with_device(
    user_id: str = Form(...),
    device_data: str = Form(...)
):
    """Analiza con diferentes dispositivos"""
    try:
        dev_list = json.loads(device_data)
        report = device_analysis.analyze_with_device(user_id, dev_list)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/medication/create-profile")
async def create_medication_profile(
    user_id: str = Form(...),
    medications: str = Form(...),
    supplements: Optional[str] = Form(None),
    skin_related_medications: Optional[str] = Form(None)
):
    """Crea perfil de medicamentos"""
    try:
        med_list = json.loads(medications)
        supp_list = json.loads(supplements) if supplements else None
        skin_list = json.loads(skin_related_medications) if skin_related_medications else None
        profile = medication_recommendations.create_medication_profile(
            user_id, med_list, supp_list, skin_list
        )
        return JSONResponse(content={"success": True, "profile": profile.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/medication/recommendations/{user_id}")
async def get_medication_recommendations(user_id: str):
    """Obtiene recomendaciones basadas en medicamentos"""
    try:
        recommendations = medication_recommendations.get_medication_recommendations(user_id)
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/hormonal/add-record")
async def add_hormonal_record(
    user_id: str = Form(...),
    record_date: str = Form(...),
    hormonal_state: str = Form(...),
    skin_condition: str = Form(...),
    cycle_day: Optional[int] = Form(None),
    symptoms: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega registro hormonal"""
    try:
        symptoms_list = json.loads(symptoms) if symptoms else None
        record = hormonal_tracker.add_hormonal_record(
            user_id, record_date, hormonal_state, skin_condition,
            cycle_day, symptoms_list, notes
        )
        return JSONResponse(content={"success": True, "record": record.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/hormonal/analyze/{user_id}")
async def analyze_hormonal_patterns(user_id: str, days: int = Query(90)):
    """Analiza patrones hormonales"""
    try:
        analysis = hormonal_tracker.analyze_hormonal_patterns(user_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ml-advanced/add-data")
async def add_ml_data_point(
    user_id: str = Form(...),
    timestamp: str = Form(...),
    metrics: str = Form(...),
    context: Optional[str] = Form(None)
):
    """Agrega punto de datos para ML avanzado"""
    try:
        metrics_dict = json.loads(metrics)
        context_dict = json.loads(context) if context else None
        advanced_ml_analysis.add_data_point(user_id, timestamp, metrics_dict, context_dict)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ml-advanced/analyze/{user_id}")
async def generate_advanced_ml_analysis(user_id: str, days: int = Query(90)):
    """Genera análisis con ML avanzado"""
    try:
        report = advanced_ml_analysis.generate_ml_analysis(user_id, days)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/environmental/add-record")
async def add_environmental_record(
    user_id: str = Form(...),
    record_date: str = Form(...),
    location: Optional[str] = Form(None),
    air_quality_index: Optional[int] = Form(None),
    pollution_level: str = Form("unknown"),
    humidity: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
    uv_index: Optional[float] = Form(None),
    pollen_level: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega registro ambiental"""
    try:
        record = environmental_tracker.add_environmental_record(
            user_id, record_date, location, air_quality_index, pollution_level,
            humidity, temperature, uv_index, pollen_level, notes
        )
        return JSONResponse(content={"success": True, "record": record.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/environmental/analyze/{user_id}")
async def analyze_environmental_impact(user_id: str, days: int = Query(30)):
    """Analiza impacto ambiental"""
    try:
        analysis = environmental_tracker.analyze_environmental_impact(user_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/routine/optimize")
async def optimize_routine(
    user_id: str = Form(...),
    products: str = Form(...),
    time_of_day: str = Form(...),
    max_duration: Optional[int] = Form(None)
):
    """Optimiza rutina de skincare"""
    try:
        products_list = json.loads(products)
        routine = routine_optimizer.optimize_routine(user_id, products_list, time_of_day, max_duration)
        return JSONResponse(content={"success": True, "routine": routine.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/concern/add")
async def add_skin_concern(
    user_id: str = Form(...),
    concern_type: str = Form(...),
    severity: str = Form(...),
    location: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega preocupación de la piel"""
    try:
        concern = skin_concern_tracker.add_concern(user_id, concern_type, severity, location, notes)
        return JSONResponse(content={"success": True, "concern": concern.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/concern/update-status")
async def update_concern_status(
    concern_id: str = Form(...),
    user_id: str = Form(...),
    new_status: str = Form(...)
):
    """Actualiza estado de preocupación"""
    try:
        success = skin_concern_tracker.update_concern_status(concern_id, user_id, new_status)
        if not success:
            raise HTTPException(status_code=404, detail="Preocupación no encontrada")
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/concern/analyze/{user_id}")
async def analyze_concerns(user_id: str):
    """Analiza preocupaciones"""
    try:
        analysis = skin_concern_tracker.analyze_concerns(user_id)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/product-effectiveness/add-usage")
async def add_product_usage(
    user_id: str = Form(...),
    product_name: str = Form(...),
    product_category: str = Form(...),
    start_date: str = Form(...),
    frequency: str = Form(...),
    application_area: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega uso de producto"""
    try:
        usage = product_effectiveness_tracker.add_product_usage(
            user_id, product_name, product_category, start_date, frequency,
            application_area, notes
        )
        return JSONResponse(content={"success": True, "usage": usage.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/product-effectiveness/add-rating")
async def add_effectiveness_rating(
    user_id: str = Form(...),
    product_usage_id: str = Form(...),
    rating_date: str = Form(...),
    effectiveness_score: int = Form(...),
    improvement_areas: Optional[str] = Form(None),
    side_effects: Optional[str] = Form(None),
    would_recommend: bool = Form(False),
    notes: Optional[str] = Form(None)
):
    """Agrega calificación de efectividad"""
    try:
        improvement_list = json.loads(improvement_areas) if improvement_areas else None
        effects_list = json.loads(side_effects) if side_effects else None
        rating = product_effectiveness_tracker.add_effectiveness_rating(
            user_id, product_usage_id, rating_date, effectiveness_score,
            improvement_list, effects_list, would_recommend, notes
        )
        return JSONResponse(content={"success": True, "rating": rating.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/product-effectiveness/report/{user_id}/{product_name}")
async def get_effectiveness_report(user_id: str, product_name: str):
    """Obtiene reporte de efectividad de producto"""
    try:
        report = product_effectiveness_tracker.generate_effectiveness_report(user_id, product_name)
        if not report:
            raise HTTPException(status_code=404, detail="Producto no encontrado")
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/natural-lighting/analyze")
async def analyze_with_natural_lighting(
    user_id: str = Form(...),
    lighting_data: str = Form(...)
):
    """Analiza con iluminación natural"""
    try:
        light_list = json.loads(lighting_data)
        report = natural_lighting_analysis.analyze_with_natural_lighting(user_id, light_list)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ethnic-skin/create-profile")
async def create_ethnic_profile(
    user_id: str = Form(...),
    skin_ethnicity: str = Form(...),
    skin_tone: str = Form(...),
    specific_concerns: Optional[str] = Form(None)
):
    """Crea perfil de piel étnica"""
    try:
        concerns_list = json.loads(specific_concerns) if specific_concerns else None
        profile = ethnic_skin_recommendations.create_ethnic_profile(
            user_id, skin_ethnicity, skin_tone, concerns_list
        )
        return JSONResponse(content={"success": True, "profile": profile.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ethnic-skin/recommendations/{user_id}")
async def get_ethnic_recommendations(user_id: str):
    """Obtiene recomendaciones basadas en piel étnica"""
    try:
        recommendations = ethnic_skin_recommendations.get_ethnic_recommendations(user_id)
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/seasonal/add-record")
async def add_seasonal_record(
    user_id: str = Form(...),
    record_date: str = Form(...),
    season: str = Form(...),
    skin_condition: str = Form(...),
    environmental_factors: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega registro estacional"""
    try:
        condition_dict = json.loads(skin_condition)
        env_dict = json.loads(environmental_factors) if environmental_factors else None
        record = seasonal_changes_tracker.add_seasonal_record(
            user_id, record_date, season, condition_dict, env_dict, notes
        )
        return JSONResponse(content={"success": True, "record": record.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/seasonal/analyze/{user_id}")
async def analyze_seasonal_patterns(user_id: str):
    """Analiza patrones estacionales"""
    try:
        analysis = seasonal_changes_tracker.analyze_seasonal_patterns(user_id)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/anonymous-comparison/add-data")
async def add_anonymous_data(
    group_id: str = Form(...),
    metrics: str = Form(...)
):
    """Agrega datos anónimos para comparación"""
    try:
        metrics_dict = json.loads(metrics)
        anonymous_comparison.add_anonymous_data(group_id, metrics_dict)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/anonymous-comparison/generate")
async def generate_anonymous_comparison(
    user_id: str = Form(...),
    user_metrics: str = Form(...),
    user_profile: str = Form(...)
):
    """Genera comparación anónima"""
    try:
        metrics_dict = json.loads(user_metrics)
        profile_dict = json.loads(user_profile)
        report = anonymous_comparison.generate_comparison(user_id, metrics_dict, profile_dict)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/distance/analyze")
async def analyze_with_distance(
    user_id: str = Form(...),
    distance_data: str = Form(...)
):
    """Analiza con diferentes distancias"""
    try:
        dist_list = json.loads(distance_data)
        report = distance_analysis.analyze_with_distance(user_id, dist_list)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/occupation/create-profile")
async def create_occupation_profile(
    user_id: str = Form(...),
    occupation_type: str = Form(...),
    work_hours: str = Form(...),
    exposure_factors: Optional[str] = Form(None)
):
    """Crea perfil de ocupación"""
    try:
        factors_list = json.loads(exposure_factors) if exposure_factors else None
        profile = occupation_recommendations.create_occupation_profile(
            user_id, occupation_type, work_hours, factors_list
        )
        return JSONResponse(content={"success": True, "profile": profile.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/occupation/recommendations/{user_id}")
async def get_occupation_recommendations(user_id: str):
    """Obtiene recomendaciones basadas en ocupación"""
    try:
        recommendations = occupation_recommendations.get_occupation_recommendations(user_id)
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/diet/add-record")
async def add_diet_record(
    user_id: str = Form(...),
    record_date: str = Form(...),
    meal_type: str = Form(...),
    foods: Optional[str] = Form(None),
    water_intake_ml: Optional[int] = Form(None),
    alcohol_consumption: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega registro de dieta"""
    try:
        foods_list = json.loads(foods) if foods else None
        record = diet_tracker.add_diet_record(
            user_id, record_date, meal_type, foods_list,
            water_intake_ml, alcohol_consumption, notes
        )
        return JSONResponse(content={"success": True, "record": record.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/diet/analyze/{user_id}")
async def analyze_diet(user_id: str, days: int = Query(30)):
    """Analiza dieta"""
    try:
        analysis = diet_tracker.analyze_diet(user_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/plateau/add-data")
async def add_plateau_data_point(
    user_id: str = Form(...),
    timestamp: str = Form(...),
    metrics: str = Form(...)
):
    """Agrega punto de datos para detección de mesetas"""
    try:
        metrics_dict = json.loads(metrics)
        plateau_detection.add_data_point(user_id, timestamp, metrics_dict)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/plateau/detect/{user_id}")
async def detect_plateaus(user_id: str, days: int = Query(60)):
    """Detecta mesetas en el progreso"""
    try:
        report = plateau_detection.detect_plateaus(user_id, days)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/trend/add-data")
async def add_trend_data_point(
    user_id: str = Form(...),
    timestamp: str = Form(...),
    metrics: str = Form(...)
):
    """Agrega punto de datos para predicción de tendencias"""
    try:
        metrics_dict = json.loads(metrics)
        trend_prediction.add_data_point(user_id, timestamp, metrics_dict)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/trend/predict/{user_id}")
async def predict_trends(user_id: str, days: int = Query(60)):
    """Predice tendencias futuras"""
    try:
        report = trend_prediction.predict_trends(user_id, days)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/weather/create-profile")
async def create_weather_profile(
    user_id: str = Form(...),
    location: str = Form(...),
    climate_type: str = Form(...),
    average_humidity: Optional[float] = Form(None),
    average_temperature: Optional[float] = Form(None),
    uv_index_range: Optional[str] = Form(None),
    seasonal_variations: bool = Form(True)
):
    """Crea perfil de clima local"""
    try:
        profile = local_weather_recommendations.create_weather_profile(
            user_id, location, climate_type, average_humidity,
            average_temperature, uv_index_range, seasonal_variations
        )
        return JSONResponse(content={"success": True, "profile": profile.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/weather/recommendations/{user_id}")
async def get_weather_recommendations(user_id: str):
    """Obtiene recomendaciones basadas en clima local"""
    try:
        recommendations = local_weather_recommendations.get_weather_recommendations(user_id)
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/custom-routine/create")
async def create_custom_routine(
    user_id: str = Form(...),
    routine_name: str = Form(...),
    steps: str = Form(...)
):
    """Crea rutina personalizada"""
    try:
        steps_list = json.loads(steps)
        routine = custom_routine_tracker.create_routine(user_id, routine_name, steps_list)
        return JSONResponse(content={"success": True, "routine": routine.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/custom-routine/record-usage")
async def record_routine_usage(
    routine_id: str = Form(...),
    user_id: str = Form(...),
    usage_date: str = Form(...),
    steps_completed: str = Form(...),
    notes: Optional[str] = Form(None)
):
    """Registra uso de rutina"""
    try:
        steps_list = json.loads(steps_completed)
        usage = custom_routine_tracker.record_usage(
            routine_id, user_id, usage_date, steps_list, notes
        )
        return JSONResponse(content={"success": True, "usage": usage.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/custom-routine/analyze/{routine_id}")
async def analyze_routine(routine_id: str, user_id: str = Query(...), days: int = Query(30)):
    """Analiza uso de rutina"""
    try:
        analysis = custom_routine_tracker.analyze_routine(routine_id, user_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/custom-routine/user/{user_id}")
async def get_user_routines(user_id: str):
    """Obtiene rutinas del usuario"""
    try:
        routines = custom_routine_tracker.get_user_routines(user_id)
        return JSONResponse(content={
            "success": True,
            "routines": [r.to_dict() for r in routines]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/product-compatibility/register")
async def register_product(
    product_id: str = Form(...),
    name: str = Form(...),
    category: str = Form(...),
    ingredients: str = Form(...),
    ph_level: Optional[float] = Form(None)
):
    """Registra producto para análisis de compatibilidad"""
    try:
        ingredients_list = json.loads(ingredients)
        product = product_compatibility.register_product(
            product_id, name, category, ingredients_list, ph_level
        )
        return JSONResponse(content={"success": True, "product": product.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/product-compatibility/check")
async def check_product_compatibility(
    user_id: str = Form(...),
    product_ids: str = Form(...)
):
    """Verifica compatibilidad de productos"""
    try:
        product_ids_list = json.loads(product_ids)
        report = product_compatibility.check_compatibility(user_id, product_ids_list)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/progress/add-data")
async def add_progress_data(
    user_id: str = Form(...),
    date: str = Form(...),
    metrics: str = Form(...),
    notes: Optional[str] = Form(None)
):
    """Agrega punto de datos de progreso"""
    try:
        metrics_dict = json.loads(metrics)
        progress_visualization.add_data_point(user_id, date, metrics_dict, notes)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/progress/report/{user_id}")
async def get_progress_report(user_id: str, days: int = Query(90)):
    """Genera reporte completo de progreso"""
    try:
        report = progress_visualization.generate_progress_report(user_id, days)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/budget/create-profile")
async def create_budget_profile(
    user_id: str = Form(...),
    monthly_budget: float = Form(...),
    currency: str = Form("USD"),
    priority_areas: Optional[str] = Form(None),
    preferred_brands: Optional[str] = Form(None)
):
    """Crea perfil de presupuesto"""
    try:
        priority_list = json.loads(priority_areas) if priority_areas else None
        brands_list = json.loads(preferred_brands) if preferred_brands else None
        profile = budget_recommendations.create_budget_profile(
            user_id, monthly_budget, currency, priority_list, brands_list
        )
        return JSONResponse(content={"success": True, "profile": profile.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/budget/register-product")
async def register_budget_product(
    product_id: str = Form(...),
    name: str = Form(...),
    category: str = Form(...),
    price: float = Form(...),
    size: str = Form(...),
    estimated_duration_days: int = Form(...),
    value_score: float = Form(0.5)
):
    """Registra producto para recomendaciones de presupuesto"""
    try:
        product = budget_recommendations.register_product(
            product_id, name, category, price, size, estimated_duration_days, value_score
        )
        return JSONResponse(content={"success": True, "product": product.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/budget/generate-routine")
async def generate_budget_routine(
    user_id: str = Form(...),
    required_categories: str = Form(...)
):
    """Genera rutina optimizada para presupuesto"""
    try:
        categories_list = json.loads(required_categories)
        routine = budget_recommendations.generate_budget_routine(user_id, categories_list)
        return JSONResponse(content={"success": True, "routine": routine.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/side-effect/add")
async def add_side_effect(
    user_id: str = Form(...),
    record_date: str = Form(...),
    product_name: str = Form(...),
    product_category: str = Form(...),
    side_effect_type: str = Form(...),
    severity: str = Form(...),
    location: Optional[str] = Form(None),
    duration_hours: Optional[int] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega registro de efecto secundario"""
    try:
        record = side_effect_tracker.add_side_effect(
            user_id, record_date, product_name, product_category,
            side_effect_type, severity, location, duration_hours, notes
        )
        return JSONResponse(content={"success": True, "record": record.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/side-effect/analyze/{user_id}")
async def analyze_side_effects(user_id: str, days: int = Query(90)):
    """Analiza efectos secundarios"""
    try:
        analysis = side_effect_tracker.analyze_side_effects(user_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/texture-ml/analyze")
async def analyze_texture_ml(
    user_id: str = Form(...),
    image_id: str = Form(...),
    image_file: UploadFile = File(...)
):
    """Analiza textura usando ML avanzado"""
    try:
        # Leer imagen
        image_bytes = await image_file.read()
        import cv2
        import numpy as np
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Convertir a escala de grises si es necesario
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2)
        
        analysis = advanced_texture_ml.analyze_texture(user_id, image_id, image_array)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

