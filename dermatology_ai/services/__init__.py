"""
Services for dermatology AI
"""

from .image_processor import ImageProcessor
from .video_processor import VideoProcessor
from .skincare_recommender import SkincareRecommender
from .history_tracker import HistoryTracker, AnalysisRecord
from .report_generator import ReportGenerator
from .visualization import VisualizationGenerator
from .database import DatabaseManager
from .analytics import AnalyticsEngine
from .alert_system import AlertSystem, Alert, AlertLevel
from .product_database import ProductDatabase, Product, ProductCategory
from .body_area_analyzer import BodyAreaAnalyzer, BodyArea
from .export_manager import ExportManager
from .webhook_manager import WebhookManager, Webhook, WebhookEvent
from .auth_manager import AuthManager, User
from .backup_manager import BackupManager
from .notification_service import NotificationService, Notification, NotificationType, NotificationPriority
from .metrics_dashboard import MetricsDashboard
from .performance_optimizer import PerformanceOptimizer, PerformanceMonitor
from .admin_api import router as admin_router
from .report_templates import ReportTemplateEngine, ReportTemplate, TemplateConfig
from .async_queue import AsyncQueue, Task, TaskStatus, TaskPriority
from .integration_service import IntegrationService, IntegrationType, IntegrationConfig
from .event_system import EventSystem, Event, EventType
from .advanced_search import AdvancedSearchEngine, SearchFilter, SearchOperator, SortOption
from .enhanced_export import EnhancedExportManager
from .collaboration_service import CollaborationService, SharedResource, SharePermission
from .tagging_system import TaggingSystem, Tag
from .model_versioning import ModelVersioning, ModelVersion, ModelStatus
from .realtime_metrics import RealtimeMetrics, MetricPoint
from .health_monitor import HealthMonitor, HealthCheck, HealthStatus
from .business_metrics import BusinessMetrics, BusinessMetric
from .batch_processor import BatchProcessor, BatchJob, BatchStatus
from .api_documentation import APIDocumentation, APIEndpoint, APIExample, HTTPMethod
from .security_enhancer import SecurityEnhancer, SecurityCheck, SecurityLevel
from .intelligent_recommender import IntelligentRecommender, Recommendation
from .feedback_system import FeedbackSystem, Feedback, FeedbackType
from .ab_testing import ABTestingSystem, ABTest, Variant
from .personalization_engine import PersonalizationEngine, UserProfile
from .gamification import GamificationSystem, Achievement, UserAchievement, AchievementType
from .challenge_system import ChallengeSystem, Challenge, UserChallenge, ChallengeStatus
from .social_features import SocialFeatures, SocialPost, UserConnection
from .trend_predictor import TrendPredictor, TrendPrediction
from .advanced_comparison import AdvancedComparison, ComparisonResult
from .enhanced_ml import EnhancedMLSystem, MLPrediction, MLModelType
from .iot_integration import IoTIntegration, IoTDevice, DeviceType
from .push_notifications import PushNotificationService, PushNotification, PushPlatform
from .advanced_reporting import AdvancedReporting, ReportConfig, ReportFormat
from .image_analysis_advanced import AdvancedImageAnalysis
from .ml_recommender import MLRecommender, MLRecommendation
from .advanced_monitoring import AdvancedMonitoring, SystemMetric
from .condition_predictor import ConditionPredictor, ConditionPrediction
from .video_analysis_advanced import AdvancedVideoAnalysis
from .learning_system import LearningSystem, LearningInsight
from .progress_analyzer import ProgressAnalyzer, ProgressMetric
from .smart_recommender import SmartRecommender, SmartRecommendation
from .intelligent_alerts import IntelligentAlertSystem, IntelligentAlert, AlertSeverity
from .predictive_analytics import PredictiveAnalytics, PredictiveInsight
from .routine_comparator import RoutineComparator, SkincareRoutine, RoutineComparison
from .product_tracker import ProductTracker, ProductUsage, ProductInsight
from .smart_reminders import SmartReminderSystem, SmartReminder, ReminderType
from .market_trends import MarketTrendsAnalyzer, MarketTrend
from .skin_goals import SkinGoalsManager, SkinGoal, GoalStatus
from .skin_journal import SkinJournal, JournalEntry
from .expert_consultation import ExpertConsultationSystem, ExpertConsultation, ConsultationStatus
from .ingredient_analyzer import IngredientAnalyzer, Ingredient, ProductIngredientAnalysis
from .custom_recipes import CustomRecipesManager, CustomRecipe
from .product_comparison import ProductComparisonSystem, ProductComparison
from .reviews_ratings import ReviewsRatingsSystem, Review, ProductRating
from .before_after_analysis import BeforeAfterAnalysis, BeforeAfterComparison
from .budget_tracker import BudgetTracker, BudgetEntry, BudgetSummary
from .community_features import CommunityFeatures, CommunityPost, PostComment
from .wearable_integration import WearableIntegration, WearableData, WearableInsight
from .weather_climate_analysis import WeatherClimateAnalysis, WeatherData, ClimateRecommendation
from .enhanced_notifications import EnhancedNotificationSystem, EnhancedNotification, NotificationType
from .ai_photo_analysis import AIPhotoAnalysisSystem, AIPhotoAnalysis
from .seasonal_recommendations import SeasonalRecommendationsSystem, SeasonalRecommendation, Season
from .allergy_tracker import AllergyTracker, AllergyRecord, AllergyProfile
from .advanced_texture_analysis import AdvancedTextureAnalysis, TextureAnalysis
from .product_needs_predictor import ProductNeedsPredictor, ProductNeed
from .habit_analyzer import HabitAnalyzer, HabitAnalysis, HabitPattern
from .personalized_coaching import PersonalizedCoaching, CoachingSession
from .medical_treatment_tracker import MedicalTreatmentTracker, MedicalTreatment, TreatmentProgress
from .visual_progress_tracker import VisualProgressTracker, VisualProgressEntry, ProgressTimeline
from .pharmacy_integration import PharmacyIntegration, Pharmacy, ProductAvailability
from .product_reminder_system import ProductReminderSystem, ProductReminder
from .ingredient_conflict_checker import IngredientConflictChecker, IngredientConflict, ProductCompatibilityCheck
from .comparative_analysis import ComparativeAnalysisSystem, ComparativeAnalysis, ComparativeInsight
from .budget_based_recommendations import BudgetBasedRecommendations, BudgetRecommendation
from .historical_photo_analysis import HistoricalPhotoAnalysis, HistoricalPhoto, PhotoTimeline
from .product_trend_analyzer import ProductTrendAnalyzer, ProductTrend, CategoryTrend
from .age_analysis import AgeAnalysisSystem, AgeAnalysis
from .age_based_recommendations import AgeBasedRecommendations, AgeBasedRecommendation
from .multi_condition_analyzer import MultiConditionAnalyzer, MultiConditionReport, ConditionAnalysis
from .successful_routines import SuccessfulRoutinesSystem, SuccessfulRoutine, RoutineMatch
from .multi_angle_analysis import MultiAngleAnalysis, MultiAngleReport, AngleAnalysis
from .lifestyle_recommendations import LifestyleRecommendations, LifestyleProfile, LifestyleRecommendation
from .medical_device_integration import MedicalDeviceIntegration, MedicalDevice, DeviceReading
from .benchmark_analysis import BenchmarkAnalysis, BenchmarkReport, Benchmark
from .lighting_analysis import LightingAnalysisSystem, ComprehensiveLightingReport, LightingAnalysis
from .genetic_recommendations import GeneticRecommendations, GeneticProfile, GeneticRecommendation
from .professional_treatment_tracker import ProfessionalTreatmentTracker, ProfessionalTreatment, TreatmentSeries
from .ai_progress_analysis import AIProgressAnalysis, AIProgressReport, AIProgressInsight
from .climate_condition_analysis import ClimateAnalysisSystem, ClimateAnalysisReport, ClimateConditionData
from .time_based_recommendations import TimeBasedRecommendations, TimeBasedRecommendation, TimeBasedRoutine
from .skin_state_analysis import SkinStateAnalysisSystem, SkinStateReport, SkinStateData
from .fitness_based_recommendations import FitnessBasedRecommendations, FitnessProfile, FitnessRecommendation
from .supplement_tracker import SupplementTracker, Supplement, SupplementReport
from .temporal_comparison import TemporalComparisonSystem, TemporalComparisonReport, TemporalComparison as TemporalComparisonData
from .resolution_analysis import ResolutionAnalysisSystem, ResolutionReport, ResolutionData
from .monthly_budget_recommendations import MonthlyBudgetRecommendations, BudgetProfile, BudgetRoutine, BudgetRecommendation
from .sleep_habit_tracker import SleepHabitTracker, SleepRecord, SleepAnalysis
from .future_prediction import FuturePredictionSystem, FuturePredictionReport, FuturePrediction as FuturePredictionData
from .format_analysis import FormatAnalysisSystem, FormatReport, FormatAnalysis
from .water_type_recommendations import WaterTypeRecommendations, WaterProfile, WaterRecommendation
from .stress_tracker import StressTracker, StressRecord, StressAnalysis
from .intelligent_alerts import IntelligentAlerts, Alert
from .device_analysis import DeviceAnalysisSystem, DeviceReport, DeviceAnalysis
from .medication_recommendations import MedicationRecommendations, MedicationProfile, MedicationRecommendation
from .hormonal_tracker import HormonalTracker, HormonalRecord, HormonalAnalysis
from .advanced_ml_analysis import AdvancedMLAnalysis, AdvancedMLReport, MLInsight
from .environmental_tracker import EnvironmentalTracker, EnvironmentalRecord, EnvironmentalAnalysis
from .routine_optimizer import RoutineOptimizer, OptimizedRoutine, RoutineStep
from .skin_concern_tracker import SkinConcernTracker, SkinConcern, ConcernAnalysis
from .product_effectiveness_tracker import ProductEffectivenessTracker, ProductUsage, EffectivenessRating, ProductEffectivenessReport
from .natural_lighting_analysis import NaturalLightingAnalysis, NaturalLightingReport, NaturalLightingData
from .ethnic_skin_recommendations import EthnicSkinRecommendations, EthnicSkinProfile, EthnicSkinRecommendation
from .seasonal_changes_tracker import SeasonalChangesTracker, SeasonalRecord, SeasonalAnalysis
from .anonymous_comparison import AnonymousComparison, AnonymousComparisonReport, ComparisonGroup
from .distance_analysis import DistanceAnalysisSystem, DistanceReport, DistanceAnalysis
from .occupation_recommendations import OccupationRecommendations, OccupationProfile, OccupationRecommendation
from .diet_tracker import DietTracker, DietRecord, DietAnalysis
from .plateau_detection import PlateauDetectionSystem, PlateauReport, PlateauData
from .trend_prediction import TrendPredictionSystem, TrendReport, TrendPrediction
from .local_weather_recommendations import LocalWeatherRecommendations, WeatherProfile, WeatherRecommendation
from .custom_routine_tracker import CustomRoutineTracker, CustomRoutine, RoutineStep, RoutineUsage, RoutineAnalysis
from .product_compatibility import ProductCompatibility, Product, ProductIngredient, CompatibilityReport, CompatibilityIssue
from .progress_visualization import ProgressVisualizationSystem, ComprehensiveProgressReport, ProgressVisualization, ProgressDataPoint
from .budget_recommendations import BudgetRecommendations, BudgetProfile, BudgetProduct, BudgetRoutine
from .side_effect_tracker import SideEffectTracker, SideEffectRecord, SideEffectAnalysis
from .advanced_texture_ml import AdvancedTextureML, MLTextureAnalysis, TextureFeature

__all__ = [
    "ImageProcessor",
    "VideoProcessor",
    "SkincareRecommender",
    "HistoryTracker",
    "AnalysisRecord",
    "ReportGenerator",
    "VisualizationGenerator",
    "DatabaseManager",
    "AnalyticsEngine",
    "AlertSystem",
    "Alert",
    "AlertLevel",
    "ProductDatabase",
    "Product",
    "ProductCategory",
    "BodyAreaAnalyzer",
    "BodyArea",
    "ExportManager",
    "WebhookManager",
    "Webhook",
    "WebhookEvent",
    "AuthManager",
    "User",
    "BackupManager",
    "NotificationService",
    "Notification",
    "NotificationType",
    "NotificationPriority",
    "MetricsDashboard",
    "PerformanceOptimizer",
    "PerformanceMonitor",
    "admin_router",
    "ReportTemplateEngine",
    "ReportTemplate",
    "TemplateConfig",
    "AsyncQueue",
    "Task",
    "TaskStatus",
    "TaskPriority",
    "IntegrationService",
    "IntegrationType",
    "IntegrationConfig",
    "EventSystem",
    "Event",
    "EventType",
    "AdvancedSearchEngine",
    "SearchFilter",
    "SearchOperator",
    "SortOption",
    "EnhancedExportManager",
    "CollaborationService",
    "SharedResource",
    "SharePermission",
    "TaggingSystem",
    "Tag",
    "ModelVersioning",
    "ModelVersion",
    "ModelStatus",
    "RealtimeMetrics",
    "MetricPoint",
    "HealthMonitor",
    "HealthCheck",
    "HealthStatus",
    "BusinessMetrics",
    "BusinessMetric",
    "BatchProcessor",
    "BatchJob",
    "BatchStatus",
    "APIDocumentation",
    "APIEndpoint",
    "APIExample",
    "HTTPMethod",
    "SecurityEnhancer",
    "SecurityCheck",
    "SecurityLevel",
    "IntelligentRecommender",
    "Recommendation",
    "FeedbackSystem",
    "Feedback",
    "FeedbackType",
    "ABTestingSystem",
    "ABTest",
    "Variant",
    "PersonalizationEngine",
    "UserProfile",
    "GamificationSystem",
    "Achievement",
    "UserAchievement",
    "AchievementType",
    "ChallengeSystem",
    "Challenge",
    "UserChallenge",
    "ChallengeStatus",
    "SocialFeatures",
    "SocialPost",
    "UserConnection",
    "TrendPredictor",
    "TrendPrediction",
    "AdvancedComparison",
    "ComparisonResult",
    "EnhancedMLSystem",
    "MLPrediction",
    "MLModelType",
    "IoTIntegration",
    "IoTDevice",
    "DeviceType",
    "PushNotificationService",
    "PushNotification",
    "PushPlatform",
    "AdvancedReporting",
    "ReportConfig",
    "ReportFormat",
    "AdvancedImageAnalysis",
    "MLRecommender",
    "MLRecommendation",
    "AdvancedMonitoring",
    "SystemMetric",
    "ConditionPredictor",
    "ConditionPrediction",
    "AdvancedVideoAnalysis",
    "LearningSystem",
    "LearningInsight",
    "ProgressAnalyzer",
    "ProgressMetric",
    "SmartRecommender",
    "SmartRecommendation",
    "IntelligentAlertSystem",
    "IntelligentAlert",
    "AlertSeverity",
    "PredictiveAnalytics",
    "PredictiveInsight",
    "RoutineComparator",
    "SkincareRoutine",
    "RoutineComparison",
    "ProductTracker",
    "ProductUsage",
    "ProductInsight",
    "SmartReminderSystem",
    "SmartReminder",
    "ReminderType",
    "MarketTrendsAnalyzer",
    "MarketTrend",
    "SkinGoalsManager",
    "SkinGoal",
    "GoalStatus",
    "SkinJournal",
    "JournalEntry",
    "ExpertConsultationSystem",
    "ExpertConsultation",
    "ConsultationStatus",
    "IngredientAnalyzer",
    "Ingredient",
    "ProductIngredientAnalysis",
    "CustomRecipesManager",
    "CustomRecipe",
    "ProductComparisonSystem",
    "ProductComparison",
    "ReviewsRatingsSystem",
    "Review",
    "ProductRating",
    "BeforeAfterAnalysis",
    "BeforeAfterComparison",
    "BudgetTracker",
    "BudgetEntry",
    "BudgetSummary",
    "CommunityFeatures",
    "CommunityPost",
    "PostComment",
    "WearableIntegration",
    "WearableData",
    "WearableInsight",
    "WeatherClimateAnalysis",
    "WeatherData",
    "ClimateRecommendation",
    "EnhancedNotificationSystem",
    "EnhancedNotification",
    "NotificationType",
    "AIPhotoAnalysisSystem",
    "AIPhotoAnalysis",
    "SeasonalRecommendationsSystem",
    "SeasonalRecommendation",
    "Season",
    "AllergyTracker",
    "AllergyRecord",
    "AllergyProfile",
    "AdvancedTextureAnalysis",
    "TextureAnalysis",
    "ProductNeedsPredictor",
    "ProductNeed",
    "HabitAnalyzer",
    "HabitAnalysis",
    "HabitPattern",
    "PersonalizedCoaching",
    "CoachingSession",
    "MedicalTreatmentTracker",
    "MedicalTreatment",
    "TreatmentProgress",
    "VisualProgressTracker",
    "VisualProgressEntry",
    "ProgressTimeline",
    "PharmacyIntegration",
    "Pharmacy",
    "ProductAvailability",
    "ProductReminderSystem",
    "ProductReminder",
    "IngredientConflictChecker",
    "IngredientConflict",
    "ProductCompatibilityCheck",
    "ComparativeAnalysisSystem",
    "ComparativeAnalysis",
    "ComparativeInsight",
    "BudgetBasedRecommendations",
    "BudgetRecommendation",
    "HistoricalPhotoAnalysis",
    "HistoricalPhoto",
    "PhotoTimeline",
    "ProductTrendAnalyzer",
    "ProductTrend",
    "CategoryTrend",
    "AgeAnalysisSystem",
    "AgeAnalysis",
    "AgeBasedRecommendations",
    "AgeBasedRecommendation",
    "MultiConditionAnalyzer",
    "MultiConditionReport",
    "ConditionAnalysis",
    "SuccessfulRoutinesSystem",
    "SuccessfulRoutine",
    "RoutineMatch",
    "MultiAngleAnalysis",
    "MultiAngleReport",
    "AngleAnalysis",
    "LifestyleRecommendations",
    "LifestyleProfile",
    "LifestyleRecommendation",
    "MedicalDeviceIntegration",
    "MedicalDevice",
    "DeviceReading",
    "BenchmarkAnalysis",
    "BenchmarkReport",
    "Benchmark",
    "LightingAnalysisSystem",
    "ComprehensiveLightingReport",
    "LightingAnalysis",
    "GeneticRecommendations",
    "GeneticProfile",
    "GeneticRecommendation",
    "ProfessionalTreatmentTracker",
    "ProfessionalTreatment",
    "TreatmentSeries",
    "AIProgressAnalysis",
    "AIProgressReport",
    "AIProgressInsight",
    "ClimateAnalysisSystem",
    "ClimateAnalysisReport",
    "ClimateConditionData",
    "TimeBasedRecommendations",
    "TimeBasedRecommendation",
    "TimeBasedRoutine",
    "AdvancedTextureML",
    "MLTextureAnalysis",
    "TextureFeature",
    "SkinStateAnalysisSystem",
    "SkinStateReport",
    "SkinStateData",
    "FitnessBasedRecommendations",
    "FitnessProfile",
    "FitnessRecommendation",
    "SupplementTracker",
    "Supplement",
    "SupplementReport",
    "TemporalComparisonSystem",
    "TemporalComparisonReport",
    "TemporalComparisonData",
    "ResolutionAnalysisSystem",
    "ResolutionReport",
    "ResolutionData",
    "MonthlyBudgetRecommendations",
    "BudgetProfile",
    "BudgetRoutine",
    "BudgetRecommendation",
    "SleepHabitTracker",
    "SleepRecord",
    "SleepAnalysis",
    "FuturePredictionSystem",
    "FuturePredictionReport",
    "FuturePredictionData",
    "FormatAnalysisSystem",
    "FormatReport",
    "FormatAnalysis",
    "WaterTypeRecommendations",
    "WaterProfile",
    "WaterRecommendation",
    "StressTracker",
    "StressRecord",
    "StressAnalysis",
    "IntelligentAlerts",
    "Alert",
    "DeviceAnalysisSystem",
    "DeviceReport",
    "DeviceAnalysis",
    "MedicationRecommendations",
    "MedicationProfile",
    "MedicationRecommendation",
    "HormonalTracker",
    "HormonalRecord",
    "HormonalAnalysis",
    "AdvancedMLAnalysis",
    "AdvancedMLReport",
    "MLInsight",
    "EnvironmentalTracker",
    "EnvironmentalRecord",
    "EnvironmentalAnalysis",
    "RoutineOptimizer",
    "OptimizedRoutine",
    "RoutineStep",
    "SkinConcernTracker",
    "SkinConcern",
    "ConcernAnalysis",
    "ProductEffectivenessTracker",
    "ProductUsage",
    "EffectivenessRating",
    "ProductEffectivenessReport",
    "NaturalLightingAnalysis",
    "NaturalLightingReport",
    "NaturalLightingData",
    "EthnicSkinRecommendations",
    "EthnicSkinProfile",
    "EthnicSkinRecommendation",
    "SeasonalChangesTracker",
    "SeasonalRecord",
    "SeasonalAnalysis",
    "AnonymousComparison",
    "AnonymousComparisonReport",
    "ComparisonGroup",
    "DistanceAnalysisSystem",
    "DistanceReport",
    "DistanceAnalysis",
    "OccupationRecommendations",
    "OccupationProfile",
    "OccupationRecommendation",
    "DietTracker",
    "DietRecord",
    "DietAnalysis",
    "PlateauDetectionSystem",
    "PlateauReport",
    "PlateauData",
    "TrendPredictionSystem",
    "TrendReport",
    "TrendPrediction",
    "LocalWeatherRecommendations",
    "WeatherProfile",
    "WeatherRecommendation",
    "CustomRoutineTracker",
    "CustomRoutine",
    "RoutineStep",
    "RoutineUsage",
    "RoutineAnalysis",
    "ProductCompatibility",
    "Product",
    "ProductIngredient",
    "CompatibilityReport",
    "CompatibilityIssue",
    "ProgressVisualizationSystem",
    "ComprehensiveProgressReport",
    "ProgressVisualization",
    "ProgressDataPoint",
    "BudgetRecommendations",
    "BudgetProfile",
    "BudgetProduct",
    "BudgetRoutine",
    "SideEffectTracker",
    "SideEffectRecord",
    "SideEffectAnalysis",
]

