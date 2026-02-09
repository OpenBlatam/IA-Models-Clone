"""
Service definitions for the dermatology API
Centralized configuration for all services
"""

from typing import Dict, Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_service_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Get all service definitions with their factory functions and dependencies
    
    Returns:
        Dictionary mapping service names to their configuration
    """
    return {
        "skin_analyzer": {
            "factory": lambda: _import_and_create("core.skin_analyzer", "SkinAnalyzer", use_advanced=True, use_cache=True),
            "group": "core"
        },
        "image_processor": {
            "factory": lambda: _import_and_create("services.image_processor", "ImageProcessor"),
            "group": "core"
        },
        "video_processor": {
            "factory": lambda: _import_and_create("services.video_processor", "VideoProcessor"),
            "group": "core"
        },
        "skincare_recommender": {
            "factory": lambda: _import_and_create("services.skincare_recommender", "SkincareRecommender"),
            "group": "recommendations"
        },
        "history_tracker": {
            "factory": lambda: _import_and_create("services.history_tracker", "HistoryTracker"),
            "group": "tracking"
        },
        "report_generator": {
            "factory": lambda: _import_and_create("services.report_generator", "ReportGenerator"),
            "group": "reporting"
        },
        "visualization_generator": {
            "factory": lambda: _import_and_create("services.visualization", "VisualizationGenerator"),
            "group": "reporting"
        },
        "db_manager": {
            "factory": lambda: _import_and_create("services.database", "DatabaseManager"),
            "group": "infrastructure"
        },
        "analytics_engine": {
            "factory": lambda: _create_analytics_engine(),
            "group": "analytics"
        },
        "alert_system": {
            "factory": lambda: _import_and_create("services.alert_system", "AlertSystem"),
            "group": "notifications"
        },
        "product_database": {
            "factory": lambda: _import_and_create("services.product_database", "ProductDatabase"),
            "group": "products"
        },
        "body_area_analyzer": {
            "factory": lambda: _import_and_create("services.body_area_analyzer", "BodyAreaAnalyzer"),
            "group": "analysis"
        },
        "export_manager": {
            "factory": lambda: _import_and_create("services.export_manager", "ExportManager"),
            "group": "export"
        },
        "webhook_manager": {
            "factory": lambda: _import_and_create("services.webhook_manager", "WebhookManager"),
            "group": "integrations"
        },
        "auth_manager": {
            "factory": lambda: _import_and_create("services.auth_manager", "AuthManager"),
            "group": "security"
        },
        "backup_manager": {
            "factory": lambda: _import_and_create("services.backup_manager", "BackupManager"),
            "group": "infrastructure"
        },
        "notification_service": {
            "factory": lambda: _import_and_create("services.notification_service", "NotificationService"),
            "group": "notifications"
        },
        "advanced_validator": {
            "factory": lambda: _import_and_create("utils.advanced_validator", "AdvancedImageValidator"),
            "group": "validation"
        },
        "metrics_dashboard": {
            "factory": lambda: _create_metrics_dashboard(),
            "group": "analytics"
        },
        "template_engine": {
            "factory": lambda: _import_and_create("services.report_templates", "ReportTemplateEngine"),
            "group": "reporting"
        },
        "async_queue": {
            "factory": lambda: _import_and_create("services.async_queue", "AsyncQueue", max_workers=5),
            "group": "infrastructure"
        },
        "integration_service": {
            "factory": lambda: _import_and_create("services.integration_service", "IntegrationService"),
            "group": "integrations"
        },
        "event_system": {
            "factory": lambda: _import_and_create("services.event_system", "EventSystem"),
            "group": "infrastructure"
        },
        "search_engine": {
            "factory": lambda: _import_and_create("services.advanced_search", "AdvancedSearchEngine"),
            "group": "search"
        },
        "enhanced_export": {
            "factory": lambda: _import_and_create("services.enhanced_export", "EnhancedExportManager"),
            "group": "export"
        },
        "collaboration_service": {
            "factory": lambda: _import_and_create("services.collaboration_service", "CollaborationService"),
            "group": "social"
        },
        "tagging_system": {
            "factory": lambda: _import_and_create("services.tagging_system", "TaggingSystem"),
            "group": "organization"
        },
        "model_versioning": {
            "factory": lambda: _import_and_create("services.model_versioning", "ModelVersioning"),
            "group": "ml"
        },
        "realtime_metrics": {
            "factory": lambda: _import_and_create("services.realtime_metrics", "RealtimeMetrics"),
            "group": "analytics"
        },
        "health_monitor": {
            "factory": lambda: _import_and_create("services.health_monitor", "HealthMonitor"),
            "group": "monitoring"
        },
        "business_metrics": {
            "factory": lambda: _import_and_create("services.business_metrics", "BusinessMetrics"),
            "group": "analytics"
        },
        "batch_processor": {
            "factory": lambda: _import_and_create("services.batch_processor", "BatchProcessor", max_workers=5),
            "group": "infrastructure"
        },
        "api_documentation": {
            "factory": lambda: _import_and_create("services.api_documentation", "APIDocumentation"),
            "group": "documentation"
        },
        "security_enhancer": {
            "factory": lambda: _import_and_create("services.security_enhancer", "SecurityEnhancer"),
            "group": "security"
        },
        "intelligent_recommender": {
            "factory": lambda: _import_and_create("services.intelligent_recommender", "IntelligentRecommender"),
            "group": "recommendations"
        },
        "feedback_system": {
            "factory": lambda: _import_and_create("services.feedback_system", "FeedbackSystem"),
            "group": "social"
        },
        "ab_testing": {
            "factory": lambda: _import_and_create("services.ab_testing", "ABTestingSystem"),
            "group": "analytics"
        },
        "personalization_engine": {
            "factory": lambda: _import_and_create("services.personalization_engine", "PersonalizationEngine"),
            "group": "recommendations"
        },
        "gamification": {
            "factory": lambda: _import_and_create("services.gamification", "GamificationSystem"),
            "group": "social"
        },
        "challenge_system": {
            "factory": lambda: _import_and_create("services.challenge_system", "ChallengeSystem"),
            "group": "social"
        },
        "social_features": {
            "factory": lambda: _import_and_create("services.social_features", "SocialFeatures"),
            "group": "social"
        },
        "trend_predictor": {
            "factory": lambda: _import_and_create("services.trend_predictor", "TrendPredictor"),
            "group": "ml"
        },
        "advanced_comparison": {
            "factory": lambda: _import_and_create("services.advanced_comparison", "AdvancedComparison"),
            "group": "analysis"
        },
        "enhanced_ml": {
            "factory": lambda: _import_and_create("services.enhanced_ml", "EnhancedMLSystem"),
            "group": "ml"
        },
        "iot_integration": {
            "factory": lambda: _import_and_create("services.iot_integration", "IoTIntegration"),
            "group": "integrations"
        },
        "push_notifications": {
            "factory": lambda: _import_and_create("services.push_notifications", "PushNotificationService"),
            "group": "notifications"
        },
        "advanced_reporting": {
            "factory": lambda: _import_and_create("services.advanced_reporting", "AdvancedReporting"),
            "group": "reporting"
        },
        "image_analysis_advanced": {
            "factory": lambda: _import_and_create("services.image_analysis_advanced", "AdvancedImageAnalysis"),
            "group": "analysis"
        },
        "ml_recommender": {
            "factory": lambda: _import_and_create("services.ml_recommender", "MLRecommender"),
            "group": "recommendations"
        },
        "advanced_monitoring": {
            "factory": lambda: _import_and_create("services.advanced_monitoring", "AdvancedMonitoring"),
            "group": "monitoring"
        },
        "condition_predictor": {
            "factory": lambda: _import_and_create("services.condition_predictor", "ConditionPredictor"),
            "group": "ml"
        },
        "video_analysis_advanced": {
            "factory": lambda: _import_and_create("services.video_analysis_advanced", "AdvancedVideoAnalysis"),
            "group": "analysis"
        },
        "learning_system": {
            "factory": lambda: _import_and_create("services.learning_system", "LearningSystem"),
            "group": "ml"
        },
        "progress_analyzer": {
            "factory": lambda: _import_and_create("services.progress_analyzer", "ProgressAnalyzer"),
            "group": "analysis"
        },
        "smart_recommender": {
            "factory": lambda: _import_and_create("services.smart_recommender", "SmartRecommender"),
            "group": "recommendations"
        },
        "intelligent_alerts": {
            "factory": lambda: _import_and_create("services.intelligent_alerts", "IntelligentAlertSystem"),
            "group": "notifications"
        },
        "predictive_analytics": {
            "factory": lambda: _import_and_create("services.predictive_analytics", "PredictiveAnalytics"),
            "group": "analytics"
        },
        "routine_comparator": {
            "factory": lambda: _import_and_create("services.routine_comparator", "RoutineComparator"),
            "group": "recommendations"
        },
        "product_tracker": {
            "factory": lambda: _import_and_create("services.product_tracker", "ProductTracker"),
            "group": "products"
        },
        "smart_reminders": {
            "factory": lambda: _import_and_create("services.smart_reminders", "SmartReminderSystem"),
            "group": "notifications"
        },
        "market_trends": {
            "factory": lambda: _import_and_create("services.market_trends", "MarketTrendsAnalyzer"),
            "group": "analytics"
        },
        "skin_goals": {
            "factory": lambda: _import_and_create("services.skin_goals", "SkinGoalsManager"),
            "group": "tracking"
        },
        "skin_journal": {
            "factory": lambda: _import_and_create("services.skin_journal", "SkinJournal"),
            "group": "tracking"
        },
        "expert_consultation": {
            "factory": lambda: _import_and_create("services.expert_consultation", "ExpertConsultationSystem"),
            "group": "social"
        },
        "ingredient_analyzer": {
            "factory": lambda: _import_and_create("services.ingredient_analyzer", "IngredientAnalyzer"),
            "group": "products"
        },
        "custom_recipes": {
            "factory": lambda: _import_and_create("services.custom_recipes", "CustomRecipesManager"),
            "group": "products"
        },
        "product_comparison": {
            "factory": lambda: _import_and_create("services.product_comparison", "ProductComparisonSystem"),
            "group": "products"
        },
        "reviews_ratings": {
            "factory": lambda: _import_and_create("services.reviews_ratings", "ReviewsRatingsSystem"),
            "group": "social"
        },
        "before_after_analysis": {
            "factory": lambda: _import_and_create("services.before_after_analysis", "BeforeAfterAnalysis"),
            "group": "analysis"
        },
        "budget_tracker": {
            "factory": lambda: _import_and_create("services.budget_tracker", "BudgetTracker"),
            "group": "tracking"
        },
        "community_features": {
            "factory": lambda: _import_and_create("services.community_features", "CommunityFeatures"),
            "group": "social"
        },
        "wearable_integration": {
            "factory": lambda: _import_and_create("services.wearable_integration", "WearableIntegration"),
            "group": "integrations"
        },
        "weather_climate": {
            "factory": lambda: _import_and_create("services.weather_climate_analysis", "WeatherClimateAnalysis"),
            "group": "analysis"
        },
        "enhanced_notifications": {
            "factory": lambda: _import_and_create("services.enhanced_notifications", "EnhancedNotificationSystem"),
            "group": "notifications"
        },
        "ai_photo_analysis": {
            "factory": lambda: _import_and_create("services.ai_photo_analysis", "AIPhotoAnalysisSystem"),
            "group": "analysis"
        },
        "seasonal_recommendations": {
            "factory": lambda: _import_and_create("services.seasonal_recommendations", "SeasonalRecommendationsSystem"),
            "group": "recommendations"
        },
        "allergy_tracker": {
            "factory": lambda: _import_and_create("services.allergy_tracker", "AllergyTracker"),
            "group": "tracking"
        },
        "advanced_texture_analysis": {
            "factory": lambda: _import_and_create("services.advanced_texture_analysis", "AdvancedTextureAnalysis"),
            "group": "analysis"
        },
        "product_needs_predictor": {
            "factory": lambda: _import_and_create("services.product_needs_predictor", "ProductNeedsPredictor"),
            "group": "recommendations"
        },
        "habit_analyzer": {
            "factory": lambda: _import_and_create("services.habit_analyzer", "HabitAnalyzer"),
            "group": "tracking"
        },
        "personalized_coaching": {
            "factory": lambda: _import_and_create("services.personalized_coaching", "PersonalizedCoaching"),
            "group": "recommendations"
        },
        "medical_treatment_tracker": {
            "factory": lambda: _import_and_create("services.medical_treatment_tracker", "MedicalTreatmentTracker"),
            "group": "tracking"
        },
        "visual_progress_tracker": {
            "factory": lambda: _import_and_create("services.visual_progress_tracker", "VisualProgressTracker"),
            "group": "tracking"
        },
        "pharmacy_integration": {
            "factory": lambda: _import_and_create("services.pharmacy_integration", "PharmacyIntegration"),
            "group": "integrations"
        },
        "product_reminder_system": {
            "factory": lambda: _import_and_create("services.product_reminder_system", "ProductReminderSystem"),
            "group": "notifications"
        },
        "ingredient_conflict_checker": {
            "factory": lambda: _import_and_create("services.ingredient_conflict_checker", "IngredientConflictChecker"),
            "group": "products"
        },
        "comparative_analysis": {
            "factory": lambda: _import_and_create("services.comparative_analysis", "ComparativeAnalysisSystem"),
            "group": "analysis"
        },
        "budget_recommendations": {
            "factory": lambda: _import_and_create("services.budget_based_recommendations", "BudgetBasedRecommendations"),
            "group": "recommendations"
        },
        "historical_photo_analysis": {
            "factory": lambda: _import_and_create("services.historical_photo_analysis", "HistoricalPhotoAnalysis"),
            "group": "analysis"
        },
        "product_trend_analyzer": {
            "factory": lambda: _import_and_create("services.product_trend_analyzer", "ProductTrendAnalyzer"),
            "group": "analytics"
        },
        "age_analysis": {
            "factory": lambda: _import_and_create("services.age_analysis", "AgeAnalysisSystem"),
            "group": "analysis"
        },
        "age_recommendations": {
            "factory": lambda: _import_and_create("services.age_based_recommendations", "AgeBasedRecommendations"),
            "group": "recommendations"
        },
        "multi_condition_analyzer": {
            "factory": lambda: _import_and_create("services.multi_condition_analyzer", "MultiConditionAnalyzer"),
            "group": "analysis"
        },
        "successful_routines": {
            "factory": lambda: _import_and_create("services.successful_routines", "SuccessfulRoutinesSystem"),
            "group": "recommendations"
        },
        "multi_angle_analysis": {
            "factory": lambda: _import_and_create("services.multi_angle_analysis", "MultiAngleAnalysis"),
            "group": "analysis"
        },
        "lifestyle_recommendations": {
            "factory": lambda: _import_and_create("services.lifestyle_recommendations", "LifestyleRecommendations"),
            "group": "recommendations"
        },
        "medical_device_integration": {
            "factory": lambda: _import_and_create("services.medical_device_integration", "MedicalDeviceIntegration"),
            "group": "integrations"
        },
        "benchmark_analysis": {
            "factory": lambda: _import_and_create("services.benchmark_analysis", "BenchmarkAnalysis"),
            "group": "analysis"
        },
        "lighting_analysis": {
            "factory": lambda: _import_and_create("services.lighting_analysis", "LightingAnalysisSystem"),
            "group": "analysis"
        },
        "genetic_recommendations": {
            "factory": lambda: _import_and_create("services.genetic_recommendations", "GeneticRecommendations"),
            "group": "recommendations"
        },
        "professional_treatment_tracker": {
            "factory": lambda: _import_and_create("services.professional_treatment_tracker", "ProfessionalTreatmentTracker"),
            "group": "tracking"
        },
        "ai_progress_analysis": {
            "factory": lambda: _import_and_create("services.ai_progress_analysis", "AIProgressAnalysis"),
            "group": "analysis"
        },
        "climate_analysis": {
            "factory": lambda: _import_and_create("services.climate_condition_analysis", "ClimateAnalysisSystem"),
            "group": "analysis"
        },
        "time_based_recommendations": {
            "factory": lambda: _import_and_create("services.time_based_recommendations", "TimeBasedRecommendations"),
            "group": "recommendations"
        },
        "side_effect_tracker": {
            "factory": lambda: _import_and_create("services.side_effect_tracker", "SideEffectTracker"),
            "group": "tracking"
        },
        "advanced_texture_ml": {
            "factory": lambda: _import_and_create("services.advanced_texture_ml", "AdvancedTextureML"),
            "group": "ml"
        },
        "skin_state_analysis": {
            "factory": lambda: _import_and_create("services.skin_state_analysis", "SkinStateAnalysisSystem"),
            "group": "analysis"
        },
        "fitness_recommendations": {
            "factory": lambda: _import_and_create("services.fitness_based_recommendations", "FitnessBasedRecommendations"),
            "group": "recommendations"
        },
        "supplement_tracker": {
            "factory": lambda: _import_and_create("services.supplement_tracker", "SupplementTracker"),
            "group": "tracking"
        },
        "temporal_comparison": {
            "factory": lambda: _import_and_create("services.temporal_comparison", "TemporalComparisonSystem"),
            "group": "analysis"
        },
        "resolution_analysis": {
            "factory": lambda: _import_and_create("services.resolution_analysis", "ResolutionAnalysisSystem"),
            "group": "analysis"
        },
        "monthly_budget_recommendations": {
            "factory": lambda: _import_and_create("services.monthly_budget_recommendations", "MonthlyBudgetRecommendations"),
            "group": "recommendations"
        },
        "sleep_habit_tracker": {
            "factory": lambda: _import_and_create("services.sleep_habit_tracker", "SleepHabitTracker"),
            "group": "tracking"
        },
        "future_prediction": {
            "factory": lambda: _import_and_create("services.future_prediction", "FuturePredictionSystem"),
            "group": "ml"
        },
        "format_analysis": {
            "factory": lambda: _import_and_create("services.format_analysis", "FormatAnalysisSystem"),
            "group": "analysis"
        },
        "water_recommendations": {
            "factory": lambda: _import_and_create("services.water_type_recommendations", "WaterTypeRecommendations"),
            "group": "recommendations"
        },
        "stress_tracker": {
            "factory": lambda: _import_and_create("services.stress_tracker", "StressTracker"),
            "group": "tracking"
        },
        "intelligent_alerts": {
            "factory": lambda: _import_and_create("services.intelligent_alerts", "IntelligentAlerts"),
            "group": "notifications"
        },
        "device_analysis": {
            "factory": lambda: _import_and_create("services.device_analysis", "DeviceAnalysisSystem"),
            "group": "analysis"
        },
        "medication_recommendations": {
            "factory": lambda: _import_and_create("services.medication_recommendations", "MedicationRecommendations"),
            "group": "recommendations"
        },
        "hormonal_tracker": {
            "factory": lambda: _import_and_create("services.hormonal_tracker", "HormonalTracker"),
            "group": "tracking"
        },
        "advanced_ml_analysis": {
            "factory": lambda: _import_and_create("services.advanced_ml_analysis", "AdvancedMLAnalysis"),
            "group": "ml"
        },
        "environmental_tracker": {
            "factory": lambda: _import_and_create("services.environmental_tracker", "EnvironmentalTracker"),
            "group": "tracking"
        },
        "routine_optimizer": {
            "factory": lambda: _import_and_create("services.routine_optimizer", "RoutineOptimizer"),
            "group": "recommendations"
        },
        "skin_concern_tracker": {
            "factory": lambda: _import_and_create("services.skin_concern_tracker", "SkinConcernTracker"),
            "group": "tracking"
        },
        "product_effectiveness_tracker": {
            "factory": lambda: _import_and_create("services.product_effectiveness_tracker", "ProductEffectivenessTracker"),
            "group": "tracking"
        },
        "natural_lighting_analysis": {
            "factory": lambda: _import_and_create("services.natural_lighting_analysis", "NaturalLightingAnalysis"),
            "group": "analysis"
        },
        "ethnic_skin_recommendations": {
            "factory": lambda: _import_and_create("services.ethnic_skin_recommendations", "EthnicSkinRecommendations"),
            "group": "recommendations"
        },
        "seasonal_changes_tracker": {
            "factory": lambda: _import_and_create("services.seasonal_changes_tracker", "SeasonalChangesTracker"),
            "group": "tracking"
        },
        "anonymous_comparison": {
            "factory": lambda: _import_and_create("services.anonymous_comparison", "AnonymousComparison"),
            "group": "analysis"
        },
        "distance_analysis": {
            "factory": lambda: _import_and_create("services.distance_analysis", "DistanceAnalysisSystem"),
            "group": "analysis"
        },
        "occupation_recommendations": {
            "factory": lambda: _import_and_create("services.occupation_recommendations", "OccupationRecommendations"),
            "group": "recommendations"
        },
        "diet_tracker": {
            "factory": lambda: _import_and_create("services.diet_tracker", "DietTracker"),
            "group": "tracking"
        },
        "plateau_detection": {
            "factory": lambda: _import_and_create("services.plateau_detection", "PlateauDetection"),
            "group": "analysis"
        },
        "trend_prediction": {
            "factory": lambda: _import_and_create("services.trend_prediction", "TrendPredictionSystem"),
            "group": "ml"
        },
        "local_weather_recommendations": {
            "factory": lambda: _import_and_create("services.local_weather_recommendations", "LocalWeatherRecommendations"),
            "group": "recommendations"
        },
        "custom_routine_tracker": {
            "factory": lambda: _import_and_create("services.custom_routine_tracker", "CustomRoutineTracker"),
            "group": "tracking"
        },
        "product_compatibility": {
            "factory": lambda: _import_and_create("services.product_compatibility", "ProductCompatibility"),
            "group": "products"
        },
        "progress_visualization": {
            "factory": lambda: _import_and_create("services.progress_visualization", "ProgressVisualizationSystem"),
            "group": "reporting"
        },
        "budget_recommendations": {
            "factory": lambda: _import_and_create("services.budget_recommendations", "BudgetRecommendations"),
            "group": "recommendations"
        },
    }


def _import_and_create(module_path: str, class_name: str, **kwargs) -> Any:
    """Safely import and create a service instance"""
    try:
        parts = module_path.split(".")
        base_module = __import__(module_path, fromlist=[parts[-1]])
        
        module = base_module
        for part in parts[1:]:
            module = getattr(module, part)
        
        service_class = getattr(module, class_name)
        return service_class(**kwargs) if kwargs else service_class()
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to import {module_path}.{class_name}: {e}")
        return None


def _create_analytics_engine() -> Any:
    """Create analytics engine with dependencies"""
    try:
        from services.database import DatabaseManager
        from services.analytics import AnalyticsEngine
        db_manager = DatabaseManager()
        return AnalyticsEngine(db_manager)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to create analytics engine: {e}")
        return None


def _create_metrics_dashboard() -> Any:
    """Create metrics dashboard with dependencies"""
    try:
        from services.database import DatabaseManager
        from services.analytics import AnalyticsEngine
        db_manager = DatabaseManager()
        analytics = AnalyticsEngine(db_manager)
        from services.metrics_dashboard import MetricsDashboard
        return MetricsDashboard(db_manager, analytics)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to create metrics dashboard: {e}")
        return None

