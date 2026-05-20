"""
Refactored Recovery API - Modular route structure
This version uses separate route modules for better organization
"""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()


def _include_route(module_name: str, tag: str) -> None:
    """
    Helper function to include a route module with fallback import paths.
    
    Args:
        module_name: Name of the route module (without .py extension)
        tag: Tag for OpenAPI documentation
    """
    import_paths = [
        f"api.routes.{module_name}",
        f".routes.{module_name}",
    ]
    
    for import_path in import_paths:
        try:
            module = __import__(import_path, fromlist=["router"], level=0 if import_path.startswith("api") else 1)
            if hasattr(module, "router"):
                router.include_router(module.router, tags=[tag])
                logger.debug(f"Included route: {module_name}")
                return
        except ImportError:
            continue
    
    logger.debug(f"Route module not found: {module_name}")


# Core route modules
_core_routes = [
    ("assessment_routes", "Assessment"),
    ("recovery_plans_routes", "Recovery Plans"),
    ("progress_routes", "Progress"),
    ("relapse_prevention_routes", "Relapse Prevention"),
    ("support_routes", "Support"),
    ("analytics_routes", "Analytics"),
    ("auth_routes", "Authentication"),
    ("users_routes", "Users"),
    ("gamification_routes", "Gamification"),
    ("emergency_routes", "Emergency"),
    ("calendar_routes", "Calendar"),
    ("chatbot_routes", "Chatbot"),
    ("community_routes", "Community"),
    ("dashboard_routes", "Dashboard"),
    ("goals_routes", "Goals"),
    ("health_tracking_routes", "Health Tracking"),
    ("medication_routes", "Medication"),
    ("notifications_routes", "Notifications"),
]

for module_name, tag in _core_routes:
    _include_route(module_name, tag)

# Additional route modules
route_modules = [
    ("sentiment_routes", "Sentiment Analysis"),
    ("mentorship_routes", "Mentorship"),
    ("wearables_routes", "Wearables"),
    ("voice_analysis_routes", "Voice Analysis"),
    ("family_tracking_routes", "Family Tracking"),
    ("intelligent_alerts_routes", "Intelligent Alerts"),
    ("virtual_therapy_routes", "Virtual Therapy"),
    ("visualization_routes", "Visualization"),
    ("virtual_economy_routes", "Virtual Economy"),
    ("withdrawal_tracking_routes", "Withdrawal Tracking"),
    ("sleep_analysis_routes", "Sleep Analysis"),
    ("challenges_routes", "Challenges"),
    ("webhooks_routes", "Webhooks"),
    ("certificates_routes", "Certificates"),
    ("backup_routes", "Backup"),
    ("social_integration_routes", "Social Integration"),
    ("nlp_analysis_routes", "NLP Analysis"),
    ("recommendations_routes", "Recommendations"),
    ("habit_tracking_routes", "Habit Tracking"),
    ("mindfulness_routes", "Mindfulness"),
    ("resource_library_routes", "Resource Library"),
    ("financial_tracking_routes", "Financial Tracking"),
    ("wellness_analysis_routes", "Wellness Analysis"),
    ("push_notifications_routes", "Push Notifications"),
    ("health_integration_routes", "Health Integration"),
    ("support_groups_routes", "Support Groups"),
    ("advanced_reporting_routes", "Advanced Reporting"),
    ("predictive_ai_routes", "Predictive AI"),
    ("temporal_patterns_routes", "Temporal Patterns"),
    ("intelligent_reminders_routes", "Intelligent Reminders"),
    ("social_relationships_routes", "Social Relationships"),
    ("realtime_coaching_routes", "Realtime Coaching"),
    ("third_party_integration_routes", "Third Party Integration"),
    ("emergency_integration_routes", "Emergency Integration"),
    ("blockchain_routes", "Blockchain"),
    ("realtime_events_routes", "Realtime Events"),
    ("social_media_analysis_routes", "Social Media Analysis"),
    ("ml_recommendations_routes", "ML Recommendations"),
    ("vr_ar_therapy_routes", "VR/AR Therapy"),
    ("advanced_biometrics_routes", "Advanced Biometrics"),
    ("voice_assistant_routes", "Voice Assistant"),
    ("purchase_patterns_routes", "Purchase Patterns"),
    ("productivity_work_routes", "Productivity & Work"),
    ("nutrition_diet_routes", "Nutrition & Diet"),
    ("exercise_tracking_routes", "Exercise Tracking"),
    ("meditation_app_routes", "Meditation App Integration"),
    ("environment_context_routes", "Environment Context"),
    ("genetic_analysis_routes", "Genetic Analysis"),
    ("medical_device_routes", "Medical Devices"),
    ("ehr_integration_routes", "EHR Integration"),
    ("telemedicine_routes", "Telemedicine"),
    ("correlation_analysis_routes", "Correlation Analysis"),
    ("long_term_prediction_routes", "Long Term Prediction"),
    ("behavioral_analysis_routes", "Behavioral Analysis"),
    ("adherence_analysis_routes", "Adherence Analysis"),
    ("symptom_tracking_routes", "Symptom Tracking"),
    ("quality_of_life_routes", "Quality of Life"),
    ("neural_network_routes", "Neural Network"),
    ("continuous_monitoring_routes", "Continuous Monitoring"),
    ("ai_sleep_analysis_routes", "AI Sleep Analysis"),
    ("voice_emotion_routes", "Voice Emotion Recognition"),
    ("wellness_app_routes", "Wellness App Integration"),
    ("activity_patterns_routes", "Activity Patterns"),
    ("health_monitoring_devices_routes", "Health Monitoring Devices"),
    ("personalized_coaching_routes", "Personalized Coaching"),
    ("social_network_analysis_routes", "Social Network Analysis"),
    ("advanced_goal_tracking_routes", "Advanced Goal Tracking"),
    ("comparative_analysis_routes", "Comparative Analysis"),
    ("resilience_analysis_routes", "Resilience Analysis"),
    ("advanced_rewards_routes", "Advanced Rewards"),
    ("alternative_therapy_routes", "Alternative Therapy"),
    ("advanced_motivation_routes", "Advanced Motivation"),
    ("advanced_relapse_tracking_routes", "Advanced Relapse Tracking"),
    ("recovery_barriers_routes", "Recovery Barriers"),
    ("advanced_stress_routes", "Advanced Stress Analysis"),
    ("advanced_social_support_routes", "Advanced Social Support"),
    ("emergency_services_routes", "Emergency Services"),
    ("advanced_medication_tracking_routes", "Advanced Medication Tracking"),
    ("sleep_pattern_analysis_routes", "Sleep Pattern Analysis"),
    ("comprehensive_wellness_routes", "Comprehensive Wellness"),
    ("group_therapy_routes", "Group Therapy"),
    ("advanced_intelligent_reminders_routes", "Advanced Intelligent Reminders"),
    ("advanced_health_device_routes", "Advanced Health Devices"),
    ("advanced_habit_analysis_routes", "Advanced Habit Analysis"),
    ("advanced_exercise_analysis_routes", "Advanced Exercise Analysis"),
    ("advanced_nutrition_analysis_routes", "Advanced Nutrition Analysis"),
    ("long_term_progress_routes", "Long Term Progress"),
    ("advanced_achievements_routes", "Advanced Achievements"),
    ("advanced_mood_analysis_routes", "Advanced Mood Analysis"),
    ("advanced_therapy_tracking_routes", "Advanced Therapy Tracking"),
    ("advanced_relationship_analysis_routes", "Advanced Relationship Analysis"),
    ("advanced_visual_progress_routes", "Advanced Visual Progress"),
    ("pdf_reports_routes", "PDF Reports"),
    ("advanced_progress_tracking_routes", "Advanced Progress Tracking"),
    ("advanced_gamification_routes", "Advanced Gamification"),
    ("advanced_data_analysis_routes", "Advanced Data Analysis"),
    ("long_term_goals_routes", "Long Term Goals"),
    ("advanced_risk_analysis_routes", "Advanced Risk Analysis"),
    ("advanced_metrics_routes", "Advanced Metrics"),
    ("intelligent_notifications_routes", "Intelligent Notifications"),
    ("advanced_medication_routes", "Advanced Medication"),
    ("iot_integration_routes", "IoT Integration"),
    ("advanced_voice_analysis_routes", "Advanced Voice Analysis"),
    ("location_tracking_routes", "Location Tracking"),
    ("image_emotion_routes", "Image Emotion Analysis"),
    ("advanced_sleep_tracking_routes", "Advanced Sleep Tracking"),
    ("ml_learning_routes", "ML Learning"),
    ("advanced_predictive_ml_routes", "Advanced Predictive ML"),
    ("advanced_emotion_tracking_routes", "Advanced Emotion Tracking"),
]

for module_name, tag in route_modules:
    _include_route(module_name, tag)

logger.info(f"Recovery API router initialized with {len(_core_routes) + len(route_modules)} route modules")
