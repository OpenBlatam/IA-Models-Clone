"""
API endpoints para el sistema de recuperación de adicciones

⚠️ DEPRECATED: This file is deprecated and kept only for backward compatibility.
Please use the modular router from `recovery_api_refactored.py` instead.

This monolithic file contains 4932+ lines and should not be used for new development.
All new endpoints should be added to the appropriate router in the `routes/` directory.
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict
from pydantic import BaseModel
from datetime import datetime

try:
    from core.addiction_analyzer import AddictionAnalyzer
    from core.recovery_planner import RecoveryPlanner
    from core.progress_tracker import ProgressTracker
    from core.relapse_prevention import RelapsePrevention
    from services.counseling_service import CounselingService
    from services.motivation_service import MotivationService
    from services.notification_service import NotificationService
    from services.analytics_service import AnalyticsService
    from services.auth_service import AuthService
    from services.report_service import ReportService
    from services.gamification_service import GamificationService
    from services.emergency_service import EmergencyService
    from services.dashboard_service import DashboardService
    from services.calendar_service import CalendarService
    from services.chatbot_service import ChatbotService
    from services.community_service import CommunityService
    from services.predictive_service import PredictiveService
    from services.health_tracking_service import HealthTrackingService
    from services.goals_service import GoalsService
    from services.wearable_service import WearableService
    from services.sentiment_service import SentimentService
    from services.mentorship_service import MentorshipService
    from services.medication_service import MedicationService
    from services.health_integration_service import HealthIntegrationService
    from services.push_notification_service import PushNotificationService
    from services.voice_analysis_service import VoiceAnalysisService
    from services.family_tracking_service import FamilyTrackingService
    from services.intelligent_alerts_service import IntelligentAlertsService
    from services.virtual_therapy_service import VirtualTherapyService
    from services.visualization_service import VisualizationService
    from services.virtual_economy_service import VirtualEconomyService
    from services.emergency_integration_service import EmergencyIntegrationService
    from services.withdrawal_tracking_service import WithdrawalTrackingService
    from services.sleep_analysis_service import SleepAnalysisService
    from services.challenge_service import ChallengeService
    from services.webhook_service import WebhookService
    from services.certificate_service import CertificateService
    from services.backup_service import BackupService
    from services.social_integration_service import SocialIntegrationService
    from services.nlp_analysis_service import NLPAnalysisService
    from services.advanced_support_groups_service import AdvancedSupportGroupsService
    from services.advanced_reporting_service import AdvancedReportingService
    from services.predictive_ai_service import PredictiveAIService
    from services.recommendation_service import RecommendationService
    from services.habit_tracking_service import HabitTrackingService
    from services.mindfulness_service import MindfulnessService
    from services.resource_library_service import ResourceLibraryService
    from services.temporal_pattern_analysis_service import TemporalPatternAnalysisService
    from services.financial_tracking_service import FinancialTrackingService
    from services.intelligent_reminders_service import IntelligentRemindersService
    from services.wellness_analysis_service import WellnessAnalysisService
    from services.social_relationships_service import SocialRelationshipsService
    from services.realtime_coaching_service import RealtimeCoachingService
    from services.third_party_integration_service import ThirdPartyIntegrationService
    from services.advanced_progress_tracking_service import AdvancedProgressTrackingService
    from services.advanced_gamification_service import AdvancedGamificationService
    from services.advanced_data_analysis_service import AdvancedDataAnalysisService
    from services.long_term_goals_service import LongTermGoalsService
    from services.advanced_risk_analysis_service import AdvancedRiskAnalysisService
    from services.advanced_metrics_service import AdvancedMetricsService
    from services.intelligent_notifications_service import IntelligentNotificationsService
    from services.advanced_medication_service import AdvancedMedicationService
    from services.iot_integration_service import IoTIntegrationService
    from services.advanced_voice_analysis_service import AdvancedVoiceAnalysisService
    from services.location_tracking_service import LocationTrackingService
    from services.image_emotion_analysis_service import ImageEmotionAnalysisService
    from services.advanced_sleep_tracking_service import AdvancedSleepTrackingService
    from services.ml_learning_service import MLLearningService
    from services.advanced_predictive_ml_service import AdvancedPredictiveMLService
    from services.blockchain_integration_service import BlockchainIntegrationService
    from services.realtime_events_service import RealtimeEventsService
    from services.social_media_analysis_service import SocialMediaAnalysisService
    from services.ml_recommendation_service import MLRecommendationService
    from services.vr_ar_therapy_service import VRARTherapyService
    from services.advanced_biometrics_service import AdvancedBiometricsService
    from services.voice_assistant_integration_service import VoiceAssistantIntegrationService
    from services.purchase_pattern_analysis_service import PurchasePatternAnalysisService
    from services.interpersonal_relationships_service import InterpersonalRelationshipsService
    from services.productivity_work_analysis_service import ProductivityWorkAnalysisService
    from services.nutrition_diet_analysis_service import NutritionDietAnalysisService
    from services.advanced_exercise_tracking_service import AdvancedExerciseTrackingService
    from services.meditation_app_integration_service import MeditationAppIntegrationService
    from services.environment_context_analysis_service import EnvironmentContextAnalysisService
    from services.advanced_habit_tracking_service import AdvancedHabitTrackingService
    from services.advanced_temporal_pattern_analysis_service import AdvancedTemporalPatternAnalysisService
    from services.genetic_predisposition_service import GeneticPredispositionService
    from services.medical_device_integration_service import MedicalDeviceIntegrationService
    from services.advanced_visual_progress_service import AdvancedVisualProgressService
    from services.ehr_integration_service import EHRIntegrationService
    from services.advanced_correlation_analysis_service import AdvancedCorrelationAnalysisService
    from services.long_term_success_prediction_service import LongTermSuccessPredictionService
    from services.advanced_behavioral_analysis_service import AdvancedBehavioralAnalysisService
    from services.telemedicine_integration_service import TelemedicineIntegrationService
    from services.advanced_intelligent_alerts_service import AdvancedIntelligentAlertsService
    from services.advanced_adherence_analysis_service import AdvancedAdherenceAnalysisService
    from services.advanced_symptom_tracking_service import AdvancedSymptomTrackingService
    from services.quality_of_life_analysis_service import QualityOfLifeAnalysisService
    from services.neural_network_analysis_service import NeuralNetworkAnalysisService
    from services.continuous_monitoring_service import ContinuousMonitoringService
    from services.ai_sleep_analysis_service import AISleepAnalysisService
    from services.advanced_emotion_tracking_service import AdvancedEmotionTrackingService
    from services.voice_emotion_recognition_service import VoiceEmotionRecognitionService
    from services.wellness_app_integration_service import WellnessAppIntegrationService
    from services.advanced_activity_pattern_analysis_service import AdvancedActivityPatternAnalysisService
    from services.health_monitoring_device_service import HealthMonitoringDeviceService
    from services.advanced_personalized_coaching_service import AdvancedPersonalizedCoachingService
    from services.advanced_social_network_analysis_service import AdvancedSocialNetworkAnalysisService
    from services.advanced_goal_tracking_service import AdvancedGoalTrackingService
    from services.comparative_progress_analysis_service import ComparativeProgressAnalysisService
    from services.resilience_analysis_service import ResilienceAnalysisService
    from services.advanced_rewards_service import AdvancedRewardsService
    from services.alternative_therapy_integration_service import AlternativeTherapyIntegrationService
    from services.advanced_motivation_analysis_service import AdvancedMotivationAnalysisService
    from services.advanced_relapse_tracking_service import AdvancedRelapseTrackingService
    from services.recovery_barriers_analysis_service import RecoveryBarriersAnalysisService
    from services.advanced_stress_analysis_service import AdvancedStressAnalysisService
    from services.advanced_social_support_service import AdvancedSocialSupportService
    from services.emergency_services_integration_service import EmergencyServicesIntegrationService
    from services.advanced_visual_progress_service import AdvancedVisualProgressService
    from services.advanced_medication_tracking_service import AdvancedMedicationTrackingService
    from services.advanced_sleep_pattern_analysis_service import AdvancedSleepPatternAnalysisService
    from services.comprehensive_wellness_analysis_service import ComprehensiveWellnessAnalysisService
    from services.advanced_intelligent_reminders_service import AdvancedIntelligentRemindersService
    from services.advanced_health_device_integration_service import AdvancedHealthDeviceIntegrationService
    from services.advanced_habit_analysis_service import AdvancedHabitAnalysisService
    from services.advanced_exercise_analysis_service import AdvancedExerciseAnalysisService
    from services.advanced_nutrition_analysis_service import AdvancedNutritionAnalysisService
    from services.long_term_progress_analysis_service import LongTermProgressAnalysisService
    from services.advanced_achievements_service import AdvancedAchievementsService
    from services.group_therapy_integration_service import GroupTherapyIntegrationService
    from services.advanced_mood_analysis_service import AdvancedMoodAnalysisService
    from services.advanced_therapy_tracking_service import AdvancedTherapyTrackingService
    from services.advanced_relationship_analysis_service import AdvancedRelationshipAnalysisService
    from models.database import DatabaseManager
    from utils.validators import AddictionTypeValidator
except ImportError:
    from ..core.addiction_analyzer import AddictionAnalyzer
    from ..core.recovery_planner import RecoveryPlanner
    from ..core.progress_tracker import ProgressTracker
    from ..core.relapse_prevention import RelapsePrevention
    from ..services.counseling_service import CounselingService
    from ..services.motivation_service import MotivationService
    from ..services.notification_service import NotificationService
    from ..services.analytics_service import AnalyticsService
    from ..services.auth_service import AuthService
    from ..services.report_service import ReportService
    from ..services.gamification_service import GamificationService
    from ..services.emergency_service import EmergencyService
    from ..services.dashboard_service import DashboardService
    from ..services.calendar_service import CalendarService
    from ..services.chatbot_service import ChatbotService
    from ..services.community_service import CommunityService
    from ..services.predictive_service import PredictiveService
    from ..services.health_tracking_service import HealthTrackingService
    from ..services.goals_service import GoalsService
    from ..services.wearable_service import WearableService
    from ..services.sentiment_service import SentimentService
    from ..services.mentorship_service import MentorshipService
    from ..services.medication_service import MedicationService
    from ..services.health_integration_service import HealthIntegrationService
    from ..services.push_notification_service import PushNotificationService
    from ..services.voice_analysis_service import VoiceAnalysisService
    from ..services.family_tracking_service import FamilyTrackingService
    from ..services.intelligent_alerts_service import IntelligentAlertsService
    from ..services.virtual_therapy_service import VirtualTherapyService
    from ..services.visualization_service import VisualizationService
    from ..services.virtual_economy_service import VirtualEconomyService
    from ..services.emergency_integration_service import EmergencyIntegrationService
    from ..services.withdrawal_tracking_service import WithdrawalTrackingService
    from ..services.sleep_analysis_service import SleepAnalysisService
    from ..services.challenge_service import ChallengeService
    from ..services.webhook_service import WebhookService
    from ..services.certificate_service import CertificateService
    from ..services.backup_service import BackupService
    from ..services.social_integration_service import SocialIntegrationService
    from ..services.nlp_analysis_service import NLPAnalysisService
    from ..services.advanced_support_groups_service import AdvancedSupportGroupsService
    from ..services.advanced_reporting_service import AdvancedReportingService
    from ..services.predictive_ai_service import PredictiveAIService
    from ..services.recommendation_service import RecommendationService
    from ..services.habit_tracking_service import HabitTrackingService
    from ..services.mindfulness_service import MindfulnessService
    from ..services.resource_library_service import ResourceLibraryService
    from ..services.temporal_pattern_analysis_service import TemporalPatternAnalysisService
    from ..services.financial_tracking_service import FinancialTrackingService
    from ..services.intelligent_reminders_service import IntelligentRemindersService
    from ..services.wellness_analysis_service import WellnessAnalysisService
    from ..services.social_relationships_service import SocialRelationshipsService
    from ..services.realtime_coaching_service import RealtimeCoachingService
    from ..services.third_party_integration_service import ThirdPartyIntegrationService
    from ..services.advanced_progress_tracking_service import AdvancedProgressTrackingService
    from ..services.advanced_gamification_service import AdvancedGamificationService
    from ..services.advanced_data_analysis_service import AdvancedDataAnalysisService
    from ..services.long_term_goals_service import LongTermGoalsService
    from ..services.advanced_risk_analysis_service import AdvancedRiskAnalysisService
    from ..services.advanced_metrics_service import AdvancedMetricsService
    from ..services.intelligent_notifications_service import IntelligentNotificationsService
    from ..services.advanced_medication_service import AdvancedMedicationService
    from ..services.iot_integration_service import IoTIntegrationService
    from ..services.advanced_voice_analysis_service import AdvancedVoiceAnalysisService
    from ..services.location_tracking_service import LocationTrackingService
    from ..services.image_emotion_analysis_service import ImageEmotionAnalysisService
    from ..services.advanced_sleep_tracking_service import AdvancedSleepTrackingService
    from ..services.ml_learning_service import MLLearningService
    from ..services.advanced_predictive_ml_service import AdvancedPredictiveMLService
    from ..services.blockchain_integration_service import BlockchainIntegrationService
    from ..services.realtime_events_service import RealtimeEventsService
    from ..services.social_media_analysis_service import SocialMediaAnalysisService
    from ..services.ml_recommendation_service import MLRecommendationService
    from ..services.vr_ar_therapy_service import VRARTherapyService
    from ..services.advanced_biometrics_service import AdvancedBiometricsService
    from ..services.voice_assistant_integration_service import VoiceAssistantIntegrationService
    from ..services.purchase_pattern_analysis_service import PurchasePatternAnalysisService
    from ..services.interpersonal_relationships_service import InterpersonalRelationshipsService
    from ..services.productivity_work_analysis_service import ProductivityWorkAnalysisService
    from ..services.nutrition_diet_analysis_service import NutritionDietAnalysisService
    from ..services.advanced_exercise_tracking_service import AdvancedExerciseTrackingService
    from ..services.meditation_app_integration_service import MeditationAppIntegrationService
    from ..services.environment_context_analysis_service import EnvironmentContextAnalysisService
    from ..services.advanced_habit_tracking_service import AdvancedHabitTrackingService
    from ..services.advanced_temporal_pattern_analysis_service import AdvancedTemporalPatternAnalysisService
    from ..services.genetic_predisposition_service import GeneticPredispositionService
    from ..services.medical_device_integration_service import MedicalDeviceIntegrationService
    from ..services.advanced_visual_progress_service import AdvancedVisualProgressService
    from ..services.ehr_integration_service import EHRIntegrationService
    from ..services.advanced_correlation_analysis_service import AdvancedCorrelationAnalysisService
    from ..services.long_term_success_prediction_service import LongTermSuccessPredictionService
    from ..services.advanced_behavioral_analysis_service import AdvancedBehavioralAnalysisService
    from ..services.telemedicine_integration_service import TelemedicineIntegrationService
    from ..services.advanced_intelligent_alerts_service import AdvancedIntelligentAlertsService
    from ..services.advanced_adherence_analysis_service import AdvancedAdherenceAnalysisService
    from ..services.advanced_symptom_tracking_service import AdvancedSymptomTrackingService
    from ..services.quality_of_life_analysis_service import QualityOfLifeAnalysisService
    from ..services.neural_network_analysis_service import NeuralNetworkAnalysisService
    from ..services.continuous_monitoring_service import ContinuousMonitoringService
    from ..services.ai_sleep_analysis_service import AISleepAnalysisService
    from ..services.advanced_emotion_tracking_service import AdvancedEmotionTrackingService
    from ..services.voice_emotion_recognition_service import VoiceEmotionRecognitionService
    from ..services.wellness_app_integration_service import WellnessAppIntegrationService
    from ..services.advanced_activity_pattern_analysis_service import AdvancedActivityPatternAnalysisService
    from ..services.health_monitoring_device_service import HealthMonitoringDeviceService
    from ..services.advanced_personalized_coaching_service import AdvancedPersonalizedCoachingService
    from ..services.advanced_social_network_analysis_service import AdvancedSocialNetworkAnalysisService
    from ..services.advanced_goal_tracking_service import AdvancedGoalTrackingService
    from ..services.comparative_progress_analysis_service import ComparativeProgressAnalysisService
    from ..services.resilience_analysis_service import ResilienceAnalysisService
    from ..services.advanced_rewards_service import AdvancedRewardsService
    from ..services.alternative_therapy_integration_service import AlternativeTherapyIntegrationService
    from ..services.advanced_motivation_analysis_service import AdvancedMotivationAnalysisService
    from ..services.advanced_relapse_tracking_service import AdvancedRelapseTrackingService
    from ..services.recovery_barriers_analysis_service import RecoveryBarriersAnalysisService
    from ..services.advanced_stress_analysis_service import AdvancedStressAnalysisService
    from ..services.advanced_social_support_service import AdvancedSocialSupportService
    from ..services.emergency_services_integration_service import EmergencyServicesIntegrationService
    from ..services.advanced_visual_progress_service import AdvancedVisualProgressService
    from ..services.advanced_medication_tracking_service import AdvancedMedicationTrackingService
    from ..services.advanced_sleep_pattern_analysis_service import AdvancedSleepPatternAnalysisService
    from ..services.comprehensive_wellness_analysis_service import ComprehensiveWellnessAnalysisService
    from ..services.advanced_intelligent_reminders_service import AdvancedIntelligentRemindersService
    from ..services.advanced_health_device_integration_service import AdvancedHealthDeviceIntegrationService
    from ..services.advanced_habit_analysis_service import AdvancedHabitAnalysisService
    from ..services.advanced_exercise_analysis_service import AdvancedExerciseAnalysisService
    from ..services.advanced_nutrition_analysis_service import AdvancedNutritionAnalysisService
    from ..services.long_term_progress_analysis_service import LongTermProgressAnalysisService
    from ..services.advanced_achievements_service import AdvancedAchievementsService
    from ..services.group_therapy_integration_service import GroupTherapyIntegrationService
    from ..services.advanced_mood_analysis_service import AdvancedMoodAnalysisService
    from ..services.advanced_therapy_tracking_service import AdvancedTherapyTrackingService
    from ..services.advanced_relationship_analysis_service import AdvancedRelationshipAnalysisService
    from ..models.database import DatabaseManager
    from ..utils.validators import AddictionTypeValidator

router = APIRouter()

# Inicializar servicios
analyzer = AddictionAnalyzer()
planner = RecoveryPlanner()
tracker = ProgressTracker()
relapse_prevention = RelapsePrevention()
counseling = CounselingService()
motivation = MotivationService()
notifications = NotificationService()
analytics = AnalyticsService()
auth = AuthService()
reports = ReportService()
gamification = GamificationService()
emergency = EmergencyService()
dashboard = DashboardService()
calendar = CalendarService()
chatbot = ChatbotService()
community = CommunityService()
predictive = PredictiveService()
health_tracking = HealthTrackingService()
goals = GoalsService()
wearable = WearableService()
sentiment = SentimentService()
mentorship = MentorshipService()
medication = MedicationService()
health_integration = HealthIntegrationService()
push_notifications = PushNotificationService()
voice_analysis = VoiceAnalysisService()
family_tracking = FamilyTrackingService()
intelligent_alerts = IntelligentAlertsService()
virtual_therapy = VirtualTherapyService()
visualization = VisualizationService()
virtual_economy = VirtualEconomyService()
emergency_integration = EmergencyIntegrationService()
withdrawal_tracking = WithdrawalTrackingService()
sleep_analysis = SleepAnalysisService()
challenges = ChallengeService()
webhooks = WebhookService()
certificates = CertificateService()
backup = BackupService()
social_integration = SocialIntegrationService()
nlp_analysis = NLPAnalysisService()
advanced_support_groups = AdvancedSupportGroupsService()
advanced_reporting = AdvancedReportingService()
predictive_ai = PredictiveAIService()
recommendations = RecommendationService()
habit_tracking = HabitTrackingService()
mindfulness = MindfulnessService()
resource_library = ResourceLibraryService()
temporal_patterns = TemporalPatternAnalysisService()
financial_tracking = FinancialTrackingService()
intelligent_reminders = IntelligentRemindersService()
wellness_analysis = WellnessAnalysisService()
social_relationships = SocialRelationshipsService()
realtime_coaching = RealtimeCoachingService()
third_party_integration = ThirdPartyIntegrationService()
advanced_progress = AdvancedProgressTrackingService()
advanced_gamification = AdvancedGamificationService()
advanced_data_analysis = AdvancedDataAnalysisService()
long_term_goals = LongTermGoalsService()
advanced_risk_analysis = AdvancedRiskAnalysisService()
advanced_metrics = AdvancedMetricsService()
intelligent_notifications = IntelligentNotificationsService()
advanced_medication = AdvancedMedicationService()
iot_integration = IoTIntegrationService()
advanced_voice = AdvancedVoiceAnalysisService()
location_tracking = LocationTrackingService()
image_emotion = ImageEmotionAnalysisService()
advanced_sleep = AdvancedSleepTrackingService()
ml_learning = MLLearningService()
advanced_predictive_ml = AdvancedPredictiveMLService()
blockchain = BlockchainIntegrationService()
realtime_events = RealtimeEventsService()
social_media_analysis = SocialMediaAnalysisService()
ml_recommendations = MLRecommendationService()
vr_ar_therapy = VRARTherapyService()
advanced_biometrics = AdvancedBiometricsService()
voice_assistant = VoiceAssistantIntegrationService()
purchase_analysis = PurchasePatternAnalysisService()
relationships = InterpersonalRelationshipsService()
productivity_work = ProductivityWorkAnalysisService()
nutrition_diet = NutritionDietAnalysisService()
advanced_exercise = AdvancedExerciseTrackingService()
meditation_apps = MeditationAppIntegrationService()
environment_context = EnvironmentContextAnalysisService()
advanced_habits = AdvancedHabitTrackingService()
temporal_patterns = AdvancedTemporalPatternAnalysisService()
genetic_analysis = GeneticPredispositionService()
medical_devices = MedicalDeviceIntegrationService()
visual_progress = AdvancedVisualProgressService()
ehr_integration = EHRIntegrationService()
correlation_analysis = AdvancedCorrelationAnalysisService()
long_term_prediction = LongTermSuccessPredictionService()
behavioral_analysis = AdvancedBehavioralAnalysisService()
telemedicine = TelemedicineIntegrationService()
intelligent_alerts = AdvancedIntelligentAlertsService()
adherence_analysis = AdvancedAdherenceAnalysisService()
symptom_tracking = AdvancedSymptomTrackingService()
quality_of_life = QualityOfLifeAnalysisService()
neural_networks = NeuralNetworkAnalysisService()
continuous_monitoring = ContinuousMonitoringService()
ai_sleep = AISleepAnalysisService()
emotion_tracking = AdvancedEmotionTrackingService()
voice_emotion = VoiceEmotionRecognitionService()
wellness_apps = WellnessAppIntegrationService()
activity_patterns = AdvancedActivityPatternAnalysisService()
health_devices = HealthMonitoringDeviceService()
personalized_coaching = AdvancedPersonalizedCoachingService()
social_network_analysis = AdvancedSocialNetworkAnalysisService()
goal_tracking = AdvancedGoalTrackingService()
comparative_analysis = ComparativeProgressAnalysisService()
resilience_analysis = ResilienceAnalysisService()
rewards_service = AdvancedRewardsService()
alternative_therapy = AlternativeTherapyIntegrationService()
motivation_analysis = AdvancedMotivationAnalysisService()
relapse_tracking = AdvancedRelapseTrackingService()
barriers_analysis = RecoveryBarriersAnalysisService()
stress_analysis = AdvancedStressAnalysisService()
social_support = AdvancedSocialSupportService()
emergency_services = EmergencyServicesIntegrationService()
visual_progress = AdvancedVisualProgressService()
medication_tracking = AdvancedMedicationTrackingService()
sleep_patterns = AdvancedSleepPatternAnalysisService()
wellness_analysis = ComprehensiveWellnessAnalysisService()
intelligent_reminders = AdvancedIntelligentRemindersService()
health_devices_advanced = AdvancedHealthDeviceIntegrationService()
habit_analysis = AdvancedHabitAnalysisService()
exercise_analysis = AdvancedExerciseAnalysisService()
nutrition_analysis = AdvancedNutritionAnalysisService()
long_term_progress = LongTermProgressAnalysisService()
achievements = AdvancedAchievementsService()
group_therapy = GroupTherapyIntegrationService()
mood_analysis = AdvancedMoodAnalysisService()
therapy_tracking = AdvancedTherapyTrackingService()
relationship_analysis = AdvancedRelationshipAnalysisService()
db_manager = DatabaseManager()


# Modelos Pydantic para requests
class AssessmentRequest(BaseModel):
    addiction_type: str
    severity: str
    frequency: str
    duration_years: Optional[float] = None
    daily_cost: Optional[float] = None
    triggers: List[str] = []
    motivations: List[str] = []
    previous_attempts: int = 0
    support_system: bool = False
    medical_conditions: List[str] = []
    additional_info: Optional[str] = None


class LogEntryRequest(BaseModel):
    user_id: str
    date: str
    mood: str
    cravings_level: int
    triggers_encountered: List[str] = []
    consumed: bool = False
    notes: Optional[str] = None


class RelapseRiskCheckRequest(BaseModel):
    user_id: str
    days_sober: int
    stress_level: int
    support_level: int
    triggers: List[str] = []
    previous_relapses: int = 0
    isolation: bool = False
    negative_thinking: bool = False
    romanticizing: bool = False
    skipping_support: bool = False


class CoachingSessionRequest(BaseModel):
    user_id: str
    topic: str
    current_situation: str
    questions: Optional[List[str]] = None


# Endpoints de Evaluación
@router.post("/assess")
async def assess_addiction(request: AssessmentRequest):
    """
    Evalúa una adicción y proporciona análisis completo
    """
    try:
        # Validar tipo de adicción
        if not AddictionTypeValidator.validate_type(request.addiction_type):
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de adicción no válido. Tipos válidos: {AddictionTypeValidator.get_valid_types()}"
            )
        
        assessment_data = request.dict()
        analysis = analyzer.assess_addiction(assessment_data)
        
        return JSONResponse(content=analysis)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en evaluación: {str(e)}")


@router.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """
    Obtiene perfil del usuario
    """
    # En implementación real, esto vendría de una base de datos
    return JSONResponse(content={
        "user_id": user_id,
        "message": "Perfil del usuario (implementar con base de datos)",
        "status": "success"
    })


@router.post("/update-profile")
async def update_profile(profile_data: Dict = Body(...)):
    """
    Actualiza perfil del usuario
    """
    return JSONResponse(content={
        "message": "Perfil actualizado",
        "status": "success",
        "data": profile_data
    })


# Endpoints de Planes de Recuperación
@router.post("/create-plan")
async def create_recovery_plan(
    user_id: str = Body(...),
    addiction_type: str = Body(...),
    assessment_data: Dict = Body(...),
    approach: Optional[str] = Body(None)
):
    """
    Crea un plan de recuperación personalizado
    """
    try:
        plan = planner.create_plan(user_id, addiction_type, assessment_data, approach)
        return JSONResponse(content=plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando plan: {str(e)}")


@router.get("/plan/{user_id}")
async def get_recovery_plan(user_id: str):
    """
    Obtiene plan de recuperación del usuario
    """
    return JSONResponse(content={
        "user_id": user_id,
        "message": "Plan de recuperación (implementar con base de datos)",
        "status": "success"
    })


@router.post("/update-plan")
async def update_recovery_plan(plan_data: Dict = Body(...)):
    """
    Actualiza plan de recuperación
    """
    return JSONResponse(content={
        "message": "Plan actualizado",
        "status": "success",
        "data": plan_data
    })


@router.get("/strategies/{addiction_type}")
async def get_strategies(addiction_type: str):
    """
    Obtiene estrategias específicas para un tipo de adicción
    """
    try:
        if not AddictionTypeValidator.validate_type(addiction_type):
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de adicción no válido"
            )
        
        strategies = planner._get_strategies(addiction_type, {})
        return JSONResponse(content={
            "addiction_type": addiction_type,
            "strategies": strategies,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estrategias: {str(e)}")


# Endpoints de Seguimiento de Progreso
@router.post("/log-entry")
async def log_daily_entry(request: LogEntryRequest):
    """
    Registra una entrada diaria
    """
    try:
        entry = tracker.log_entry(
            request.user_id,
            request.date,
            request.mood,
            request.cravings_level,
            request.triggers_encountered,
            request.consumed,
            request.notes
        )
        return JSONResponse(content={
            "message": "Entrada registrada exitosamente",
            "entry": entry,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando entrada: {str(e)}")


@router.get("/progress/{user_id}")
async def get_progress(
    user_id: str,
    start_date: Optional[str] = Query(None)
):
    """
    Obtiene progreso del usuario
    """
    try:
        start = None
        if start_date:
            start = datetime.fromisoformat(start_date)
        
        progress = tracker.get_progress(user_id, start, [])
        return JSONResponse(content=progress)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo progreso: {str(e)}")


@router.get("/stats/{user_id}")
async def get_stats(user_id: str):
    """
    Obtiene estadísticas detalladas del usuario
    """
    try:
        stats = tracker.get_stats(user_id, [])
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


@router.get("/timeline/{user_id}")
async def get_timeline(user_id: str):
    """
    Obtiene línea de tiempo de progreso
    """
    try:
        timeline = tracker.get_timeline(user_id, [])
        return JSONResponse(content={
            "user_id": user_id,
            "timeline": timeline,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo timeline: {str(e)}")


# Endpoints de Prevención de Recaídas
@router.post("/check-relapse-risk")
async def check_relapse_risk(request: RelapseRiskCheckRequest):
    """
    Evalúa riesgo de recaída
    """
    try:
        current_state = {
            "stress_level": request.stress_level,
            "support_level": request.support_level,
            "triggers": request.triggers,
            "isolation": request.isolation,
            "negative_thinking": request.negative_thinking,
            "romanticizing": request.romanticizing,
            "skipping_support": request.skipping_support
        }
        
        risk_analysis = relapse_prevention.check_relapse_risk(
            request.user_id,
            request.days_sober,
            current_state,
            None
        )
        
        return JSONResponse(content=risk_analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando riesgo: {str(e)}")


@router.get("/triggers/{user_id}")
async def get_triggers(user_id: str):
    """
    Obtiene triggers identificados del usuario
    """
    return JSONResponse(content={
        "user_id": user_id,
        "triggers": [],
        "message": "Triggers del usuario (implementar con base de datos)",
        "status": "success"
    })


@router.post("/coping-strategies")
async def get_coping_strategies(
    situation: str = Body(...),
    trigger_type: Optional[str] = Body(None)
):
    """
    Obtiene estrategias de afrontamiento para una situación
    """
    try:
        strategies = relapse_prevention.get_coping_strategies(situation, trigger_type)
        return JSONResponse(content={
            "situation": situation,
            "trigger_type": trigger_type,
            "strategies": strategies,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estrategias: {str(e)}")


@router.post("/emergency-plan")
async def generate_emergency_plan(
    user_id: str = Body(...),
    current_situation: Dict = Body(...)
):
    """
    Genera plan de emergencia
    """
    try:
        plan = relapse_prevention.generate_emergency_plan(user_id, current_situation)
        return JSONResponse(content=plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando plan de emergencia: {str(e)}")


# Endpoints de Soporte y Motivación
@router.post("/coaching-session")
async def coaching_session(request: CoachingSessionRequest):
    """
    Sesión de coaching personalizado
    """
    try:
        session = counseling.create_coaching_session(
            request.user_id,
            request.topic,
            request.current_situation,
            request.questions
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en sesión de coaching: {str(e)}")


@router.get("/motivation/{user_id}")
async def get_motivation(user_id: str):
    """
    Obtiene mensajes motivacionales personalizados
    """
    try:
        messages = motivation.get_motivational_messages(user_id, {})
        return JSONResponse(content=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo motivación: {str(e)}")


@router.post("/celebrate-milestone")
async def celebrate_milestone(
    user_id: str = Body(...),
    milestone_days: int = Body(...)
):
    """
    Celebra un logro/hito
    """
    try:
        celebration = motivation.celebrate_milestone(user_id, milestone_days)
        return JSONResponse(content=celebration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error celebrando hito: {str(e)}")


@router.get("/achievements/{user_id}")
async def get_achievements(user_id: str):
    """
    Obtiene logros del usuario
    """
    try:
        achievements = motivation.get_achievements(user_id)
        return JSONResponse(content=achievements)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo logros: {str(e)}")


# Endpoints de Análisis y Reportes
@router.get("/analytics/{user_id}")
async def get_analytics(user_id: str):
    """
    Obtiene análisis completo del usuario
    """
    try:
        progress = tracker.get_progress(user_id, None, [])
        stats = tracker.get_stats(user_id, [])
        
        return JSONResponse(content={
            "user_id": user_id,
            "progress": progress,
            "statistics": stats,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo análisis: {str(e)}")


@router.post("/generate-report")
async def generate_report(
    user_id: str = Body(...),
    report_type: str = Body("summary")
):
    """
    Genera reporte detallado
    """
    try:
        progress = tracker.get_progress(user_id, None, [])
        stats = tracker.get_stats(user_id, [])
        
        report = {
            "user_id": user_id,
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "progress_summary": progress,
            "detailed_statistics": stats,
            "recommendations": []
        }
        
        return JSONResponse(content=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando reporte: {str(e)}")


@router.get("/insights/{user_id}")
async def get_insights(user_id: str):
    """
    Obtiene insights personalizados
    """
    try:
        progress = tracker.get_progress(user_id, None, [])
        stats = tracker.get_stats(user_id, [])
        
        insights = {
            "user_id": user_id,
            "key_insights": [
                f"Has estado sobrio por {progress.get('days_sober', 0)} días",
                "Continúa con tu plan de recuperación",
                "Mantén contacto regular con tu sistema de apoyo"
            ],
            "trends": stats.get("trends", {}),
            "recommendations": []
        }
        
        return JSONResponse(content=insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo insights: {str(e)}")


# Endpoints de Notificaciones
@router.get("/notifications/{user_id}")
async def get_notifications(user_id: str):
    """
    Obtiene notificaciones pendientes del usuario
    """
    try:
        pending = notifications.get_pending_notifications(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "notifications": pending,
            "unread_count": len([n for n in pending if not n.get("read", False)]),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo notificaciones: {str(e)}")


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """
    Marca una notificación como leída
    """
    try:
        success = notifications.mark_notification_read(notification_id)
        return JSONResponse(content={
            "notification_id": notification_id,
            "marked_read": success,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marcando notificación: {str(e)}")


@router.get("/reminders/{user_id}")
async def get_reminders(user_id: str):
    """
    Obtiene recordatorios diarios del usuario
    """
    try:
        reminders = notifications.get_daily_reminders(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "reminders": reminders,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recordatorios: {str(e)}")


# Endpoints de Análisis Avanzado
@router.get("/analytics/advanced/{user_id}")
async def get_advanced_analytics(user_id: str):
    """
    Obtiene análisis avanzado completo del usuario
    """
    try:
        # En implementación real, esto vendría de la base de datos
        entries = []
        analytics_data = analytics.generate_comprehensive_analytics(user_id, entries)
        return JSONResponse(content=analytics_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo análisis avanzado: {str(e)}")


# Endpoints de Usuario
@router.post("/users/create")
async def create_user(
    user_id: str = Body(...),
    email: Optional[str] = Body(None),
    name: Optional[str] = Body(None)
):
    """
    Crea un nuevo usuario
    """
    try:
        user = db_manager.create_user(user_id, email, name)
        return JSONResponse(content={
            "user_id": user.id,
            "email": user.email,
            "name": user.name,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando usuario: {str(e)}")


@router.get("/users/{user_id}")
async def get_user(user_id: str):
    """
    Obtiene información del usuario
    """
    try:
        user = db_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        return JSONResponse(content={
            "user_id": user.id,
            "email": user.email,
            "name": user.name,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "status": "success"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo usuario: {str(e)}")


# Endpoint de Exportación
@router.get("/export/{user_id}")
async def export_user_data(
    user_id: str,
    format: str = Query("json", regex="^(json|csv)$")
):
    """
    Exporta datos del usuario en formato JSON o CSV
    """
    try:
        # En implementación real, esto exportaría datos reales
        export_data = {
            "user_id": user_id,
            "format": format,
            "exported_at": datetime.now().isoformat(),
            "data": {
                "message": "Datos de exportación (implementar con datos reales de BD)"
            }
        }
        
        if format == "csv":
            # Convertir a CSV (implementación simplificada)
            return JSONResponse(content={
                "message": "Exportación CSV (implementar conversión)",
                "data": export_data
            })
        
        return JSONResponse(content=export_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando datos: {str(e)}")


# Endpoints de Autenticación
@router.post("/auth/register")
async def register(
    user_id: str = Body(...),
    email: Optional[str] = Body(None),
    password: Optional[str] = Body(None),
    name: Optional[str] = Body(None)
):
    """
    Registra un nuevo usuario
    """
    try:
        # Crear usuario
        user = db_manager.create_user(user_id, email, name)
        
        # Hash password si se proporciona
        hashed_password = None
        if password:
            hashed_password = auth.hash_password(password)
        
        # Crear token de acceso
        token_data = {"sub": user_id, "email": email}
        access_token = auth.create_access_token(data=token_data)
        
        return JSONResponse(content={
            "user_id": user.id,
            "email": user.email,
            "access_token": access_token,
            "token_type": "bearer",
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando usuario: {str(e)}")


@router.post("/auth/login")
async def login(
    user_id: str = Body(...),
    password: Optional[str] = Body(None)
):
    """
    Inicia sesión y obtiene token
    """
    try:
        user = db_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        # En implementación real, verificar password aquí
        token_data = {"sub": user_id, "email": user.email}
        access_token = auth.create_access_token(data=token_data)
        
        return JSONResponse(content={
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": user_id,
            "status": "success"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en login: {str(e)}")


# Endpoints de Reportes PDF
@router.get("/reports/pdf/{user_id}")
async def generate_pdf_report(user_id: str):
    """
    Genera reporte PDF del usuario
    """
    try:
        # Obtener datos del usuario
        user = db_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        progress = tracker.get_progress(user_id, None, [])
        analytics_data = analytics.generate_comprehensive_analytics(user_id, [])
        
        user_data = {
            "name": user.name,
            "email": user.email
        }
        
        pdf_bytes = reports.generate_pdf_report(user_id, user_data, progress, analytics_data)
        
        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=recovery_report_{user_id}.pdf"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando PDF: {str(e)}")


# Endpoints de Gamificación
@router.get("/gamification/points/{user_id}")
async def get_user_points(user_id: str):
    """
    Obtiene puntos y nivel del usuario
    """
    try:
        progress = tracker.get_progress(user_id, None, [])
        # En implementación real, obtener datos reales
        points = gamification.calculate_points(
            days_sober=progress.get("days_sober", 0),
            entries_count=0,
            milestones_achieved=0,
            coaching_sessions=0
        )
        
        return JSONResponse(content=points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo puntos: {str(e)}")


@router.get("/gamification/achievements/{user_id}")
async def get_user_achievements(user_id: str):
    """
    Obtiene logros del usuario
    """
    try:
        progress = tracker.get_progress(user_id, None, [])
        achievements = gamification.check_achievements(
            user_id=user_id,
            days_sober=progress.get("days_sober", 0),
            current_streak=progress.get("streak_days", 0),
            entries_count=0
        )
        
        return JSONResponse(content={
            "user_id": user_id,
            "achievements": achievements,
            "total_achievements": len(achievements),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo logros: {str(e)}")


@router.get("/gamification/leaderboard")
async def get_leaderboard(limit: int = Query(10, ge=1, le=100)):
    """
    Obtiene tabla de clasificación
    """
    try:
        # En implementación real, obtener datos de múltiples usuarios
        users_data = []
        leaderboard = gamification.get_leaderboard(users_data, limit)
        
        return JSONResponse(content={
            "leaderboard": leaderboard,
            "limit": limit,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo leaderboard: {str(e)}")


# Endpoints de Emergencia
@router.post("/emergency/contact")
async def create_emergency_contact(
    user_id: str = Body(...),
    name: str = Body(...),
    relationship: str = Body(...),
    phone: str = Body(...),
    email: Optional[str] = Body(None),
    is_primary: bool = Body(False)
):
    """
    Crea un contacto de emergencia
    """
    try:
        contact = emergency.create_emergency_contact(
            user_id, name, relationship, phone, email, is_primary
        )
        return JSONResponse(content=contact)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando contacto: {str(e)}")


@router.get("/emergency/contacts/{user_id}")
async def get_emergency_contacts(user_id: str):
    """
    Obtiene contactos de emergencia del usuario
    """
    try:
        contacts = emergency.get_emergency_contacts(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "contacts": contacts,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo contactos: {str(e)}")


@router.post("/emergency/trigger")
async def trigger_emergency(
    user_id: str = Body(...),
    risk_level: str = Body(...),
    situation: str = Body(...)
):
    """
    Activa protocolo de emergencia
    """
    try:
        protocol = emergency.trigger_emergency_protocol(user_id, risk_level, situation)
        return JSONResponse(content=protocol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activando protocolo: {str(e)}")


@router.get("/emergency/resources")
async def get_crisis_resources(location: Optional[str] = Query(None)):
    """
    Obtiene recursos de crisis
    """
    try:
        resources = emergency.get_crisis_resources(location)
        return JSONResponse(content={
            "resources": resources,
            "location": location,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recursos: {str(e)}")


# Endpoints de Dashboard
@router.get("/dashboard/{user_id}")
async def get_dashboard(user_id: str):
    """
    Obtiene datos completos del dashboard
    """
    try:
        progress = tracker.get_progress(user_id, None, [])
        stats = tracker.get_stats(user_id, [])
        analytics_data = analytics.generate_comprehensive_analytics(user_id, [])
        
        dashboard_data = dashboard.get_dashboard_data(
            user_id=user_id,
            entries=[],
            progress_data=progress,
            analytics_data=analytics_data
        )
        
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo dashboard: {str(e)}")


# Endpoints de Calendario
@router.post("/calendar/event")
async def create_calendar_event(
    user_id: str = Body(...),
    event_type: str = Body(...),
    title: str = Body(...),
    description: str = Body(...),
    scheduled_time: str = Body(...),
    repeat_daily: bool = Body(False),
    repeat_weekly: bool = Body(False),
    reminder_minutes: int = Body(15)
):
    """
    Crea un evento en el calendario
    """
    try:
        scheduled = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
        event = calendar.create_event(
            user_id, event_type, title, description, scheduled,
            repeat_daily, repeat_weekly, reminder_minutes
        )
        return JSONResponse(content=event)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando evento: {str(e)}")


@router.get("/calendar/upcoming/{user_id}")
async def get_upcoming_events(
    user_id: str,
    days_ahead: int = Query(7, ge=1, le=30)
):
    """
    Obtiene eventos próximos del usuario
    """
    try:
        events = calendar.get_upcoming_events(user_id, days_ahead)
        return JSONResponse(content={
            "user_id": user_id,
            "events": events,
            "days_ahead": days_ahead,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo eventos: {str(e)}")


@router.post("/calendar/daily-reminders/{user_id}")
async def create_daily_reminders(user_id: str):
    """
    Crea recordatorios diarios automáticos
    """
    try:
        reminders = calendar.create_daily_reminders(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "reminders_created": len(reminders),
            "reminders": reminders,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando recordatorios: {str(e)}")


# Endpoints de Chatbot
@router.post("/chatbot/message")
async def send_chatbot_message(
    user_id: str = Body(...),
    message: str = Body(...),
    context: Optional[Dict] = Body(None)
):
    """Envía un mensaje al chatbot y recibe respuesta"""
    try:
        response = chatbot.process_message(user_id, message, context)
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando mensaje: {str(e)}")


@router.post("/chatbot/start")
async def start_chatbot_conversation(
    user_id: str = Body(...),
    user_data: Optional[Dict] = Body(None)
):
    """Inicia una nueva conversación con el chatbot"""
    try:
        welcome = chatbot.start_conversation(user_id, user_data)
        return JSONResponse(content=welcome)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando conversación: {str(e)}")


@router.get("/chatbot/history/{user_id}")
async def get_chatbot_history(user_id: str, limit: int = Query(20, ge=1, le=100)):
    """Obtiene historial de conversación con el chatbot"""
    try:
        history = chatbot.get_conversation_history(user_id, limit)
        return JSONResponse(content={
            "user_id": user_id,
            "history": history,
            "limit": limit,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo historial: {str(e)}")


# Endpoints de Comunidad
@router.post("/community/post")
async def create_community_post(
    user_id: str = Body(...),
    post_type: str = Body(...),
    title: str = Body(...),
    content: str = Body(...),
    is_anonymous: bool = Body(False)
):
    """Crea una publicación en la comunidad"""
    try:
        post = community.create_post(user_id, post_type, title, content, is_anonymous)
        return JSONResponse(content=post)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando publicación: {str(e)}")


@router.get("/community/posts")
async def get_community_posts(
    post_type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Obtiene publicaciones de la comunidad"""
    try:
        posts = community.get_community_posts(post_type, limit, offset)
        return JSONResponse(content={
            "posts": posts,
            "total": len(posts),
            "limit": limit,
            "offset": offset,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo publicaciones: {str(e)}")


# Endpoints de Análisis Predictivo
@router.post("/predictive/relapse-risk")
async def predict_relapse_risk(
    user_id: str = Body(...),
    historical_data: List[Dict] = Body(...),
    current_state: Dict = Body(...)
):
    """Predice riesgo de recaída usando ML"""
    try:
        prediction = predictive.predict_relapse_risk(user_id, historical_data, current_state)
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


# Endpoints de Seguimiento de Salud
@router.post("/health/metric")
async def record_health_metric(
    user_id: str = Body(...),
    metric_type: str = Body(...),
    value: float = Body(...),
    unit: str = Body(...),
    notes: Optional[str] = Body(None)
):
    """Registra una métrica de salud"""
    try:
        metric = health_tracking.record_health_metric(user_id, metric_type, value, unit, notes)
        return JSONResponse(content=metric)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando métrica: {str(e)}")


@router.get("/health/summary/{user_id}")
async def get_health_summary(
    user_id: str,
    days_sober: int = Query(...),
    addiction_type: str = Query(...)
):
    """Obtiene resumen de salud del usuario"""
    try:
        summary = health_tracking.get_health_summary(user_id, days_sober, addiction_type)
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo resumen: {str(e)}")


# Endpoints de Metas
@router.post("/goals/create")
async def create_goal(
    user_id: str = Body(...),
    goal_type: str = Body(...),
    title: str = Body(...),
    description: str = Body(...),
    target_date: str = Body(...),
    target_value: Optional[float] = Body(None)
):
    """Crea una nueva meta"""
    try:
        goal = goals.create_goal(user_id, goal_type, title, description, target_date, target_value)
        return JSONResponse(content=goal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando meta: {str(e)}")


@router.get("/goals/{user_id}")
async def get_user_goals(user_id: str, status: Optional[str] = Query(None)):
    """Obtiene metas del usuario"""
    try:
        user_goals = goals.get_user_goals(user_id, status)
        return JSONResponse(content={
            "user_id": user_id,
            "goals": user_goals,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo metas: {str(e)}")


# Endpoints de Wearables
@router.post("/wearable/register")
async def register_wearable(
    user_id: str = Body(...),
    device_type: str = Body(...),
    device_name: str = Body(...),
    device_id: str = Body(...)
):
    """Registra un dispositivo wearable"""
    try:
        device = wearable.register_device(user_id, device_type, device_name, device_id)
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")


# Endpoints de Análisis de Sentimientos
@router.post("/sentiment/analyze")
async def analyze_sentiment(
    text: str = Body(...),
    context: Optional[Dict] = Body(None)
):
    """Analiza el sentimiento de un texto"""
    try:
        analysis = sentiment.analyze_sentiment(text, context)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando sentimiento: {str(e)}")


@router.post("/sentiment/journal-entry")
async def analyze_journal_entry(
    user_id: str = Body(...),
    entry_text: str = Body(...),
    entry_date: str = Body(...)
):
    """Analiza una entrada de diario"""
    try:
        analysis = sentiment.analyze_journal_entry(user_id, entry_text, entry_date)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando entrada: {str(e)}")


@router.get("/sentiment/trend/{user_id}")
async def get_emotional_trend(user_id: str):
    """Obtiene tendencia emocional del usuario"""
    try:
        sentiment_data = []
        trend = sentiment.track_emotional_trend(user_id, sentiment_data)
        return JSONResponse(content=trend)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo tendencia: {str(e)}")


# Endpoints de Mentoría
@router.post("/mentorship/request")
async def create_mentorship_request(
    mentee_id: str = Body(...),
    preferences: Dict = Body(...),
    goals: List[str] = Body(...)
):
    """Crea una solicitud de mentoría"""
    try:
        request = mentorship.create_mentorship_request(mentee_id, preferences, goals)
        return JSONResponse(content=request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando solicitud: {str(e)}")


@router.get("/mentorship/available")
async def get_available_mentors(
    addiction_type: Optional[str] = Query(None),
    experience_min: Optional[int] = Query(None)
):
    """Obtiene mentores disponibles"""
    try:
        mentors = mentorship.get_available_mentors(addiction_type, experience_min)
        return JSONResponse(content={
            "mentors": mentors,
            "total": len(mentors),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo mentores: {str(e)}")


# Endpoints de Medicamentos
@router.post("/medication/add")
async def add_medication(
    user_id: str = Body(...),
    medication_name: str = Body(...),
    dosage: str = Body(...),
    frequency: str = Body(...),
    start_date: str = Body(...)
):
    """Agrega un medicamento al seguimiento"""
    try:
        med = medication.add_medication(user_id, medication_name, dosage, frequency, start_date)
        return JSONResponse(content=med)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error agregando medicamento: {str(e)}")


@router.get("/medication/schedule/{user_id}")
async def get_medication_schedule(user_id: str):
    """Obtiene horario de medicamentos"""
    try:
        schedule = medication.get_medication_schedule(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "schedule": schedule,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo horario: {str(e)}")


# Endpoints de Integración con Apps de Salud
@router.post("/health-app/connect")
async def connect_health_app(
    user_id: str = Body(...),
    app_type: str = Body(...),
    access_token: str = Body(...)
):
    """Conecta una app de salud"""
    try:
        connection = health_integration.connect_health_app(user_id, app_type, access_token)
        return JSONResponse(content=connection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando app: {str(e)}")


@router.post("/health-app/sync")
async def sync_health_data(
    user_id: str = Body(...),
    app_type: str = Body(...),
    data_types: List[str] = Body(...)
):
    """Sincroniza datos de salud"""
    try:
        sync_result = health_integration.sync_health_data(user_id, app_type, data_types)
        return JSONResponse(content=sync_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sincronizando datos: {str(e)}")


# Endpoints de Notificaciones Push
@router.post("/push/register-device")
async def register_push_device(
    user_id: str = Body(...),
    device_token: str = Body(...),
    platform: str = Body(...)
):
    """Registra dispositivo para notificaciones push"""
    try:
        device = push_notifications.register_device(user_id, device_token, platform)
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")


@router.post("/push/send")
async def send_push_notification(
    user_id: str = Body(...),
    title: str = Body(...),
    body: str = Body(...),
    priority: str = Body("normal")
):
    """Envía una notificación push"""
    try:
        notification = push_notifications.send_notification(user_id, title, body, priority)
        return JSONResponse(content=notification)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enviando notificación: {str(e)}")


# Endpoints de Análisis de Voz
@router.post("/voice/analyze")
async def analyze_voice(
    user_id: str = Body(...),
    duration_seconds: float = Body(...),
    metadata: Optional[Dict] = Body(None)
):
    """Analiza una grabación de voz"""
    try:
        # En implementación real, recibiría audio_data como archivo
        audio_data = b""  # Placeholder
        analysis = voice_analysis.analyze_voice_recording(user_id, audio_data, duration_seconds, metadata)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando voz: {str(e)}")


# Endpoints de Seguimiento Familiar
@router.post("/family/add-member")
async def add_family_member(
    user_id: str = Body(...),
    family_member_name: str = Body(...),
    relationship: str = Body(...),
    email: Optional[str] = Body(None),
    can_view_progress: bool = Body(True)
):
    """Agrega un miembro de la familia"""
    try:
        member = family_tracking.add_family_member(
            user_id, family_member_name, relationship, email, None, can_view_progress
        )
        return JSONResponse(content=member)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error agregando familiar: {str(e)}")


@router.get("/family/dashboard/{user_id}")
async def get_family_dashboard(
    user_id: str,
    family_member_id: Optional[str] = Query(None)
):
    """Obtiene dashboard para familiares"""
    try:
        dashboard = family_tracking.get_family_dashboard(user_id, family_member_id)
        return JSONResponse(content=dashboard)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo dashboard: {str(e)}")


# Endpoints de Alertas Inteligentes
@router.post("/alerts/evaluate")
async def evaluate_alerts(
    user_id: str = Body(...),
    user_data: Dict = Body(...),
    recent_activity: List[Dict] = Body(...)
):
    """Evalúa condiciones y genera alertas"""
    try:
        alerts = intelligent_alerts.evaluate_alert_conditions(user_id, user_data, recent_activity)
        return JSONResponse(content={
            "user_id": user_id,
            "alerts": alerts,
            "total": len(alerts),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando alertas: {str(e)}")


@router.get("/alerts/active/{user_id}")
async def get_active_alerts(
    user_id: str,
    severity: Optional[str] = Query(None)
):
    """Obtiene alertas activas"""
    try:
        alerts = intelligent_alerts.get_active_alerts(user_id, severity)
        return JSONResponse(content={
            "user_id": user_id,
            "alerts": alerts,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo alertas: {str(e)}")


# Endpoints de Terapias Virtuales
@router.post("/therapy/schedule")
async def schedule_therapy(
    user_id: str = Body(...),
    therapy_type: str = Body(...),
    scheduled_time: str = Body(...),
    duration_minutes: int = Body(60)
):
    """Programa una sesión de terapia"""
    try:
        scheduled = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
        session = virtual_therapy.schedule_therapy_session(
            user_id, therapy_type, None, scheduled, duration_minutes
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error programando sesión: {str(e)}")


@router.get("/therapy/therapists")
async def get_available_therapists(
    therapy_type: Optional[str] = Query(None),
    specialization: Optional[str] = Query(None)
):
    """Obtiene terapeutas disponibles"""
    try:
        therapists = virtual_therapy.get_available_therapists(therapy_type, specialization)
        return JSONResponse(content={
            "therapists": therapists,
            "total": len(therapists),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo terapeutas: {str(e)}")


# Endpoints de Visualizaciones
@router.post("/visualization/progress-chart")
async def generate_progress_chart(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    chart_type: str = Body("line")
):
    """Genera gráfico de progreso"""
    try:
        chart = visualization.generate_progress_chart(user_id, data, chart_type)
        return JSONResponse(content=chart)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando gráfico: {str(e)}")


@router.post("/visualization/radar")
async def generate_radar_chart(
    user_id: str = Body(...),
    metrics: Dict = Body(...)
):
    """Genera gráfico de radar"""
    try:
        chart = visualization.generate_radar_chart(user_id, metrics)
        return JSONResponse(content=chart)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando gráfico: {str(e)}")


# Endpoints de Economía Virtual
@router.post("/economy/earn-points")
async def earn_points(
    user_id: str = Body(...),
    action_type: str = Body(...),
    amount: int = Body(...),
    description: str = Body(...)
):
    """Otorga puntos al usuario"""
    try:
        transaction = virtual_economy.earn_points(user_id, action_type, amount, description)
        return JSONResponse(content=transaction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error otorgando puntos: {str(e)}")


@router.get("/economy/balance/{user_id}")
async def get_user_balance(user_id: str):
    """Obtiene balance de puntos del usuario"""
    try:
        balance = virtual_economy.get_user_balance(user_id)
        return JSONResponse(content=balance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo balance: {str(e)}")


@router.get("/economy/rewards")
async def get_rewards_catalog(
    category: Optional[str] = Query(None),
    max_points: Optional[int] = Query(None)
):
    """Obtiene catálogo de recompensas"""
    try:
        rewards = virtual_economy.get_rewards_catalog(category, max_points)
        return JSONResponse(content={
            "rewards": rewards,
            "total": len(rewards),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recompensas: {str(e)}")


# Endpoints de Integración con Emergencias
@router.get("/emergency/services")
async def get_emergency_services(
    location: Optional[str] = Query(None),
    service_type: Optional[str] = Query(None)
):
    """Obtiene servicios de emergencia disponibles"""
    try:
        services = emergency_integration.get_emergency_services(location, service_type)
        return JSONResponse(content={
            "services": services,
            "location": location,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo servicios: {str(e)}")


@router.get("/emergency/crisis-resources")
async def get_crisis_resources(location: Optional[str] = Query(None)):
    """Obtiene recursos de crisis"""
    try:
        resources = emergency_integration.get_crisis_resources(location)
        return JSONResponse(content=resources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recursos: {str(e)}")


# Endpoints de Síntomas de Abstinencia
@router.post("/withdrawal/record-symptom")
async def record_withdrawal_symptom(
    user_id: str = Body(...),
    symptom_name: str = Body(...),
    severity: str = Body(...),
    notes: Optional[str] = Body(None)
):
    """Registra un síntoma de abstinencia"""
    try:
        symptom = withdrawal_tracking.record_symptom(user_id, symptom_name, severity, notes)
        return JSONResponse(content=symptom)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando síntoma: {str(e)}")


@router.get("/withdrawal/timeline/{user_id}")
async def get_withdrawal_timeline(
    user_id: str,
    addiction_type: str = Query(...),
    days_sober: int = Query(...)
):
    """Obtiene línea de tiempo de síntomas de abstinencia"""
    try:
        timeline = withdrawal_tracking.get_withdrawal_timeline(user_id, addiction_type, days_sober)
        return JSONResponse(content=timeline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo timeline: {str(e)}")


# Endpoints de Análisis de Sueño
@router.post("/sleep/analyze-patterns")
async def analyze_sleep_patterns(
    user_id: str = Body(...),
    sleep_data: List[Dict] = Body(...)
):
    """Analiza patrones de sueño"""
    try:
        analysis = sleep_analysis.analyze_sleep_patterns(user_id, sleep_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando sueño: {str(e)}")


@router.post("/sleep/correlate")
async def correlate_sleep_with_recovery(
    user_id: str = Body(...),
    sleep_data: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Correlaciona sueño con recuperación"""
    try:
        correlation = sleep_analysis.correlate_sleep_with_recovery(user_id, sleep_data, recovery_data)
        return JSONResponse(content=correlation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error correlacionando datos: {str(e)}")


# Endpoints de Desafíos
@router.post("/challenges/create")
async def create_challenge(
    user_id: str = Body(...),
    challenge_type: str = Body(...),
    title: str = Body(...),
    description: str = Body(...),
    duration_days: int = Body(...)
):
    """Crea un nuevo desafío"""
    try:
        challenge = challenges.create_challenge(
            user_id, challenge_type, title, description, duration_days
        )
        return JSONResponse(content=challenge)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando desafío: {str(e)}")


@router.get("/challenges/available/{user_id}")
async def get_available_challenges(
    user_id: str,
    challenge_type: Optional[str] = Query(None)
):
    """Obtiene desafíos disponibles"""
    try:
        available = challenges.get_available_challenges(user_id, challenge_type)
        return JSONResponse(content={
            "user_id": user_id,
            "challenges": available,
            "total": len(available),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo desafíos: {str(e)}")


# Endpoints de Webhooks
@router.post("/webhooks/register")
async def register_webhook(
    user_id: str = Body(...),
    url: str = Body(...),
    event_types: List[str] = Body(...),
    secret: Optional[str] = Body(None)
):
    """Registra un webhook"""
    try:
        webhook = webhooks.register_webhook(user_id, url, event_types, secret)
        return JSONResponse(content=webhook)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando webhook: {str(e)}")


@router.get("/webhooks/{user_id}")
async def get_user_webhooks(user_id: str, active_only: bool = Query(True)):
    """Obtiene webhooks del usuario"""
    try:
        webhooks_list = webhooks.get_webhooks(user_id, active_only)
        return JSONResponse(content={
            "user_id": user_id,
            "webhooks": webhooks_list,
            "total": len(webhooks_list),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo webhooks: {str(e)}")


# Endpoints de Certificados
@router.post("/certificates/generate")
async def generate_certificate(
    user_id: str = Body(...),
    certificate_type: str = Body(...),
    title: str = Body(...),
    description: str = Body(...)
):
    """Genera un certificado"""
    try:
        certificate = certificates.generate_certificate(
            user_id, certificate_type, title, description
        )
        return JSONResponse(content=certificate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando certificado: {str(e)}")


@router.get("/certificates/{user_id}")
async def get_user_certificates(
    user_id: str,
    certificate_type: Optional[str] = Query(None)
):
    """Obtiene certificados del usuario"""
    try:
        certs = certificates.get_user_certificates(user_id, certificate_type)
        return JSONResponse(content={
            "user_id": user_id,
            "certificates": certs,
            "total": len(certs),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo certificados: {str(e)}")


# Endpoints de Backup
@router.post("/backup/create")
async def create_backup(
    user_id: str = Body(...),
    backup_type: str = Body("full"),
    include_data: Optional[List[str]] = Body(None)
):
    """Crea un backup de datos"""
    try:
        backup_result = backup.create_backup(user_id, backup_type, include_data)
        return JSONResponse(content=backup_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando backup: {str(e)}")


@router.post("/backup/restore")
async def restore_backup(
    user_id: str = Body(...),
    backup_id: str = Body(...)
):
    """Restaura datos desde backup"""
    try:
        restore_result = backup.restore_backup(user_id, backup_id)
        return JSONResponse(content=restore_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restaurando backup: {str(e)}")


# Endpoints de Integración Social
@router.post("/social/connect")
async def connect_social_account(
    user_id: str = Body(...),
    platform: str = Body(...),
    access_token: str = Body(...)
):
    """Conecta cuenta de red social"""
    try:
        connection = social_integration.connect_social_account(user_id, platform, access_token)
        return JSONResponse(content=connection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando cuenta: {str(e)}")


@router.post("/social/share-milestone")
async def share_milestone(
    user_id: str = Body(...),
    milestone_data: Dict = Body(...),
    platforms: List[str] = Body(...)
):
    """Comparte hito en redes sociales"""
    try:
        share_result = social_integration.share_milestone(user_id, milestone_data, platforms)
        return JSONResponse(content=share_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error compartiendo hito: {str(e)}")


# Endpoints de Análisis NLP
@router.post("/nlp/analyze-text")
async def analyze_text(
    text: str = Body(...),
    analysis_type: str = Body("comprehensive")
):
    """Analiza texto con NLP"""
    try:
        analysis = nlp_analysis.analyze_text(text, analysis_type)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando texto: {str(e)}")


@router.post("/nlp/extract-insights")
async def extract_insights(
    text: str = Body(...),
    context: Optional[Dict] = Body(None)
):
    """Extrae insights del texto"""
    try:
        insights = nlp_analysis.extract_insights(text, context)
        return JSONResponse(content=insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extrayendo insights: {str(e)}")


# Endpoints de Grupos de Apoyo Avanzado
@router.post("/support-groups/create")
async def create_support_group(
    creator_id: str = Body(...),
    name: str = Body(...),
    description: str = Body(...),
    group_type: str = Body("public")
):
    """Crea un grupo de apoyo"""
    try:
        group = advanced_support_groups.create_support_group(
            creator_id, name, description, group_type
        )
        return JSONResponse(content=group)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando grupo: {str(e)}")


@router.get("/support-groups/search")
async def search_support_groups(
    query: Optional[str] = Query(None),
    addiction_type: Optional[str] = Query(None)
):
    """Busca grupos de apoyo"""
    try:
        groups = advanced_support_groups.search_groups(query, addiction_type)
        return JSONResponse(content={
            "groups": groups,
            "total": len(groups),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error buscando grupos: {str(e)}")


# Endpoints de Reportes Avanzados
@router.post("/reports/comprehensive")
async def generate_comprehensive_report(
    user_id: str = Body(...),
    start_date: Optional[str] = Body(None),
    end_date: Optional[str] = Body(None)
):
    """Genera reporte comprensivo"""
    try:
        report = advanced_reporting.generate_comprehensive_report(
            user_id, start_date, end_date
        )
        return JSONResponse(content=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando reporte: {str(e)}")


@router.post("/reports/export")
async def export_report(
    user_id: str = Body(...),
    report_data: Dict = Body(...),
    format: str = Body("pdf")
):
    """Exporta reporte en formato específico"""
    try:
        export_result = advanced_reporting.export_report(user_id, report_data, format)
        return JSONResponse(content=export_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando reporte: {str(e)}")


# Endpoints de IA Predictiva
@router.post("/predictive/success-probability")
async def predict_success_probability(
    user_id: str = Body(...),
    days_sober: int = Body(...),
    historical_data: Dict = Body(...)
):
    """Predice probabilidad de éxito en recuperación"""
    try:
        prediction = predictive_ai.predict_success_probability(user_id, days_sober, historical_data)
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo probabilidad: {str(e)}")


@router.post("/predictive/relapse-window")
async def predict_relapse_window(
    user_id: str = Body(...),
    current_data: Dict = Body(...)
):
    """Predice ventana de riesgo de recaída"""
    try:
        prediction = predictive_ai.predict_relapse_window(user_id, current_data)
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo ventana: {str(e)}")


# Endpoints de Recomendaciones
@router.post("/recommendations/personalized")
async def get_personalized_recommendations(
    user_id: str = Body(...),
    context: Dict = Body(...)
):
    """Obtiene recomendaciones personalizadas"""
    try:
        recs = recommendations.get_personalized_recommendations(user_id, context)
        return JSONResponse(content=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recomendaciones: {str(e)}")


@router.get("/recommendations/resources/{user_id}")
async def get_resource_recommendations(
    user_id: str,
    resource_type: Optional[str] = Query(None)
):
    """Obtiene recomendaciones de recursos"""
    try:
        resources = recommendations.get_resource_recommendations(user_id, resource_type)
        return JSONResponse(content={
            "user_id": user_id,
            "resources": resources,
            "total": len(resources),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recursos: {str(e)}")


# Endpoints de Seguimiento de Hábitos
@router.post("/habits/create")
async def create_habit(
    user_id: str = Body(...),
    name: str = Body(...),
    description: str = Body(...),
    frequency: str = Body("daily")
):
    """Crea un nuevo hábito"""
    try:
        habit = habit_tracking.create_habit(user_id, name, description, frequency)
        return JSONResponse(content=habit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando hábito: {str(e)}")


@router.post("/habits/log-completion")
async def log_habit_completion(
    habit_id: str = Body(...),
    user_id: str = Body(...),
    value: Optional[float] = Body(None)
):
    """Registra completación de hábito"""
    try:
        completion = habit_tracking.log_habit_completion(habit_id, user_id, value)
        return JSONResponse(content=completion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando completación: {str(e)}")


# Endpoints de Mindfulness
@router.post("/mindfulness/start-session")
async def start_meditation_session(
    user_id: str = Body(...),
    meditation_type: str = Body(...),
    duration_minutes: int = Body(10)
):
    """Inicia sesión de meditación"""
    try:
        session = mindfulness.start_meditation_session(user_id, meditation_type, duration_minutes)
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando sesión: {str(e)}")


@router.get("/mindfulness/programs/{user_id}")
async def get_meditation_programs(
    user_id: str,
    difficulty: Optional[str] = Query(None)
):
    """Obtiene programas de meditación"""
    try:
        programs = mindfulness.get_meditation_programs(user_id, difficulty)
        return JSONResponse(content={
            "user_id": user_id,
            "programs": programs,
            "total": len(programs),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo programas: {str(e)}")


# Endpoints de Biblioteca de Recursos
@router.get("/resources/library")
async def get_resources(
    resource_type: Optional[str] = Query(None),
    topic: Optional[str] = Query(None)
):
    """Obtiene recursos educativos"""
    try:
        resources = resource_library.get_resources(resource_type, topic)
        return JSONResponse(content={
            "resources": resources,
            "total": len(resources),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recursos: {str(e)}")


@router.get("/resources/search")
async def search_resources(query: str = Query(...)):
    """Busca recursos educativos"""
    try:
        results = resource_library.search_resources(query)
        return JSONResponse(content={
            "query": query,
            "results": results,
            "total": len(results),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error buscando recursos: {str(e)}")


# Endpoints de Análisis de Patrones Temporales
@router.post("/patterns/daily")
async def analyze_daily_patterns(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    metric: str = Body("mood")
):
    """Analiza patrones diarios"""
    try:
        analysis = temporal_patterns.analyze_daily_patterns(user_id, data, metric)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/patterns/weekly")
async def analyze_weekly_patterns(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    metric: str = Body("check_ins")
):
    """Analiza patrones semanales"""
    try:
        analysis = temporal_patterns.analyze_weekly_patterns(user_id, data, metric)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


# Endpoints de Seguimiento Financiero
@router.post("/financial/calculate-savings")
async def calculate_savings(
    user_id: str = Body(...),
    addiction_type: str = Body(...),
    days_sober: int = Body(...),
    daily_cost: float = Body(...)
):
    """Calcula ahorros por días de sobriedad"""
    try:
        savings = financial_tracking.calculate_savings(user_id, addiction_type, days_sober, daily_cost)
        return JSONResponse(content=savings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando ahorros: {str(e)}")


@router.get("/financial/summary/{user_id}")
async def get_financial_summary(user_id: str, days: int = Query(30)):
    """Obtiene resumen financiero"""
    try:
        summary = financial_tracking.get_financial_summary(user_id, days)
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo resumen: {str(e)}")


# Endpoints de Recordatorios Inteligentes
@router.post("/reminders/create")
async def create_reminder(
    user_id: str = Body(...),
    reminder_type: str = Body(...),
    title: str = Body(...),
    message: str = Body(...),
    scheduled_time: str = Body(...)
):
    """Crea un recordatorio"""
    try:
        reminder = intelligent_reminders.create_reminder(
            user_id, reminder_type, title, message, scheduled_time
        )
        return JSONResponse(content=reminder)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando recordatorio: {str(e)}")


@router.get("/reminders/upcoming/{user_id}")
async def get_upcoming_reminders(user_id: str, hours_ahead: int = Query(24)):
    """Obtiene recordatorios próximos"""
    try:
        reminders = intelligent_reminders.get_upcoming_reminders(user_id, hours_ahead)
        return JSONResponse(content={
            "user_id": user_id,
            "reminders": reminders,
            "total": len(reminders),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recordatorios: {str(e)}")


# Endpoints de Análisis de Bienestar
@router.post("/wellness/calculate-score")
async def calculate_wellness_score(
    user_id: str = Body(...),
    metrics: Dict = Body(...)
):
    """Calcula puntuación de bienestar general"""
    try:
        score = wellness_analysis.calculate_wellness_score(user_id, metrics)
        return JSONResponse(content=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando bienestar: {str(e)}")


@router.post("/wellness/trends")
async def analyze_wellness_trends(
    user_id: str = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Analiza tendencias de bienestar"""
    try:
        trends = wellness_analysis.analyze_wellness_trends(user_id, historical_data)
        return JSONResponse(content=trends)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando tendencias: {str(e)}")


# Endpoints de Relaciones Sociales
@router.post("/relationships/add")
async def add_relationship(
    user_id: str = Body(...),
    contact_name: str = Body(...),
    relationship_type: str = Body(...),
    contact_info: Dict = Body(...)
):
    """Agrega una relación"""
    try:
        relationship = social_relationships.add_relationship(
            user_id, contact_name, relationship_type, contact_info
        )
        return JSONResponse(content=relationship)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error agregando relación: {str(e)}")


@router.get("/relationships/network/{user_id}")
async def get_support_network(user_id: str):
    """Obtiene red de apoyo del usuario"""
    try:
        network = social_relationships.get_support_network(user_id)
        return JSONResponse(content=network)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo red: {str(e)}")


# Endpoints de Coaching en Tiempo Real
@router.post("/coaching/start-session")
async def start_coaching_session(
    user_id: str = Body(...),
    session_type: str = Body("general")
):
    """Inicia sesión de coaching"""
    try:
        session = realtime_coaching.start_coaching_session(user_id, session_type)
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando sesión: {str(e)}")


@router.post("/coaching/send-message")
async def send_coaching_message(
    session_id: str = Body(...),
    user_id: str = Body(...),
    message: str = Body(...)
):
    """Envía mensaje en sesión de coaching"""
    try:
        response = realtime_coaching.send_coaching_message(session_id, user_id, message)
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enviando mensaje: {str(e)}")


# Endpoints de Integración con Apps de Terceros
@router.post("/integrations/connect")
async def connect_integration(
    user_id: str = Body(...),
    integration_type: str = Body(...),
    app_name: str = Body(...),
    credentials: Dict = Body(...)
):
    """Conecta una integración"""
    try:
        integration = third_party_integration.connect_integration(
            user_id, integration_type, app_name, credentials
        )
        return JSONResponse(content=integration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando integración: {str(e)}")


@router.get("/integrations/available")
async def get_available_integrations(integration_type: Optional[str] = Query(None)):
    """Obtiene integraciones disponibles"""
    try:
        integrations = third_party_integration.get_available_integrations(integration_type)
        return JSONResponse(content={
            "integrations": integrations,
            "total": len(integrations),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo integraciones: {str(e)}")


# Endpoints de Seguimiento Visual Avanzado
@router.post("/progress/visualization")
async def generate_progress_visualization(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    visualization_type: str = Body("comprehensive")
):
    """Genera visualización avanzada de progreso"""
    try:
        visualization = advanced_progress.generate_progress_visualization(
            user_id, data, visualization_type
        )
        return JSONResponse(content=visualization)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando visualización: {str(e)}")


@router.post("/progress/comparison")
async def generate_comparison_view(
    user_id: str = Body(...),
    period1_data: List[Dict] = Body(...),
    period2_data: List[Dict] = Body(...)
):
    """Genera vista comparativa entre períodos"""
    try:
        comparison = advanced_progress.generate_comparison_view(
            user_id, period1_data, period2_data
        )
        return JSONResponse(content=comparison)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando comparación: {str(e)}")


# Endpoints de Gamificación Avanzada
@router.post("/gamification/award-achievement")
async def award_achievement(
    user_id: str = Body(...),
    achievement_id: str = Body(...)
):
    """Otorga un logro"""
    try:
        achievement = advanced_gamification.award_achievement(user_id, achievement_id)
        return JSONResponse(content=achievement)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error otorgando logro: {str(e)}")


@router.get("/gamification/user-level/{user_id}")
async def get_user_level(user_id: str, total_points: int = Query(...)):
    """Obtiene nivel del usuario"""
    try:
        level_info = advanced_gamification.calculate_user_level(user_id, total_points)
        return JSONResponse(content=level_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo nivel: {str(e)}")


# Endpoints de Análisis de Datos Avanzado
@router.post("/analysis/comprehensive")
async def perform_comprehensive_analysis(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    analysis_type: str = Body("full")
):
    """Realiza análisis comprensivo"""
    try:
        analysis = advanced_data_analysis.perform_comprehensive_analysis(
            user_id, data, analysis_type
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error realizando análisis: {str(e)}")


@router.post("/analysis/behavioral-patterns")
async def analyze_behavioral_patterns(
    user_id: str = Body(...),
    behavioral_data: List[Dict] = Body(...)
):
    """Analiza patrones de comportamiento"""
    try:
        patterns = advanced_data_analysis.analyze_behavioral_patterns(
            user_id, behavioral_data
        )
        return JSONResponse(content=patterns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


# Endpoints de Objetivos a Largo Plazo
@router.post("/long-term-goals/create")
async def create_long_term_goal(
    user_id: str = Body(...),
    title: str = Body(...),
    description: str = Body(...),
    target_date: str = Body(...)
):
    """Crea un objetivo a largo plazo"""
    try:
        goal = long_term_goals.create_long_term_goal(
            user_id, title, description, target_date
        )
        return JSONResponse(content=goal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando objetivo: {str(e)}")


@router.post("/long-term-goals/update-progress")
async def update_goal_progress(
    goal_id: str = Body(...),
    user_id: str = Body(...),
    progress: float = Body(...)
):
    """Actualiza progreso de objetivo"""
    try:
        result = long_term_goals.update_goal_progress(goal_id, user_id, progress)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando progreso: {str(e)}")


# Endpoints de Análisis de Riesgo Avanzado
@router.post("/risk/assessment")
async def perform_risk_assessment(
    user_id: str = Body(...),
    current_data: Dict = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Realiza evaluación de riesgo completa"""
    try:
        assessment = advanced_risk_analysis.perform_risk_assessment(
            user_id, current_data, historical_data
        )
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando riesgo: {str(e)}")


@router.post("/risk/predict-relapse")
async def predict_relapse_probability(
    user_id: str = Body(...),
    time_horizon: int = Body(7),
    current_data: Optional[Dict] = Body(None)
):
    """Predice probabilidad de recaída"""
    try:
        prediction = advanced_risk_analysis.predict_relapse_probability(
            user_id, time_horizon, current_data
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo recaída: {str(e)}")


# Endpoints de Métricas Avanzadas
@router.post("/metrics/kpis")
async def calculate_recovery_kpis(
    user_id: str = Body(...),
    data: List[Dict] = Body(...)
):
    """Calcula KPIs de recuperación"""
    try:
        kpis = advanced_metrics.calculate_recovery_kpis(user_id, data)
        return JSONResponse(content=kpis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando KPIs: {str(e)}")


@router.get("/metrics/dashboard/{user_id}")
async def generate_metrics_dashboard(
    user_id: str,
    period: str = Query("30d")
):
    """Genera dashboard de métricas"""
    try:
        dashboard = advanced_metrics.generate_metrics_dashboard(user_id, period)
        return JSONResponse(content=dashboard)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando dashboard: {str(e)}")


# Endpoints de Notificaciones Inteligentes
@router.post("/notifications/intelligent")
async def create_intelligent_notification(
    user_id: str = Body(...),
    notification_type: str = Body(...),
    context: Dict = Body(...)
):
    """Crea notificación inteligente"""
    try:
        notification = intelligent_notifications.create_intelligent_notification(
            user_id, notification_type, context
        )
        return JSONResponse(content=notification)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando notificación: {str(e)}")


@router.get("/notifications/preferences/{user_id}")
async def get_notification_preferences(user_id: str):
    """Obtiene preferencias de notificación"""
    try:
        preferences = intelligent_notifications.get_notification_preferences(user_id)
        return JSONResponse(content=preferences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo preferencias: {str(e)}")


# Endpoints de Medicamentos Avanzados
@router.post("/medication/advanced/add")
async def add_medication_advanced(
    user_id: str = Body(...),
    name: str = Body(...),
    dosage: str = Body(...),
    frequency: str = Body(...),
    start_date: str = Body(...)
):
    """Agrega un medicamento con seguimiento avanzado"""
    try:
        medication = advanced_medication.add_medication(
            user_id, name, dosage, frequency, start_date
        )
        return JSONResponse(content=medication)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error agregando medicamento: {str(e)}")


@router.get("/medication/advanced/adherence/{medication_id}")
async def get_medication_adherence(
    medication_id: str,
    user_id: str = Query(...),
    days: int = Query(30)
):
    """Obtiene adherencia a medicamento"""
    try:
        adherence = advanced_medication.get_medication_adherence(medication_id, user_id, days)
        return JSONResponse(content=adherence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo adherencia: {str(e)}")


# Endpoints de Integración IoT
@router.post("/iot/register-device")
async def register_iot_device(
    user_id: str = Body(...),
    device_type: str = Body(...),
    device_name: str = Body(...),
    device_id: str = Body(...),
    connection_info: Dict = Body(...)
):
    """Registra dispositivo IoT"""
    try:
        device = iot_integration.register_iot_device(
            user_id, device_type, device_name, device_id, connection_info
        )
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")


@router.get("/iot/health-metrics/{user_id}")
async def get_iot_health_metrics(
    user_id: str,
    device_type: Optional[str] = Query(None),
    days: int = Query(7)
):
    """Obtiene métricas de salud de dispositivos IoT"""
    try:
        metrics = iot_integration.get_iot_health_metrics(user_id, device_type, days)
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo métricas: {str(e)}")


# Endpoints de Análisis de Voz Avanzado
@router.post("/voice/advanced/analyze")
async def analyze_voice_advanced(
    user_id: str = Body(...),
    audio_data: str = Body(...),  # Base64 encoded
    metadata: Optional[Dict] = Body(None)
):
    """Analiza grabación de voz avanzada"""
    try:
        # Decodificar base64 en implementación real
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        analysis = advanced_voice.analyze_voice_recording(user_id, audio_bytes, metadata)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando voz: {str(e)}")


@router.post("/voice/advanced/detect-stress")
async def detect_voice_stress(
    user_id: str = Body(...),
    audio_data: str = Body(...)
):
    """Detecta estrés en la voz"""
    try:
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        stress_analysis = advanced_voice.detect_voice_stress(user_id, audio_bytes)
        return JSONResponse(content=stress_analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detectando estrés: {str(e)}")


# Endpoints de Seguimiento de Ubicación
@router.post("/location/add")
async def add_location(
    user_id: str = Body(...),
    location_type: str = Body(...),
    name: str = Body(...),
    latitude: float = Body(...),
    longitude: float = Body(...)
):
    """Agrega una ubicación"""
    try:
        location = location_tracking.add_location(
            user_id, location_type, name, latitude, longitude
        )
        return JSONResponse(content=location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error agregando ubicación: {str(e)}")


@router.post("/location/check-proximity")
async def check_location_proximity(
    user_id: str = Body(...),
    current_latitude: float = Body(...),
    current_longitude: float = Body(...)
):
    """Verifica proximidad a ubicaciones registradas"""
    try:
        proximity = location_tracking.check_location_proximity(
            user_id, current_latitude, current_longitude
        )
        return JSONResponse(content=proximity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verificando proximidad: {str(e)}")


# Endpoints de Análisis de Imágenes Emocionales
@router.post("/image/analyze-emotions")
async def analyze_image_emotions(
    user_id: str = Body(...),
    image_data: str = Body(...),  # Base64 encoded
    metadata: Optional[Dict] = Body(None)
):
    """Analiza emociones en una imagen"""
    try:
        import base64
        image_bytes = base64.b64decode(image_data)
        
        analysis = image_emotion.analyze_image_emotions(user_id, image_bytes, metadata)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando imagen: {str(e)}")


# Endpoints de Seguimiento de Sueño Avanzado
@router.post("/sleep/advanced/record")
async def record_sleep_data_advanced(
    user_id: str = Body(...),
    sleep_start: str = Body(...),
    sleep_end: str = Body(...),
    sleep_stages: Optional[Dict] = Body(None)
):
    """Registra datos de sueño avanzado"""
    try:
        sleep_record = advanced_sleep.record_sleep_data(
            user_id, sleep_start, sleep_end, sleep_stages
        )
        return JSONResponse(content=sleep_record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando sueño: {str(e)}")


@router.post("/sleep/advanced/analyze-patterns")
async def analyze_sleep_patterns_advanced(
    user_id: str = Body(...),
    sleep_data: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones de sueño avanzado"""
    try:
        analysis = advanced_sleep.analyze_sleep_patterns_advanced(user_id, sleep_data, days)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


# Endpoints de Aprendizaje Automático
@router.post("/ml/train-model")
async def train_personalized_model(
    user_id: str = Body(...),
    training_data: List[Dict] = Body(...),
    model_type: str = Body("relapse_prediction")
):
    """Entrena modelo personalizado para usuario"""
    try:
        model = ml_learning.train_personalized_model(user_id, training_data, model_type)
        return JSONResponse(content=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")


@router.post("/ml/predict")
async def predict_with_ml_model(
    user_id: str = Body(...),
    model_id: str = Body(...),
    input_features: Dict = Body(...)
):
    """Predice usando modelo ML"""
    try:
        prediction = ml_learning.predict_with_ml_model(user_id, model_id, input_features)
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo: {str(e)}")


# Endpoints de Análisis Predictivo ML Avanzado
@router.post("/predictive-ml/recovery-trajectory")
async def predict_recovery_trajectory(
    user_id: str = Body(...),
    historical_data: List[Dict] = Body(...),
    prediction_horizon: int = Body(30)
):
    """Predice trayectoria de recuperación"""
    try:
        trajectory = advanced_predictive_ml.predict_recovery_trajectory(
            user_id, historical_data, prediction_horizon
        )
        return JSONResponse(content=trajectory)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo trayectoria: {str(e)}")


@router.post("/predictive-ml/long-term-outcome")
async def predict_long_term_outcome(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Predice resultado a largo plazo"""
    try:
        outcome = advanced_predictive_ml.predict_long_term_outcome(
            user_id, current_state, historical_data
        )
        return JSONResponse(content=outcome)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo resultado: {str(e)}")


# Endpoints de Blockchain
@router.post("/blockchain/mint-nft")
async def mint_achievement_nft(
    user_id: str = Body(...),
    achievement_id: str = Body(...),
    achievement_data: Dict = Body(...)
):
    """Crea NFT de logro en blockchain"""
    try:
        nft = blockchain.mint_achievement_nft(user_id, achievement_id, achievement_data)
        return JSONResponse(content=nft)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando NFT: {str(e)}")


@router.post("/blockchain/create-certificate")
async def create_certificate_on_blockchain(
    user_id: str = Body(...),
    certificate_id: str = Body(...),
    certificate_data: Dict = Body(...)
):
    """Crea certificado en blockchain"""
    try:
        certificate = blockchain.create_certificate_on_blockchain(
            user_id, certificate_id, certificate_data
        )
        return JSONResponse(content=certificate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando certificado: {str(e)}")


# Endpoints de Eventos en Tiempo Real
@router.post("/events/log")
async def log_event(
    user_id: str = Body(...),
    event_type: str = Body(...),
    event_data: Dict = Body(...)
):
    """Registra un evento"""
    try:
        event = realtime_events.log_event(user_id, event_type, event_data)
        return JSONResponse(content=event)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando evento: {str(e)}")


@router.get("/events/recent/{user_id}")
async def get_recent_events(
    user_id: str,
    event_types: Optional[str] = Query(None),
    limit: int = Query(50)
):
    """Obtiene eventos recientes"""
    try:
        event_types_list = event_types.split(",") if event_types else None
        events = realtime_events.get_recent_events(user_id, event_types_list, limit)
        return JSONResponse(content={
            "user_id": user_id,
            "events": events,
            "total": len(events),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo eventos: {str(e)}")


# Endpoints de Análisis de Redes Sociales
@router.post("/social-media/analyze")
async def analyze_social_activity(
    user_id: str = Body(...),
    social_posts: List[Dict] = Body(...),
    platform: str = Body("general")
):
    """Analiza actividad en redes sociales"""
    try:
        analysis = social_media_analysis.analyze_social_activity(
            user_id, social_posts, platform
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando actividad: {str(e)}")


@router.post("/social-media/detect-triggers")
async def detect_social_triggers(
    user_id: str = Body(...),
    social_content: List[Dict] = Body(...)
):
    """Detecta triggers en contenido social"""
    try:
        triggers = social_media_analysis.detect_social_triggers(user_id, social_content)
        return JSONResponse(content=triggers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detectando triggers: {str(e)}")


# Endpoints de Recomendaciones ML
@router.post("/ml-recommendations/get")
async def get_ml_recommendations(
    user_id: str = Body(...),
    user_profile: Dict = Body(...),
    context: Dict = Body(...)
):
    """Obtiene recomendaciones basadas en ML"""
    try:
        recommendations = ml_recommendations.get_ml_recommendations(
            user_id, user_profile, context
        )
        return JSONResponse(content=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recomendaciones: {str(e)}")


@router.post("/ml-recommendations/collaborative")
async def get_collaborative_recommendations(
    user_id: str = Body(...),
    similar_users: List[str] = Body(...)
):
    """Obtiene recomendaciones colaborativas"""
    try:
        recommendations = ml_recommendations.get_collaborative_recommendations(
            user_id, similar_users
        )
        return JSONResponse(content={
            "user_id": user_id,
            "recommendations": recommendations,
            "total": len(recommendations),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recomendaciones: {str(e)}")


# Endpoints de Terapia VR/AR
@router.post("/vr-ar/create-session")
async def create_vr_therapy_session(
    user_id: str = Body(...),
    therapy_type: str = Body(...),
    duration_minutes: int = Body(30)
):
    """Crea sesión de terapia VR/AR"""
    try:
        session = vr_ar_therapy.create_therapy_session(
            user_id, therapy_type, None, duration_minutes
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando sesión: {str(e)}")


@router.get("/vr-ar/scenarios")
async def get_available_scenarios(
    therapy_type: Optional[str] = Query(None)
):
    """Obtiene escenarios VR/AR disponibles"""
    try:
        scenarios = vr_ar_therapy.get_available_scenarios(therapy_type)
        return JSONResponse(content={
            "scenarios": scenarios,
            "total": len(scenarios),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo escenarios: {str(e)}")


# Endpoints de Biometría Avanzada
@router.post("/biometrics/record")
async def record_biometric_data(
    user_id: str = Body(...),
    biometric_type: str = Body(...),
    measurements: Dict = Body(...)
):
    """Registra datos biométricos"""
    try:
        record = advanced_biometrics.record_biometric_data(
            user_id, biometric_type, measurements
        )
        return JSONResponse(content=record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando datos: {str(e)}")


@router.post("/biometrics/analyze-trends")
async def analyze_biometric_trends(
    user_id: str = Body(...),
    biometric_data: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza tendencias biométricas"""
    try:
        analysis = advanced_biometrics.analyze_biometric_trends(
            user_id, biometric_data, days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando tendencias: {str(e)}")


# Endpoints de Asistentes de Voz
@router.post("/voice-assistant/register")
async def register_voice_assistant(
    user_id: str = Body(...),
    assistant_type: str = Body(...),
    device_id: str = Body(...),
    connection_info: Dict = Body(...)
):
    """Registra asistente de voz"""
    try:
        assistant = voice_assistant.register_voice_assistant(
            user_id, assistant_type, device_id, connection_info
        )
        return JSONResponse(content=assistant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando asistente: {str(e)}")


@router.post("/voice-assistant/process-command")
async def process_voice_command(
    user_id: str = Body(...),
    assistant_type: str = Body(...),
    command: str = Body(...)
):
    """Procesa comando de voz"""
    try:
        response = voice_assistant.process_voice_command(
            user_id, assistant_type, command
        )
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando comando: {str(e)}")


# Endpoints de Análisis de Compras
@router.post("/purchases/record")
async def record_purchase(
    user_id: str = Body(...),
    purchase_data: Dict = Body(...)
):
    """Registra una compra"""
    try:
        purchase = purchase_analysis.record_purchase(user_id, purchase_data)
        return JSONResponse(content=purchase)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando compra: {str(e)}")


@router.post("/purchases/analyze-patterns")
async def analyze_purchase_patterns(
    user_id: str = Body(...),
    purchases: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones de compra"""
    try:
        analysis = purchase_analysis.analyze_purchase_patterns(
            user_id, purchases, days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


# Endpoints de Relaciones Interpersonales
@router.post("/relationships/add")
async def add_relationship(
    user_id: str = Body(...),
    relationship_type: str = Body(...),
    person_name: str = Body(...),
    quality: str = Body("good")
):
    """Agrega una relación"""
    try:
        relationship = relationships.add_relationship(
            user_id, relationship_type, person_name, quality
        )
        return JSONResponse(content=relationship)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error agregando relación: {str(e)}")


@router.post("/relationships/analyze-network")
async def analyze_relationship_network(
    user_id: str = Body(...),
    relationships: List[Dict] = Body(...)
):
    """Analiza red de relaciones"""
    try:
        analysis = relationships.analyze_relationship_network(user_id, relationships)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando red: {str(e)}")


# Endpoints de Productividad y Trabajo
@router.post("/productivity/record-work")
async def record_work_session(
    user_id: str = Body(...),
    work_data: Dict = Body(...)
):
    """Registra sesión de trabajo"""
    try:
        session = productivity_work.record_work_session(user_id, work_data)
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando sesión: {str(e)}")


@router.post("/productivity/analyze-work-patterns")
async def analyze_work_patterns(
    user_id: str = Body(...),
    work_sessions: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones de trabajo"""
    try:
        analysis = productivity_work.analyze_work_patterns(
            user_id, work_sessions, days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


# Endpoints de Nutrición y Dieta
@router.post("/nutrition/record-meal")
async def record_meal(
    user_id: str = Body(...),
    meal_data: Dict = Body(...)
):
    """Registra una comida"""
    try:
        meal = nutrition_diet.record_meal(user_id, meal_data)
        return JSONResponse(content=meal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando comida: {str(e)}")


@router.post("/nutrition/analyze-patterns")
async def analyze_nutrition_patterns(
    user_id: str = Body(...),
    meals: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones nutricionales"""
    try:
        analysis = nutrition_diet.analyze_nutrition_patterns(user_id, meals, days)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


# Endpoints de Ejercicio Avanzado
@router.post("/exercise/record-session")
async def record_exercise_session(
    user_id: str = Body(...),
    exercise_data: Dict = Body(...)
):
    """Registra sesión de ejercicio"""
    try:
        session = advanced_exercise.record_exercise_session(user_id, exercise_data)
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando sesión: {str(e)}")


@router.post("/exercise/analyze-patterns")
async def analyze_exercise_patterns(
    user_id: str = Body(...),
    exercise_sessions: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones de ejercicio"""
    try:
        analysis = advanced_exercise.analyze_exercise_patterns(
            user_id, exercise_sessions, days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


# Endpoints de Apps de Meditación
@router.post("/meditation/connect-app")
async def connect_meditation_app(
    user_id: str = Body(...),
    app_type: str = Body(...),
    connection_info: Dict = Body(...)
):
    """Conecta app de meditación"""
    try:
        connection = meditation_apps.connect_meditation_app(
            user_id, app_type, connection_info
        )
        return JSONResponse(content=connection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando app: {str(e)}")


@router.post("/meditation/analyze-impact")
async def analyze_meditation_impact(
    user_id: str = Body(...),
    meditation_data: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Analiza impacto de meditación en recuperación"""
    try:
        analysis = meditation_apps.analyze_meditation_impact(
            user_id, meditation_data, recovery_data
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando impacto: {str(e)}")


# Endpoints de Análisis de Entorno
@router.post("/environment/record-context")
async def record_environment_context(
    user_id: str = Body(...),
    context_data: Dict = Body(...)
):
    """Registra contexto de entorno"""
    try:
        context = environment_context.record_environment_context(user_id, context_data)
        return JSONResponse(content=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando contexto: {str(e)}")


@router.post("/environment/predict-risk")
async def predict_environment_risk(
    user_id: str = Body(...),
    current_context: Dict = Body(...)
):
    """Predice riesgo de entorno"""
    try:
        prediction = environment_context.predict_environment_risk(user_id, current_context)
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo riesgo: {str(e)}")


# Endpoints de Hábitos Avanzados
@router.post("/habits/create")
async def create_habit(
    user_id: str = Body(...),
    habit_name: str = Body(...),
    habit_type: str = Body(...),
    target_frequency: str = Body("daily")
):
    """Crea un hábito"""
    try:
        habit = advanced_habits.create_habit(
            user_id, habit_name, habit_type, target_frequency
        )
        return JSONResponse(content=habit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando hábito: {str(e)}")


@router.post("/habits/analyze-performance")
async def analyze_habit_performance(
    user_id: str = Body(...),
    habit_id: str = Body(...),
    completions: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza rendimiento de hábito"""
    try:
        analysis = advanced_habits.analyze_habit_performance(
            user_id, habit_id, completions, days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando rendimiento: {str(e)}")


# Endpoints de Análisis Temporal Avanzado
@router.post("/temporal/analyze-daily-patterns")
async def analyze_daily_patterns(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    metric: str = Body("mood")
):
    """Analiza patrones diarios"""
    try:
        analysis = temporal_patterns.analyze_daily_patterns(user_id, data, metric)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/temporal/analyze-weekly-patterns")
async def analyze_weekly_patterns(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    metric: str = Body("mood")
):
    """Analiza patrones semanales"""
    try:
        analysis = temporal_patterns.analyze_weekly_patterns(user_id, data, metric)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


# Endpoints de Análisis Genético
@router.post("/genetic/analyze")
async def analyze_genetic_data(
    user_id: str = Body(...),
    genetic_data: Dict = Body(...)
):
    """Analiza datos genéticos"""
    try:
        analysis = genetic_analysis.analyze_genetic_data(user_id, genetic_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando datos genéticos: {str(e)}")


@router.post("/genetic/predict-risk")
async def predict_genetic_risk(
    user_id: str = Body(...),
    genetic_profile: Dict = Body(...),
    addiction_type: str = Body(...)
):
    """Predice riesgo genético"""
    try:
        prediction = genetic_analysis.predict_genetic_risk(
            user_id, genetic_profile, addiction_type
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo riesgo: {str(e)}")


# Endpoints de Dispositivos Médicos
@router.post("/medical-devices/register")
async def register_medical_device(
    user_id: str = Body(...),
    device_type: str = Body(...),
    device_id: str = Body(...),
    device_info: Dict = Body(...)
):
    """Registra dispositivo médico"""
    try:
        device = medical_devices.register_medical_device(
            user_id, device_type, device_id, device_info
        )
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")


@router.post("/medical-devices/analyze-data")
async def analyze_medical_device_data(
    user_id: str = Body(...),
    device_type: str = Body(...),
    measurements: List[Dict] = Body(...)
):
    """Analiza datos de dispositivo médico"""
    try:
        analysis = medical_devices.analyze_medical_device_data(
            user_id, device_type, measurements
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando datos: {str(e)}")


# Endpoints de Progreso Visual Avanzado
@router.post("/visual-progress/generate")
async def generate_progress_visualization(
    user_id: str = Body(...),
    progress_data: List[Dict] = Body(...),
    visualization_type: str = Body("comprehensive")
):
    """Genera visualización de progreso"""
    try:
        visualization = visual_progress.generate_progress_visualization(
            user_id, progress_data, visualization_type
        )
        return JSONResponse(content=visualization)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando visualización: {str(e)}")


@router.post("/visual-progress/create-timeline")
async def create_progress_timeline(
    user_id: str = Body(...),
    timeline_data: List[Dict] = Body(...)
):
    """Crea línea de tiempo de progreso"""
    try:
        timeline = visual_progress.create_progress_timeline(user_id, timeline_data)
        return JSONResponse(content=timeline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando línea de tiempo: {str(e)}")


# Endpoints de Integración EHR
@router.post("/ehr/connect")
async def connect_ehr_system(
    user_id: str = Body(...),
    ehr_system: str = Body(...),
    connection_credentials: Dict = Body(...)
):
    """Conecta sistema EHR"""
    try:
        connection = ehr_integration.connect_ehr_system(
            user_id, ehr_system, connection_credentials
        )
        return JSONResponse(content=connection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando EHR: {str(e)}")


@router.get("/ehr/medical-history/{user_id}")
async def get_medical_history(
    user_id: str,
    ehr_system: str = Query(...)
):
    """Obtiene historial médico"""
    try:
        history = ehr_integration.get_medical_history(user_id, ehr_system)
        return JSONResponse(content=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo historial: {str(e)}")


# Endpoints de Análisis de Correlaciones
@router.post("/correlations/analyze-multivariate")
async def analyze_multivariate_correlations(
    user_id: str = Body(...),
    variables: Dict[str, List[float]] = Body(...)
):
    """Analiza correlaciones multivariadas"""
    try:
        analysis = correlation_analysis.analyze_multivariate_correlations(
            user_id, variables
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando correlaciones: {str(e)}")


# Endpoints de Predicción de Éxito a Largo Plazo
@router.post("/prediction/long-term-success")
async def predict_long_term_success(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    historical_data: List[Dict] = Body(...),
    prediction_horizon_years: int = Body(5)
):
    """Predice éxito a largo plazo"""
    try:
        prediction = long_term_prediction.predict_long_term_success(
            user_id, current_state, historical_data, prediction_horizon_years
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo éxito: {str(e)}")


@router.post("/prediction/milestone-achievement")
async def predict_milestone_achievement(
    user_id: str = Body(...),
    milestone: str = Body(...),
    current_progress: Dict = Body(...)
):
    """Predice logro de hito"""
    try:
        prediction = long_term_prediction.predict_milestone_achievement(
            user_id, milestone, current_progress
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo hito: {str(e)}")


# Endpoints de Análisis de Comportamiento Avanzado
@router.post("/behavioral/analyze-patterns")
async def analyze_behavioral_patterns(
    user_id: str = Body(...),
    behavioral_data: List[Dict] = Body(...),
    analysis_type: str = Body("comprehensive")
):
    """Analiza patrones de comportamiento"""
    try:
        analysis = behavioral_analysis.analyze_behavioral_patterns(
            user_id, behavioral_data, analysis_type
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/behavioral/detect-anomalies")
async def detect_behavioral_anomalies(
    user_id: str = Body(...),
    current_behavior: Dict = Body(...),
    historical_patterns: List[Dict] = Body(...)
):
    """Detecta anomalías de comportamiento"""
    try:
        anomalies = behavioral_analysis.detect_behavioral_anomalies(
            user_id, current_behavior, historical_patterns
        )
        return JSONResponse(content=anomalies)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detectando anomalías: {str(e)}")


# Endpoints de Telemedicina
@router.post("/telemedicine/schedule-session")
async def schedule_telemedicine_session(
    user_id: str = Body(...),
    provider: str = Body(...),
    session_type: str = Body(...),
    scheduled_time: str = Body(...)
):
    """Programa sesión de telemedicina"""
    try:
        session = telemedicine.schedule_telemedicine_session(
            user_id, provider, session_type, scheduled_time
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error programando sesión: {str(e)}")


@router.get("/telemedicine/available-providers/{user_id}")
async def get_available_providers(
    user_id: str,
    specialty: Optional[str] = Query(None)
):
    """Obtiene proveedores disponibles"""
    try:
        providers = telemedicine.get_available_providers(user_id, specialty)
        return JSONResponse(content={
            "user_id": user_id,
            "providers": providers,
            "total": len(providers),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo proveedores: {str(e)}")


# Endpoints de Alertas Inteligentes
@router.post("/alerts/create")
async def create_intelligent_alert(
    user_id: str = Body(...),
    alert_type: str = Body(...),
    context: Dict = Body(...)
):
    """Crea alerta inteligente"""
    try:
        alert = intelligent_alerts.create_intelligent_alert(
            user_id, alert_type, context
        )
        return JSONResponse(content=alert)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando alerta: {str(e)}")


@router.post("/alerts/evaluate-conditions")
async def evaluate_alert_conditions(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Evalúa condiciones de alerta"""
    try:
        alerts = intelligent_alerts.evaluate_alert_conditions(
            user_id, current_state, historical_data
        )
        return JSONResponse(content={
            "user_id": user_id,
            "alerts": alerts,
            "total": len(alerts),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando condiciones: {str(e)}")


# Endpoints de Análisis de Adherencia
@router.post("/adherence/calculate-rate")
async def calculate_adherence_rate(
    user_id: str = Body(...),
    expected_actions: List[Dict] = Body(...),
    completed_actions: List[Dict] = Body(...),
    period_days: int = Body(30)
):
    """Calcula tasa de adherencia"""
    try:
        analysis = adherence_analysis.calculate_adherence_rate(
            user_id, expected_actions, completed_actions, period_days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando adherencia: {str(e)}")


@router.post("/adherence/predict-risk")
async def predict_adherence_risk(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    historical_adherence: List[Dict] = Body(...)
):
    """Predice riesgo de no adherencia"""
    try:
        prediction = adherence_analysis.predict_adherence_risk(
            user_id, current_state, historical_adherence
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo riesgo: {str(e)}")


# Endpoints de Seguimiento de Síntomas
@router.post("/symptoms/record")
async def record_symptom(
    user_id: str = Body(...),
    symptom_data: Dict = Body(...)
):
    """Registra un síntoma"""
    try:
        symptom = symptom_tracking.record_symptom(user_id, symptom_data)
        return JSONResponse(content=symptom)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando síntoma: {str(e)}")


@router.post("/symptoms/analyze-patterns")
async def analyze_symptom_patterns(
    user_id: str = Body(...),
    symptoms: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones de síntomas"""
    try:
        analysis = symptom_tracking.analyze_symptom_patterns(user_id, symptoms, days)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


# Endpoints de Calidad de Vida
@router.post("/quality-of-life/assess")
async def assess_quality_of_life(
    user_id: str = Body(...),
    qol_data: Dict = Body(...)
):
    """Evalúa calidad de vida"""
    try:
        assessment = quality_of_life.assess_quality_of_life(user_id, qol_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando calidad de vida: {str(e)}")


@router.post("/quality-of-life/analyze-trends")
async def analyze_qol_trends(
    user_id: str = Body(...),
    qol_assessments: List[Dict] = Body(...)
):
    """Analiza tendencias de calidad de vida"""
    try:
        analysis = quality_of_life.analyze_qol_trends(user_id, qol_assessments)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando tendencias: {str(e)}")


# Endpoints de Redes Neuronales
@router.post("/neural-network/train")
async def train_neural_network(
    user_id: str = Body(...),
    training_data: List[Dict] = Body(...),
    network_architecture: str = Body("deep_lstm")
):
    """Entrena red neuronal"""
    try:
        network = neural_networks.train_neural_network(
            user_id, training_data, network_architecture
        )
        return JSONResponse(content=network)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando red: {str(e)}")


@router.post("/neural-network/predict")
async def predict_with_neural_network(
    network_id: str = Body(...),
    user_id: str = Body(...),
    input_features: Dict = Body(...)
):
    """Predice usando red neuronal"""
    try:
        prediction = neural_networks.predict_with_neural_network(
            network_id, user_id, input_features
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo: {str(e)}")


# Endpoints de Monitoreo Continuo
@router.post("/monitoring/start-continuous")
async def start_continuous_monitoring(
    user_id: str = Body(...),
    monitoring_config: Dict = Body(...)
):
    """Inicia monitoreo continuo"""
    try:
        monitoring = continuous_monitoring.start_continuous_monitoring(
            user_id, monitoring_config
        )
        return JSONResponse(content=monitoring)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando monitoreo: {str(e)}")


@router.get("/monitoring/realtime-metrics/{user_id}")
async def get_realtime_metrics(
    user_id: str,
    monitoring_id: str = Query(...)
):
    """Obtiene métricas en tiempo real"""
    try:
        metrics = continuous_monitoring.get_realtime_metrics(user_id, monitoring_id)
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo métricas: {str(e)}")


# Endpoints de Análisis de Sueño con IA
@router.post("/ai-sleep/analyze")
async def analyze_sleep_with_ai(
    user_id: str = Body(...),
    sleep_data: Dict = Body(...),
    ai_model: str = Body("advanced")
):
    """Analiza sueño con IA"""
    try:
        analysis = ai_sleep.analyze_sleep_with_ai(user_id, sleep_data, ai_model)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando sueño: {str(e)}")


@router.post("/ai-sleep/predict-quality")
async def predict_sleep_quality_with_ai(
    user_id: str = Body(...),
    current_factors: Dict = Body(...),
    sleep_history: List[Dict] = Body(...)
):
    """Predice calidad de sueño con IA"""
    try:
        prediction = ai_sleep.predict_sleep_quality_with_ai(
            user_id, current_factors, sleep_history
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo calidad: {str(e)}")


# Endpoints de Seguimiento de Emociones Avanzado
@router.post("/emotions/record")
async def record_emotion(
    user_id: str = Body(...),
    emotion_data: Dict = Body(...)
):
    """Registra una emoción"""
    try:
        emotion = emotion_tracking.record_emotion(user_id, emotion_data)
        return JSONResponse(content=emotion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando emoción: {str(e)}")


@router.post("/emotions/analyze-patterns")
async def analyze_emotion_patterns(
    user_id: str = Body(...),
    emotions: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones emocionales"""
    try:
        analysis = emotion_tracking.analyze_emotion_patterns(user_id, emotions, days)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


# Endpoints de Reconocimiento Emocional en Voz
@router.post("/voice-emotion/analyze")
async def analyze_voice_emotions(
    user_id: str = Body(...),
    audio_data: str = Body(...),  # Base64 encoded
    metadata: Optional[Dict] = Body(None)
):
    """Analiza emociones en la voz"""
    try:
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        analysis = voice_emotion.analyze_voice_emotions(user_id, audio_bytes, metadata)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando emociones: {str(e)}")


@router.post("/voice-emotion/detect-risk")
async def detect_emotional_risk_from_voice(
    user_id: str = Body(...),
    audio_data: str = Body(...)
):
    """Detecta riesgo emocional desde voz"""
    try:
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        risk_analysis = voice_emotion.detect_emotional_risk_from_voice(user_id, audio_bytes)
        return JSONResponse(content=risk_analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detectando riesgo: {str(e)}")


# Endpoints de Apps de Bienestar
@router.post("/wellness/connect-app")
async def connect_wellness_app(
    user_id: str = Body(...),
    app_type: str = Body(...),
    connection_info: Dict = Body(...)
):
    """Conecta app de bienestar"""
    try:
        connection = wellness_apps.connect_wellness_app(
            user_id, app_type, connection_info
        )
        return JSONResponse(content=connection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando app: {str(e)}")


@router.post("/wellness/analyze-impact")
async def analyze_wellness_impact(
    user_id: str = Body(...),
    wellness_data: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Analiza impacto de bienestar en recuperación"""
    try:
        analysis = wellness_apps.analyze_wellness_impact(
            user_id, wellness_data, recovery_data
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando impacto: {str(e)}")


# Endpoints de Análisis de Patrones de Actividad
@router.post("/activity-patterns/analyze")
async def analyze_activity_patterns(
    user_id: str = Body(...),
    activity_data: List[Dict] = Body(...),
    analysis_type: str = Body("comprehensive")
):
    """Analiza patrones de actividad"""
    try:
        analysis = activity_patterns.analyze_activity_patterns(
            user_id, activity_data, analysis_type
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/activity-patterns/predict-outcome")
async def predict_activity_outcome(
    user_id: str = Body(...),
    current_activities: Dict = Body(...),
    activity_history: List[Dict] = Body(...)
):
    """Predice resultado de actividad"""
    try:
        prediction = activity_patterns.predict_activity_outcome(
            user_id, current_activities, activity_history
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo resultado: {str(e)}")


# Endpoints de Dispositivos de Monitoreo de Salud
@router.post("/health-devices/register")
async def register_health_device(
    user_id: str = Body(...),
    device_type: str = Body(...),
    device_id: str = Body(...),
    device_info: Dict = Body(...)
):
    """Registra dispositivo de salud"""
    try:
        device = health_devices.register_health_device(
            user_id, device_type, device_id, device_info
        )
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")


@router.post("/health-devices/record-reading")
async def record_health_reading(
    user_id: str = Body(...),
    device_id: str = Body(...),
    reading_data: Dict = Body(...)
):
    """Registra lectura de salud"""
    try:
        reading = health_devices.record_health_reading(
            user_id, device_id, reading_data
        )
        return JSONResponse(content=reading)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando lectura: {str(e)}")


# Endpoints de Coaching Personalizado
@router.post("/coaching/create-plan")
async def create_coaching_plan(
    user_id: str = Body(...),
    user_profile: Dict = Body(...),
    goals: List[str] = Body(...)
):
    """Crea plan de coaching personalizado"""
    try:
        plan = personalized_coaching.create_coaching_plan(
            user_id, user_profile, goals
        )
        return JSONResponse(content=plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando plan: {str(e)}")


@router.post("/coaching/provide-session")
async def provide_coaching_session(
    user_id: str = Body(...),
    session_context: Dict = Body(...)
):
    """Proporciona sesión de coaching"""
    try:
        session = personalized_coaching.provide_coaching_session(
            user_id, session_context
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error proporcionando sesión: {str(e)}")


# Endpoints de Análisis de Redes Sociales
@router.post("/social-network/analyze")
async def analyze_social_network(
    user_id: str = Body(...),
    network_data: Dict = Body(...)
):
    """Analiza red social"""
    try:
        analysis = social_network_analysis.analyze_social_network(
            user_id, network_data
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando red: {str(e)}")


@router.post("/social-network/predict-influence")
async def predict_social_influence(
    user_id: str = Body(...),
    network_data: Dict = Body(...)
):
    """Predice influencia social"""
    try:
        prediction = social_network_analysis.predict_social_influence(
            user_id, network_data
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo influencia: {str(e)}")


# Endpoints de Seguimiento de Objetivos Avanzado
@router.post("/goals/create-advanced")
async def create_advanced_goal(
    user_id: str = Body(...),
    goal_data: Dict = Body(...)
):
    """Crea objetivo avanzado"""
    try:
        goal = goal_tracking.create_advanced_goal(user_id, goal_data)
        return JSONResponse(content=goal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando objetivo: {str(e)}")


@router.post("/goals/analyze-performance")
async def analyze_goal_performance(
    user_id: str = Body(...),
    goals: List[Dict] = Body(...)
):
    """Analiza rendimiento de objetivos"""
    try:
        analysis = goal_tracking.analyze_goal_performance(user_id, goals)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando rendimiento: {str(e)}")


# Endpoints de Análisis Comparativo
@router.post("/comparative/compare-periods")
async def compare_periods(
    user_id: str = Body(...),
    period1_data: List[Dict] = Body(...),
    period2_data: List[Dict] = Body(...),
    metrics: List[str] = Body(...)
):
    """Compara dos períodos"""
    try:
        comparison = comparative_analysis.compare_periods(
            user_id, period1_data, period2_data, metrics
        )
        return JSONResponse(content=comparison)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparando períodos: {str(e)}")


@router.post("/comparative/compare-with-baseline")
async def compare_with_baseline(
    user_id: str = Body(...),
    current_data: Dict = Body(...),
    baseline_data: Dict = Body(...),
    metrics: List[str] = Body(...)
):
    """Compara con línea base"""
    try:
        comparison = comparative_analysis.compare_with_baseline(
            user_id, current_data, baseline_data, metrics
        )
        return JSONResponse(content=comparison)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparando con línea base: {str(e)}")


# Endpoints de Análisis de Resiliencia
@router.post("/resilience/assess")
async def assess_resilience(
    user_id: str = Body(...),
    resilience_data: Dict = Body(...)
):
    """Evalúa resiliencia"""
    try:
        assessment = resilience_analysis.assess_resilience(user_id, resilience_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando resiliencia: {str(e)}")


@router.post("/resilience/predict-outcome")
async def predict_resilience_outcome(
    user_id: str = Body(...),
    current_resilience: Dict = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Predice resultado de resiliencia"""
    try:
        prediction = resilience_analysis.predict_resilience_outcome(
            user_id, current_resilience, historical_data
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo resultado: {str(e)}")


# Endpoints de Recompensas Avanzado
@router.post("/rewards/award")
async def award_reward(
    user_id: str = Body(...),
    reward_id: str = Body(...),
    achievement_data: Dict = Body(...)
):
    """Otorga recompensa"""
    try:
        reward = rewards_service.award_reward(user_id, reward_id, achievement_data)
        return JSONResponse(content=reward)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error otorgando recompensa: {str(e)}")


@router.post("/rewards/analyze-impact")
async def analyze_reward_impact(
    user_id: str = Body(...),
    rewards: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Analiza impacto de recompensas"""
    try:
        analysis = rewards_service.analyze_reward_impact(
            user_id, rewards, recovery_data
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando impacto: {str(e)}")


# Endpoints de Terapias Alternativas
@router.post("/alternative-therapy/recommend")
async def recommend_therapy(
    user_id: str = Body(...),
    user_profile: Dict = Body(...),
    current_state: Dict = Body(...)
):
    """Recomienda terapia alternativa"""
    try:
        recommendation = alternative_therapy.recommend_therapy(
            user_id, user_profile, current_state
        )
        return JSONResponse(content=recommendation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recomendando terapia: {str(e)}")


@router.post("/alternative-therapy/track-session")
async def track_therapy_session(
    user_id: str = Body(...),
    therapy_type: str = Body(...),
    session_data: Dict = Body(...)
):
    """Rastrea sesión de terapia"""
    try:
        session = alternative_therapy.track_therapy_session(
            user_id, therapy_type, session_data
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando sesión: {str(e)}")


# Endpoints de Análisis de Motivación
@router.post("/motivation/assess")
async def assess_motivation(
    user_id: str = Body(...),
    motivation_data: Dict = Body(...)
):
    """Evalúa motivación"""
    try:
        assessment = motivation_analysis.assess_motivation(user_id, motivation_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando motivación: {str(e)}")


@router.post("/motivation/predict-drop")
async def predict_motivation_drop(
    user_id: str = Body(...),
    current_motivation: Dict = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Predice caída de motivación"""
    try:
        prediction = motivation_analysis.predict_motivation_drop(
            user_id, current_motivation, historical_data
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo caída: {str(e)}")


# Endpoints de Seguimiento de Recaídas
@router.post("/relapse/record")
async def record_relapse(
    user_id: str = Body(...),
    relapse_data: Dict = Body(...)
):
    """Registra recaída"""
    try:
        relapse = relapse_tracking.record_relapse(user_id, relapse_data)
        return JSONResponse(content=relapse)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando recaída: {str(e)}")


@router.post("/relapse/predict-risk")
async def predict_relapse_risk(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    relapse_history: List[Dict] = Body(...)
):
    """Predice riesgo de recaída"""
    try:
        prediction = relapse_tracking.predict_relapse_risk(
            user_id, current_state, relapse_history
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo riesgo: {str(e)}")


# Endpoints de Análisis de Barreras
@router.post("/barriers/identify")
async def identify_barriers(
    user_id: str = Body(...),
    user_data: Dict = Body(...)
):
    """Identifica barreras de recuperación"""
    try:
        barriers = barriers_analysis.identify_barriers(user_id, user_data)
        return JSONResponse(content=barriers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error identificando barreras: {str(e)}")


@router.post("/barriers/suggest-solutions")
async def suggest_barrier_solutions(
    user_id: str = Body(...),
    barrier: Dict = Body(...)
):
    """Sugiere soluciones para barrera"""
    try:
        solutions = barriers_analysis.suggest_barrier_solutions(user_id, barrier)
        return JSONResponse(content=solutions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sugiriendo soluciones: {str(e)}")


# Endpoints de Análisis de Estrés
@router.post("/stress/assess")
async def assess_stress(
    user_id: str = Body(...),
    stress_data: Dict = Body(...)
):
    """Evalúa estrés"""
    try:
        assessment = stress_analysis.assess_stress(user_id, stress_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando estrés: {str(e)}")


@router.post("/stress/predict-episode")
async def predict_stress_episode(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    stress_history: List[Dict] = Body(...)
):
    """Predice episodio de estrés"""
    try:
        prediction = stress_analysis.predict_stress_episode(
            user_id, current_state, stress_history
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo episodio: {str(e)}")


# Endpoints de Apoyo Social
@router.post("/social-support/assess")
async def assess_social_support(
    user_id: str = Body(...),
    support_data: Dict = Body(...)
):
    """Evalúa apoyo social"""
    try:
        assessment = social_support.assess_social_support(user_id, support_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando apoyo: {str(e)}")


@router.post("/social-support/recommend-resources")
async def recommend_support_resources(
    user_id: str = Body(...),
    user_profile: Dict = Body(...)
):
    """Recomienda recursos de apoyo"""
    try:
        resources = social_support.recommend_support_resources(user_id, user_profile)
        return JSONResponse(content=resources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recomendando recursos: {str(e)}")


# Endpoints de Servicios de Emergencia
@router.post("/emergency/trigger")
async def trigger_emergency(
    user_id: str = Body(...),
    emergency_type: str = Body(...),
    emergency_data: Dict = Body(...)
):
    """Activa emergencia"""
    try:
        emergency = emergency_services.trigger_emergency(
            user_id, emergency_type, emergency_data
        )
        return JSONResponse(content=emergency)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activando emergencia: {str(e)}")


@router.post("/emergency/resources")
async def get_emergency_resources(
    user_id: str = Body(...),
    location: Optional[Dict] = Body(None)
):
    """Obtiene recursos de emergencia"""
    try:
        resources = emergency_services.get_emergency_resources(user_id, location)
        return JSONResponse(content=resources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recursos: {str(e)}")


# Endpoints de Progreso Visual
@router.post("/visual-progress/generate-timeline")
async def generate_progress_timeline(
    user_id: str = Body(...),
    progress_data: List[Dict] = Body(...)
):
    """Genera línea de tiempo de progreso"""
    try:
        timeline = visual_progress.generate_progress_timeline(user_id, progress_data)
        return JSONResponse(content=timeline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando línea de tiempo: {str(e)}")


@router.post("/visual-progress/create-chart")
async def create_progress_chart(
    user_id: str = Body(...),
    metrics: List[str] = Body(...),
    time_period: str = Body("30_days")
):
    """Crea gráfico de progreso"""
    try:
        chart = visual_progress.create_progress_chart(user_id, metrics, time_period)
        return JSONResponse(content=chart)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando gráfico: {str(e)}")


# Endpoints de Seguimiento de Medicamentos
@router.post("/medications/register")
async def register_medication(
    user_id: str = Body(...),
    medication_data: Dict = Body(...)
):
    """Registra medicamento"""
    try:
        medication = medication_tracking.register_medication(user_id, medication_data)
        return JSONResponse(content=medication)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando medicamento: {str(e)}")


@router.post("/medications/analyze-adherence")
async def analyze_medication_adherence(
    user_id: str = Body(...),
    medication_id: str = Body(...),
    doses: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza adherencia a medicamentos"""
    try:
        analysis = medication_tracking.analyze_medication_adherence(
            user_id, medication_id, doses, days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando adherencia: {str(e)}")


# Endpoints de Análisis de Patrones de Sueño
@router.post("/sleep-patterns/analyze")
async def analyze_sleep_patterns(
    user_id: str = Body(...),
    sleep_data: List[Dict] = Body(...)
):
    """Analiza patrones de sueño"""
    try:
        analysis = sleep_patterns.analyze_sleep_patterns(user_id, sleep_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/sleep-patterns/correlate-with-recovery")
async def correlate_sleep_with_recovery(
    user_id: str = Body(...),
    sleep_data: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Correlaciona sueño con recuperación"""
    try:
        correlation = sleep_patterns.correlate_sleep_with_recovery(
            user_id, sleep_data, recovery_data
        )
        return JSONResponse(content=correlation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error correlacionando: {str(e)}")


# Endpoints de Bienestar Integral
@router.post("/wellness/assess-comprehensive")
async def assess_comprehensive_wellness(
    user_id: str = Body(...),
    wellness_data: Dict = Body(...)
):
    """Evalúa bienestar integral"""
    try:
        assessment = wellness_analysis.assess_comprehensive_wellness(user_id, wellness_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando bienestar: {str(e)}")


# Endpoints de Recordatorios Inteligentes
@router.post("/reminders/create-intelligent")
async def create_intelligent_reminder(
    user_id: str = Body(...),
    reminder_data: Dict = Body(...)
):
    """Crea recordatorio inteligente"""
    try:
        reminder = intelligent_reminders.create_intelligent_reminder(user_id, reminder_data)
        return JSONResponse(content=reminder)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando recordatorio: {str(e)}")


@router.post("/reminders/optimize-timing")
async def optimize_reminder_timing(
    user_id: str = Body(...),
    reminder_type: str = Body(...),
    user_patterns: Dict = Body(...)
):
    """Optimiza horario de recordatorio"""
    try:
        optimization = intelligent_reminders.optimize_reminder_timing(
            user_id, reminder_type, user_patterns
        )
        return JSONResponse(content=optimization)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizando horario: {str(e)}")


# Endpoints de Dispositivos de Salud Avanzado
@router.post("/health-devices-advanced/register")
async def register_health_device_advanced(
    user_id: str = Body(...),
    device_data: Dict = Body(...)
):
    """Registra dispositivo de salud avanzado"""
    try:
        device = health_devices_advanced.register_device(user_id, device_data)
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")


@router.post("/health-devices-advanced/analyze-data")
async def analyze_device_data(
    user_id: str = Body(...),
    device_id: str = Body(...),
    data_points: List[Dict] = Body(...)
):
    """Analiza datos del dispositivo"""
    try:
        analysis = health_devices_advanced.analyze_device_data(
            user_id, device_id, data_points
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando datos: {str(e)}")


# Endpoints de Análisis de Hábitos
@router.post("/habits/analyze")
async def analyze_habits(
    user_id: str = Body(...),
    habits: List[Dict] = Body(...)
):
    """Analiza hábitos"""
    try:
        analysis = habit_analysis.analyze_habits(user_id, habits)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando hábitos: {str(e)}")


# Endpoints de Análisis de Ejercicio
@router.post("/exercise/analyze-patterns")
async def analyze_exercise_patterns(
    user_id: str = Body(...),
    exercise_data: List[Dict] = Body(...)
):
    """Analiza patrones de ejercicio"""
    try:
        analysis = exercise_analysis.analyze_exercise_patterns(user_id, exercise_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


# Endpoints de Análisis de Nutrición
@router.post("/nutrition/analyze-patterns")
async def analyze_nutrition_patterns(
    user_id: str = Body(...),
    nutrition_data: List[Dict] = Body(...)
):
    """Analiza patrones de nutrición"""
    try:
        analysis = nutrition_analysis.analyze_nutrition_patterns(user_id, nutrition_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/nutrition/assess-adequacy")
async def assess_nutritional_adequacy(
    user_id: str = Body(...),
    daily_intake: Dict = Body(...),
    nutritional_requirements: Dict = Body(...)
):
    """Evalúa adecuación nutricional"""
    try:
        assessment = nutrition_analysis.assess_nutritional_adequacy(
            user_id, daily_intake, nutritional_requirements
        )
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando adecuación: {str(e)}")


# Endpoints de Progreso a Largo Plazo
@router.post("/long-term-progress/analyze")
async def analyze_long_term_progress(
    user_id: str = Body(...),
    progress_data: List[Dict] = Body(...),
    time_period_months: int = Body(12)
):
    """Analiza progreso a largo plazo"""
    try:
        analysis = long_term_progress.analyze_long_term_progress(
            user_id, progress_data, time_period_months
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando progreso: {str(e)}")


@router.post("/long-term-progress/predict-outcome")
async def predict_long_term_outcome(
    user_id: str = Body(...),
    current_progress: Dict = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Predice resultado a largo plazo"""
    try:
        prediction = long_term_progress.predict_long_term_outcome(
            user_id, current_progress, historical_data
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo resultado: {str(e)}")


# Endpoints de Logros
@router.post("/achievements/award")
async def award_achievement(
    user_id: str = Body(...),
    achievement_data: Dict = Body(...)
):
    """Otorga logro"""
    try:
        achievement = achievements.award_achievement(user_id, achievement_data)
        return JSONResponse(content=achievement)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error otorgando logro: {str(e)}")


@router.post("/achievements/check-eligibility")
async def check_achievement_eligibility(
    user_id: str = Body(...),
    achievement_criteria: Dict = Body(...),
    user_data: Dict = Body(...)
):
    """Verifica elegibilidad para logro"""
    try:
        eligibility = achievements.check_achievement_eligibility(
            user_id, achievement_criteria, user_data
        )
        return JSONResponse(content=eligibility)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verificando elegibilidad: {str(e)}")


# Endpoints de Terapias Grupales
@router.post("/group-therapy/find-groups")
async def find_suitable_groups(
    user_id: str = Body(...),
    user_profile: Dict = Body(...),
    preferences: Dict = Body(...)
):
    """Encuentra grupos adecuados"""
    try:
        groups = group_therapy.find_suitable_groups(
            user_id, user_profile, preferences
        )
        return JSONResponse(content=groups)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encontrando grupos: {str(e)}")


@router.post("/group-therapy/track-participation")
async def track_group_participation(
    user_id: str = Body(...),
    group_id: str = Body(...),
    sessions: List[Dict] = Body(...)
):
    """Rastrea participación en grupo"""
    try:
        participation = group_therapy.track_group_participation(
            user_id, group_id, sessions
        )
        return JSONResponse(content=participation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rastreando participación: {str(e)}")


# Endpoints de Análisis de Estado de Ánimo
@router.post("/mood/analyze-patterns")
async def analyze_mood_patterns(
    user_id: str = Body(...),
    mood_data: List[Dict] = Body(...)
):
    """Analiza patrones de estado de ánimo"""
    try:
        analysis = mood_analysis.analyze_mood_patterns(user_id, mood_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/mood/predict-episode")
async def predict_mood_episode(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    mood_history: List[Dict] = Body(...)
):
    """Predice episodio de estado de ánimo"""
    try:
        prediction = mood_analysis.predict_mood_episode(
            user_id, current_state, mood_history
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo episodio: {str(e)}")


# Endpoints de Seguimiento de Terapia
@router.post("/therapy/track-session")
async def track_therapy_session(
    user_id: str = Body(...),
    session_data: Dict = Body(...)
):
    """Rastrea sesión de terapia"""
    try:
        session = therapy_tracking.track_therapy_session(user_id, session_data)
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando sesión: {str(e)}")


@router.post("/therapy/recommend-adjustments")
async def recommend_therapy_adjustments(
    user_id: str = Body(...),
    current_therapy: Dict = Body(...),
    progress_data: List[Dict] = Body(...)
):
    """Recomienda ajustes de terapia"""
    try:
        recommendations = therapy_tracking.recommend_therapy_adjustments(
            user_id, current_therapy, progress_data
        )
        return JSONResponse(content=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recomendando ajustes: {str(e)}")


# Endpoints de Análisis de Relaciones
@router.post("/relationships/analyze")
async def analyze_relationships(
    user_id: str = Body(...),
    relationships: List[Dict] = Body(...)
):
    """Analiza relaciones"""
    try:
        analysis = relationship_analysis.analyze_relationships(user_id, relationships)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando relaciones: {str(e)}")


@router.post("/relationships/assess-impact")
async def assess_relationship_impact(
    user_id: str = Body(...),
    relationships: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Evalúa impacto de relaciones en recuperación"""
    try:
        impact = relationship_analysis.assess_relationship_impact(
            user_id, relationships, recovery_data
        )
        return JSONResponse(content=impact)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando impacto: {str(e)}")

