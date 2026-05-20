"""
Wellness domain services
"""

from services.domains import register_service

try:
    from services.wellness_analysis_service import WellnessAnalysisService
    from services.comprehensive_wellness_analysis_service import ComprehensiveWellnessAnalysisService
    from services.sleep_analysis_service import SleepAnalysisService
    from services.ai_sleep_analysis_service import AISleepAnalysisService
    from services.mindfulness_service import MindfulnessService
    from services.meditation_app_integration_service import MeditationAppIntegrationService
    
    def register_services():
        register_service("wellness", "wellness", WellnessAnalysisService)
        register_service("wellness", "comprehensive", ComprehensiveWellnessAnalysisService)
        register_service("wellness", "sleep", SleepAnalysisService)
        register_service("wellness", "ai_sleep", AISleepAnalysisService)
        register_service("wellness", "mindfulness", MindfulnessService)
        register_service("wellness", "meditation", MeditationAppIntegrationService)
except ImportError:
    pass



