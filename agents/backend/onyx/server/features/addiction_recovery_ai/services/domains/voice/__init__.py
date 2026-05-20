"""
Voice analysis domain services
"""

from services.domains import register_service

try:
    from services.voice_analysis_service import VoiceAnalysisService
    from services.advanced_voice_analysis_service import AdvancedVoiceAnalysisService
    from services.voice_emotion_recognition_service import VoiceEmotionRecognitionService
    from services.voice_assistant_integration_service import VoiceAssistantIntegrationService
    
    def register_services():
        register_service("voice", "analysis", VoiceAnalysisService)
        register_service("voice", "advanced", AdvancedVoiceAnalysisService)
        register_service("voice", "emotion", VoiceEmotionRecognitionService)
        register_service("voice", "assistant", VoiceAssistantIntegrationService)
except ImportError:
    pass



