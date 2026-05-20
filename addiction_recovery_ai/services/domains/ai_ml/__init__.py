"""
AI/ML domain services
"""

from services.domains import register_service

try:
    from services.predictive_ai_service import PredictiveAIService
    from services.ml_learning_service import MLLearningService
    from services.ml_recommendation_service import MLRecommendationService
    from services.neural_network_analysis_service import NeuralNetworkAnalysisService
    from services.advanced_predictive_ml_service import AdvancedPredictiveMLService
    from services.nlp_analysis_service import NLPAnalysisService
    
    def register_services():
        register_service("ai_ml", "predictive_ai", PredictiveAIService)
        register_service("ai_ml", "ml_learning", MLLearningService)
        register_service("ai_ml", "ml_recommendation", MLRecommendationService)
        register_service("ai_ml", "neural_network", NeuralNetworkAnalysisService)
        register_service("ai_ml", "advanced_predictive_ml", AdvancedPredictiveMLService)
        register_service("ai_ml", "nlp", NLPAnalysisService)
except ImportError:
    pass



