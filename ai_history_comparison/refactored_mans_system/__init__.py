"""
Refactored MANS System - Unified Advanced Technology Architecture

This module provides a completely refactored and unified system that integrates:
- Advanced AI with neural networks and deep learning
- Generative AI and large language models
- Computer vision and image processing
- Natural language processing
- Reinforcement learning and transfer learning
- Federated learning and explainable AI
- AI ethics and safety
- Space technology with satellite communication
- Space-based data processing and orbital mechanics
- Space weather monitoring and debris tracking
- Interplanetary networking and space-based AI

All systems are now unified under a single, cohesive architecture.
"""

from .unified_mans_config import (
    UnifiedMANSConfig,
    AdvancedAIConfig,
    SpaceTechnologyConfig,
    NeuralNetworkConfig,
    GenerativeAIConfig,
    ComputerVisionConfig,
    NLPConfig,
    ReinforcementLearningConfig,
    TransferLearningConfig,
    FederatedLearningConfig,
    ExplainableAIConfig,
    AIEthicsConfig,
    AISafetyConfig,
    SatelliteCommunicationConfig,
    SpaceWeatherConfig,
    SpaceDebrisConfig,
    InterplanetaryNetworkingConfig
)

from .unified_mans_manager import (
    UnifiedMANSManager,
    get_unified_mans_manager,
    initialize_mans_system,
    shutdown_mans_system
)

from .unified_mans_services import (
    UnifiedMANSService,
    AdvancedAIService,
    SpaceTechnologyService,
    NeuralNetworkService,
    GenerativeAIService,
    ComputerVisionService,
    NLPService,
    ReinforcementLearningService,
    TransferLearningService,
    FederatedLearningService,
    ExplainableAIService,
    AIEthicsService,
    AISafetyService,
    SatelliteCommunicationService,
    SpaceWeatherService,
    SpaceDebrisService,
    InterplanetaryNetworkingService
)

from .unified_mans_api import (
    unified_mans_router,
    get_mans_service
)

__all__ = [
    # Configuration
    "UnifiedMANSConfig",
    "AdvancedAIConfig",
    "SpaceTechnologyConfig",
    "NeuralNetworkConfig",
    "GenerativeAIConfig",
    "ComputerVisionConfig",
    "NLPConfig",
    "ReinforcementLearningConfig",
    "TransferLearningConfig",
    "FederatedLearningConfig",
    "ExplainableAIConfig",
    "AIEthicsConfig",
    "AISafetyConfig",
    "SatelliteCommunicationConfig",
    "SpaceWeatherConfig",
    "SpaceDebrisConfig",
    "InterplanetaryNetworkingConfig",
    
    # Manager
    "UnifiedMANSManager",
    "get_unified_mans_manager",
    "initialize_mans_system",
    "shutdown_mans_system",
    
    # Services
    "UnifiedMANSService",
    "AdvancedAIService",
    "SpaceTechnologyService",
    "NeuralNetworkService",
    "GenerativeAIService",
    "ComputerVisionService",
    "NLPService",
    "ReinforcementLearningService",
    "TransferLearningService",
    "FederatedLearningService",
    "ExplainableAIService",
    "AIEthicsService",
    "AISafetyService",
    "SatelliteCommunicationService",
    "SpaceWeatherService",
    "SpaceDebrisService",
    "InterplanetaryNetworkingService",
    
    # API
    "unified_mans_router",
    "get_mans_service"
]





















