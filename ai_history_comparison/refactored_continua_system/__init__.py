"""
Refactored CONTINUA System - Unified Advanced Technology Architecture

This module provides a completely refactored and unified system that integrates:
- 5G Technology with ultra-low latency and massive IoT
- Metaverse with virtual worlds and immersive experiences
- Web3/DeFi with decentralized finance and blockchain
- Neural Interfaces with brain-computer interfaces
- Swarm Intelligence with multi-agent coordination
- Biometric Systems with advanced authentication
- Autonomous Systems with self-managing capabilities
- Space Technology with satellite and orbital systems
- AI Agents with intelligent coordination
- Quantum AI with quantum computing capabilities
- Advanced AI with cutting-edge machine learning

All systems are now unified under a single, cohesive architecture.
"""

from .unified_continua_config import (
    UnifiedContinuaConfig,
    FiveGConfig,
    MetaverseConfig,
    Web3Config,
    NeuralInterfaceConfig,
    SwarmIntelligenceConfig,
    BiometricConfig,
    AutonomousConfig,
    SpaceTechnologyConfig,
    AIAgentsConfig,
    QuantumAIConfig,
    AdvancedAIConfig
)

from .unified_continua_manager import (
    UnifiedContinuaManager,
    get_unified_continua_manager,
    initialize_continua_system,
    shutdown_continua_system
)

from .unified_continua_services import (
    UnifiedContinuaService,
    FiveGService,
    MetaverseService,
    Web3Service,
    NeuralInterfaceService,
    SwarmIntelligenceService,
    BiometricService,
    AutonomousService,
    SpaceTechnologyService,
    AIAgentsService,
    QuantumAIService,
    AdvancedAIService
)

from .unified_continua_api import (
    unified_continua_router,
    get_continua_service
)

__all__ = [
    # Configuration
    "UnifiedContinuaConfig",
    "FiveGConfig",
    "MetaverseConfig", 
    "Web3Config",
    "NeuralInterfaceConfig",
    "SwarmIntelligenceConfig",
    "BiometricConfig",
    "AutonomousConfig",
    "SpaceTechnologyConfig",
    "AIAgentsConfig",
    "QuantumAIConfig",
    "AdvancedAIConfig",
    
    # Manager
    "UnifiedContinuaManager",
    "get_unified_continua_manager",
    "initialize_continua_system",
    "shutdown_continua_system",
    
    # Services
    "UnifiedContinuaService",
    "FiveGService",
    "MetaverseService",
    "Web3Service",
    "NeuralInterfaceService",
    "SwarmIntelligenceService",
    "BiometricService",
    "AutonomousService",
    "SpaceTechnologyService",
    "AIAgentsService",
    "QuantumAIService",
    "AdvancedAIService",
    
    # API
    "unified_continua_router",
    "get_continua_service"
]





















