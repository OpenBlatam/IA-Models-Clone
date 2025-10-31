"""
Blaze AI Modular System v8.2.0

This module provides a completely modular architecture where each component
is self-contained and can be used independently or as part of the system.
"""

from .base import BaseModule, ModuleConfig, ModuleStatus
from .cache import CacheModule, CacheConfig
from .monitoring import MonitoringModule, MonitoringConfig
from .optimization import OptimizationModule, OptimizationConfig
from .storage import StorageModule, StorageConfig
from .execution import ExecutionModule, ExecutionConfig
from .engines import EnginesModule, EngineModuleConfig
from .ml import MLModule, MLModuleConfig
from .data_analysis import DataAnalysisModule, DataAnalysisModuleConfig
from .ai_intelligence import AIIntelligenceModule, AIIntelligenceConfig
from .api_rest import APIRESTModule, APIRESTConfig
from .security import SecurityModule, SecurityConfig, create_security_module, create_security_module_with_defaults
from .distributed_processing import DistributedProcessingModule, DistributedProcessingConfig, create_distributed_processing_module, create_distributed_processing_module_with_defaults
from .edge_computing import EdgeComputingModule, EdgeComputingConfig, create_edge_computing_module, create_edge_computing_module_with_defaults
from .blockchain import BlockchainModule, BlockchainConfig, create_blockchain_module, create_blockchain_module_with_defaults
from .iot_advanced import IoTAdvancedModule, IoTAdvancedConfig, create_iot_advanced_module, create_iot_advanced_module_with_defaults
from .federated_learning import FederatedLearningModule, FederatedLearningConfig, create_federated_learning_module, create_federated_learning_module_with_defaults
from .cloud_integration import CloudIntegrationModule, CloudIntegrationConfig, create_cloud_integration_module, create_cloud_integration_module_with_defaults
from .zero_knowledge_proofs import ZeroKnowledgeProofsModule, ZKProofConfig, create_zero_knowledge_proofs_module, create_zero_knowledge_proofs_module_with_defaults
from .quantum_computing import QuantumComputingModule, QuantumConfig, create_quantum_computing_module, create_quantum_computing_module_with_defaults
from .advanced_analytics import AdvancedAnalyticsModule, AnalyticsConfig, create_advanced_analytics_module, create_advanced_analytics_module_with_defaults
from .registry import ModuleRegistry

__all__ = [
    # Base classes
    "BaseModule",
    "ModuleConfig",
    "ModuleStatus",

    # Core modules
    "CacheModule",
    "MonitoringModule",
    "OptimizationModule",
    "StorageModule",
    "ExecutionModule",
    "EnginesModule",
    "MLModule",
    "DataAnalysisModule",
    "AIIntelligenceModule",
    "APIRESTModule",
    "SecurityModule",
    "DistributedProcessingModule",
    "EdgeComputingModule",
    "BlockchainModule",
    "IoTAdvancedModule",
    "FederatedLearningModule",
                         "CloudIntegrationModule",
                         "ZeroKnowledgeProofsModule",
                         "QuantumComputingModule",
                         "AdvancedAnalyticsModule",

    # Configuration classes
    "CacheConfig",
    "MonitoringConfig",
    "OptimizationConfig",
    "StorageConfig",
    "ExecutionConfig",
    "EngineModuleConfig",
    "MLModuleConfig",
    "DataAnalysisModuleConfig",
                    "AIIntelligenceConfig",
                "APIRESTConfig",
                                        "SecurityConfig",
                            "DistributedProcessingConfig",
                            "EdgeComputingConfig",
                                                         "BlockchainConfig",
                             "IoTAdvancedConfig",
                             "FederatedLearningConfig",
                             "CloudIntegrationConfig",
                             "ZKProofConfig",
                             "QuantumConfig",
                             "AnalyticsConfig",

    # Registry
    "ModuleRegistry"
]
