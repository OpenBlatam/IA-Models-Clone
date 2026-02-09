"""
Integration Module for Deep Learning Generator

Handles integration with external systems and advanced features.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AdvancedFeaturesIntegrator:
    """
    Handles integration of advanced features and external systems.
    """
    
    @staticmethod
    def get_pipeline_info() -> Dict[str, Any]:
        """
        Get information about available advanced pipelines.
        
        Returns:
            Dictionary with pipeline information
        """
        try:
            # Try relative import first
            from ...robot_movement_ai.core.architecture.pipelines import get_pipeline_info
            pipeline_info = get_pipeline_info()
            return {
                "pipelines_available": pipeline_info.get("modules", {}),
                "total_pipeline_modules": len(pipeline_info.get("modules", {})),
                "available": True
            }
        except ImportError:
            return {
                "pipelines_available": {},
                "total_pipeline_modules": 0,
                "available": False
            }
    
    @staticmethod
    def integrate_pipelines(config: Dict[str, Any]) -> None:
        """
        Integrate advanced pipelines into configuration.
        
        Args:
            config: Configuration dictionary to update
        """
        try:
            # Try relative import first
            from ...robot_movement_ai.core.architecture.pipelines import (
                TrainingPipeline,
                InferencePipeline,
                TransferLearningManager,
                AutoMLPipeline,
                KnowledgeDistillation,
                ReinforcementLearningWrapper
            )
            config["advanced_pipelines"] = {
                "training": TrainingPipeline,
                "inference": InferencePipeline,
                "transfer_learning": TransferLearningManager,
                "automl": AutoMLPipeline,
                "knowledge_distillation": KnowledgeDistillation,
                "reinforcement_learning": ReinforcementLearningWrapper
            }
            logger.info("Advanced pipelines integrated into generator")
        except ImportError:
            logger.debug("Advanced pipelines not available, continuing without them")
            config["advanced_pipelines"] = None

