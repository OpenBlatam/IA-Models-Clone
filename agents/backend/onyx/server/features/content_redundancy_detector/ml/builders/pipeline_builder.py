"""
Pipeline Builder
Builder for creating pipelines
"""

from typing import Optional, Dict, Any
from pathlib import Path
import logging

from ..pipelines import TrainingPipeline, InferencePipeline
from ..builders import ModelBuilder, TrainerBuilder
from ..config import ConfigBuilder, ConfigLoader, ConfigValidator

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Builder for creating pipelines
    """
    
    @staticmethod
    def build_training_pipeline(
        config_path: Optional[Path] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        config_builder: Optional[ConfigBuilder] = None,
    ) -> TrainingPipeline:
        """
        Build training pipeline
        
        Args:
            config_path: Path to config file
            config_dict: Configuration dictionary
            config_builder: Config builder instance
            
        Returns:
            Training pipeline
        """
        # Get config
        if config_builder is not None:
            config = config_builder.build()
        elif config_dict is not None:
            config = config_dict
        elif config_path is not None:
            config = ConfigLoader.load_yaml(config_path)
        else:
            raise ValueError("Must provide config_path, config_dict, or config_builder")
        
        # Validate config
        validator = ConfigValidator()
        if not validator.validate_config(config):
            errors = validator.get_errors()
            raise ValueError(f"Invalid configuration: {errors}")
        
        # Create pipeline
        if config_path:
            pipeline = TrainingPipeline(config_path=config_path)
        else:
            pipeline = TrainingPipeline(config_dict=config)
        
        pipeline.setup()
        
        return pipeline
    
    @staticmethod
    def build_inference_pipeline(
        model_path: Optional[Path] = None,
        model=None,
        config_path: Optional[Path] = None,
    ) -> InferencePipeline:
        """
        Build inference pipeline
        
        Args:
            model_path: Path to model checkpoint
            model: Model instance
            config_path: Path to config file
            
        Returns:
            Inference pipeline
        """
        return InferencePipeline(
            model_path=model_path,
            model=model,
            config_path=config_path,
        )



