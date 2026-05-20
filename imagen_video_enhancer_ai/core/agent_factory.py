"""
Agent Factory for Imagen Video Enhancer AI
==========================================

Factory pattern for creating and configuring agents.
"""

import logging
from typing import Optional
from pathlib import Path

from ..config.enhancer_config import EnhancerConfig
from .constants import (
    DEFAULT_MAX_PARALLEL_TASKS,
    DEFAULT_OUTPUT_DIR,
    OUTPUT_DIRECTORIES
)
from .helpers import create_output_directories

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory for creating EnhancerAgent instances.
    
    Provides a clean way to create and configure agents
    with proper initialization order.
    """
    
    @staticmethod
    def create_agent(
        config: Optional[EnhancerConfig] = None,
        max_parallel_tasks: int = DEFAULT_MAX_PARALLEL_TASKS,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        debug: bool = False
    ):
        """
        Create and configure an EnhancerAgent.
        
        Args:
            config: Optional configuration
            max_parallel_tasks: Maximum parallel tasks
            output_dir: Output directory
            debug: Debug mode
            
        Returns:
            Configured EnhancerAgent instance
        """
        from .enhancer_agent import EnhancerAgent
        
        # Validate and prepare configuration
        final_config = config or EnhancerConfig()
        final_config.validate()
        
        # Prepare output directories
        output_path = Path(output_dir)
        output_dirs = create_output_directories(output_path, OUTPUT_DIRECTORIES)
        
        # Create agent with prepared configuration
        agent = EnhancerAgent(
            config=final_config,
            max_parallel_tasks=max_parallel_tasks,
            output_dir=str(output_path),
            debug=debug
        )
        
        logger.info(
            f"Created EnhancerAgent with {max_parallel_tasks} max parallel tasks, "
            f"output dir: {output_dir}"
        )
        
        return agent
    
    @staticmethod
    def create_from_config_file(config_path: str, **kwargs):
        """
        Create agent from configuration file.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Additional arguments for agent creation
            
        Returns:
            Configured EnhancerAgent instance
        """
        import json
        from pathlib import Path
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        config = EnhancerConfig.from_dict(config_data)
        
        return AgentFactory.create_agent(config=config, **kwargs)




