"""
Agent Builder
=============

Builder pattern for constructing EnhancerAgent with all components.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .enhancer_agent import EnhancerAgent
from .module_system import ModuleRegistry, ModuleConfig
from ..config.enhancer_config import EnhancerConfig
from .constants import (
    DEFAULT_MAX_PARALLEL_TASKS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_CACHE_TTL_HOURS,
    DEFAULT_MAX_RETRIES
)

logger = logging.getLogger(__name__)


class AgentBuilder:
    """Builder for EnhancerAgent."""
    
    def __init__(self):
        """Initialize agent builder."""
        self.config: Optional[EnhancerConfig] = None
        self.max_parallel_tasks = DEFAULT_MAX_PARALLEL_TASKS
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.debug = False
        self.modules: ModuleRegistry = ModuleRegistry()
    
    def with_config(self, config: EnhancerConfig) -> "AgentBuilder":
        """
        Set agent configuration.
        
        Args:
            config: Enhancer configuration
            
        Returns:
            Self for chaining
        """
        self.config = config
        return self
    
    def with_max_parallel_tasks(self, max_tasks: int) -> "AgentBuilder":
        """
        Set maximum parallel tasks.
        
        Args:
            max_tasks: Maximum parallel tasks
            
        Returns:
            Self for chaining
        """
        self.max_parallel_tasks = max_tasks
        return self
    
    def with_output_dir(self, output_dir: str) -> "AgentBuilder":
        """
        Set output directory.
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Self for chaining
        """
        self.output_dir = output_dir
        return self
    
    def with_debug(self, debug: bool = True) -> "AgentBuilder":
        """
        Enable debug mode.
        
        Args:
            debug: Debug flag
            
        Returns:
            Self for chaining
        """
        self.debug = debug
        return self
    
    def with_module(
        self,
        module_name: str,
        enabled: bool = True,
        priority: int = 0,
        dependencies: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> "AgentBuilder":
        """
        Configure a module.
        
        Args:
            module_name: Module name
            enabled: Whether module is enabled
            priority: Module priority
            dependencies: Module dependencies
            config: Module configuration
            
        Returns:
            Self for chaining
        """
        module_config = ModuleConfig(
            name=module_name,
            enabled=enabled,
            priority=priority,
            dependencies=dependencies or [],
            config=config or {}
        )
        self.modules.module_configs[module_name] = module_config
        return self
    
    def build(self) -> EnhancerAgent:
        """
        Build and return EnhancerAgent.
        
        Returns:
            Configured EnhancerAgent instance
        """
        config = self.config or EnhancerConfig()
        config.validate()
        
        agent = EnhancerAgent(
            config=config,
            max_parallel_tasks=self.max_parallel_tasks,
            output_dir=self.output_dir,
            debug=self.debug
        )
        
        logger.info("Agent built successfully")
        return agent
    
    @classmethod
    def create_default(cls) -> "AgentBuilder":
        """
        Create builder with default configuration.
        
        Returns:
            AgentBuilder instance
        """
        return cls()
    
    @classmethod
    def from_config_file(cls, config_file: Path) -> "AgentBuilder":
        """
        Create builder from config file.
        
        Args:
            config_file: Path to config file
            
        Returns:
            AgentBuilder instance
        """
        # This would load config from file
        # For now, return default builder
        return cls()




