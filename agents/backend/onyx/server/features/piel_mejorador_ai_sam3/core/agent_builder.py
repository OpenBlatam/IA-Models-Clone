"""
Agent Builder for Piel Mejorador AI SAM3
=========================================

Builder pattern for constructing PielMejoradorAgent.
"""

import logging
from typing import Optional
from pathlib import Path

from ..config.piel_mejorador_config import PielMejoradorConfig
from .service_factory import ServiceFactory
from .config_validator import ConfigValidator
from .piel_mejorador_agent import PielMejoradorAgent

logger = logging.getLogger(__name__)


class AgentBuilder:
    """
    Builder for PielMejoradorAgent.
    
    Provides fluent interface for agent construction.
    """
    
    def __init__(self):
        """Initialize builder."""
        self._config: Optional[PielMejoradorConfig] = None
        self._max_parallel_tasks: int = 5
        self._output_dir: str = "piel_mejorador_output"
        self._debug: bool = False
    
    def with_config(self, config: PielMejoradorConfig) -> "AgentBuilder":
        """Set configuration."""
        self._config = config
        return self
    
    def with_max_parallel_tasks(self, max_tasks: int) -> "AgentBuilder":
        """Set max parallel tasks."""
        self._max_parallel_tasks = max_tasks
        return self
    
    def with_output_dir(self, output_dir: str) -> "AgentBuilder":
        """Set output directory."""
        self._output_dir = output_dir
        return self
    
    def with_debug(self, debug: bool = True) -> "AgentBuilder":
        """Enable debug mode."""
        self._debug = debug
        return self
    
    def build(self) -> PielMejoradorAgent:
        """
        Build the agent.
        
        Returns:
            Configured PielMejoradorAgent
        """
        config = self._config or PielMejoradorConfig()
        
        # Validate configuration
        validation_result = ConfigValidator.validate_config(config)
        if not validation_result.valid:
            logger.error(f"Configuration validation failed: {validation_result.errors}")
            if validation_result.errors:
                raise ValueError(f"Invalid configuration: {', '.join(validation_result.errors)}")
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        # Create agent with factory
        agent = PielMejoradorAgent(
            config=config,
            max_parallel_tasks=self._max_parallel_tasks,
            output_dir=self._output_dir,
            debug=self._debug,
            factory=ServiceFactory()
        )
        
        return agent




