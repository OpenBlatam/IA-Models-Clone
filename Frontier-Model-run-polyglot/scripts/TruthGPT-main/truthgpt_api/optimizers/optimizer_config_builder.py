"""
Optimizer Config Builder
========================

Builds and manages optimizer configuration dictionaries.
"""

from typing import Dict, Any, Optional
from .core_detector import is_optimization_core_available
from .paper_integration import is_paper_registry_available
from .optimizer_constants import AMSGRAD_SUPPORTED, normalize_optimizer_type, BACKEND_OPTIMIZATION_CORE, BACKEND_PYTORCH, normalize_optimizer_type


class OptimizerConfigBuilder:
    """
    Builds optimizer configuration dictionaries.
    
    Responsibilities:
    - Build configuration dictionaries
    - Add AMSGrad information
    - Add backend information
    - Extract core optimizer configs
    """
    
    @staticmethod
    def build_base_config(
        optimizer_type: str,
        learning_rate: float,
        use_core: bool,
        using_core: bool,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build base configuration dictionary.
        
        Args:
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            use_core: Whether to use optimization_core
            using_core: Whether actually using optimization_core
            kwargs: Optimizer parameters
        
        Returns:
            Base configuration dictionary
        """
        return {
            'optimizer_type': optimizer_type,
            'learning_rate': learning_rate,
            'use_core': use_core,
            'core_available': is_optimization_core_available(),
            'using_core': using_core,
            **kwargs
        }
    
    @staticmethod
    def add_amsgrad_info(config: Dict[str, Any], optimizer_type: str, kwargs: Dict[str, Any]) -> None:
        """
        Add AMSGrad information to configuration if applicable.
        
        Args:
            config: Configuration dictionary to modify
            optimizer_type: Type of optimizer
            kwargs: Optimizer parameters
        """
        if normalize_optimizer_type(optimizer_type) not in AMSGRAD_SUPPORTED:
            return
        
        amsgrad_enabled = kwargs.get('amsgrad', False)
        config['amsgrad'] = amsgrad_enabled
        
        if amsgrad_enabled:
            config['variant'] = 'AMSGrad'
            config['description'] = 'Adam with AMSGrad variant (maintains max of second moment)'
            config['benefits'] = [
                'More stable gradient estimates',
                'Better for non-stationary objectives',
                'Can help with convergence issues'
            ]
    
    @staticmethod
    def add_backend_info(config: Dict[str, Any], core_optimizer: Optional[Any], logger) -> None:
        """
        Add backend information to configuration.
        
        Args:
            config: Configuration dictionary to modify
            core_optimizer: Core optimizer instance or None
            logger: Logger instance
        """
        if core_optimizer is not None:
            config['backend'] = BACKEND_OPTIMIZATION_CORE
            if hasattr(core_optimizer, 'get_config'):
                try:
                    config['core_config'] = core_optimizer.get_config()
                except Exception as e:
                    logger.debug(f"Failed to get core config: {e}")
        else:
            config['backend'] = BACKEND_PYTORCH
    
    @classmethod
    def build_full_config(
        cls,
        optimizer_type: str,
        learning_rate: float,
        use_core: bool,
        using_core: bool,
        kwargs: Dict[str, Any],
        core_optimizer: Optional[Any],
        logger
    ) -> Dict[str, Any]:
        """
        Build complete configuration dictionary.
        
        Args:
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            use_core: Whether to use optimization_core
            using_core: Whether actually using optimization_core
            kwargs: Optimizer parameters
            core_optimizer: Core optimizer instance or None
            logger: Logger instance
        
        Returns:
            Complete configuration dictionary
        """
        config = cls.build_base_config(
            optimizer_type, learning_rate, use_core, using_core, kwargs
        )
        
        cls.add_amsgrad_info(config, optimizer_type, kwargs)
        cls.add_backend_info(config, core_optimizer, logger)
        
        return config
    
    @classmethod
    def build_config(
        cls,
        optimizer_type: str,
        learning_rate: float,
        use_core: bool,
        core_optimizer: Optional[Any],
        kwargs: Dict[str, Any],
        paper_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Build optimizer configuration dictionary (simplified interface).
        
        Args:
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            use_core: Whether to use optimization_core
            core_optimizer: Core optimizer instance or None
            kwargs: Optimizer parameters
            paper_manager: PaperIntegrationManager instance if available
        
        Returns:
            Complete configuration dictionary
        """
        import logging
        logger = logging.getLogger(__name__)
        
        using_core = core_optimizer is not None
        config = cls.build_full_config(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            use_core=use_core,
            using_core=using_core,
            kwargs=kwargs,
            core_optimizer=core_optimizer,
            logger=logger
        )
        
        # Add paper info if available
        if paper_manager and is_paper_registry_available():
            try:
                config['paper_enhanced'] = paper_manager.is_enhanced
                if paper_manager.paper_params:
                    config['paper_params'] = paper_manager.paper_params
            except Exception:
                pass
        
        return config

