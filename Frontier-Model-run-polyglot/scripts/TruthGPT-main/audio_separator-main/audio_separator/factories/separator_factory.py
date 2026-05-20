"""
Factory for creating audio separators.
Refactored to use factory pattern for better extensibility.
"""

from typing import Dict, Any, Optional, Type
from ..separator.audio_separator import AudioSeparator
from ..separator.batch_separator import BatchSeparator
from ..separator.base_separator import BaseSeparator
from ..exceptions import AudioValidationError
from ..logger import logger


class SeparatorFactory:
    """
    Factory for creating separator instances.
    
    Provides:
    - Centralized creation logic
    - Type registration
    - Configuration management
    """
    
    _separator_types: Dict[str, Type[BaseSeparator]] = {
        "audio": AudioSeparator,
        "batch": BatchSeparator,
    }
    
    @classmethod
    def create(
        cls,
        separator_type: str = DEFAULT_SEPARATOR_TYPE,
        model_type: str = DEFAULT_MODEL_TYPE,
        model_kwargs: Optional[Dict[str, Any]] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        **kwargs
    ) -> BaseSeparator:
        """
        Create a separator instance.
        
        Args:
            separator_type: Type of separator ('audio' or 'batch')
            model_type: Type of model to use
            model_kwargs: Model initialization arguments
            sample_rate: Target sample rate
            **kwargs: Additional separator arguments
            
        Returns:
            Separator instance
            
        Raises:
            AudioValidationError: If separator type is invalid
        """
        if separator_type not in cls._separator_types:
            raise AudioValidationError(
                f"Unknown separator type: {separator_type}. "
                f"Available types: {list(cls._separator_types.keys())}",
                component="SeparatorFactory",
                error_code=ERROR_CODE_INVALID_SEPARATOR_TYPE
            )
        
        separator_class = cls._separator_types[separator_type]
        
        logger.debug(f"Creating {separator_type} separator with model_type={model_type}")
        
        if separator_type == "audio":
            return separator_class(
                model_type=model_type,
                model_kwargs=model_kwargs,
                sample_rate=sample_rate,
                **kwargs
            )
        elif separator_type == "batch":
            return separator_class(
                model_type=model_type,
                model_kwargs=model_kwargs,
                sample_rate=sample_rate,
                **kwargs
            )
        else:
            return separator_class(**kwargs)
    
    @classmethod
    def register_separator(
        cls,
        separator_type: str,
        separator_class: Type[BaseSeparator]
    ):
        """
        Register a new separator type.
        
        Args:
            separator_type: Type identifier
            separator_class: Separator class
        """
        cls._separator_types[separator_type] = separator_class
        logger.info(f"Registered separator type: {separator_type}")

