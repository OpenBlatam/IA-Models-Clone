"""
Preprocessing Pipeline

Composable preprocessing pipelines.
"""

import logging
from typing import List, Callable, Any

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Composable preprocessing pipeline."""
    
    def __init__(self, steps: List[Callable]):
        """
        Initialize preprocessing pipeline.
        
        Args:
            steps: List of preprocessing functions
        """
        self.steps = steps
    
    def __call__(self, data: Any) -> Any:
        """
        Apply preprocessing pipeline.
        
        Args:
            data: Input data
            
        Returns:
            Preprocessed data
        """
        for step in self.steps:
            data = step(data)
        
        return data
    
    def add_step(self, step: Callable) -> 'PreprocessingPipeline':
        """
        Add preprocessing step.
        
        Args:
            step: Preprocessing function
            
        Returns:
            Self for chaining
        """
        self.steps.append(step)
        return self


def compose_preprocessing_pipeline(
    *steps: Callable
) -> PreprocessingPipeline:
    """
    Compose preprocessing pipeline from steps.
    
    Args:
        *steps: Preprocessing functions
        
    Returns:
        PreprocessingPipeline instance
    """
    return PreprocessingPipeline(list(steps))



