"""
Base execution strategy interface
Strategy pattern for different model execution modes
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Awaitable
from ...api.schemas import ModelConfig, ModelResponse


class ExecutionStrategy(ABC):
    """Base class for execution strategies"""
    
    @abstractmethod
    async def execute(
        self,
        models: List[ModelConfig],
        prompt: str,
        execute_model_func: Callable[[ModelConfig, str], Awaitable[ModelResponse]],
        timeout: Optional[float] = None,
        **kwargs
    ) -> List[ModelResponse]:
        """
        Execute models according to strategy
        
        Args:
            models: List of model configurations
            prompt: Input prompt
            execute_model_func: Function to execute a single model
            timeout: Optional timeout for execution
            **kwargs: Additional parameters
            
        Returns:
            List of model responses
        """
        pass
    
    def get_name(self) -> str:
        """Get strategy name"""
        return self.__class__.__name__




