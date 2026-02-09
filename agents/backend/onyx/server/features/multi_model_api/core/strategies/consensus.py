"""
Consensus execution strategy
Executes models in parallel and applies consensus algorithm
"""

import logging
from typing import List, Optional, Callable, Awaitable, Dict
from ...api.schemas import ModelConfig, ModelResponse
from .base import ExecutionStrategy
from .parallel import ParallelStrategy

logger = logging.getLogger(__name__)


class ConsensusStrategy(ExecutionStrategy):
    """Execute models in parallel and apply consensus"""
    
    def __init__(self):
        self.parallel_strategy = ParallelStrategy()
    
    async def execute(
        self,
        models: List[ModelConfig],
        prompt: str,
        execute_model_func: Callable[[ModelConfig, str], Awaitable[ModelResponse]],
        timeout: Optional[float] = None,
        consensus_method: str = "majority",
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> List[ModelResponse]:
        """
        Execute models in parallel (consensus uses parallel execution)
        
        Note: Consensus aggregation is handled by the service layer,
        this strategy just executes models in parallel
        """
        # Consensus strategy executes models in parallel
        # The actual consensus algorithm is applied in the service layer
        return await self.parallel_strategy.execute(
            models=models,
            prompt=prompt,
            execute_model_func=execute_model_func,
            timeout=timeout,
            **kwargs
        )




