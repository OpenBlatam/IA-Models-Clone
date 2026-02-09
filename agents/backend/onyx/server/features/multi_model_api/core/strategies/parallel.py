"""
Parallel execution strategy
Executes all models concurrently
"""

import asyncio
import logging
from typing import List, Optional, Callable, Awaitable
from ...api.schemas import ModelConfig, ModelResponse
from ...api.helpers import get_enabled_models, calculate_response_stats
from .base import ExecutionStrategy

logger = logging.getLogger(__name__)


class ParallelStrategy(ExecutionStrategy):
    """Execute models in parallel"""
    
    async def execute(
        self,
        models: List[ModelConfig],
        prompt: str,
        execute_model_func: Callable[[ModelConfig, str], Awaitable[ModelResponse]],
        timeout: Optional[float] = None,
        **kwargs
    ) -> List[ModelResponse]:
        """Execute models in parallel with optimized error handling"""
        enabled_models = get_enabled_models(models)
        
        if not enabled_models:
            return []
        
        # Create actual Task objects for proper cancellation
        tasks = [
            asyncio.create_task(execute_model_func(model, prompt, **kwargs))
            for model in enabled_models
        ]
        
        # Execute with optional timeout
        if timeout:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Parallel execution timed out after {timeout}s",
                    extra={
                        "timeout": timeout,
                        "models_count": len(enabled_models),
                        "models": [m.model_type.value for m in enabled_models]
                    }
                )
                # Cancel remaining tasks to free resources
                for task in tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass  # Ignore cancellation errors
                
                # Return timeout responses
                return [
                    ModelResponse(
                        model_type=model.model_type,
                        response="",
                        success=False,
                        error=f"Timeout after {timeout}s"
                    )
                    for model in enabled_models
                ]
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        responses = []
        for model, result in zip(enabled_models, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Model {model.model_type} failed: {result}",
                    extra={
                        "model_type": model.model_type.value,
                        "error_type": type(result).__name__
                    },
                    exc_info=isinstance(result, BaseException)
                )
                responses.append(ModelResponse(
                    model_type=model.model_type,
                    response="",
                    success=False,
                    error=str(result)
                ))
            else:
                responses.append(result)
        
        # Log summary
        success_count, _ = calculate_response_stats(responses)
        logger.info(
            f"Parallel execution completed: {success_count}/{len(responses)} successful",
            extra={
                "success_count": success_count,
                "total_count": len(responses),
                "strategy": "parallel"
            }
        )
        
        return responses

