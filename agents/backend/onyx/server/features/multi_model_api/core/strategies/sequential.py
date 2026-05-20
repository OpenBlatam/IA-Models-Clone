"""
Sequential execution strategy
Executes models one by one in order
"""

import asyncio
import logging
from typing import List, Optional, Callable, Awaitable
from ...api.schemas import ModelConfig, ModelResponse
from ...api.helpers import get_enabled_models, calculate_response_stats
from .base import ExecutionStrategy

logger = logging.getLogger(__name__)


class SequentialStrategy(ExecutionStrategy):
    """Execute models sequentially"""
    
    async def execute(
        self,
        models: List[ModelConfig],
        prompt: str,
        execute_model_func: Callable[[ModelConfig, str], Awaitable[ModelResponse]],
        timeout: Optional[float] = None,
        **kwargs
    ) -> List[ModelResponse]:
        """Execute models sequentially with error handling"""
        enabled_models = get_enabled_models(models)
        if not enabled_models:
            return []
        
        responses = []
        total_models = len(enabled_models)
        
        for i, model in enumerate(enabled_models):
            try:
                # Execute with optional timeout
                execute_coro = execute_model_func(model, prompt, **kwargs)
                if timeout:
                    response = await asyncio.wait_for(execute_coro, timeout=timeout)
                else:
                    response = await execute_coro
                
                responses.append(response)
                
                # Log progress (only for debug to reduce overhead)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Sequential execution: {i + 1}/{total_models} completed",
                        extra={
                            "model_type": model.model_type.value,
                            "progress": f"{i + 1}/{total_models}",
                            "strategy": "sequential"
                        }
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Model {model.model_type} timed out after {timeout}s",
                    extra={
                        "model_type": model.model_type.value,
                        "timeout": timeout,
                        "strategy": "sequential"
                    }
                )
                responses.append(ModelResponse(
                    model_type=model.model_type,
                    response="",
                    success=False,
                    error=f"Timeout after {timeout}s"
                ))
            except Exception as e:
                logger.error(
                    f"Model {model.model_type} failed: {e}",
                    extra={
                        "model_type": model.model_type.value,
                        "error_type": type(e).__name__,
                        "strategy": "sequential"
                    },
                    exc_info=True
                )
                responses.append(ModelResponse(
                    model_type=model.model_type,
                    response="",
                    success=False,
                    error=str(e)
                ))
        
        # Log summary
        success_count, _ = calculate_response_stats(responses)
        logger.info(
            f"Sequential execution completed: {success_count}/{len(responses)} successful",
            extra={
                "success_count": success_count,
                "total_count": len(responses),
                "strategy": "sequential"
            }
        )
        
        return responses

