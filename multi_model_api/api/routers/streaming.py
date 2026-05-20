"""
Streaming router for Multi-Model API
Handles Server-Sent Events (SSE) streaming endpoints
"""

import asyncio
import logging
import uuid
import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ...api.schemas import MultiModelRequest, ModelResponse
from ...api.dependencies import get_execution_service, check_rate_limit
from ...api.helpers import validate_rate_limit, get_client_identifier
from ...core.services import ExecutionService, ValidationService
from ...core.performance import fast_json_dumps
from ...core.consensus import apply_consensus
from ...api.helpers import get_weights_map, build_response_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multi-model", tags=["Streaming"])


@router.post("/execute/stream")
async def execute_multi_model_stream(
    request: MultiModelRequest,
    http_request: Request,
    execution_service: ExecutionService = Depends(get_execution_service)
):
    """
    Execute multiple AI models with streaming responses (SSE)
    
    Returns Server-Sent Events stream with:
    - Model responses as they complete
    - Progress updates
    - Final aggregated response
    
    Args:
        request: Multi-model request
        http_request: FastAPI Request object
        execution_service: Execution service instance
        
    Returns:
        StreamingResponse with SSE events
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Rate limiting
    rate_limit_info = await check_rate_limit(http_request, "execute")
    validate_rate_limit(rate_limit_info)
    
    enabled_models = [m for m in request.models if m.is_enabled]
    if not enabled_models:
        from ...api.exceptions import ValidationException
        raise ValidationException(
            message="At least one model must be enabled",
            field="models"
        )
    
    async def generate_stream():
        """Generate SSE stream"""
        try:
            # Send start event
            yield f"data: {fast_json_dumps({'type': 'start', 'request_id': request_id, 'models_count': len(enabled_models)})}\n\n"
            
            timeout = request.timeout or ValidationService.DEFAULT_TIMEOUT
            responses = []
            
            # Get repository from service
            repository = execution_service.model_repository
            
            if request.strategy == "parallel":
                # Create tasks for all models
                tasks = {
                    asyncio.create_task(repository.execute_model(
                        model.model_type,
                        request.prompt,
                        timeout=timeout,
                        temperature=model.temperature,
                        max_tokens=model.max_tokens,
                        **(model.custom_params or {})
                    )): model
                    for model in enabled_models
                }
                
                # Process tasks as they complete
                try:
                    for task in asyncio.as_completed(tasks.keys()):
                        model = tasks[task]
                        try:
                            response = await asyncio.wait_for(task, timeout=timeout)
                            responses.append(response)
                            
                            response_data: Dict[str, Any] = {
                                'type': 'model_response',
                                'model_type': model.model_type.value,
                                'response': response.response if response.success else None,
                                'success': response.success,
                                'error': response.error,
                                'latency_ms': response.latency_ms,
                                'tokens_used': response.tokens_used
                            }
                            yield f"data: {fast_json_dumps(response_data)}\n\n"
                        except Exception as e:
                            error_response = ModelResponse(
                                model_type=model.model_type,
                                response="",
                                success=False,
                                error=str(e)
                            )
                            responses.append(error_response)
                            yield f"data: {fast_json_dumps({'type': 'model_error', 'model_type': model.model_type.value, 'error': str(e)})}\n\n"
                except asyncio.TimeoutError:
                    # Handle remaining tasks
                    for task, model in tasks.items():
                        if not task.done():
                            error_response = ModelResponse(
                                model_type=model.model_type,
                                response="",
                                success=False,
                                error=f"Timeout after {timeout}s"
                            )
                            responses.append(error_response)
                            yield f"data: {fast_json_dumps({'type': 'model_error', 'model_type': model.model_type.value, 'error': f'Timeout after {timeout}s'})}\n\n"
            
            elif request.strategy == "sequential":
                for model in enabled_models:
                    try:
                        response = await asyncio.wait_for(
                            repository.execute_model(
                                model.model_type,
                                request.prompt,
                                timeout=timeout,
                                temperature=model.temperature,
                                max_tokens=model.max_tokens,
                                **(model.custom_params or {})
                            ),
                            timeout=timeout
                        )
                        responses.append(response)
                        
                        response_data: Dict[str, Any] = {
                            'type': 'model_response',
                            'model_type': model.model_type.value,
                            'response': response.response if response.success else None,
                            'success': response.success,
                            'error': response.error,
                            'latency_ms': response.latency_ms,
                            'tokens_used': response.tokens_used
                        }
                        yield f"data: {fast_json_dumps(response_data)}\n\n"
                    except Exception as e:
                        error_response = ModelResponse(
                            model_type=model.model_type,
                            response="",
                            success=False,
                            error=str(e)
                        )
                        responses.append(error_response)
                        yield f"data: {fast_json_dumps({'type': 'model_error', 'model_type': model.model_type.value, 'error': str(e)})}\n\n"
            
            # Apply consensus if needed
            if request.strategy == "consensus":
                successful_responses = [r for r in responses if r.success and r.response]
                if successful_responses:
                    weights_map = get_weights_map(enabled_models)
                    consensus_method = request.consensus_method or "majority"
                    consensus_result = apply_consensus(
                        successful_responses,
                        consensus_method,
                        weights_map
                    )
                    yield f"data: {fast_json_dumps({'type': 'consensus', 'result': consensus_result, 'method': consensus_method})}\n\n"
            
            # Aggregate responses
            weights_map = get_weights_map(enabled_models)
            consensus_method = request.consensus_method or "majority"
            aggregated_response = execution_service.consensus_service.aggregate_responses(
                responses=responses,
                strategy=request.strategy,
                consensus_method=consensus_method,
                weights=weights_map,
                enabled_models=enabled_models
            )
            
            # Build final response data
            response_data = build_response_data(
                request_id,
                request,
                responses,
                aggregated_response,
                start_time,
                enabled_models
            )
            
            # Send complete event
            yield f"data: {fast_json_dumps({'type': 'complete', 'request_id': request_id, 'aggregated_response': aggregated_response, 'total_tokens': response_data['total_tokens'], 'total_latency_ms': response_data['total_latency_ms'], 'success_count': response_data['success_count'], 'failure_count': response_data['failure_count'], 'timestamp': response_data['timestamp']})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error for request {request_id}: {e}", exc_info=True)
            yield f"data: {fast_json_dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )




