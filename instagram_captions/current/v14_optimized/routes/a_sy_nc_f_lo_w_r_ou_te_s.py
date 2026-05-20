from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, AsyncGenerator
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from core.async_flow_manager import (
from core.shared_resources import get_shared_resources, database_session, http_client
from core.blocking_operations_limiter import (
from typing import Any, List, Dict, Optional
"""
Async Flow Routes for Instagram Captions API v14.0

Specialized routes demonstrating async and non-blocking flows:
- Async pipelines for caption generation
- Event-driven processing
- Reactive flows for dynamic content
- Async streams for real-time processing
- State machines for workflow management
"""


# Import async flow components
    AsyncFlowManager, AsyncPipeline, AsyncStream, EventBus, ReactiveFlow, AsyncStateMachine,
    FlowType, FlowState, FlowConfig, FlowMetrics,
    get_flow_manager, flow_context, non_blocking_operation,
    non_blocking_call, non_blocking_batch, non_blocking_stream
)

# Import shared resources

# Import blocking operations limiter
    blocking_limiter, limit_blocking_operations, OperationType
)

logger = logging.getLogger(__name__)

# Create router
async_flow_router = APIRouter(prefix="/async-flows", tags=["async-flows"])

# Security
security = HTTPBearer()

# Dependency for API key validation
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key"""
    # Simplified validation - in real app, validate against database
    if not credentials.credentials or len(credentials.credentials) < 10:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Dependency to extract user identifier
async def get_user_identifier(request: Request, api_key: str = Depends(verify_api_key)) -> str:
    """Extract user identifier from request or API key"""
    return api_key[:16]  # Use first 16 characters as identifier


# =============================================================================
# ASYNC PIPELINE ROUTES
# =============================================================================

@async_flow_router.post("/pipelines/caption-generation")
@limit_blocking_operations(
    operation_type=OperationType.CAPTION_GENERATION,
    identifier="async_pipeline_generation",
    user_id_param="user_id"
)
async def create_caption_pipeline(
    request: Request,
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> Dict[str, Any]:
    """
    Create an async pipeline for caption generation
    
    Demonstrates non-blocking pipeline processing with multiple stages:
    1. Content validation
    2. AI model loading
    3. Caption generation
    4. Post-processing
    5. Caching
    """
    
    try:
        # Get flow manager
        flow_manager = await get_flow_manager()
        
        # Create pipeline
        pipeline = await flow_manager.create_pipeline(f"caption_pipeline_{user_id}")
        
        # Add pipeline stages
        pipeline.add_stage(
            validate_content_stage,
            config={"timeout": 5.0, "retries": 2}
        ).add_stage(
            load_ai_model_stage,
            config={"timeout": 10.0, "retries": 1}
        ).add_stage(
            generate_caption_stage,
            config={"timeout": 15.0, "retries": 3}
        ).add_stage(
            post_process_stage,
            config={"timeout": 3.0, "retries": 1}
        ).add_stage(
            cache_result_stage,
            config={"timeout": 2.0, "retries": 1}
        )
        
        return {
            "success": True,
            "pipeline_id": f"caption_pipeline_{user_id}",
            "stages": len(pipeline.stages),
            "message": "Caption generation pipeline created successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to create caption pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline creation failed: {str(e)}")


@async_flow_router.post("/pipelines/execute")
async def execute_pipeline(
    pipeline_name: str,
    content_description: str,
    style: str = "casual",
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> Dict[str, Any]:
    """
    Execute a caption generation pipeline
    
    Demonstrates non-blocking pipeline execution with progress tracking.
    """
    
    try:
        # Get flow manager
        flow_manager = await get_flow_manager()
        
        # Prepare input data
        input_data = {
            "content_description": content_description,
            "style": style,
            "user_id": user_id,
            "timestamp": time.time()
        }
        
        # Execute pipeline
        async with non_blocking_operation():
            result = await flow_manager.execute_flow(
                FlowType.PIPELINE,
                pipeline_name,
                input_data
            )
        
        return {
            "success": True,
            "pipeline_name": pipeline_name,
            "result": result,
            "processing_time": time.time() - input_data["timestamp"],
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


# =============================================================================
# ASYNC STREAM ROUTES
# =============================================================================

@async_flow_router.post("/streams/real-time-captions")
async def create_real_time_caption_stream(
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> Dict[str, Any]:
    """
    Create a real-time caption generation stream
    
    Demonstrates non-blocking streaming with backpressure control.
    """
    
    try:
        # Get flow manager
        flow_manager = await get_flow_manager()
        
        # Create stream
        stream = await flow_manager.create_stream(
            f"caption_stream_{user_id}",
            max_buffer_size=100
        )
        
        return {
            "success": True,
            "stream_id": f"caption_stream_{user_id}",
            "buffer_size": stream.max_buffer_size,
            "message": "Real-time caption stream created successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to create caption stream: {e}")
        raise HTTPException(status_code=500, detail=f"Stream creation failed: {str(e)}")


@async_flow_router.post("/streams/produce")
async def produce_to_stream(
    stream_name: str,
    content_description: str,
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> Dict[str, Any]:
    """
    Produce content to a stream for processing
    
    Demonstrates non-blocking stream production with backpressure handling.
    """
    
    try:
        # Get flow manager
        flow_manager = await get_flow_manager()
        
        # Get stream
        if stream_name not in flow_manager.streams:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        stream = flow_manager.streams[stream_name]
        
        # Prepare content
        content = {
            "content_description": content_description,
            "user_id": user_id,
            "timestamp": time.time()
        }
        
        # Produce to stream
        async with non_blocking_operation():
            success = await stream.produce(content)
        
        if success:
            return {
                "success": True,
                "stream_name": stream_name,
                "message": "Content produced to stream successfully",
                "timestamp": time.time()
            }
        else:
            return {
                "success": False,
                "stream_name": stream_name,
                "message": "Stream buffer full - backpressure applied",
                "timestamp": time.time()
            }
        
    except Exception as e:
        logger.error(f"Stream production failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stream production failed: {str(e)}")


@async_flow_router.get("/streams/consume/{stream_name}")
async def consume_from_stream(
    stream_name: str,
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> StreamingResponse:
    """
    Consume from a stream with Server-Sent Events
    
    Demonstrates non-blocking stream consumption with real-time updates.
    """
    
    try:
        # Get flow manager
        flow_manager = await get_flow_manager()
        
        # Get stream
        if stream_name not in flow_manager.streams:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        stream = flow_manager.streams[stream_name]
        
        async def generate_stream_data():
            """Generate stream data as Server-Sent Events"""
            try:
                async for item in stream.consume():
                    # Process item (generate caption)
                    caption = await generate_caption_for_stream_item(item)
                    
                    # Send as SSE
                    yield f"data: {json.dumps(caption)}\n\n"
                    
            except Exception as e:
                logger.error(f"Stream consumption error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream_data(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Stream consumption failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stream consumption failed: {str(e)}")


# =============================================================================
# REACTIVE FLOW ROUTES
# =============================================================================

@async_flow_router.post("/reactive/create")
async def create_reactive_flow(
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> Dict[str, Any]:
    """
    Create a reactive flow for dynamic caption generation
    
    Demonstrates reactive programming patterns with automatic dependency management.
    """
    
    try:
        # Get flow manager
        flow_manager = await get_flow_manager()
        
        # Create reactive flow
        flow = await flow_manager.create_reactive_flow(f"caption_flow_{user_id}")
        
        # Add reactive computations
        flow.add_computation("user_preferences", get_user_preferences, [])
        flow.add_computation("content_analysis", analyze_content, ["user_preferences"])
        flow.add_computation("caption_generation", generate_reactive_caption, ["content_analysis"])
        flow.add_computation("optimization", optimize_caption, ["caption_generation"])
        
        return {
            "success": True,
            "flow_id": f"caption_flow_{user_id}",
            "computations": ["user_preferences", "content_analysis", "caption_generation", "optimization"],
            "message": "Reactive flow created successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to create reactive flow: {e}")
        raise HTTPException(status_code=500, detail=f"Reactive flow creation failed: {str(e)}")


@async_flow_router.post("/reactive/compute")
async def compute_reactive_result(
    flow_name: str,
    computation_name: str,
    content_description: str,
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> Dict[str, Any]:
    """
    Compute a reactive result with automatic dependency resolution
    
    Demonstrates non-blocking reactive computation with dependency tracking.
    """
    
    try:
        # Get flow manager
        flow_manager = await get_flow_manager()
        
        # Get reactive flow
        if flow_name not in flow_manager.reactive_flows:
            raise HTTPException(status_code=404, detail="Reactive flow not found")
        
        flow = flow_manager.reactive_flows[flow_name]
        
        # Set context for computation
        flow_context.set({
            "content_description": content_description,
            "user_id": user_id,
            "timestamp": time.time()
        })
        
        # Compute result
        async with non_blocking_operation():
            result = await flow.get(computation_name)
        
        return {
            "success": True,
            "flow_name": flow_name,
            "computation_name": computation_name,
            "result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Reactive computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reactive computation failed: {str(e)}")


# =============================================================================
# STATE MACHINE ROUTES
# =============================================================================

@async_flow_router.post("/state-machines/caption-workflow")
async def create_caption_workflow_state_machine(
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> Dict[str, Any]:
    """
    Create a state machine for caption generation workflow
    
    Demonstrates non-blocking state transitions with async handlers.
    """
    
    try:
        # Get flow manager
        flow_manager = await get_flow_manager()
        
        # Create state machine
        state_machine = await flow_manager.create_state_machine(
            f"caption_workflow_{user_id}",
            initial_state="idle"
        )
        
        # Add states
        state_machine.add_state("idle").add_state("validating").add_state("generating").add_state("completed").add_state("failed")
        
        # Add transitions
        state_machine.add_transition("idle", "validating", "start_generation")
        state_machine.add_transition("validating", "generating", "validation_passed")
        state_machine.add_transition("validating", "failed", "validation_failed")
        state_machine.add_transition("generating", "completed", "generation_success")
        state_machine.add_transition("generating", "failed", "generation_failed")
        state_machine.add_transition("completed", "idle", "reset")
        state_machine.add_transition("failed", "idle", "reset")
        
        # Add state handlers
        state_machine.set_state_handler("validating", handle_validating_state)
        state_machine.set_state_handler("generating", handle_generating_state)
        state_machine.set_state_handler("completed", handle_completed_state)
        state_machine.set_state_handler("failed", handle_failed_state)
        
        return {
            "success": True,
            "state_machine_id": f"caption_workflow_{user_id}",
            "current_state": state_machine.get_current_state(),
            "available_states": list(state_machine.states.keys()),
            "message": "Caption workflow state machine created successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to create state machine: {e}")
        raise HTTPException(status_code=500, detail=f"State machine creation failed: {str(e)}")


@async_flow_router.post("/state-machines/trigger")
async def trigger_state_transition(
    state_machine_name: str,
    trigger: str,
    context: Optional[Dict[str, Any]] = None,
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> Dict[str, Any]:
    """
    Trigger a state transition in the workflow
    
    Demonstrates non-blocking state transitions with context handling.
    """
    
    try:
        # Get flow manager
        flow_manager = await get_flow_manager()
        
        # Get state machine
        if state_machine_name not in flow_manager.state_machines:
            raise HTTPException(status_code=404, detail="State machine not found")
        
        state_machine = flow_manager.state_machines[state_machine_name]
        
        # Prepare context
        transition_context = context or {}
        transition_context.update({
            "user_id": user_id,
            "timestamp": time.time()
        })
        
        # Trigger transition
        async with non_blocking_operation():
            success = await state_machine.trigger(trigger, transition_context)
        
        if success:
            return {
                "success": True,
                "state_machine_name": state_machine_name,
                "trigger": trigger,
                "new_state": state_machine.get_current_state(),
                "state_history": state_machine.get_state_history(),
                "timestamp": time.time()
            }
        else:
            return {
                "success": False,
                "state_machine_name": state_machine_name,
                "trigger": trigger,
                "current_state": state_machine.get_current_state(),
                "message": "Invalid transition for current state",
                "timestamp": time.time()
            }
        
    except Exception as e:
        logger.error(f"State transition failed: {e}")
        raise HTTPException(status_code=500, detail=f"State transition failed: {str(e)}")


# =============================================================================
# EVENT-DRIVEN ROUTES
# =============================================================================

@async_flow_router.post("/events/publish")
async def publish_event(
    event_type: str,
    event_data: Dict[str, Any],
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> Dict[str, Any]:
    """
    Publish an event to the event bus
    
    Demonstrates non-blocking event-driven processing.
    """
    
    try:
        # Get flow manager
        flow_manager = await get_flow_manager()
        
        # Add user context to event data
        event_data["user_id"] = user_id
        event_data["timestamp"] = time.time()
        
        # Publish event
        async with non_blocking_operation():
            await flow_manager.event_bus.publish(event_type, event_data)
        
        return {
            "success": True,
            "event_type": event_type,
            "event_data": event_data,
            "message": "Event published successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Event publishing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Event publishing failed: {str(e)}")


@async_flow_router.post("/events/subscribe")
async def subscribe_to_event(
    event_type: str,
    handler_name: str,
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> Dict[str, Any]:
    """
    Subscribe to an event type
    
    Demonstrates non-blocking event subscription and handling.
    """
    
    try:
        # Get flow manager
        flow_manager = await get_flow_manager()
        
        # Create handler function
        async def event_handler(event_data: Dict[str, Any]):
            """Handle incoming events"""
            logger.info(f"Handling {event_type} event for user {user_id}")
            
            # Process event based on type
            if event_type == "caption_request":
                await handle_caption_request_event(event_data)
            elif event_type == "content_update":
                await handle_content_update_event(event_data)
            else:
                logger.info(f"Unknown event type: {event_type}")
        
        # Subscribe to event
        flow_manager.event_bus.subscribe(event_type, event_handler)
        
        return {
            "success": True,
            "event_type": event_type,
            "handler_name": handler_name,
            "user_id": user_id,
            "message": "Event subscription created successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Event subscription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Event subscription failed: {str(e)}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Pipeline stage functions
async def validate_content_stage(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Validate content stage"""
    content = data.get("content_description", "")
    if not content or len(content) < 10:
        raise ValueError("Content description too short")
    
    data["validated"] = True
    data["validation_time"] = time.time()
    return data

async def load_ai_model_stage(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Load AI model stage"""
    # Simulate AI model loading
    await asyncio.sleep(0.1)
    data["model_loaded"] = True
    data["model_load_time"] = time.time()
    return data

async def generate_caption_stage(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate caption stage"""
    content = data.get("content_description", "")
    style = data.get("style", "casual")
    
    # Simulate caption generation
    await asyncio.sleep(0.2)
    caption = f"Generated caption for: {content} (Style: {style})"
    
    data["caption"] = caption
    data["generation_time"] = time.time()
    return data

async def post_process_stage(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process caption stage"""
    caption = data.get("caption", "")
    
    # Simulate post-processing
    await asyncio.sleep(0.05)
    processed_caption = f"✨ {caption} ✨"
    
    data["processed_caption"] = processed_caption
    data["post_process_time"] = time.time()
    return data

async def cache_result_stage(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Cache result stage"""
    # Simulate caching
    await asyncio.sleep(0.01)
    data["cached"] = True
    data["cache_time"] = time.time()
    return data

# Reactive computation functions
async def get_user_preferences() -> Dict[str, Any]:
    """Get user preferences"""
    await asyncio.sleep(0.1)
    return {"style": "casual", "hashtag_count": 15}

async def analyze_content(preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze content based on preferences"""
    await asyncio.sleep(0.1)
    return {"analysis": "content_analyzed", "preferences": preferences}

async def generate_reactive_caption(analysis: Dict[str, Any]) -> str:
    """Generate caption reactively"""
    await asyncio.sleep(0.2)
    return f"Reactive caption: {analysis['analysis']}"

async def optimize_caption(caption: str) -> str:
    """Optimize caption"""
    await asyncio.sleep(0.1)
    return f"Optimized: {caption}"

# State machine handlers
async def handle_validating_state(context: Dict[str, Any]):
    """Handle validating state"""
    logger.info("Validating content...")
    await asyncio.sleep(0.1)

async def handle_generating_state(context: Dict[str, Any]):
    """Handle generating state"""
    logger.info("Generating caption...")
    await asyncio.sleep(0.2)

async def handle_completed_state(context: Dict[str, Any]):
    """Handle completed state"""
    logger.info("Caption generation completed")

async def handle_failed_state(context: Dict[str, Any]):
    """Handle failed state"""
    logger.error("Caption generation failed")

# Event handlers
async def handle_caption_request_event(event_data: Dict[str, Any]):
    """Handle caption request event"""
    logger.info(f"Processing caption request: {event_data}")

async def handle_content_update_event(event_data: Dict[str, Any]):
    """Handle content update event"""
    logger.info(f"Processing content update: {event_data}")

# Stream processing
async def generate_caption_for_stream_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Generate caption for stream item"""
    content = item.get("content_description", "")
    user_id = item.get("user_id", "")
    
    # Simulate caption generation
    await asyncio.sleep(0.1)
    caption = f"Stream caption for: {content}"
    
    return {
        "caption": caption,
        "user_id": user_id,
        "timestamp": time.time()
    } 