from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
from typing import Any, Dict, Optional
from datetime import datetime
from ...models import VideoRequest, VideoResponse
from .generator import OnyxVideoGenerator
from .core.models import GeneratorStatus
from typing import Any, List, Dict, Optional
import logging
"""
Onyx Video Workflow - Utilities

Utility functions and helpers for the Onyx video workflow system.
"""




# Global generator instances
onyx_video_generators: Dict[str, OnyxVideoGenerator] = {}


async def initialize_onyx_video_system(workflow_type: str = "default") -> None:
    """Initialize Onyx video system."""
    if workflow_type not in onyx_video_generators:
        generator = OnyxVideoGenerator(workflow_type)
        await generator.initialize()
        onyx_video_generators[workflow_type] = generator


async def generate_onyx_video(request: VideoRequest, workflow_type: str = "default") -> VideoResponse:
    """Generate video using Onyx system."""
    if workflow_type not in onyx_video_generators:
        await initialize_onyx_video_system(workflow_type)
    
    generator = onyx_video_generators[workflow_type]
    return await generator.generate_video(request)


async def generate_onyx_video_with_vision(request: VideoRequest, image_data: bytes, workflow_type: str = "default") -> VideoResponse:
    """Generate video with vision using Onyx system."""
    if workflow_type not in onyx_video_generators:
        await initialize_onyx_video_system(workflow_type)
    
    generator = onyx_video_generators[workflow_type]
    return await generator.generate_video_with_vision(request, image_data)


async def get_onyx_video_status(workflow_type: str = "default") -> Dict[str, Any]:
    """Get Onyx video system status."""
    if workflow_type not in onyx_video_generators:
        return {"error": f"Workflow type {workflow_type} not initialized"}
    
    generator = onyx_video_generators[workflow_type]
    status = await generator.get_generator_status()
    return status.__dict__


async def switch_workflow_type(current_type: str, new_type: str) -> bool:
    """Switch workflow type for a generator."""
    try:
        if current_type not in onyx_video_generators:
            await initialize_onyx_video_system(current_type)
        
        generator = onyx_video_generators[current_type]
        await generator.switch_workflow_type(new_type)
        
        # Update the dictionary key
        onyx_video_generators[new_type] = generator
        if current_type != new_type:
            del onyx_video_generators[current_type]
        
        return True
        
    except Exception as e:
        return False


async def get_available_workflow_types() -> Dict[str, str]:
    """Get available workflow types and their descriptions."""
    if not onyx_video_generators:
        # Return default types if no generators are initialized
        return {
            "default": "Standard video generation workflow",
            "vision": "Vision-based video generation workflow",
            "quick": "Quick video generation workflow",
            "advanced": "Advanced video generation with additional processing"
        }
    
    # Get from the first available generator
    first_generator = next(iter(onyx_video_generators.values()))
    return await first_generator.get_available_workflow_types()


async def cleanup_onyx_video_system(workflow_type: Optional[str] = None) -> None:
    """Cleanup Onyx video system."""
    if workflow_type:
        if workflow_type in onyx_video_generators:
            await onyx_video_generators[workflow_type].cleanup()
            del onyx_video_generators[workflow_type]
    else:
        # Cleanup all generators
        for generator in onyx_video_generators.values():
            await generator.cleanup()
        onyx_video_generators.clear()


async async def create_video_request(
    input_text: str,
    user_id: str,
    quality: str = "medium",
    duration: int = 60,
    output_format: str = "mp4"
) -> VideoRequest:
    """Create a video request with default values."""
    return VideoRequest(
        request_id=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        input_text=input_text,
        user_id=user_id,
        quality=quality,
        duration=duration,
        output_format=output_format
    )


async def batch_generate_videos(requests: list[VideoRequest], workflow_type: str = "default") -> list[VideoResponse]:
    """Generate multiple videos in batch."""
    if workflow_type not in onyx_video_generators:
        await initialize_onyx_video_system(workflow_type)
    
    generator = onyx_video_generators[workflow_type]
    
    # Generate videos concurrently
    tasks = [generator.generate_video(request) for request in requests]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    results = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            # Create error response
            error_response = VideoResponse(
                request_id=requests[i].request_id,
                status="failed",
                output_url="",
                metadata={"error": str(response)}
            )
            results.append(error_response)
        else:
            results.append(response)
    
    return results


async def get_generator_instance(workflow_type: str = "default") -> Optional[OnyxVideoGenerator]:
    """Get a generator instance for the specified workflow type."""
    if workflow_type not in onyx_video_generators:
        await initialize_onyx_video_system(workflow_type)
    
    return onyx_video_generators.get(workflow_type)


async def health_check(workflow_type: str = "default") -> Dict[str, Any]:
    """Perform a health check on the video system."""
    try:
        if workflow_type not in onyx_video_generators:
            return {
                "status": "not_initialized",
                "workflow_type": workflow_type,
                "timestamp": datetime.now().isoformat()
            }
        
        generator = onyx_video_generators[workflow_type]
        status = await generator.get_generator_status()
        
        health = {
            "status": "healthy",
            "workflow_type": workflow_type,
            "timestamp": datetime.now().isoformat(),
            "generator_status": status.__dict__
        }
        
        return health
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "workflow_type": workflow_type,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


async def get_system_statistics() -> Dict[str, Any]:
    """Get system-wide statistics."""
    stats = {
        "total_generators": len(onyx_video_generators),
        "workflow_types": list(onyx_video_generators.keys()),
        "timestamp": datetime.now().isoformat()
    }
    
    # Get individual generator statistics
    generator_stats = {}
    for workflow_type, generator in onyx_video_generators.items():
        try:
            status = await generator.get_generator_status()
            generator_stats[workflow_type] = status.__dict__
        except Exception as e:
            generator_stats[workflow_type] = {"error": str(e)}
    
    stats["generator_statistics"] = generator_stats
    return stats 