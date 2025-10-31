from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
from pathlib import Path
from ..api.main import OnyxAIVideoSystem, get_system, generate_video, generate_video_with_vision
from ..core.models import VideoRequest, VideoQuality, VideoFormat
from ..config.config_manager import OnyxConfigManager
from ..utils.logger import setup_logger
from typing import Any, List, Dict, Optional
import logging
"""
Onyx AI Video System - Basic Usage Example

This example demonstrates basic usage of the Onyx AI Video system
including initialization, video generation, and system management.
"""


# Import the main system components


async def basic_video_generation():
    """Example of basic video generation."""
    print("=== Basic Video Generation Example ===")
    
    # Initialize system
    system = OnyxAIVideoSystem()
    await system.initialize()
    
    # Create video request
    request = VideoRequest(
        input_text="Create a short video about artificial intelligence and its impact on modern technology",
        user_id="example_user_001",
        quality=VideoQuality.MEDIUM,
        duration=30,
        output_format=VideoFormat.MP4
    )
    
    print(f"Generating video for request: {request.request_id}")
    print(f"Input text: {request.input_text[:50]}...")
    
    # Generate video
    start_time = time.time()
    response = await system.generate_video(request)
    generation_time = time.time() - start_time
    
    print(f"Video generated successfully!")
    print(f"Request ID: {response.request_id}")
    print(f"Status: {response.status}")
    print(f"Generation time: {generation_time:.2f} seconds")
    
    if response.output_url:
        print(f"Output URL: {response.output_url}")
    if response.output_path:
        print(f"Output Path: {response.output_path}")
    if response.duration:
        print(f"Video Duration: {response.duration} seconds")
    if response.file_size:
        print(f"File Size: {response.file_size} bytes")
    
    # Shutdown system
    await system.shutdown()


async def video_with_vision():
    """Example of video generation with vision capabilities."""
    print("\n=== Video Generation with Vision Example ===")
    
    # Initialize system
    system = OnyxAIVideoSystem()
    await system.initialize()
    
    # Create video request
    request = VideoRequest(
        input_text="Analyze this image and create a video explaining what you see",
        user_id="example_user_002",
        quality=VideoQuality.HIGH,
        duration=45,
        output_format=VideoFormat.MP4
    )
    
    # Load sample image (you would replace this with your actual image)
    image_path = Path(__file__).parent / "sample_image.jpg"
    if image_path.exists():
        with open(image_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            image_data = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        print(f"Generating video with vision for request: {request.request_id}")
        print(f"Input text: {request.input_text}")
        print(f"Image file: {image_path}")
        
        # Generate video with vision
        start_time = time.time()
        response = await system.generate_video_with_vision(request, image_data)
        generation_time = time.time() - start_time
        
        print(f"Video with vision generated successfully!")
        print(f"Request ID: {response.request_id}")
        print(f"Status: {response.status}")
        print(f"Generation time: {generation_time:.2f} seconds")
        
        if response.output_url:
            print(f"Output URL: {response.output_url}")
        if response.output_path:
            print(f"Output Path: {response.output_path}")
    else:
        print("Sample image not found, skipping vision example")
    
    # Shutdown system
    await system.shutdown()


async def batch_processing():
    """Example of batch video processing."""
    print("\n=== Batch Processing Example ===")
    
    # Initialize system
    system = OnyxAIVideoSystem()
    await system.initialize()
    
    # Create multiple requests
    requests = [
        VideoRequest(
            input_text="Create a video about machine learning basics",
            user_id="batch_user_001",
            quality=VideoQuality.MEDIUM,
            duration=60
        ),
        VideoRequest(
            input_text="Explain deep learning concepts in a video",
            user_id="batch_user_001",
            quality=VideoQuality.MEDIUM,
            duration=60
        ),
        VideoRequest(
            input_text="Show how neural networks work",
            user_id="batch_user_001",
            quality=VideoQuality.MEDIUM,
            duration=60
        )
    ]
    
    print(f"Processing {len(requests)} videos in batch...")
    
    # Process requests
    responses = []
    for i, request in enumerate(requests, 1):
        print(f"Processing request {i}/{len(requests)}: {request.request_id}")
        
        start_time = time.time()
        response = await system.generate_video(request)
        generation_time = time.time() - start_time
        
        responses.append(response)
        print(f"  Completed in {generation_time:.2f} seconds - Status: {response.status}")
    
    # Summary
    successful = sum(1 for r in responses if r.status == "completed")
    print(f"\nBatch processing completed!")
    print(f"Successful: {successful}/{len(responses)}")
    print(f"Failed: {len(responses) - successful}/{len(responses)}")
    
    # Shutdown system
    await system.shutdown()


async def system_monitoring():
    """Example of system monitoring and metrics."""
    print("\n=== System Monitoring Example ===")
    
    # Initialize system
    system = OnyxAIVideoSystem()
    await system.initialize()
    
    # Generate some test videos first
    for i in range(3):
        request = VideoRequest(
            input_text=f"Test video {i+1} for monitoring",
            user_id="monitor_user_001",
            quality=VideoQuality.LOW,
            duration=10
        )
        await system.generate_video(request)
    
    # Get system status
    print("=== System Status ===")
    status = await system.get_system_status()
    print(f"Status: {status.status}")
    print(f"Version: {status.version}")
    print(f"Uptime: {status.uptime:.1f} seconds")
    print(f"Total Requests: {status.request_count}")
    print(f"Error Rate: {status.error_rate:.2f}%")
    print(f"Active Plugins: {status.active_plugins}/{status.total_plugins}")
    
    if status.cpu_usage:
        print(f"CPU Usage: {status.cpu_usage}%")
    if status.memory_usage:
        print(f"Memory Usage: {status.memory_usage}%")
    if status.gpu_usage:
        print(f"GPU Usage: {status.gpu_usage}%")
    
    # Get performance metrics
    print("\n=== Performance Metrics ===")
    metrics = await system.get_metrics()
    print(f"Total Requests: {metrics.total_requests}")
    print(f"Successful: {metrics.successful_requests}")
    print(f"Failed: {metrics.failed_requests}")
    print(f"Average Processing Time: {metrics.avg_processing_time:.3f}s")
    print(f"Min Processing Time: {metrics.min_processing_time:.3f}s" if metrics.min_processing_time else "N/A")
    print(f"Max Processing Time: {metrics.max_processing_time:.3f}s" if metrics.max_processing_time else "N/A")
    
    if metrics.plugin_executions:
        print("\nPlugin Executions:")
        for plugin, count in metrics.plugin_executions.items():
            errors = metrics.plugin_errors.get(plugin, 0)
            print(f"  {plugin}: {count} executions, {errors} errors")
    
    # Get active requests
    print("\n=== Active Requests ===")
    active_requests = await system.get_active_requests()
    if active_requests:
        for request_id, info in active_requests.items():
            duration = info.get('end_time', time.time()) - info['start_time']
            print(f"  {request_id}: {info['status']} ({duration:.1f}s) - User: {info['user_id']}")
    else:
        print("  No active requests")
    
    # Shutdown system
    await system.shutdown()


async def plugin_management():
    """Example of plugin management."""
    print("\n=== Plugin Management Example ===")
    
    # Initialize system
    system = OnyxAIVideoSystem()
    await system.initialize()
    
    # List plugins
    print("=== Available Plugins ===")
    plugins = system.plugin_manager.get_plugins()
    for name, plugin in plugins.items():
        status_icon = "✓" if plugin.status == "active" else "✗" if plugin.status == "error" else "-"
        print(f"{status_icon} {name} v{plugin.version} ({plugin.status})")
        if plugin.description:
            print(f"    {plugin.description}")
    
    # Get detailed plugin information
    if plugins:
        plugin_name = list(plugins.keys())[0]
        print(f"\n=== Plugin Details: {plugin_name} ===")
        plugin = plugins[plugin_name]
        print(f"Version: {plugin.version}")
        print(f"Status: {plugin.status}")
        print(f"Category: {plugin.category}")
        print(f"Description: {plugin.description or 'N/A'}")
        print(f"Author: {plugin.author or 'N/A'}")
        print(f"Execution Count: {plugin.execution_count}")
        print(f"Success Count: {plugin.success_count}")
        print(f"Error Count: {plugin.error_count}")
        if plugin.avg_execution_time:
            print(f"Average Execution Time: {plugin.avg_execution_time:.3f}s")
    
    # Shutdown system
    await system.shutdown()


async def configuration_management():
    """Example of configuration management."""
    print("\n=== Configuration Management Example ===")
    
    # Initialize system
    system = OnyxAIVideoSystem()
    await system.initialize()
    
    # Show current configuration
    print("=== Current Configuration ===")
    config = system.config
    print(f"System Name: {config.system_name}")
    print(f"Version: {config.version}")
    print(f"Environment: {config.environment}")
    print(f"Debug Mode: {config.debug}")
    
    print(f"\nLogging Level: {config.logging.level}")
    print(f"Use Onyx Logging: {config.logging.use_onyx_logging}")
    
    print(f"\nLLM Provider: {config.llm.provider}")
    print(f"LLM Model: {config.llm.model}")
    print(f"LLM Temperature: {config.llm.temperature}")
    
    print(f"\nDefault Video Quality: {config.video.default_quality}")
    print(f"Default Video Format: {config.video.default_format}")
    print(f"Default Duration: {config.video.default_duration}s")
    print(f"Max Duration: {config.video.max_duration}s")
    
    print(f"\nPlugins Directory: {config.plugins.plugins_directory}")
    print(f"Auto Load Plugins: {config.plugins.auto_load}")
    print(f"Max Workers: {config.plugins.max_workers}")
    
    print(f"\nPerformance Monitoring: {config.performance.enable_monitoring}")
    print(f"Cache Enabled: {config.performance.cache_enabled}")
    print(f"GPU Enabled: {config.performance.gpu_enabled}")
    
    print(f"\nSecurity Encryption: {config.security.enable_encryption}")
    print(f"Input Validation: {config.security.validate_input}")
    print(f"Rate Limiting: {config.security.rate_limit_enabled}")
    
    print(f"\nOnyx Integration:")
    print(f"  Use Onyx Logging: {config.onyx.use_onyx_logging}")
    print(f"  Use Onyx LLM: {config.onyx.use_onyx_llm}")
    print(f"  Use Onyx Telemetry: {config.onyx.use_onyx_telemetry}")
    print(f"  Use Onyx Encryption: {config.onyx.use_onyx_encryption}")
    print(f"  Use Onyx Threading: {config.onyx.use_onyx_threading}")
    print(f"  Use Onyx Retry: {config.onyx.use_onyx_retry}")
    print(f"  Use Onyx GPU: {config.onyx.use_onyx_gpu}")
    
    # Shutdown system
    await system.shutdown()


async def error_handling():
    """Example of error handling."""
    print("\n=== Error Handling Example ===")
    
    # Initialize system
    system = OnyxAIVideoSystem()
    await system.initialize()
    
    # Test invalid input
    print("Testing invalid input...")
    try:
        request = VideoRequest(
            input_text="",  # Empty input
            user_id="error_user_001"
        )
        response = await system.generate_video(request)
    except Exception as e:
        print(f"Expected error for empty input: {type(e).__name__}: {e}")
    
    # Test very long input
    print("Testing very long input...")
    try:
        long_text = "A" * 50000  # Very long text
        request = VideoRequest(
            input_text=long_text,
            user_id="error_user_001"
        )
        response = await system.generate_video(request)
    except Exception as e:
        print(f"Expected error for long input: {type(e).__name__}: {e}")
    
    # Test invalid user access
    print("Testing invalid user access...")
    try:
        request = VideoRequest(
            input_text="Test video",
            user_id="invalid_user"
        )
        response = await system.generate_video(request)
    except Exception as e:
        print(f"Expected error for invalid access: {type(e).__name__}: {e}")
    
    # Shutdown system
    await system.shutdown()


async def main():
    """Main example function."""
    print("Onyx AI Video System - Basic Usage Examples")
    print("=" * 50)
    
    # Setup logging
    setup_logger(level="INFO")
    
    try:
        # Run examples
        await basic_video_generation()
        await video_with_vision()
        await batch_processing()
        await system_monitoring()
        await plugin_management()
        await configuration_management()
        await error_handling()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        raise


match __name__:
    case "__main__":
    asyncio.run(main()) 