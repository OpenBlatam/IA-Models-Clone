from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from onyx.utils.logger import setup_logger
from onyx.utils.timing import time_function
from onyx.utils.telemetry import TelemetryLogger
from onyx.utils.gpu_utils import get_gpu_info, is_gpu_available
from onyx.core.functions import format_response, handle_error
from ..onyx_main import get_system, OnyxAIVideoSystem
from ..onyx_config import get_config, validate_config, get_config_summary
from ..onyx_video_workflow import onyx_video_generator
from ..onyx_plugin_manager import onyx_plugin_manager
from ..core.onyx_integration import onyx_integration
from ..models import VideoRequest, VideoResponse
            from ..onyx_plugin_manager import OnyxPluginContext
            from onyx.utils.threadpool_concurrency import run_functions_in_parallel, FunctionCall
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Onyx AI Video System - Usage Examples

Comprehensive examples demonstrating how to use the Onyx-adapted AI Video system
with various features, configurations, and integrations.
"""


# Onyx imports

# Local imports

logger = setup_logger(__name__)


class OnyxAIVideoExamples:
    """
    Comprehensive examples for Onyx AI Video System.
    
    Demonstrates various usage patterns, configurations, and integrations
    with the Onyx ecosystem.
    """
    
    def __init__(self) -> Any:
        self.logger = setup_logger("onyx_ai_video_examples")
        self.telemetry = TelemetryLogger()
        self.system: Optional[OnyxAIVideoSystem] = None
        self.examples_dir = Path("examples")
        self.examples_dir.mkdir(exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the examples system."""
        try:
            self.logger.info("Initializing Onyx AI Video Examples")
            
            # Initialize system
            self.system = await get_system()
            
            # Show system information
            await self._show_system_info()
            
            self.logger.info("Examples system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Examples initialization failed: {e}")
            raise
    
    async def _show_system_info(self) -> None:
        """Show system information."""
        print("\n" + "="*60)
        print("🚀 ONYX AI VIDEO SYSTEM - EXAMPLES")
        print("="*60)
        
        # Get system status
        status = await self.system.get_system_status()
        print(f"System Status: {status['status']}")
        print(f"GPU Available: {status['gpu_available']}")
        print(f"Plugins Loaded: {status['components']['plugins']['total_plugins']}")
        
        # Get configuration
        config = get_config()
        print(f"Environment: {config.environment}")
        print(f"Max Workers: {config.max_workers}")
        print(f"Default Quality: {config.default_quality}")
        
        if status['gpu_available']:
            gpu_info = status['gpu_info']
            print(f"GPU: {gpu_info.get('name', 'Unknown')}")
            print(f"GPU Memory: {gpu_info.get('memory_total', 0)} MB")
        
        print("="*60 + "\n")
    
    async def run_all_examples(self) -> None:
        """Run all examples."""
        try:
            self.logger.info("Running all Onyx AI Video examples...")
            
            # Basic examples
            await self.example_basic_video_generation()
            await self.example_video_with_plugins()
            await self.example_video_with_vision()
            
            # Advanced examples
            await self.example_batch_processing()
            await self.example_custom_configuration()
            await self.example_plugin_management()
            
            # Integration examples
            await self.example_onyx_integration()
            await self.example_performance_monitoring()
            await self.example_error_handling()
            
            # System examples
            await self.example_system_management()
            
            self.logger.info("All examples completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Examples execution failed: {e}")
            raise
    
    async def example_basic_video_generation(self) -> None:
        """Example 1: Basic video generation."""
        print("\n" + "="*40)
        print("📹 EXAMPLE 1: BASIC VIDEO GENERATION")
        print("="*40)
        
        try:
            # Create video request
            request = VideoRequest(
                input_text="Crea un video educativo sobre inteligencia artificial y sus aplicaciones en la vida cotidiana",
                user_id="example_user_001",
                request_id=f"basic_example_{int(time.time())}",
                quality="medium",
                duration=60,
                output_format="mp4"
            )
            
            print(f"Generating video: {request.request_id}")
            print(f"Input text: {request.input_text[:100]}...")
            
            # Generate video
            with time_function("basic_video_generation"):
                response = await self.system.generate_video(request)
            
            # Show results
            print(f"✅ Video generated successfully!")
            print(f"Request ID: {response.request_id}")
            print(f"Status: {response.status}")
            print(f"Output URL: {response.output_url}")
            
            if response.metadata:
                print(f"Metadata keys: {list(response.metadata.keys())}")
            
            # Save example result
            self._save_example_result("basic_video_generation", {
                "request": request.__dict__,
                "response": response.__dict__,
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Basic video generation failed: {e}")
            self._save_example_result("basic_video_generation", {
                "error": str(e),
                "success": False
            })
    
    async def example_video_with_plugins(self) -> None:
        """Example 2: Video generation with plugins."""
        print("\n" + "="*40)
        print("🔌 EXAMPLE 2: VIDEO WITH PLUGINS")
        print("="*40)
        
        try:
            # Create video request with plugins
            request = VideoRequest(
                input_text="Crea un video promocional sobre una nueva tecnología revolucionaria",
                user_id="example_user_002",
                request_id=f"plugins_example_{int(time.time())}",
                quality="high",
                duration=90,
                output_format="mp4",
                plugins=["content_analyzer", "visual_enhancer"]
            )
            
            print(f"Generating video with plugins: {request.request_id}")
            print(f"Plugins: {request.plugins}")
            
            # Generate video
            with time_function("plugins_video_generation"):
                response = await self.system.generate_video(request)
            
            # Show results
            print(f"✅ Video with plugins generated successfully!")
            print(f"Request ID: {response.request_id}")
            print(f"Status: {response.status}")
            
            # Show plugin results
            if response.metadata and "plugin_results" in response.metadata:
                plugin_results = response.metadata["plugin_results"]
                print(f"Plugin Results:")
                for plugin_name, result in plugin_results.items():
                    print(f"  {plugin_name}: {type(result).__name__}")
                    if isinstance(result, dict):
                        print(f"    Keys: {list(result.keys())}")
            
            # Save example result
            self._save_example_result("video_with_plugins", {
                "request": request.__dict__,
                "response": response.__dict__,
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Video with plugins failed: {e}")
            self._save_example_result("video_with_plugins", {
                "error": str(e),
                "success": False
            })
    
    async def example_video_with_vision(self) -> None:
        """Example 3: Video generation with vision capabilities."""
        print("\n" + "="*40)
        print("👁️ EXAMPLE 3: VIDEO WITH VISION")
        print("="*40)
        
        try:
            # Create sample image data (simulated)
            image_data = b"fake_image_data_for_example"
            
            # Create video request
            request = VideoRequest(
                input_text="Analiza esta imagen y crea un video explicativo sobre su contenido",
                user_id="example_user_003",
                request_id=f"vision_example_{int(time.time())}",
                quality="high",
                duration=75,
                output_format="mp4"
            )
            
            print(f"Generating video with vision: {request.request_id}")
            print(f"Image size: {len(image_data)} bytes")
            
            # Generate video with vision
            with time_function("vision_video_generation"):
                response = await self.system.generate_video_with_vision(request, image_data)
            
            # Show results
            print(f"✅ Vision video generated successfully!")
            print(f"Request ID: {response.request_id}")
            print(f"Status: {response.status}")
            print(f"Output URL: {response.output_url}")
            
            if response.metadata:
                print(f"Vision analysis: {response.metadata.get('vision_analysis', 'N/A')}")
            
            # Save example result
            self._save_example_result("video_with_vision", {
                "request": request.__dict__,
                "response": response.__dict__,
                "image_size": len(image_data),
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Vision video generation failed: {e}")
            self._save_example_result("video_with_vision", {
                "error": str(e),
                "success": False
            })
    
    async def example_batch_processing(self) -> None:
        """Example 4: Batch video processing."""
        print("\n" + "="*40)
        print("📦 EXAMPLE 4: BATCH PROCESSING")
        print("="*40)
        
        try:
            # Create multiple requests
            requests = [
                VideoRequest(
                    input_text="Video sobre tecnología",
                    user_id="batch_user",
                    request_id=f"batch_001_{int(time.time())}",
                    quality="medium",
                    duration=30
                ),
                VideoRequest(
                    input_text="Video sobre ciencia",
                    user_id="batch_user",
                    request_id=f"batch_002_{int(time.time())}",
                    quality="medium",
                    duration=30
                ),
                VideoRequest(
                    input_text="Video sobre arte",
                    user_id="batch_user",
                    request_id=f"batch_003_{int(time.time())}",
                    quality="medium",
                    duration=30
                )
            ]
            
            print(f"Processing {len(requests)} videos in batch...")
            
            # Process batch
            results = []
            with time_function("batch_processing"):
                for request in requests:
                    try:
                        response = await self.system.generate_video(request)
                        results.append({
                            "request_id": request.request_id,
                            "status": response.status,
                            "success": True
                        })
                        print(f"  ✅ {request.request_id}: {response.status}")
                    except Exception as e:
                        results.append({
                            "request_id": request.request_id,
                            "error": str(e),
                            "success": False
                        })
                        print(f"  ❌ {request.request_id}: {e}")
            
            # Show batch results
            successful = sum(1 for r in results if r["success"])
            print(f"\nBatch completed: {successful}/{len(requests)} successful")
            
            # Save example result
            self._save_example_result("batch_processing", {
                "total_requests": len(requests),
                "successful": successful,
                "results": results,
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            self._save_example_result("batch_processing", {
                "error": str(e),
                "success": False
            })
    
    async def example_custom_configuration(self) -> None:
        """Example 5: Custom configuration usage."""
        print("\n" + "="*40)
        print("⚙️ EXAMPLE 5: CUSTOM CONFIGURATION")
        print("="*40)
        
        try:
            # Get current configuration
            config = get_config()
            config_summary = get_config_summary(config)
            
            print("Current Configuration:")
            print(f"  Environment: {config.environment}")
            print(f"  Max Workers: {config.max_workers}")
            print(f"  Default Quality: {config.default_quality}")
            print(f"  GPU Available: {is_gpu_available()}")
            
            # Validate configuration
            issues = validate_config(config)
            if issues:
                print(f"Configuration Issues: {issues}")
            else:
                print("✅ Configuration is valid")
            
            # Show Onyx integration status
            print("\nOnyx Integration Status:")
            onyx_integration = config_summary['onyx_integration']
            for key, value in onyx_integration.items():
                status = "✅" if value else "❌"
                print(f"  {key}: {status}")
            
            # Save example result
            self._save_example_result("custom_configuration", {
                "config_summary": config_summary,
                "validation_issues": issues,
                "gpu_available": is_gpu_available(),
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Custom configuration example failed: {e}")
            self._save_example_result("custom_configuration", {
                "error": str(e),
                "success": False
            })
    
    async def example_plugin_management(self) -> None:
        """Example 6: Plugin management."""
        print("\n" + "="*40)
        print("🔌 EXAMPLE 6: PLUGIN MANAGEMENT")
        print("="*40)
        
        try:
            # Get plugin information
            plugin_infos = await onyx_plugin_manager.get_all_plugins_info()
            plugin_status = await onyx_plugin_manager.get_manager_status()
            
            print(f"Total Plugins: {plugin_status['total_plugins']}")
            print(f"Enabled Plugins: {plugin_status['enabled_plugins']}")
            print(f"Initialized Plugins: {plugin_status['initialized_plugins']}")
            
            print("\nPlugin Details:")
            for info in plugin_infos:
                enabled = "✅" if info.enabled else "❌"
                gpu_required = "Yes" if info.gpu_required else "No"
                print(f"  {enabled} {info.name} ({info.version}) - {info.category}")
                print(f"    Description: {info.description}")
                print(f"    GPU Required: {gpu_required}")
                print(f"    Timeout: {info.timeout}s")
            
            # Test plugin execution
            print("\nTesting plugin execution...")
            
            test_request = VideoRequest(
                input_text="Test plugin execution",
                user_id="plugin_test_user",
                request_id="plugin_test_001"
            )
            
            context = OnyxPluginContext(
                request=test_request,
                gpu_available=is_gpu_available()
            )
            
            # Execute plugins
            plugin_results = await onyx_plugin_manager.execute_plugins(
                context, 
                ["content_analyzer"]
            )
            
            print(f"Plugin execution results: {len(plugin_results)} plugins")
            for plugin_name, result in plugin_results.items():
                print(f"  {plugin_name}: {type(result).__name__}")
            
            # Save example result
            self._save_example_result("plugin_management", {
                "plugin_infos": [info.__dict__ for info in plugin_infos],
                "plugin_status": plugin_status,
                "plugin_results": plugin_results,
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Plugin management example failed: {e}")
            self._save_example_result("plugin_management", {
                "error": str(e),
                "success": False
            })
    
    async def example_onyx_integration(self) -> None:
        """Example 7: Onyx integration features."""
        print("\n" + "="*40)
        print("🔗 EXAMPLE 7: ONYX INTEGRATION")
        print("="*40)
        
        try:
            # Test Onyx LLM integration
            print("Testing Onyx LLM integration...")
            llm = await onyx_integration.llm_manager.get_default_llm()
            print(f"✅ LLM initialized: {type(llm).__name__}")
            
            # Test Onyx threading
            print("Testing Onyx threading...")
            
            def test_function(x) -> Any:
                return x * 2
            
            function_calls = [FunctionCall(test_function, i) for i in range(5)]
            results = run_functions_in_parallel(function_calls)
            print(f"✅ Threading test results: {results}")
            
            # Test Onyx GPU utilities
            print("Testing Onyx GPU utilities...")
            gpu_available = is_gpu_available()
            print(f"GPU Available: {gpu_available}")
            
            if gpu_available:
                gpu_info = get_gpu_info()
                print(f"GPU Info: {gpu_info}")
            
            # Test Onyx security
            print("Testing Onyx security...")
            is_valid = await onyx_integration.security_manager.validate_access(
                "test_user", "test_resource"
            )
            print(f"Security validation: {is_valid}")
            
            # Test Onyx performance
            print("Testing Onyx performance utilities...")
            with onyx_integration.performance_manager.time_operation("test_operation"):
                time.sleep(0.1)  # Simulate work
            
            # Save example result
            self._save_example_result("onyx_integration", {
                "llm_available": llm is not None,
                "threading_working": len(results) == 5,
                "gpu_available": gpu_available,
                "gpu_info": gpu_info if gpu_available else None,
                "security_working": is_valid,
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Onyx integration example failed: {e}")
            self._save_example_result("onyx_integration", {
                "error": str(e),
                "success": False
            })
    
    async def example_performance_monitoring(self) -> None:
        """Example 8: Performance monitoring."""
        print("\n" + "="*40)
        print("📊 EXAMPLE 8: PERFORMANCE MONITORING")
        print("="*40)
        
        try:
            # Get system metrics
            metrics = await self.system.get_metrics()
            
            print("System Metrics:")
            ai_video_metrics = metrics.get('ai_video', {})
            if ai_video_metrics:
                print(f"  Request Count: {ai_video_metrics['request_count']}")
                print(f"  Error Count: {ai_video_metrics['error_count']}")
                print(f"  Error Rate: {ai_video_metrics['error_rate']:.2%}")
                print(f"  Uptime: {ai_video_metrics['uptime']:.2f} seconds")
            
            # Get performance metrics
            performance_metrics = ai_video_metrics.get('performance_metrics', {})
            if performance_metrics:
                print("\nPerformance Metrics:")
                for operation, metrics_data in performance_metrics.items():
                    print(f"  {operation}:")
                    print(f"    Count: {metrics_data['count']}")
                    success_rate = metrics_data['success_count'] / max(metrics_data['count'], 1)
                    print(f"    Success Rate: {success_rate:.2%}")
                    print(f"    Avg Duration: {metrics_data['avg_duration']:.2f}s")
            
            # Test performance monitoring
            print("\nTesting performance monitoring...")
            with time_function("performance_test"):
                # Simulate some work
                await asyncio.sleep(0.5)
            
            # Get cache information
            cache_size = len(onyx_integration.performance_manager._cache)
            print(f"Cache Size: {cache_size}")
            
            # Save example result
            self._save_example_result("performance_monitoring", {
                "metrics": metrics,
                "cache_size": cache_size,
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Performance monitoring example failed: {e}")
            self._save_example_result("performance_monitoring", {
                "error": str(e),
                "success": False
            })
    
    async def example_error_handling(self) -> None:
        """Example 9: Error handling."""
        print("\n" + "="*40)
        print("⚠️ EXAMPLE 9: ERROR HANDLING")
        print("="*40)
        
        try:
            # Test invalid request
            print("Testing invalid request handling...")
            try:
                invalid_request = VideoRequest(
                    input_text="",  # Empty text
                    user_id="error_test_user",
                    request_id="error_test_001"
                )
                
                response = await self.system.generate_video(invalid_request)
                print("❌ Expected error but got success")
                
            except Exception as e:
                print(f"✅ Correctly caught error: {type(e).__name__}: {e}")
            
            # Test timeout handling
            print("\nTesting timeout handling...")
            try:
                # Create a request that might timeout
                timeout_request = VideoRequest(
                    input_text="A very long text that might cause timeout issues " * 100,
                    user_id="timeout_test_user",
                    request_id="timeout_test_001",
                    quality="ultra",
                    duration=300
                )
                
                response = await self.system.generate_video(timeout_request)
                print(f"✅ Timeout test completed: {response.status}")
                
            except Exception as e:
                print(f"✅ Timeout handled: {type(e).__name__}: {e}")
            
            # Test plugin error handling
            print("\nTesting plugin error handling...")
            try:
                plugin_request = VideoRequest(
                    input_text="Test plugin error handling",
                    user_id="plugin_error_user",
                    request_id="plugin_error_001",
                    plugins=["non_existent_plugin"]
                )
                
                response = await self.system.generate_video(plugin_request)
                print(f"✅ Plugin error handled: {response.status}")
                
            except Exception as e:
                print(f"✅ Plugin error caught: {type(e).__name__}: {e}")
            
            # Save example result
            self._save_example_result("error_handling", {
                "error_handling_working": True,
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Error handling example failed: {e}")
            self._save_example_result("error_handling", {
                "error": str(e),
                "success": False
            })
    
    async def example_system_management(self) -> None:
        """Example 10: System management."""
        print("\n" + "="*40)
        print("🔧 EXAMPLE 10: SYSTEM MANAGEMENT")
        print("="*40)
        
        try:
            # Get system status
            status = await self.system.get_system_status()
            
            print("System Status:")
            print(f"  Status: {status['status']}")
            print(f"  Uptime: {status['uptime']:.2f} seconds")
            print(f"  Request Count: {status['request_count']}")
            print(f"  Error Count: {status['error_count']}")
            print(f"  Error Rate: {status['error_rate']:.2%}")
            
            # Component status
            components = status['components']
            print(f"\nComponent Status:")
            print(f"  Integration: {components['integration']['status']}")
            print(f"  Workflow: {components['workflow']['status']}")
            print(f"  Plugins: {components['plugins']['total_plugins']} loaded")
            
            # Performance status
            performance = status['performance']
            if performance:
                print(f"\nPerformance Status:")
                for operation, metrics in performance.items():
                    print(f"  {operation}: {metrics['total_requests']} requests, "
                          f"{metrics['success_rate']:.2%} success rate")
            
            # Health check
            print("\nRunning health check...")
            health_issues = []
            
            if status['status'] != 'operational':
                health_issues.append(f"System status: {status['status']}")
            
            if status['error_rate'] > 0.1:
                health_issues.append(f"High error rate: {status['error_rate']:.2%}")
            
            if health_issues:
                print("❌ Health check failed:")
                for issue in health_issues:
                    print(f"  - {issue}")
            else:
                print("✅ Health check passed")
            
            # Save example result
            self._save_example_result("system_management", {
                "status": status,
                "health_issues": health_issues,
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"System management example failed: {e}")
            self._save_example_result("system_management", {
                "error": str(e),
                "success": False
            })
    
    def _save_example_result(self, example_name: str, result: Dict[str, Any]) -> None:
        """Save example result to file."""
        try:
            result_file = self.examples_dir / f"{example_name}_result.json"
            with open(result_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(result, f, indent=2, default=str)
            
            self.logger.info(f"Example result saved: {result_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save example result: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup examples system."""
        try:
            if self.system:
                await self.system.shutdown()
            
            self.logger.info("Examples system cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Examples cleanup failed: {e}")


async def main() -> None:
    """Main entry point for examples."""
    try:
        # Create examples
        examples = OnyxAIVideoExamples()
        
        # Initialize
        await examples.initialize()
        
        # Run examples
        await examples.run_all_examples()
        
        # Cleanup
        await examples.cleanup()
        
        print("\n" + "="*60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check the 'examples/' directory for detailed results.")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Examples execution failed: {e}")
        raise


match __name__:
    case "__main__":
    asyncio.run(main()) 