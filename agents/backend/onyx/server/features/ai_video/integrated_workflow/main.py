from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import argparse
import logging
import json
from typing import Optional
from ..integrated_workflow.core.workflow import IntegratedVideoWorkflow
from ..integrated_workflow.core.models import WorkflowConfiguration
from ..integrated_workflow.utils import (
from ..plugins import ManagerConfig, ValidationLevel
from typing import Any, List, Dict, Optional
"""
Integrated Workflow - Main Entry Point

Main entry point for the integrated AI video workflow system.
"""


# Import integrated workflow
    create_integrated_workflow,
    quick_video_generation,
    batch_video_generation,
    load_config_from_file,
    create_workflow_report,
    format_duration
)

# Import plugin system

logger = logging.getLogger(__name__)


async def main():
    """Example usage of the integrated workflow."""
    parser = argparse.ArgumentParser(description="Integrated AI Video Workflow")
    parser.add_argument("url", nargs="?", help="Source URL for video generation")
    parser.add_argument("--avatar", help="Avatar to use for video generation")
    parser.add_argument("--workflow-id", help="Custom workflow ID")
    parser.add_argument("--config", help="Plugin configuration file")
    parser.add_argument("--stats", action="store_true", help="Show plugin statistics")
    parser.add_argument("--health", action="store_true", help="Show health report")
    parser.add_argument("--batch", nargs="+", help="Process multiple URLs")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent workflows for batch processing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Load configuration if provided
    plugin_config = None
    if args.config:
        config_data = load_config_from_file(args.config)
        if config_data:
            plugin_config = ManagerConfig(**config_data)
    
    # Create workflow
    workflow = await create_integrated_workflow(plugin_config)
    
    try:
        # Show statistics if requested
        if args.stats:
            stats = workflow.get_plugin_stats()
            print("Plugin Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return
        
        # Show health report if requested
        if args.health:
            health = workflow.get_health_report()
            print("Health Report:")
            print(f"  Status: {health.status}")
            print(f"  Plugin Manager: {'✅' if health.plugin_manager_healthy else '❌'}")
            print(f"  Workflow: {'✅' if health.workflow_initialized else '❌'}")
            print(f"  State Repository: {'✅' if health.state_repository_available else '❌'}")
            print(f"  Plugins Loaded: {health.plugins_loaded}")
            print(f"  Plugins Healthy: {health.plugins_healthy}")
            print(f"  Plugins Failed: {health.plugins_failed}")
            
            if health.recommendations:
                print("  Recommendations:")
                for rec in health.recommendations:
                    print(f"    - {rec}")
            return
        
        # Batch processing
        if args.batch:
            print(f"🎬 Starting batch video generation for {len(args.batch)} URLs...")
            results = await batch_video_generation(
                args.batch,
                avatar=args.avatar,
                plugin_config=plugin_config,
                max_concurrent=args.max_concurrent
            )
            
            # Create report
            report = create_workflow_report(results)
            print("\n📊 Batch Processing Report:")
            print(f"  Total Workflows: {report['summary']['total_workflows']}")
            print(f"  Successful: {report['summary']['successful_workflows']}")
            print(f"  Failed: {report['summary']['failed_workflows']}")
            print(f"  Success Rate: {report['summary']['success_rate']:.1f}%")
            print(f"  Total Time: {report['timing']['total_time']}")
            print(f"  Average Time: {report['timing']['average_time']}")
            
            # Show individual results
            print("\n📋 Individual Results:")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result.source_url}")
                print(f"     Status: {result.status.value}")
                print(f"     Time: {format_duration(result.total_time)}")
                if result.video_url:
                    print(f"     Video: {result.video_url}")
                if result.error:
                    print(f"     Error: {result.error}")
                print()
            return
        
        # Single URL processing
        if not args.url:
            print("❌ Please provide a URL or use --batch for multiple URLs")
            return
        
        # Execute workflow
        print(f"🎬 Starting video generation for: {args.url}")
        result = await workflow.execute_workflow(
            args.url,
            workflow_id=args.workflow_id,
            avatar=args.avatar,
            plugin_config=plugin_config
        )
        
        print(f"✅ Video generation completed!")
        print(f"  Workflow ID: {result.workflow_id}")
        print(f"  Total time: {format_duration(result.total_time)}")
        print(f"  Extraction time: {format_duration(result.extraction_time)}")
        print(f"  Suggestions time: {format_duration(result.suggestions_time)}")
        print(f"  Generation time: {format_duration(result.generation_time)}")
        print(f"  Video URL: {result.video_url}")
        
        if result.ai_video:
            print(f"  Title: {result.ai_video.title}")
            print(f"  Duration: {result.ai_video.duration}s")
            print(f"  Resolution: {result.ai_video.resolution}")
        
        # Show plugin information
        if result.loaded_plugins:
            print(f"  Plugins Loaded: {', '.join(result.loaded_plugins)}")
        if result.active_plugins:
            print(f"  Active Plugins: {', '.join(result.active_plugins)}")
        if result.plugin_errors:
            print(f"  Plugin Errors: {len(result.plugin_errors)}")
        
    finally:
        await workflow.shutdown()


match __name__:
    case "__main__":
    asyncio.run(main()) 