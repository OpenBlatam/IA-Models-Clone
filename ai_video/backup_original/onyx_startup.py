from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import argparse
import json
import sys
import signal
import time
from pathlib import Path
from typing import Dict, Any, Optional
from onyx.utils.logger import setup_logger
from onyx.utils.timing import time_function
from onyx.utils.telemetry import TelemetryLogger
from onyx.utils.gpu_utils import get_gpu_info, is_gpu_available
from onyx.core.functions import format_response, handle_error
from .onyx_config import get_config, save_config, validate_config, get_config_summary, create_default_config
from .onyx_main import OnyxAIVideoSystem, get_system, shutdown_system
from .onyx_video_workflow import onyx_video_generator
from .onyx_plugin_manager import onyx_plugin_manager
from .core.onyx_integration import onyx_integration
            from .models import VideoRequest
                from .models import VideoRequest
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Onyx AI Video System - Startup Script

Command-line interface and startup script for the Onyx-adapted AI Video system.
Provides initialization, configuration management, and system control.
"""


# Onyx imports

# Local imports

logger = setup_logger(__name__)


class OnyxAIVideoCLI:
    """
    Command-line interface for Onyx AI Video System.
    
    Provides comprehensive CLI for system management, configuration,
    and video generation operations.
    """
    
    def __init__(self) -> Any:
        self.logger = setup_logger("onyx_ai_video_cli")
        self.telemetry = TelemetryLogger()
        self.system: Optional[OnyxAIVideoSystem] = None
    
    async def initialize(self) -> None:
        """Initialize the CLI and system."""
        try:
            self.logger.info("Initializing Onyx AI Video CLI")
            
            # Initialize system
            self.system = await get_system()
            
            self.logger.info("Onyx AI Video CLI initialized successfully")
            
        except Exception as e:
            self.logger.error(f"CLI initialization failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the CLI and system."""
        try:
            if self.system:
                await self.system.shutdown()
            
            self.logger.info("Onyx AI Video CLI shutdown completed")
            
        except Exception as e:
            self.logger.error(f"CLI shutdown failed: {e}")
    
    async def run_command(self, args: argparse.Namespace) -> int:
        """Run CLI command."""
        try:
            command = args.command
            
            if command == "start":
                return await self._start_system(args)
            elif command == "status":
                return await self._show_status(args)
            elif command == "metrics":
                return await self._show_metrics(args)
            elif command == "config":
                return await self._handle_config(args)
            elif command == "generate":
                return await self._generate_video(args)
            elif command == "plugins":
                return await self._handle_plugins(args)
            elif command == "test":
                return await self._run_tests(args)
            elif command == "health":
                return await self._health_check(args)
            else:
                self.logger.error(f"Unknown command: {command}")
                return 1
                
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return 1
    
    async def _start_system(self, args: argparse.Namespace) -> int:
        """Start the AI Video system."""
        try:
            self.logger.info("Starting Onyx AI Video System...")
            
            # Show startup information
            await self._show_startup_info()
            
            # Keep system running
            self.logger.info("System is running. Press Ctrl+C to stop.")
            
            # Setup signal handlers
            def signal_handler(signum, frame) -> Any:
                self.logger.info(f"Received signal {signum}, shutting down...")
                asyncio.create_task(self.shutdown())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Run forever
            while True:
                await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            return 0
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            return 1
    
    async def _show_startup_info(self) -> None:
        """Show startup information."""
        try:
            # Get system status
            status = await self.system.get_system_status()
            
            # Get configuration summary
            config = get_config()
            config_summary = get_config_summary(config)
            
            print("\n" + "="*60)
            print("🚀 ONYX AI VIDEO SYSTEM")
            print("="*60)
            print(f"Status: {status['status']}")
            print(f"Environment: {config.environment}")
            print(f"GPU Available: {status['gpu_available']}")
            print(f"Plugins Loaded: {status['components']['plugins']['total_plugins']}")
            print(f"Uptime: {status['uptime']:.2f} seconds")
            print("="*60)
            
            if status['gpu_available']:
                gpu_info = status['gpu_info']
                print(f"GPU: {gpu_info.get('name', 'Unknown')}")
                print(f"Memory: {gpu_info.get('memory_total', 0)} MB")
            
            print(f"Output Directory: {config.output_directory}")
            print(f"Temp Directory: {config.temp_directory}")
            print(f"Cache Directory: {config.cache_directory}")
            print("="*60 + "\n")
            
        except Exception as e:
            self.logger.error(f"Failed to show startup info: {e}")
    
    async def _show_status(self, args: argparse.Namespace) -> int:
        """Show system status."""
        try:
            status = await self.system.get_system_status()
            
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                await self._print_status_table(status)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return 1
    
    async def _print_status_table(self, status: Dict[str, Any]) -> None:
        """Print status in table format."""
        print("\n" + "="*60)
        print("📊 ONYX AI VIDEO SYSTEM STATUS")
        print("="*60)
        
        # System info
        print(f"Status: {status['status']}")
        print(f"Uptime: {status['uptime']:.2f} seconds")
        print(f"Request Count: {status['request_count']}")
        print(f"Error Count: {status['error_count']}")
        print(f"Error Rate: {status['error_rate']:.2%}")
        
        # GPU info
        print(f"\nGPU Available: {status['gpu_available']}")
        if status['gpu_available'] and status['gpu_info']:
            gpu_info = status['gpu_info']
            print(f"GPU Name: {gpu_info.get('name', 'Unknown')}")
            print(f"GPU Memory: {gpu_info.get('memory_total', 0)} MB")
        
        # Component info
        components = status['components']
        print(f"\nComponents:")
        print(f"  Integration: {components['integration']['status']}")
        print(f"  Workflow: {components['workflow']['status']}")
        print(f"  Plugins: {components['plugins']['total_plugins']} loaded")
        
        # Performance info
        performance = status['performance']
        if performance:
            print(f"\nPerformance:")
            for operation, metrics in performance.items():
                print(f"  {operation}:")
                print(f"    Requests: {metrics['total_requests']}")
                print(f"    Success Rate: {metrics['success_rate']:.2%}")
                print(f"    Avg Duration: {metrics['avg_duration']:.2f}s")
        
        print("="*60 + "\n")
    
    async def _show_metrics(self, args: argparse.Namespace) -> int:
        """Show system metrics."""
        try:
            metrics = await self.system.get_metrics()
            
            if args.json:
                print(json.dumps(metrics, indent=2))
            else:
                await self._print_metrics_table(metrics)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
            return 1
    
    async def _print_metrics_table(self, metrics: Dict[str, Any]) -> None:
        """Print metrics in table format."""
        print("\n" + "="*60)
        print("📈 ONYX AI VIDEO SYSTEM METRICS")
        print("="*60)
        
        # AI Video metrics
        ai_video_metrics = metrics.get('ai_video', {})
        if ai_video_metrics:
            print("AI Video Metrics:")
            print(f"  Request Count: {ai_video_metrics['request_count']}")
            print(f"  Error Count: {ai_video_metrics['error_count']}")
            print(f"  Error Rate: {ai_video_metrics['error_rate']:.2%}")
            print(f"  Uptime: {ai_video_metrics['uptime']:.2f} seconds")
        
        # Performance metrics
        performance_metrics = ai_video_metrics.get('performance_metrics', {})
        if performance_metrics:
            print("\nPerformance Metrics:")
            for operation, metrics_data in performance_metrics.items():
                print(f"  {operation}:")
                print(f"    Count: {metrics_data['count']}")
                print(f"    Success Rate: {metrics_data['success_count'] / max(metrics_data['count'], 1):.2%}")
                print(f"    Avg Duration: {metrics_data['avg_duration']:.2f}s")
        
        print("="*60 + "\n")
    
    async def _handle_config(self, args: argparse.Namespace) -> int:
        """Handle configuration commands."""
        try:
            config_action = args.config_action
            
            if config_action == "show":
                config = get_config()
                config_summary = get_config_summary(config)
                
                if args.json:
                    print(json.dumps(config_summary, indent=2))
                else:
                    await self._print_config_table(config_summary)
                
            elif config_action == "validate":
                config = get_config()
                issues = validate_config(config)
                
                if issues:
                    print("Configuration issues found:")
                    for issue in issues:
                        print(f"  ❌ {issue}")
                    return 1
                else:
                    print("✅ Configuration is valid")
                
            elif config_action == "create":
                config_path = args.config_path or "config/onyx_ai_video.json"
                create_default_config(config_path)
                print(f"✅ Default configuration created: {config_path}")
                
            elif config_action == "save":
                config = get_config()
                config_path = args.config_path or "config/onyx_ai_video.json"
                save_config(config, config_path)
                print(f"✅ Configuration saved: {config_path}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Configuration handling failed: {e}")
            return 1
    
    async def _print_config_table(self, config_summary: Dict[str, Any]) -> None:
        """Print configuration in table format."""
        print("\n" + "="*60)
        print("⚙️ ONYX AI VIDEO CONFIGURATION")
        print("="*60)
        
        # Environment
        print(f"Environment: {config_summary['environment']}")
        print(f"Debug Mode: {config_summary['debug_mode']}")
        
        # Onyx Integration
        print(f"\nOnyx Integration:")
        onyx_integration = config_summary['onyx_integration']
        for key, value in onyx_integration.items():
            status = "✅" if value else "❌"
            print(f"  {key}: {status}")
        
        # Performance
        print(f"\nPerformance:")
        performance = config_summary['performance']
        for key, value in performance.items():
            print(f"  {key}: {value}")
        
        # Video Generation
        print(f"\nVideo Generation:")
        video_gen = config_summary['video_generation']
        for key, value in video_gen.items():
            print(f"  {key}: {value}")
        
        # LLM
        print(f"\nLLM Settings:")
        llm = config_summary['llm']
        for key, value in llm.items():
            print(f"  {key}: {value}")
        
        # Plugins
        print(f"\nPlugins:")
        plugins = config_summary['plugins']
        for key, value in plugins.items():
            print(f"  {key}: {value}")
        
        # Security
        print(f"\nSecurity:")
        security = config_summary['security']
        for key, value in security.items():
            print(f"  {key}: {value}")
        
        # Storage
        print(f"\nStorage:")
        storage = config_summary['storage']
        for key, value in storage.items():
            print(f"  {key}: {value}")
        
        print("="*60 + "\n")
    
    async def _generate_video(self, args: argparse.Namespace) -> int:
        """Generate video from command line."""
        try:
            
            # Create video request
            request = VideoRequest(
                input_text=args.input_text,
                user_id=args.user_id or "cli_user",
                request_id=args.request_id or f"cli_{int(time.time())}",
                quality=args.quality or "medium",
                duration=args.duration or 60,
                output_format=args.output_format or "mp4",
                plugins=args.plugins.split(",") if args.plugins else None
            )
            
            self.logger.info(f"Generating video: {request.request_id}")
            
            # Generate video
            with time_function("cli_video_generation"):
                response = await self.system.generate_video(request)
            
            # Show results
            print(f"\n✅ Video generated successfully!")
            print(f"Request ID: {response.request_id}")
            print(f"Status: {response.status}")
            print(f"Output URL: {response.output_url}")
            
            if response.metadata:
                print(f"Metadata: {json.dumps(response.metadata, indent=2)}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}")
            return 1
    
    async def _handle_plugins(self, args: argparse.Namespace) -> int:
        """Handle plugin commands."""
        try:
            plugin_action = args.plugin_action
            
            if plugin_action == "list":
                plugin_infos = await onyx_plugin_manager.get_all_plugins_info()
                
                if args.json:
                    print(json.dumps([info.__dict__ for info in plugin_infos], indent=2))
                else:
                    await self._print_plugins_table(plugin_infos)
                
            elif plugin_action == "status":
                status = await onyx_plugin_manager.get_manager_status()
                
                if args.json:
                    print(json.dumps(status, indent=2))
                else:
                    await self._print_plugin_status_table(status)
                
            elif plugin_action == "enable":
                plugin_name = args.plugin_name
                success = await onyx_plugin_manager.enable_plugin(plugin_name)
                
                if success:
                    print(f"✅ Plugin enabled: {plugin_name}")
                else:
                    print(f"❌ Failed to enable plugin: {plugin_name}")
                    return 1
                
            elif plugin_action == "disable":
                plugin_name = args.plugin_name
                success = await onyx_plugin_manager.disable_plugin(plugin_name)
                
                if success:
                    print(f"✅ Plugin disabled: {plugin_name}")
                else:
                    print(f"❌ Failed to disable plugin: {plugin_name}")
                    return 1
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Plugin handling failed: {e}")
            return 1
    
    async def _print_plugins_table(self, plugin_infos: list) -> None:
        """Print plugins in table format."""
        print("\n" + "="*80)
        print("🔌 ONYX AI VIDEO PLUGINS")
        print("="*80)
        print(f"{'Name':<20} {'Version':<10} {'Category':<15} {'Enabled':<8} {'GPU':<5}")
        print("-"*80)
        
        for info in plugin_infos:
            enabled = "✅" if info.enabled else "❌"
            gpu_required = "Yes" if info.gpu_required else "No"
            print(f"{info.name:<20} {info.version:<10} {info.category:<15} {enabled:<8} {gpu_required:<5}")
        
        print("="*80 + "\n")
    
    async def _print_plugin_status_table(self, status: Dict[str, Any]) -> None:
        """Print plugin status in table format."""
        print("\n" + "="*60)
        print("🔌 ONYX AI VIDEO PLUGIN STATUS")
        print("="*60)
        
        print(f"Total Plugins: {status['total_plugins']}")
        print(f"Enabled Plugins: {status['enabled_plugins']}")
        print(f"Initialized Plugins: {status['initialized_plugins']}")
        print(f"GPU Available: {status['gpu_available']}")
        
        if status['plugins']:
            print(f"\nPlugin Details:")
            for plugin in status['plugins']:
                status_icon = "✅" if plugin['enabled'] else "❌"
                print(f"  {status_icon} {plugin['name']} ({plugin['version']}) - {plugin['category']}")
        
        print("="*60 + "\n")
    
    async def _run_tests(self, args: argparse.Namespace) -> int:
        """Run system tests."""
        try:
            self.logger.info("Running system tests...")
            
            # Test 1: System status
            status = await self.system.get_system_status()
            assert status['status'] in ['operational', 'degraded'], "Invalid system status"
            print("✅ System status test passed")
            
            # Test 2: Configuration
            config = get_config()
            issues = validate_config(config)
            assert not issues, f"Configuration issues: {issues}"
            print("✅ Configuration test passed")
            
            # Test 3: Plugin system
            plugin_status = await onyx_plugin_manager.get_manager_status()
            assert plugin_status['total_plugins'] >= 0, "Plugin system not working"
            print("✅ Plugin system test passed")
            
            # Test 4: Video generation (if requested)
            if args.test_generation:
                
                test_request = VideoRequest(
                    input_text="Test video generation",
                    user_id="test_user",
                    request_id="test_001"
                )
                
                response = await self.system.generate_video(test_request)
                assert response.status == "completed", "Video generation failed"
                print("✅ Video generation test passed")
            
            print("\n🎉 All tests passed!")
            return 0
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return 1
    
    async def _health_check(self, args: argparse.Namespace) -> int:
        """Run health check."""
        try:
            self.logger.info("Running health check...")
            
            # Check system status
            status = await self.system.get_system_status()
            
            # Check configuration
            config = get_config()
            issues = validate_config(config)
            
            # Check GPU
            gpu_available = is_gpu_available()
            
            # Determine overall health
            health_issues = []
            
            if status['status'] != 'operational':
                health_issues.append(f"System status: {status['status']}")
            
            if issues:
                health_issues.extend(issues)
            
            if status['error_rate'] > 0.1:  # 10% error rate threshold
                health_issues.append(f"High error rate: {status['error_rate']:.2%}")
            
            # Print results
            if args.json:
                health_result = {
                    "healthy": len(health_issues) == 0,
                    "issues": health_issues,
                    "status": status,
                    "gpu_available": gpu_available,
                    "config_issues": issues
                }
                print(json.dumps(health_result, indent=2))
            else:
                if health_issues:
                    print("❌ Health check failed:")
                    for issue in health_issues:
                        print(f"  - {issue}")
                    return 1
                else:
                    print("✅ Health check passed")
                    print(f"System Status: {status['status']}")
                    print(f"GPU Available: {gpu_available}")
                    print(f"Error Rate: {status['error_rate']:.2%}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return 1


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Onyx AI Video System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  onyx-ai-video start                    # Start the system
  onyx-ai-video status                   # Show system status
  onyx-ai-video config show              # Show configuration
  onyx-ai-video generate "Create a video" # Generate video
  onyx-ai-video plugins list             # List plugins
  onyx-ai-video test                     # Run tests
  onyx-ai-video health                   # Health check
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the AI Video system')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show system metrics')
    metrics_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_action', help='Configuration actions')
    
    config_show_parser = config_subparsers.add_parser('show', help='Show configuration')
    config_show_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    config_validate_parser = config_subparsers.add_parser('validate', help='Validate configuration')
    
    config_create_parser = config_subparsers.add_parser('create', help='Create default configuration')
    config_create_parser.add_argument('--config-path', help='Configuration file path')
    
    config_save_parser = config_subparsers.add_parser('save', help='Save configuration')
    config_save_parser.add_argument('--config-path', help='Configuration file path')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate video')
    generate_parser.add_argument('input_text', help='Input text for video generation')
    generate_parser.add_argument('--user-id', help='User ID')
    generate_parser.add_argument('--request-id', help='Request ID')
    generate_parser.add_argument('--quality', choices=['low', 'medium', 'high', 'ultra'], help='Video quality')
    generate_parser.add_argument('--duration', type=int, help='Video duration in seconds')
    generate_parser.add_argument('--output-format', help='Output format')
    generate_parser.add_argument('--plugins', help='Comma-separated list of plugins')
    
    # Plugins command
    plugins_parser = subparsers.add_parser('plugins', help='Plugin management')
    plugins_subparsers = plugins_parser.add_subparsers(dest='plugin_action', help='Plugin actions')
    
    plugins_list_parser = plugins_subparsers.add_parser('list', help='List plugins')
    plugins_list_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    plugins_status_parser = plugins_subparsers.add_parser('status', help='Show plugin status')
    plugins_status_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    plugins_enable_parser = plugins_subparsers.add_parser('enable', help='Enable plugin')
    plugins_enable_parser.add_argument('plugin_name', help='Plugin name')
    
    plugins_disable_parser = plugins_subparsers.add_parser('disable', help='Disable plugin')
    plugins_disable_parser.add_argument('plugin_name', help='Plugin name')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    test_parser.add_argument('--test-generation', action='store_true', help='Test video generation')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Health check')
    health_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    return parser


async def main() -> int:
    """Main entry point."""
    try:
        # Create parser
        parser = create_parser()
        args = parser.parse_args()
        
        # Check if command is provided
        if not args.command:
            parser.print_help()
            return 1
        
        # Initialize CLI
        cli = OnyxAIVideoCLI()
        await cli.initialize()
        
        # Run command
        result = await cli.run_command(args)
        
        # Shutdown CLI
        await cli.shutdown()
        
        return result
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        return 0
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        return 1


match __name__:
    case "__main__":
    sys.exit(asyncio.run(main())) 