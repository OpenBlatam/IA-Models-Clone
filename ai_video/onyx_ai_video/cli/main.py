from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import argparse
import json
import sys
import os
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging
from ..api.main import OnyxAIVideoSystem, get_system, shutdown_system
from ..core.models import VideoRequest, VideoResponse, SystemStatus, PerformanceMetrics
from ..config.config_manager import OnyxConfigManager, create_config_file
from ..utils.logger import OnyxLogger, setup_logger
from ..utils.performance import get_performance_monitor
from ..utils.security import get_security_manager
            import time
from typing import Any, List, Dict, Optional
"""
Onyx AI Video System - CLI

Command-line interface for the Onyx AI Video system with system
management, video generation, and plugin administration.
"""




class OnyxAIVideoCLI:
    """
    Command-line interface for Onyx AI Video system.
    
    Provides comprehensive CLI for system management, video generation,
    plugin administration, and monitoring.
    """
    
    def __init__(self) -> Any:
        self.system: Optional[OnyxAIVideoSystem] = None
        self.logger = OnyxLogger("ai_video_cli")
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="Onyx AI Video System - Command Line Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Initialize system
  python -m onyx_ai_video.cli.main init --config config.yaml
  
  # Generate video
  python -m onyx_ai_video.cli.main generate --input "Create a video about AI" --user-id user123
  
  # Check system status
  python -m onyx_ai_video.cli.main status
  
  # List plugins
  python -m onyx_ai_video.cli.main plugins list
  
  # Monitor performance
  python -m onyx_ai_video.cli.main monitor --interval 30
            """
        )
        
        # Global options
        parser.add_argument(
            '--config', '-c',
            type=str,
            help='Configuration file path'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output in JSON format'
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Init command
        init_parser = subparsers.add_parser('init', help='Initialize system')
        init_parser.add_argument(
            '--force',
            action='store_true',
            help='Force reinitialization'
        )
        init_parser.add_argument(
            '--create-config',
            type=str,
            help='Create default configuration file'
        )
        
        # Generate command
        generate_parser = subparsers.add_parser('generate', help='Generate video')
        generate_parser.add_argument(
            '--input', '-i',
            type=str,
            required=True,
            help='Input text for video generation'
        )
        generate_parser.add_argument(
            '--user-id', '-u',
            type=str,
            required=True,
            help='User identifier'
        )
        generate_parser.add_argument(
            '--quality',
            choices=['low', 'medium', 'high', 'ultra'],
            default='medium',
            help='Video quality'
        )
        generate_parser.add_argument(
            '--duration',
            type=int,
            default=60,
            help='Video duration in seconds'
        )
        generate_parser.add_argument(
            '--format',
            choices=['mp4', 'avi', 'mov', 'webm', 'mkv'],
            default='mp4',
            help='Output video format'
        )
        generate_parser.add_argument(
            '--plugins',
            type=str,
            help='Comma-separated list of plugins to use'
        )
        generate_parser.add_argument(
            '--vision',
            type=str,
            help='Path to image file for vision processing'
        )
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Show system status')
        status_parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed status'
        )
        
        # Metrics command
        metrics_parser = subparsers.add_parser('metrics', help='Show performance metrics')
        metrics_parser.add_argument(
            '--period',
            type=int,
            default=3600,
            help='Metrics period in seconds'
        )
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_command', help='Config commands')
        
        config_show_parser = config_subparsers.add_parser('show', help='Show configuration')
        config_reload_parser = config_subparsers.add_parser('reload', help='Reload configuration')
        config_create_parser = config_subparsers.add_parser('create', help='Create configuration file')
        config_create_parser.add_argument('path', type=str, help='Configuration file path')
        config_create_parser.add_argument(
            '--format',
            choices=['yaml', 'json'],
            default='yaml',
            help='Configuration file format'
        )
        
        # Plugins command
        plugins_parser = subparsers.add_parser('plugins', help='Plugin management')
        plugins_subparsers = plugins_parser.add_subparsers(dest='plugins_command', help='Plugin commands')
        
        plugins_list_parser = plugins_subparsers.add_parser('list', help='List plugins')
        plugins_list_parser.add_argument(
            '--status',
            choices=['active', 'inactive', 'error'],
            help='Filter by status'
        )
        
        plugins_info_parser = plugins_subparsers.add_parser('info', help='Show plugin information')
        plugins_info_parser.add_argument('name', type=str, help='Plugin name')
        
        plugins_enable_parser = plugins_subparsers.add_parser('enable', help='Enable plugin')
        plugins_enable_parser.add_argument('name', type=str, help='Plugin name')
        
        plugins_disable_parser = plugins_subparsers.add_parser('disable', help='Disable plugin')
        plugins_disable_parser.add_argument('name', type=str, help='Plugin name')
        
        plugins_reload_parser = plugins_subparsers.add_parser('reload', help='Reload plugins')
        
        # Monitor command
        monitor_parser = subparsers.add_parser('monitor', help='Real-time monitoring')
        monitor_parser.add_argument(
            '--interval',
            type=int,
            default=5,
            help='Update interval in seconds'
        )
        monitor_parser.add_argument(
            '--metrics',
            type=str,
            help='Comma-separated list of metrics to monitor'
        )
        
        # Requests command
        requests_parser = subparsers.add_parser('requests', help='Request management')
        requests_subparsers = requests_parser.add_subparsers(dest='requests_command', help='Request commands')
        
        requests_list_parser = requests_subparsers.add_parser('list', help='List active requests')
        requests_cancel_parser = requests_subparsers.add_parser('cancel', help='Cancel request')
        requests_cancel_parser.add_argument('request_id', type=str, help='Request ID')
        requests_cancel_parser.add_argument('user_id', type=str, help='User ID')
        
        # Security command
        security_parser = subparsers.add_parser('security', help='Security management')
        security_subparsers = security_parser.add_subparsers(dest='security_command', help='Security commands')
        
        security_status_parser = security_subparsers.add_parser('status', help='Show security status')
        security_cleanup_parser = security_subparsers.add_parser('cleanup', help='Clean up expired access')
        
        # Shutdown command
        shutdown_parser = subparsers.add_parser('shutdown', help='Shutdown system')
        shutdown_parser.add_argument(
            '--force',
            action='store_true',
            help='Force shutdown'
        )
        
        return parser
    
    async def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI.
        
        Args:
            args: Command line arguments
            
        Returns:
            Exit code
        """
        try:
            parsed_args = self.parser.parse_args(args)
            
            # Setup logging
            if parsed_args.verbose:
                setup_logger(level="DEBUG")
            else:
                setup_logger(level="INFO")
            
            # Handle commands
            if parsed_args.command == 'init':
                return await self._handle_init(parsed_args)
            elif parsed_args.command == 'generate':
                return await self._handle_generate(parsed_args)
            elif parsed_args.command == 'status':
                return await self._handle_status(parsed_args)
            elif parsed_args.command == 'metrics':
                return await self._handle_metrics(parsed_args)
            elif parsed_args.command == 'config':
                return await self._handle_config(parsed_args)
            elif parsed_args.command == 'plugins':
                return await self._handle_plugins(parsed_args)
            elif parsed_args.command == 'monitor':
                return await self._handle_monitor(parsed_args)
            elif parsed_args.command == 'requests':
                return await self._handle_requests(parsed_args)
            elif parsed_args.command == 'security':
                return await self._handle_security(parsed_args)
            elif parsed_args.command == 'shutdown':
                return await self._handle_shutdown(parsed_args)
            else:
                self.parser.print_help()
                return 1
            
        except KeyboardInterrupt:
            self.logger.info("Operation cancelled by user")
            return 130
        except Exception as e:
            self.logger.error(f"CLI error: {e}")
            return 1
    
    async def _handle_init(self, args) -> int:
        """Handle init command."""
        try:
            if args.create_config:
                create_config_file(args.create_config)
                self.logger.info(f"Configuration file created: {args.create_config}")
                return 0
            
            self.logger.info("Initializing Onyx AI Video System")
            self.system = await get_system(args.config)
            self.logger.info("System initialized successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return 1
    
    async def _handle_generate(self, args) -> int:
        """Handle generate command."""
        try:
            if not self.system:
                self.system = await get_system(args.config)
            
            # Create video request
            request = VideoRequest(
                input_text=args.input,
                user_id=args.user_id,
                quality=args.quality,
                duration=args.duration,
                output_format=args.format,
                plugins=args.plugins.split(',') if args.plugins else None
            )
            
            # Generate video
            if args.vision:
                # Load image data
                with open(args.vision, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    image_data = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                
                response = await self.system.generate_video_with_vision(request, image_data)
            else:
                response = await self.system.generate_video(request)
            
            # Output result
            if args.json:
                print(json.dumps(response.dict(), indent=2, default=str))
            else:
                print(f"Video generated successfully!")
                print(f"Request ID: {response.request_id}")
                print(f"Status: {response.status}")
                if response.output_url:
                    print(f"Output URL: {response.output_url}")
                if response.output_path:
                    print(f"Output Path: {response.output_path}")
                if response.duration:
                    print(f"Duration: {response.duration}s")
                if response.file_size:
                    print(f"File Size: {response.file_size} bytes")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}")
            return 1
    
    async def _handle_status(self, args) -> int:
        """Handle status command."""
        try:
            if not self.system:
                self.system = await get_system(args.config)
            
            status = await self.system.get_system_status()
            
            if args.json:
                print(json.dumps(status.dict(), indent=2, default=str))
            else:
                print("=== Onyx AI Video System Status ===")
                print(f"Status: {status.status}")
                print(f"Version: {status.version}")
                print(f"Uptime: {status.uptime:.1f} seconds")
                print(f"Total Requests: {status.request_count}")
                print(f"Error Rate: {status.error_rate:.2f}%")
                print(f"Active Plugins: {status.active_plugins}/{status.total_plugins}")
                
                if args.detailed:
                    print("\n=== Component Status ===")
                    for component, info in status.components.items():
                        print(f"{component}: {info['status']}")
                    
                    if status.cpu_usage:
                        print(f"\nCPU Usage: {status.cpu_usage}%")
                    if status.memory_usage:
                        print(f"Memory Usage: {status.memory_usage}%")
                    if status.gpu_usage:
                        print(f"GPU Usage: {status.gpu_usage}%")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return 1
    
    async def _handle_metrics(self, args) -> int:
        """Handle metrics command."""
        try:
            if not self.system:
                self.system = await get_system(args.config)
            
            metrics = await self.system.get_metrics()
            
            if args.json:
                print(json.dumps(metrics.dict(), indent=2, default=str))
            else:
                print("=== Performance Metrics ===")
                print(f"Total Requests: {metrics.total_requests}")
                print(f"Successful: {metrics.successful_requests}")
                print(f"Failed: {metrics.failed_requests}")
                print(f"Success Rate: {(metrics.successful_requests/metrics.total_requests*100):.2f}%" if metrics.total_requests > 0 else "N/A")
                print(f"Average Processing Time: {metrics.avg_processing_time:.3f}s")
                print(f"Min Processing Time: {metrics.min_processing_time:.3f}s" if metrics.min_processing_time else "N/A")
                print(f"Max Processing Time: {metrics.max_processing_time:.3f}s" if metrics.max_processing_time else "N/A")
                
                if metrics.plugin_executions:
                    print("\n=== Plugin Executions ===")
                    for plugin, count in metrics.plugin_executions.items():
                        errors = metrics.plugin_errors.get(plugin, 0)
                        print(f"{plugin}: {count} executions, {errors} errors")
                
                if metrics.cache_hits > 0 or metrics.cache_misses > 0:
                    total_cache = metrics.cache_hits + metrics.cache_misses
                    cache_hit_rate = (metrics.cache_hits / total_cache * 100) if total_cache > 0 else 0
                    print(f"\nCache Hit Rate: {cache_hit_rate:.2f}%")
                    print(f"Cache Size: {metrics.cache_size}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Metrics retrieval failed: {e}")
            return 1
    
    async def _handle_config(self, args) -> int:
        """Handle config command."""
        try:
            if args.config_command == 'show':
                if not self.system:
                    self.system = await get_system(args.config)
                
                config_dict = self.system.config.dict()
                
                if args.json:
                    print(json.dumps(config_dict, indent=2, default=str))
                else:
                    print("=== Configuration ===")
                    for section, values in config_dict.items():
                        if isinstance(values, dict):
                            print(f"\n[{section}]")
                            for key, value in values.items():
                                print(f"  {key}: {value}")
                        else:
                            print(f"{section}: {values}")
                
            elif args.config_command == 'reload':
                if not self.system:
                    self.system = await get_system(args.config)
                
                await self.system.reload_config()
                self.logger.info("Configuration reloaded successfully")
                
            elif args.config_command == 'create':
                create_config_file(args.path)
                self.logger.info(f"Configuration file created: {args.path}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Configuration operation failed: {e}")
            return 1
    
    async def _handle_plugins(self, args) -> int:
        """Handle plugins command."""
        try:
            if not self.system:
                self.system = await get_system(args.config)
            
            if args.plugins_command == 'list':
                plugins = self.system.plugin_manager.get_plugins()
                
                if args.json:
                    print(json.dumps([p.dict() for p in plugins.values()], indent=2, default=str))
                else:
                    print("=== Plugins ===")
                    for name, plugin in plugins.items():
                        status_icon = "✓" if plugin.status == "active" else "✗" if plugin.status == "error" else "-"
                        print(f"{status_icon} {name} v{plugin.version} ({plugin.status})")
                        if plugin.description:
                            print(f"    {plugin.description}")
                
            elif args.plugins_command == 'info':
                plugins = self.system.plugin_manager.get_plugins()
                plugin = plugins.get(args.name)
                
                if not plugin:
                    self.logger.error(f"Plugin not found: {args.name}")
                    return 1
                
                if args.json:
                    print(json.dumps(plugin.dict(), indent=2, default=str))
                else:
                    print(f"=== Plugin: {args.name} ===")
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
                
            elif args.plugins_command == 'enable':
                await self.system.plugin_manager.enable_plugin(args.name)
                self.logger.info(f"Plugin enabled: {args.name}")
                
            elif args.plugins_command == 'disable':
                await self.system.plugin_manager.disable_plugin(args.name)
                self.logger.info(f"Plugin disabled: {args.name}")
                
            elif args.plugins_command == 'reload':
                await self.system.plugin_manager.reload_plugins()
                self.logger.info("Plugins reloaded successfully")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Plugin operation failed: {e}")
            return 1
    
    async def _handle_monitor(self, args) -> int:
        """Handle monitor command."""
        try:
            if not self.system:
                self.system = await get_system(args.config)
            
            
            print("=== Real-time Monitoring ===")
            print("Press Ctrl+C to stop")
            
            while True:
                try:
                    # Clear screen (works on most terminals)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    # Get status and metrics
                    status = await self.system.get_system_status()
                    metrics = await self.system.get_metrics()
                    
                    print(f"=== Onyx AI Video System Monitor ===")
                    print(f"Status: {status.status}")
                    print(f"Uptime: {status.uptime:.1f}s")
                    print(f"Requests: {metrics.total_requests} (Success: {metrics.successful_requests}, Failed: {metrics.failed_requests})")
                    print(f"Error Rate: {(metrics.failed_requests/metrics.total_requests*100):.2f}%" if metrics.total_requests > 0 else "Error Rate: 0%")
                    print(f"Avg Processing Time: {metrics.avg_processing_time:.3f}s")
                    
                    if status.cpu_usage:
                        print(f"CPU: {status.cpu_usage}%")
                    if status.memory_usage:
                        print(f"Memory: {status.memory_usage}%")
                    if status.gpu_usage:
                        print(f"GPU: {status.gpu_usage}%")
                    
                    print(f"Active Plugins: {status.active_plugins}/{status.total_plugins}")
                    print(f"Cache Hit Rate: {(metrics.cache_hits/(metrics.cache_hits+metrics.cache_misses)*100):.2f}%" if (metrics.cache_hits+metrics.cache_misses) > 0 else "Cache Hit Rate: N/A")
                    
                    print(f"\nLast Updated: {datetime.now().strftime('%H:%M:%S')}")
                    
                    time.sleep(args.interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Monitor error: {e}")
                    time.sleep(args.interval)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Monitoring failed: {e}")
            return 1
    
    async async def _handle_requests(self, args) -> int:
        """Handle requests command."""
        try:
            if not self.system:
                self.system = await get_system(args.config)
            
            if args.requests_command == 'list':
                active_requests = await self.system.get_active_requests()
                
                if args.json:
                    print(json.dumps(active_requests, indent=2, default=str))
                else:
                    print("=== Active Requests ===")
                    if not active_requests:
                        print("No active requests")
                    else:
                        for request_id, info in active_requests.items():
                            duration = info.get('end_time', time.time()) - info['start_time']
                            print(f"{request_id}: {info['status']} ({duration:.1f}s) - User: {info['user_id']}")
                
            elif args.requests_command == 'cancel':
                success = await self.system.cancel_request(args.request_id, args.user_id)
                
                if success:
                    self.logger.info(f"Request cancelled: {args.request_id}")
                else:
                    self.logger.error(f"Failed to cancel request: {args.request_id}")
                    return 1
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Request operation failed: {e}")
            return 1
    
    async def _handle_security(self, args) -> int:
        """Handle security command."""
        try:
            if not self.system:
                self.system = await get_system(args.config)
            
            if args.security_command == 'status':
                security_status = get_security_manager().get_security_status()
                
                if args.json:
                    print(json.dumps(security_status, indent=2, default=str))
                else:
                    print("=== Security Status ===")
                    for key, value in security_status.items():
                        print(f"{key}: {value}")
                
            elif args.security_command == 'cleanup':
                get_security_manager().cleanup_expired_access()
                self.logger.info("Expired access tokens cleaned up")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Security operation failed: {e}")
            return 1
    
    async def _handle_shutdown(self, args) -> int:
        """Handle shutdown command."""
        try:
            if not self.system:
                self.system = await get_system(args.config)
            
            if args.force:
                self.logger.warning("Force shutdown requested")
            
            await self.system.shutdown()
            self.logger.info("System shutdown completed")
            return 0
            
        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")
            return 1


def main():
    """Main CLI entry point."""
    cli = OnyxAIVideoCLI()
    exit_code = asyncio.run(cli.run())
    sys.exit(exit_code)


match __name__:
    case "__main__":
    main() 