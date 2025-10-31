from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import argparse
import signal
import json
from typing import Dict, Any
from onyx.utils.gpu_utils import is_gpu_available
from ..onyx_config import get_config, save_config, validate_config, get_config_summary, create_default_config
from ..onyx_plugin_manager import onyx_plugin_manager
from .display import (
        from ..models import VideoRequest
            from ..models import VideoRequest
from typing import Any, List, Dict, Optional
import logging
"""
Onyx AI Video System - CLI Commands

Individual command implementations for the Onyx AI Video CLI.
"""


# Onyx imports

# Local imports
    show_startup_info, print_status_table, print_metrics_table,
    print_config_table, print_plugins_table, print_plugin_status_table
)


async def start_system(cli, args: argparse.Namespace) -> int:
    """Start the AI Video system."""
    try:
        cli.logger.info("Starting Onyx AI Video System...")
        
        # Show startup information
        await show_startup_info(cli.system)
        
        # Keep system running
        cli.logger.info("System is running. Press Ctrl+C to stop.")
        
        # Setup signal handlers
        def signal_handler(signum, frame) -> Any:
            cli.logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(cli.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run forever
        while True:
            await asyncio.sleep(1)
        
    except KeyboardInterrupt:
        cli.logger.info("Received keyboard interrupt")
        return 0
    except Exception as e:
        cli.logger.error(f"System startup failed: {e}")
        return 1


async def show_status(cli, args: argparse.Namespace) -> int:
    """Show system status."""
    try:
        status = await cli.system.get_system_status()
        
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            await print_status_table(status)
        
        return 0
        
    except Exception as e:
        cli.logger.error(f"Status check failed: {e}")
        return 1


async def show_metrics(cli, args: argparse.Namespace) -> int:
    """Show system metrics."""
    try:
        metrics = await cli.system.get_system_metrics()
        
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            await print_metrics_table(metrics)
        
        return 0
        
    except Exception as e:
        cli.logger.error(f"Metrics check failed: {e}")
        return 1


async def handle_config(cli, args: argparse.Namespace) -> int:
    """Handle configuration commands."""
    try:
        config_action = args.config_action
        
        if config_action == "show":
            config = get_config()
            config_summary = get_config_summary(config)
            
            if args.json:
                print(json.dumps(config_summary, indent=2))
            else:
                await print_config_table(config_summary)
        
        elif config_action == "validate":
            config = get_config()
            issues = validate_config(config)
            
            if issues:
                print("❌ Configuration validation failed:")
                for issue in issues:
                    print(f"  - {issue}")
                return 1
            else:
                print("✅ Configuration validation passed")
        
        elif config_action == "create":
            config_path = getattr(args, 'config_path', None)
            config = create_default_config()
            
            if config_path:
                save_config(config, config_path)
                print(f"✅ Default configuration created at: {config_path}")
            else:
                save_config(config)
                print("✅ Default configuration created")
        
        elif config_action == "save":
            config_path = getattr(args, 'config_path', None)
            config = get_config()
            
            if config_path:
                save_config(config, config_path)
                print(f"✅ Configuration saved to: {config_path}")
            else:
                save_config(config)
                print("✅ Configuration saved")
        
        return 0
        
    except Exception as e:
        cli.logger.error(f"Configuration handling failed: {e}")
        return 1


async def generate_video(cli, args: argparse.Namespace) -> int:
    """Generate video."""
    try:
        
        # Create video request
        request = VideoRequest(
            input_text=args.input_text,
            user_id=getattr(args, 'user_id', 'cli_user'),
            request_id=getattr(args, 'request_id', None)
        )
        
        # Set additional parameters
        if hasattr(args, 'quality') and args.quality:
            request.quality = args.quality
        
        if hasattr(args, 'duration') and args.duration:
            request.duration = args.duration
        
        if hasattr(args, 'output_format') and args.output_format:
            request.output_format = args.output_format
        
        if hasattr(args, 'plugins') and args.plugins:
            request.plugins = args.plugins.split(',')
        
        # Generate video
        cli.logger.info(f"Generating video for: {args.input_text}")
        response = await cli.system.generate_video(request)
        
        if response.status == "completed":
            print(f"✅ Video generated successfully: {response.output_path}")
            return 0
        else:
            print(f"❌ Video generation failed: {response.error}")
            return 1
        
    except Exception as e:
        cli.logger.error(f"Video generation failed: {e}")
        return 1


async def handle_plugins(cli, args: argparse.Namespace) -> int:
    """Handle plugin commands."""
    try:
        plugin_action = args.plugin_action
        
        if plugin_action == "list":
            plugin_infos = await onyx_plugin_manager.get_all_plugins_info()
            
            if args.json:
                print(json.dumps([info.__dict__ for info in plugin_infos], indent=2))
            else:
                await print_plugins_table(plugin_infos)
        
        elif plugin_action == "status":
            status = await onyx_plugin_manager.get_manager_status()
            
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                await print_plugin_status_table(status)
        
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
        cli.logger.error(f"Plugin handling failed: {e}")
        return 1


async def run_tests(cli, args: argparse.Namespace) -> int:
    """Run system tests."""
    try:
        cli.logger.info("Running system tests...")
        
        # Test 1: System status
        status = await cli.system.get_system_status()
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
            
            response = await cli.system.generate_video(test_request)
            assert response.status == "completed", "Video generation failed"
            print("✅ Video generation test passed")
        
        print("\n🎉 All tests passed!")
        return 0
        
    except Exception as e:
        cli.logger.error(f"Test failed: {e}")
        return 1


async def health_check(cli, args: argparse.Namespace) -> int:
    """Run health check."""
    try:
        cli.logger.info("Running health check...")
        
        # Check system status
        status = await cli.system.get_system_status()
        
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
        cli.logger.error(f"Health check failed: {e}")
        return 1 