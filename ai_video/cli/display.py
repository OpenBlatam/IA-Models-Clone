from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, Any, List
from ..onyx_config import get_config, get_config_summary
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Onyx AI Video System - CLI Display Functions

Display and formatting functions for the Onyx AI Video CLI.
"""



async def show_startup_info(system) -> None:
    """Show startup information."""
    try:
        # Get system status
        status = await system.get_system_status()
        
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
        print(f"Failed to show startup info: {e}")


async def print_status_table(status: Dict[str, Any]) -> None:
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
    
    # Components info
    if 'components' in status:
        components = status['components']
        print(f"\nComponents:")
        for component, info in components.items():
            if isinstance(info, dict):
                print(f"  {component}: {info.get('status', 'unknown')}")
            else:
                print(f"  {component}: {info}")
    
    print("="*60 + "\n")


async def print_metrics_table(metrics: Dict[str, Any]) -> None:
    """Print metrics in table format."""
    print("\n" + "="*60)
    print("📈 ONYX AI VIDEO SYSTEM METRICS")
    print("="*60)
    
    # Performance metrics
    if 'performance' in metrics:
        perf = metrics['performance']
        print(f"Average Response Time: {perf.get('avg_response_time', 0):.2f} ms")
        print(f"Max Response Time: {perf.get('max_response_time', 0):.2f} ms")
        print(f"Requests per Second: {perf.get('requests_per_second', 0):.2f}")
    
    # Resource metrics
    if 'resources' in metrics:
        resources = metrics['resources']
        print(f"\nResource Usage:")
        print(f"  CPU Usage: {resources.get('cpu_percent', 0):.1f}%")
        print(f"  Memory Usage: {resources.get('memory_percent', 0):.1f}%")
        print(f"  GPU Usage: {resources.get('gpu_percent', 0):.1f}%")
        print(f"  GPU Memory: {resources.get('gpu_memory_percent', 0):.1f}%")
    
    # Video generation metrics
    if 'video_generation' in metrics:
        vg = metrics['video_generation']
        print(f"\nVideo Generation:")
        print(f"  Total Generated: {vg.get('total_generated', 0)}")
        print(f"  Success Rate: {vg.get('success_rate', 0):.2%}")
        print(f"  Average Duration: {vg.get('avg_duration', 0):.2f} seconds")
    
    print("="*60 + "\n")


async def print_config_table(config_summary: Dict[str, Any]) -> None:
    """Print configuration in table format."""
    print("\n" + "="*80)
    print("⚙️ ONYX AI VIDEO SYSTEM CONFIGURATION")
    print("="*80)
    
    # Basic settings
    print(f"{'Setting':<30} {'Value':<50}")
    print("-"*80)
    
    for key, value in config_summary.items():
        if isinstance(value, dict):
            print(f"{key:<30} {'(Complex Object)'}")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key:<28} {str(sub_value):<50}")
        else:
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 48:
                str_value = str_value[:45] + "..."
            print(f"{key:<30} {str_value:<50}")
    
    print("="*80 + "\n")


async def print_plugins_table(plugin_infos: List[Any]) -> None:
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


async def print_plugin_status_table(status: Dict[str, Any]) -> None:
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