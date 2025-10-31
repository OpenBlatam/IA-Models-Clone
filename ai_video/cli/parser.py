from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import argparse
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Onyx AI Video System - CLI Argument Parser

Command-line argument parser for the Onyx AI Video CLI.
"""



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