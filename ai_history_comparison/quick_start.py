#!/usr/bin/env python3
"""
Quick Start Script for AI History Analysis System
================================================

This script provides a quick way to start the AI History Analysis System
with different configurations and modes.

Usage:
    python quick_start.py --mode full          # Start complete system
    python quick_start.py --mode api           # Start API only
    python quick_start.py --mode dashboard     # Start dashboard only
    python quick_start.py --mode demo          # Run demonstration
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from ai_history_comparison.comprehensive_system import (
    get_comprehensive_system, SystemConfiguration
)
from ai_history_comparison.examples.complete_system_example import CompleteSystemExample

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_default_config() -> SystemConfiguration:
    """Create default system configuration"""
    return SystemConfiguration(
        enable_api=True,
        enable_dashboard=True,
        enable_ml_predictor=True,
        enable_alerts=True,
        enable_integration=True,
        api_port=8002,
        dashboard_port=8003,
        ml_model_storage_path="ml_models",
        alert_cooldown_minutes=30,
        health_check_interval=60,
        performance_monitoring=True,
        auto_scaling=False,
        backup_enabled=True,
        log_level="INFO"
    )


async def start_full_system(config: SystemConfiguration):
    """Start the complete system with API and dashboard"""
    logger.info("🚀 Starting Complete AI History Analysis System...")
    
    system = get_comprehensive_system(config)
    
    try:
        # Start system in background
        asyncio.create_task(system.start())
        
        # Start API and dashboard concurrently
        await asyncio.gather(
            system.run_api(),
            system.run_dashboard()
        )
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, shutting down...")
    finally:
        await system.shutdown()


async def start_api_only(config: SystemConfiguration):
    """Start only the API server"""
    logger.info("🌐 Starting API Server...")
    
    config.enable_dashboard = False
    system = get_comprehensive_system(config)
    
    try:
        await system.start()
        await system.run_api()
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, shutting down...")
    finally:
        await system.shutdown()


async def start_dashboard_only(config: SystemConfiguration):
    """Start only the dashboard"""
    logger.info("📊 Starting Dashboard...")
    
    config.enable_api = False
    system = get_comprehensive_system(config)
    
    try:
        await system.start()
        await system.run_dashboard()
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, shutting down...")
    finally:
        await system.shutdown()


async def run_demonstration():
    """Run the complete system demonstration"""
    logger.info("🎬 Running System Demonstration...")
    
    example = CompleteSystemExample()
    await example.run_complete_demonstration()


def print_system_info():
    """Print system information and available modes"""
    print("""
🤖 AI History Analysis System - Quick Start
==========================================

Available Modes:
  full        - Start complete system (API + Dashboard)
  api         - Start API server only
  dashboard   - Start dashboard only
  demo        - Run system demonstration

Default Ports:
  API Server:     http://localhost:8002
  Dashboard:      http://localhost:8003
  API Docs:       http://localhost:8002/docs

Features:
  ✅ Real-time performance monitoring
  ✅ Machine learning predictions
  ✅ Intelligent alerts and notifications
  ✅ Model comparison and benchmarking
  ✅ WebSocket dashboard with live updates
  ✅ REST API with 20+ endpoints
  ✅ Integration with workflow chains

Quick Commands:
  python quick_start.py --mode full
  python quick_start.py --mode api --api-port 8002
  python quick_start.py --mode dashboard --dashboard-port 8003
  python quick_start.py --mode demo

For more information, see:
  - README.md
  - SYSTEM_OVERVIEW.md
  - examples/complete_system_example.py
""")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Quick Start Script for AI History Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py --mode full
  python quick_start.py --mode api --api-port 8002
  python quick_start.py --mode dashboard --dashboard-port 8003
  python quick_start.py --mode demo
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "api", "dashboard", "demo", "info"],
        default="full",
        help="System mode to run (default: full)"
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=8002,
        help="API server port (default: 8002)"
    )
    
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8003,
        help="Dashboard server port (default: 8003)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    
    parser.add_argument(
        "--ml-models-path",
        type=str,
        default="ml_models",
        help="Path to store ML models (default: ml_models)"
    )
    
    args = parser.parse_args()
    
    # Print system info if requested
    if args.mode == "info":
        print_system_info()
        return
    
    # Create configuration
    config = create_default_config()
    config.api_port = args.api_port
    config.dashboard_port = args.dashboard_port
    config.log_level = args.log_level
    config.ml_model_storage_path = args.ml_models_path
    
    # Load configuration from file if provided
    if args.config:
        import json
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.get("system_configuration", {}).items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            logger.info(f"📁 Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            return
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create ML models directory
    os.makedirs(config.ml_model_storage_path, exist_ok=True)
    
    # Run the selected mode
    try:
        if args.mode == "full":
            await start_full_system(config)
        elif args.mode == "api":
            await start_api_only(config)
        elif args.mode == "dashboard":
            await start_dashboard_only(config)
        elif args.mode == "demo":
            await run_demonstration()
    
    except Exception as e:
        logger.error(f"❌ Error running system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
        sys.exit(0)

























