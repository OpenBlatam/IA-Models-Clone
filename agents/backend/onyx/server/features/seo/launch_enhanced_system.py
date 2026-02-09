#!/usr/bin/env python3
"""
Enhanced SEO System Launch Script
Provides easy startup with various configuration options and monitoring
"""

import argparse
import os
import sys
import logging
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhanced_seo_system.log')
        ]
    )

def load_environment_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {}
    
    # Model configuration
    config['model_name'] = os.getenv('SEO_MODEL_NAME', 'microsoft/DialoGPT-medium')
    config['max_length'] = int(os.getenv('SEO_MAX_LENGTH', '512'))
    config['batch_size'] = int(os.getenv('SEO_BATCH_SIZE', '8'))
    
    # Performance settings
    config['enable_caching'] = os.getenv('SEO_ENABLE_CACHING', 'true').lower() == 'true'
    config['cache_size'] = int(os.getenv('SEO_CACHE_SIZE', '1000'))
    config['enable_async'] = os.getenv('SEO_ENABLE_ASYNC', 'true').lower() == 'true'
    config['max_concurrent_requests'] = int(os.getenv('SEO_MAX_CONCURRENT', '10'))
    
    # Monitoring settings
    config['enable_profiling'] = os.getenv('SEO_ENABLE_PROFILING', 'true').lower() == 'true'
    config['enable_metrics'] = os.getenv('SEO_ENABLE_METRICS', 'true').lower() == 'true'
    config['log_level'] = os.getenv('SEO_LOG_LEVEL', 'INFO')
    
    # Error handling
    config['max_retries'] = int(os.getenv('SEO_MAX_RETRIES', '3'))
    config['retry_delay'] = float(os.getenv('SEO_RETRY_DELAY', '1.0'))
    config['enable_circuit_breaker'] = os.getenv('SEO_ENABLE_CIRCUIT_BREAKER', 'true').lower() == 'true'
    
    return config

def create_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Create configuration from command line arguments."""
    config = load_environment_config()
    
    # Override with command line arguments
    if args.model_name:
        config['model_name'] = args.model_name
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.cache_size:
        config['cache_size'] = args.cache_size
    if args.max_concurrent:
        config['max_concurrent_requests'] = args.max_concurrent
    if args.log_level:
        config['log_level'] = args.log_level
    if args.disable_caching:
        config['enable_caching'] = False
    if args.disable_async:
        config['enable_async'] = False
    if args.disable_profiling:
        config['enable_profiling'] = False
    if args.disable_metrics:
        config['enable_metrics'] = False
    if args.disable_circuit_breaker:
        config['enable_circuit_breaker'] = False
    
    return config

def print_startup_banner(config: Dict[str, Any]) -> None:
    """Print startup banner with configuration."""
    print("=" * 80)
    print("🚀 Enhanced SEO Engine - Production-Ready System")
    print("=" * 80)
    print()
    print("📋 Configuration:")
    print(f"   Model: {config['model_name']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Cache Size: {config['cache_size']}")
    print(f"   Max Concurrent: {config['max_concurrent_requests']}")
    print(f"   Log Level: {config['log_level']}")
    print()
    print("⚙️ Features:")
    print(f"   Caching: {'✅ Enabled' if config['enable_caching'] else '❌ Disabled'}")
    print(f"   Async Processing: {'✅ Enabled' if config['enable_async'] else '❌ Disabled'}")
    print(f"   Profiling: {'✅ Enabled' if config['enable_profiling'] else '❌ Disabled'}")
    print(f"   Metrics: {'✅ Enabled' if config['enable_metrics'] else '❌ Disabled'}")
    print(f"   Circuit Breaker: {'✅ Enabled' if config['enable_circuit_breaker'] else '❌ Disabled'}")
    print()
    print("🌐 Interface will be available at: http://localhost:7860")
    print("📊 Monitoring dashboard: http://localhost:7860")
    print("=" * 80)
    print()

def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    required_packages = [
        'torch', 'transformers', 'gradio', 'numpy', 'pandas',
        'psutil', 'plotly', 'asyncio', 'aiohttp'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print()
        print("💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print()
        print("   Or install all dependencies with:")
        print("   pip install -r requirements_enhanced.txt")
        return False
    
    return True

def check_system_requirements() -> bool:
    """Check system requirements."""
    import psutil
    
    # Check memory
    memory = psutil.virtual_memory()
    if memory.total < 4 * 1024 * 1024 * 1024:  # 4GB
        print("⚠️  Warning: Less than 4GB RAM detected. Performance may be limited.")
    
    # Check CPU cores
    cpu_count = psutil.cpu_count()
    if cpu_count < 2:
        print("⚠️  Warning: Less than 2 CPU cores detected. Performance may be limited.")
    
    # Check disk space
    disk = psutil.disk_usage('/')
    if disk.free < 2 * 1024 * 1024 * 1024:  # 2GB
        print("❌ Error: Less than 2GB free disk space. Please free up space.")
        return False
    
    return True

def run_tests_if_requested() -> bool:
    """Run tests if requested."""
    try:
        from test_enhanced_seo_system import run_tests
        print("🧪 Running system tests...")
        success = run_tests()
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        return success
    except ImportError:
        print("⚠️  Test module not found. Skipping tests.")
        return True

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\n🛑 Received shutdown signal. Cleaning up...")
    sys.exit(0)

def main():
    """Main launch function."""
    parser = argparse.ArgumentParser(
        description="Launch Enhanced SEO Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with default settings
  python launch_enhanced_system.py

  # Launch with custom configuration
  python launch_enhanced_system.py --model-name microsoft/DialoGPT-large --batch-size 16

  # Launch in development mode
  python launch_enhanced_system.py --dev --log-level DEBUG

  # Launch with tests
  python launch_enhanced_system.py --run-tests

  # Launch with minimal features
  python launch_enhanced_system.py --disable-caching --disable-async --disable-profiling
        """
    )
    
    # Configuration options
    parser.add_argument('--model-name', type=str, help='Model name to use')
    parser.add_argument('--batch-size', type=int, help='Batch size for processing')
    parser.add_argument('--cache-size', type=int, help='Cache size')
    parser.add_argument('--max-concurrent', type=int, help='Maximum concurrent requests')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    
    # Feature toggles
    parser.add_argument('--disable-caching', action='store_true', help='Disable caching')
    parser.add_argument('--disable-async', action='store_true', help='Disable async processing')
    parser.add_argument('--disable-profiling', action='store_true', help='Disable profiling')
    parser.add_argument('--disable-metrics', action='store_true', help='Disable metrics collection')
    parser.add_argument('--disable-circuit-breaker', action='store_true', help='Disable circuit breaker')
    
    # Mode options
    parser.add_argument('--dev', action='store_true', help='Development mode')
    parser.add_argument('--production', action='store_true', help='Production mode')
    parser.add_argument('--run-tests', action='store_true', help='Run tests before launching')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies only')
    parser.add_argument('--check-system', action='store_true', help='Check system requirements only')
    
    # Server options
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=7860, help='Port to bind to')
    parser.add_argument('--share', action='store_true', help='Create public link')
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.check_deps:
        if check_dependencies():
            print("✅ All dependencies are installed!")
        else:
            sys.exit(1)
        return
    
    if args.check_system:
        if check_system_requirements():
            print("✅ System requirements met!")
        else:
            sys.exit(1)
        return
    
    # Check dependencies and system requirements
    if not check_dependencies():
        sys.exit(1)
    
    if not check_system_requirements():
        sys.exit(1)
    
    # Run tests if requested
    if args.run_tests:
        if not run_tests_if_requested():
            print("❌ Tests failed. Exiting.")
            sys.exit(1)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Adjust configuration based on mode
    if args.dev:
        config['log_level'] = 'DEBUG'
        config['enable_profiling'] = True
        config['enable_metrics'] = True
        print("🔧 Development mode enabled")
    
    if args.production:
        config['log_level'] = 'INFO'
        config['enable_profiling'] = False
        config['enable_metrics'] = True
        print("🏭 Production mode enabled")
    
    # Setup logging
    setup_logging(config['log_level'])
    logger = logging.getLogger(__name__)
    
    # Print startup banner
    print_startup_banner(config)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Import and launch the interface
        logger.info("Starting Enhanced SEO Engine...")
        
        # Import the enhanced interface
        from enhanced_gradio_interface import create_interface
        
        # Create and launch interface
        interface = create_interface()
        
        logger.info(f"Launching interface on {args.host}:{args.port}")
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.dev,
            show_error=True,
            quiet=False
        )
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        print("❌ Failed to import required modules. Please check your installation.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to launch system: {e}")
        print(f"❌ Failed to launch system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
