"""
BUL Optimized Startup Script
============================

Quick start script for the optimized BUL system.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from bul_optimized import BULSystem
from config_optimized import get_config

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('bul.log')
        ]
    )

def main():
    """Main startup function."""
    print("🚀 Starting BUL - Business Universal Language (Optimized)")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = get_config()
        
        # Validate configuration
        errors = config.validate_config()
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   - {error}")
            print("\n💡 Please check your .env file or environment variables")
            return 1
        
        # Create and start system
        print("📋 Configuration loaded successfully")
        print(f"   - API Host: {config.api_host}")
        print(f"   - API Port: {config.api_port}")
        print(f"   - Debug Mode: {config.debug_mode}")
        print(f"   - Output Directory: {config.output_directory}")
        print(f"   - Enabled Areas: {', '.join(config.enabled_business_areas)}")
        
        print("\n🔧 Initializing BUL system...")
        system = BULSystem(config.to_dict())
        
        print("✅ System initialized successfully")
        print(f"\n🌐 Starting server on http://{config.api_host}:{config.api_port}")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("❤️  Health Check: http://localhost:8000/health")
        print("\n⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start the system
        system.run(
            host=config.api_host,
            port=config.api_port,
            debug=config.debug_mode
        )
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Shutdown requested by user")
        logger.info("System shutdown requested")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting system: {e}")
        logger.error(f"System startup error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
