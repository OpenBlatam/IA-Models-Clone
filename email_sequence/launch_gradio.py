from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import argparse
import os
import sys
import logging
from pathlib import Path
from gradio_app import main as launch_app
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Gradio Application Launcher

Simple launcher script for the Email Sequence AI Gradio application
with configuration options and deployment utilities.
"""


# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment variables and configuration"""
    
    # Default configuration
    default_config = {
        'GRADIO_SERVER_NAME': '0.0.0.0',
        'GRADIO_SERVER_PORT': '7860',
        'GRADIO_SHARE': 'false',
        'GRADIO_DEBUG': 'false',
        'LOG_LEVEL': 'INFO',
        'SAVE_PATH': './outputs'
    }
    
    # Set environment variables if not already set
    for key, value in default_config.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key}={value}")


def create_output_directories():
    """Create necessary output directories"""
    
    directories = [
        './outputs',
        './outputs/sequences',
        './outputs/evaluations',
        './outputs/training',
        './outputs/gradients',
        './checkpoints'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def check_dependencies():
    """Check if all required dependencies are installed"""
    
    required_packages = [
        'gradio',
        'torch',
        'numpy',
        'pandas',
        'plotly',
        'nltk',
        'textstat',
        'textblob'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages:")
        logger.error("pip install -r requirements/gradio_requirements.txt")
        return False
    
    return True


def main():
    """Main launcher function"""
    
    parser = argparse.ArgumentParser(
        description="Launch Email Sequence AI Gradio Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_gradio.py                    # Launch with default settings
  python launch_gradio.py --port 8080        # Launch on port 8080
  python launch_gradio.py --share            # Enable public sharing
  python launch_gradio.py --debug            # Enable debug mode
  python launch_gradio.py --check-deps       # Check dependencies only
        """
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run the server on (default: 7860)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='Enable public sharing (creates a public URL)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory for generated files (default: ./outputs)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Setup environment
    setup_environment()
    
    # Update environment variables based on arguments
    os.environ['GRADIO_SERVER_PORT'] = str(args.port)
    os.environ['GRADIO_SERVER_NAME'] = args.host
    os.environ['GRADIO_SHARE'] = str(args.share).lower()
    os.environ['GRADIO_DEBUG'] = str(args.debug).lower()
    os.environ['SAVE_PATH'] = args.output_dir
    
    # Create output directories
    create_output_directories()
    
    # Check dependencies
    if not check_dependencies():
        if args.check_deps:
            sys.exit(1)
        logger.error("Dependencies check failed. Please install missing packages.")
        sys.exit(1)
    
    if args.check_deps:
        logger.info("All dependencies are installed!")
        sys.exit(0)
    
    # Display startup information
    logger.info("=" * 60)
    logger.info("Email Sequence AI - Gradio Application")
    logger.info("=" * 60)
    logger.info(f"Server: {args.host}:{args.port}")
    logger.info(f"Share: {args.share}")
    logger.info(f"Debug: {args.debug}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info("=" * 60)
    
    try:
        # Launch the application
        launch_app()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error launching application: {e}")
        sys.exit(1)


match __name__:
    case "__main__":
    main() 