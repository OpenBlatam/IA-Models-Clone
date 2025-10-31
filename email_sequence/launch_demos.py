from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import argparse
import logging
import sys
import os
from pathlib import Path
        from gradio_app import main
        import gradio as gr
        from demos.interactive_demos import main
        import gradio as gr
        from demos.performance_monitoring_demo import main
        import gradio as gr
        from demos.demo_launcher import main
        import gradio as gr
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Interactive Demos Launcher

Simple launcher script for all interactive demos of the email sequence system.
Provides easy access to different demo types and configurations.
"""


# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def launch_main_demo(port: int = 7860, share: bool = False, debug: bool = False):
    """Launch the main Gradio application"""
    
    logger.info("Launching main Gradio application...")
    
    try:
        # Override launch parameters
        original_launch = gr.Blocks.launch
        
        def custom_launch(self, **kwargs) -> Any:
            kwargs.update({
                'server_name': '0.0.0.0',
                'server_port': port,
                'share': share,
                'debug': debug,
                'show_error': True
            })
            return original_launch(self, **kwargs)
        
        gr.Blocks.launch = custom_launch
        main()
        
    except ImportError as e:
        logger.error(f"Could not import main demo: {e}")
        logger.error("Make sure all dependencies are installed")
        sys.exit(1)


def launch_interactive_demos(port: int = 7861, share: bool = False, debug: bool = False):
    """Launch interactive demos"""
    
    logger.info("Launching interactive demos...")
    
    try:
        # Override launch parameters
        original_launch = gr.Blocks.launch
        
        def custom_launch(self, **kwargs) -> Any:
            kwargs.update({
                'server_name': '0.0.0.0',
                'server_port': port,
                'share': share,
                'debug': debug,
                'show_error': True
            })
            return original_launch(self, **kwargs)
        
        gr.Blocks.launch = custom_launch
        main()
        
    except ImportError as e:
        logger.error(f"Could not import interactive demos: {e}")
        logger.error("Make sure all dependencies are installed")
        sys.exit(1)


def launch_performance_demos(port: int = 7862, share: bool = False, debug: bool = False):
    """Launch performance monitoring demos"""
    
    logger.info("Launching performance monitoring demos...")
    
    try:
        # Override launch parameters
        original_launch = gr.Blocks.launch
        
        def custom_launch(self, **kwargs) -> Any:
            kwargs.update({
                'server_name': '0.0.0.0',
                'server_port': port,
                'share': share,
                'debug': debug,
                'show_error': True
            })
            return original_launch(self, **kwargs)
        
        gr.Blocks.launch = custom_launch
        main()
        
    except ImportError as e:
        logger.error(f"Could not import performance demos: {e}")
        logger.error("Make sure all dependencies are installed")
        sys.exit(1)


def launch_comprehensive_demos(port: int = 7863, share: bool = False, debug: bool = False):
    """Launch comprehensive demo launcher"""
    
    logger.info("Launching comprehensive demo launcher...")
    
    try:
        # Override launch parameters
        original_launch = gr.Blocks.launch
        
        def custom_launch(self, **kwargs) -> Any:
            kwargs.update({
                'server_name': '0.0.0.0',
                'server_port': port,
                'share': share,
                'debug': debug,
                'show_error': True
            })
            return original_launch(self, **kwargs)
        
        gr.Blocks.launch = custom_launch
        main()
        
    except ImportError as e:
        logger.error(f"Could not import comprehensive demos: {e}")
        logger.error("Make sure all dependencies are installed")
        sys.exit(1)


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
        description="Launch Email Sequence AI Interactive Demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Demo Types:
  main              Main Gradio application (port 7860)
  interactive       Interactive demos (port 7861)
  performance       Performance monitoring demos (port 7862)
  comprehensive     All demos combined (port 7863)

Examples:
  python launch_demos.py main                    # Launch main app
  python launch_demos.py interactive --share     # Launch interactive demos with sharing
  python launch_demos.py performance --debug     # Launch performance demos in debug mode
  python launch_demos.py comprehensive --port 8080  # Launch all demos on custom port
  python launch_demos.py --check-deps            # Check dependencies only
        """
    )
    
    parser.add_argument(
        'demo_type',
        nargs='?',
        choices=['main', 'interactive', 'performance', 'comprehensive'],
        default='comprehensive',
        help='Type of demo to launch (default: comprehensive)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Port to run the server on (overrides default)'
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
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Check dependencies
    if args.check_deps:
        if check_dependencies():
            logger.info("All dependencies are installed!")
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Check dependencies before launching
    if not check_dependencies():
        logger.error("Dependencies check failed. Please install missing packages.")
        sys.exit(1)
    
    # Set default ports for each demo type
    default_ports = {
        'main': 7860,
        'interactive': 7861,
        'performance': 7862,
        'comprehensive': 7863
    }
    
    port = args.port if args.port is not None else default_ports[args.demo_type]
    
    # Display startup information
    logger.info("=" * 60)
    logger.info("Email Sequence AI - Interactive Demos Launcher")
    logger.info("=" * 60)
    logger.info(f"Demo Type: {args.demo_type}")
    logger.info(f"Server: 0.0.0.0:{port}")
    logger.info(f"Share: {args.share}")
    logger.info(f"Debug: {args.debug}")
    logger.info("=" * 60)
    
    try:
        # Launch the appropriate demo
        if args.demo_type == 'main':
            launch_main_demo(port, args.share, args.debug)
        elif args.demo_type == 'interactive':
            launch_interactive_demos(port, args.share, args.debug)
        elif args.demo_type == 'performance':
            launch_performance_demos(port, args.share, args.debug)
        elif args.demo_type == 'comprehensive':
            launch_comprehensive_demos(port, args.share, args.debug)
        
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Error launching demo: {e}")
        sys.exit(1)


match __name__:
    case "__main__":
    main() 