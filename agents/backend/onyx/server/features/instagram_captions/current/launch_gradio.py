#!/usr/bin/env python3
"""
Gradio Interface Launcher
Simple script to launch the NLP Gradio interface
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from gradio_integration_system import GradioNLPSystem, GradioConfig
except ImportError as e:
    print(f"Error importing Gradio system: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements_nlp_optimized.txt")
    print("pip install gradio plotly psutil")
    sys.exit(1)

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gradio_launcher.log'),
            logging.StreamHandler()
        ]
    )

def create_config(args) -> GradioConfig:
    """Create Gradio configuration from command line arguments"""
    return GradioConfig(
        theme=args.theme,
        title=args.title,
        description=args.description,
        allow_flagging=args.allow_flagging,
        cache_examples=args.cache_examples,
        max_threads=args.max_threads,
        show_error=args.show_error,
        height=args.height,
        width=args.width
    )

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Launch NLP Gradio Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_gradio.py
  python launch_gradio.py --port 8080 --share
  python launch_gradio.py --theme dark --title "My NLP System"
  python launch_gradio.py --verbose --host 0.0.0.0
        """
    )
    
    # Gradio configuration options
    parser.add_argument('--theme', default='default', 
                       choices=['default', 'dark', 'light'],
                       help='Gradio theme (default: default)')
    parser.add_argument('--title', default='Advanced NLP System',
                       help='Interface title (default: Advanced NLP System)')
    parser.add_argument('--description', 
                       default='Comprehensive NLP platform with text generation, analysis, and training capabilities',
                       help='Interface description')
    parser.add_argument('--allow-flagging', default='never',
                       choices=['never', 'auto', 'manual'],
                       help='Flagging behavior (default: never)')
    parser.add_argument('--cache-examples', action='store_true',
                       help='Cache examples for faster loading')
    parser.add_argument('--max-threads', type=int, default=40,
                       help='Maximum threads (default: 40)')
    parser.add_argument('--show-error', action='store_true', default=True,
                       help='Show error details (default: True)')
    parser.add_argument('--height', type=int, default=600,
                       help='Interface height (default: 600)')
    parser.add_argument('--width', type=int, default=800,
                       help='Interface width (default: 800)')
    
    # Server configuration options
    parser.add_argument('--host', default='0.0.0.0',
                       help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=7860,
                       help='Server port (default: 7860)')
    parser.add_argument('--share', action='store_true',
                       help='Create public link for sharing')
    parser.add_argument('--auth', nargs=2, metavar=('USERNAME', 'PASSWORD'),
                       help='Basic authentication (username password)')
    parser.add_argument('--ssl-verify', action='store_true',
                       help='Verify SSL certificates')
    
    # Debug options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting NLP Gradio Interface...")
        
        # Create configuration
        config = create_config(args)
        logger.info(f"Configuration: {config}")
        
        # Initialize Gradio system
        logger.info("Initializing Gradio NLP System...")
        gradio_system = GradioNLPSystem(config)
        
        # Create interface
        logger.info("Creating Gradio interface...")
        interface = gradio_system.create_interface()
        
        # Prepare launch arguments
        launch_kwargs = {
            'server_name': args.host,
            'server_port': args.port,
            'share': args.share,
            'show_error': args.show_error,
            'debug': args.debug
        }
        
        if args.auth:
            launch_kwargs['auth'] = tuple(args.auth)
        
        if args.ssl_verify:
            launch_kwargs['ssl_verify'] = True
        
        # Display launch information
        logger.info("=" * 60)
        logger.info("🎯 NLP Gradio Interface Configuration:")
        logger.info(f"   Title: {config.title}")
        logger.info(f"   Theme: {config.theme}")
        logger.info(f"   Host: {args.host}")
        logger.info(f"   Port: {args.port}")
        logger.info(f"   Share: {args.share}")
        logger.info(f"   Auth: {'Yes' if args.auth else 'No'}")
        logger.info(f"   Debug: {args.debug}")
        logger.info("=" * 60)
        
        # Launch interface
        logger.info("Launching interface...")
        interface.launch(**launch_kwargs)
        
    except KeyboardInterrupt:
        logger.info("🛑 Interface stopped by user")
    except Exception as e:
        logger.error(f"❌ Error launching interface: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()




