#!/usr/bin/env python3
"""
Enhanced Gradio Interface Launcher
Launch the user-friendly NLP platform with enhanced UX/UI
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
    from enhanced_gradio_interface import EnhancedGradioNLPSystem, EnhancedGradioConfig
except ImportError as e:
    print(f"❌ Error importing Enhanced Gradio system: {e}")
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
            logging.FileHandler('enhanced_gradio_launcher.log'),
            logging.StreamHandler()
        ]
    )

def create_enhanced_config(args) -> EnhancedGradioConfig:
    """Create enhanced Gradio configuration from command line arguments"""
    return EnhancedGradioConfig(
        theme=args.theme,
        title=args.title,
        description=args.description,
        allow_flagging=args.allow_flagging,
        cache_examples=args.cache_examples,
        max_threads=args.max_threads,
        show_error=args.show_error,
        height=args.height,
        width=args.width,
        show_tips=args.show_tips,
        enable_animations=args.enable_animations
    )

def print_welcome_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🤖 Advanced NLP Platform                  ║
    ║                                                              ║
    ║  Experience the power of AI with our comprehensive NLP      ║
    ║  platform. Generate text, analyze sentiment, classify       ║
    ║  content, and train custom models - all through an          ║
    ║  intuitive and user-friendly interface.                     ║
    ║                                                              ║
    ║  🎯 Features:                                                ║
    ║  • Text Generation with AI models                           ║
    ║  • Sentiment Analysis & Emotion Detection                   ║
    ║  • Text Classification with Custom Labels                   ║
    ║  • Model Training & Fine-tuning                             ║
    ║  • Performance Benchmarking                                 ║
    ║  • System Resource Monitoring                               ║
    ║                                                              ║
    ║  🚀 Enhanced UX/UI with:                                     ║
    ║  • Intuitive interface design                               ║
    ║  • Real-time visualizations                                 ║
    ║  • Helpful error messages                                   ║
    ║  • Performance indicators                                   ║
    ║  • Interactive charts and graphs                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_feature_showcase():
    """Print feature showcase"""
    showcase = """
    🎨 **Feature Showcase:**
    
    📝 **Text Generation**
       • Create creative stories, articles, and content
       • Adjust creativity with temperature controls
       • Choose from multiple AI models (GPT-2, BERT, etc.)
       • Real-time generation metrics
    
    🎭 **Sentiment Analysis**
       • Analyze emotional tone of any text
       • Get detailed sentiment breakdown
       • Visual sentiment distribution charts
       • Confidence scoring for accuracy
    
    🏷️ **Text Classification**
       • Categorize text with custom labels
       • Multi-label classification support
       • Interactive confidence score visualization
       • Easy label management
    
    🎯 **Model Training**
       • Train custom models on your data
       • Real-time training progress monitoring
       • Interactive training metrics
       • Model performance optimization
    
    ⚡ **Performance Benchmarking**
       • Test system performance
       • Compare different models
       • Detailed performance metrics
       • Throughput analysis
    
    🖥️ **System Monitoring**
       • Real-time resource usage
       • GPU/CPU utilization tracking
       • Memory and performance monitoring
       • System health indicators
    """
    print(showcase)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Launch Enhanced NLP Gradio Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 **Examples:**
  python launch_enhanced_gradio.py
  python launch_enhanced_gradio.py --port 8080 --share
  python launch_enhanced_gradio.py --theme dark --title "My AI Platform"
  python launch_enhanced_gradio.py --verbose --host 0.0.0.0

🎨 **Features:**
  • Enhanced user interface with emojis and better styling
  • Real-time visualizations and performance indicators
  • Helpful error messages and input validation
  • Interactive charts and progress tracking
  • Comprehensive model capabilities showcase
        """
    )
    
    # Enhanced Gradio configuration options
    parser.add_argument('--theme', default='default', 
                       choices=['default', 'dark', 'light'],
                       help='Gradio theme (default: default)')
    parser.add_argument('--title', default='🤖 Advanced NLP Platform',
                       help='Interface title (default: 🤖 Advanced NLP Platform)')
    parser.add_argument('--description', 
                       default='Experience the power of AI with our comprehensive NLP platform. Generate text, analyze sentiment, classify content, and train custom models - all through an intuitive interface.',
                       help='Interface description')
    parser.add_argument('--allow-flagging', default='never',
                       choices=['never', 'auto', 'manual'],
                       help='Flagging behavior (default: never)')
    parser.add_argument('--cache-examples', action='store_true', default=True,
                       help='Cache examples for faster loading (default: True)')
    parser.add_argument('--max-threads', type=int, default=40,
                       help='Maximum threads (default: 40)')
    parser.add_argument('--show-error', action='store_true', default=True,
                       help='Show error details (default: True)')
    parser.add_argument('--height', type=int, default=800,
                       help='Interface height (default: 800)')
    parser.add_argument('--width', type=int, default=1200,
                       help='Interface width (default: 1200)')
    parser.add_argument('--show-tips', action='store_true', default=True,
                       help='Show helpful tips (default: True)')
    parser.add_argument('--enable-animations', action='store_true', default=True,
                       help='Enable interface animations (default: True)')
    
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
    parser.add_argument('--showcase', action='store_true',
                       help='Show feature showcase')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Print welcome banner
        print_welcome_banner()
        
        # Show feature showcase if requested
        if args.showcase:
            print_feature_showcase()
        
        logger.info("🚀 Starting Enhanced NLP Gradio Interface...")
        
        # Create enhanced configuration
        config = create_enhanced_config(args)
        logger.info(f"Configuration: {config}")
        
        # Initialize Enhanced Gradio system
        logger.info("Initializing Enhanced Gradio NLP System...")
        enhanced_system = EnhancedGradioNLPSystem(config)
        
        # Create enhanced interface
        logger.info("Creating Enhanced Gradio interface...")
        interface = enhanced_system.create_enhanced_interface()
        
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
        logger.info("=" * 70)
        logger.info("🎯 Enhanced NLP Gradio Interface Configuration:")
        logger.info(f"   🎨 Title: {config.title}")
        logger.info(f"   🎭 Theme: {config.theme}")
        logger.info(f"   🌐 Host: {args.host}")
        logger.info(f"   🔌 Port: {args.port}")
        logger.info(f"   🔗 Share: {args.share}")
        logger.info(f"   🔐 Auth: {'Yes' if args.auth else 'No'}")
        logger.info(f"   🐛 Debug: {args.debug}")
        logger.info(f"   💡 Tips: {config.show_tips}")
        logger.info(f"   ✨ Animations: {config.enable_animations}")
        logger.info("=" * 70)
        
        # Print usage tips
        print("""
💡 **Quick Start Tips:**
   • Start with the "📝 Text Generation" tab to see AI in action
   • Try the "🎭 Sentiment Analysis" for understanding text emotions
   • Use "🏷️ Text Classification" to categorize content
   • Experiment with "🎯 Model Training" for custom models
   • Check "⚡ Performance" to benchmark your system
   • Monitor "🖥️ System Info" for resource usage

🎨 **Enhanced Features:**
   • Intuitive interface with emojis and clear labels
   • Real-time visualizations and progress indicators
   • Helpful error messages and validation feedback
   • Interactive charts and performance metrics
   • Comprehensive model capabilities showcase
        """)
        
        # Launch enhanced interface
        logger.info("Launching enhanced interface...")
        interface.launch(**launch_kwargs)
        
    except KeyboardInterrupt:
        logger.info("🛑 Enhanced interface stopped by user")
        print("\n👋 Thanks for using the Advanced NLP Platform!")
    except Exception as e:
        logger.error(f"❌ Error launching enhanced interface: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()




