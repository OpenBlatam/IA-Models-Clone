#!/usr/bin/env python3
"""
Advanced Error Handling and Debugging System Launcher
Launch script for the comprehensive debugging and error handling system
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_error_handling_debugging_system import (
    AdvancedErrorHandlingGradioInterface,
    DebugLevel,
    create_advanced_debugging_app
)


def print_welcome_banner():
    """Print welcome banner with system information"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🔧 Advanced Error Handling & Debugging System                              ║
║                                                                              ║
║  Comprehensive error analysis, performance profiling, and debugging tools   ║
║  for AI/ML applications with intelligent error pattern detection.           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_feature_showcase():
    """Print feature showcase"""
    features = """
🎯 **Key Features:**

🔍 **Advanced Error Analysis**
   • Error pattern detection and categorization
   • Automatic severity assessment
   • Context-aware solution suggestions
   • Prevention strategy recommendations

📊 **Performance Profiling**
   • Function-level execution profiling
   • Memory usage and leak detection
   • CPU usage tracking and analysis
   • GPU memory monitoring (CUDA)

🧠 **Intelligent Debugging**
   • Configurable debug levels (Basic → Full)
   • Call stack analysis and variable inspection
   • Enhanced exceptions with rich context
   • Comprehensive debug logging

🖥️ **System Monitoring**
   • Real-time resource tracking
   • Memory leak detection
   • Garbage collection analysis
   • Performance trend analysis

🎨 **Gradio Interface**
   • Interactive debugging dashboard
   • Real-time performance monitoring
   • Error analysis and pattern detection
   • System resource visualization
    """
    print(features)


def print_debug_levels():
    """Print available debug levels"""
    levels = """
🔧 **Available Debug Levels:**

1. **BASIC** (Production)
   • Minimal overhead (~1-2% performance impact)
   • Basic error handling and logging
   • Essential error information

2. **DETAILED** (Development)
   • Low overhead (~5-10% performance impact)
   • Enhanced error analysis
   • Call stack tracking and variable inspection

3. **PROFILING** (Performance Analysis)
   • Medium overhead (~15-25% performance impact)
   • Detailed performance profiling
   • cProfile integration and function statistics

4. **MEMORY** (Memory Debugging)
   • Low overhead (~5-10% performance impact)
   • Memory-specific debugging
   • Memory leak detection and analysis

5. **THREADING** (Thread Analysis)
   • Medium overhead (~10-20% performance impact)
   • Threading analysis and safety checking
   • Deadlock detection

6. **FULL** (Complete Debugging)
   • High overhead (~30-50% performance impact)
   • Complete debugging suite
   • Maximum debugging information
    """
    print(levels)


def setup_logging(log_level: str, log_file: str = None):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )


def validate_debug_level(debug_level: str) -> DebugLevel:
    """Validate and convert debug level string to enum"""
    debug_level_map = {
        'basic': DebugLevel.BASIC,
        'detailed': DebugLevel.DETAILED,
        'profiling': DebugLevel.PROFILING,
        'memory': DebugLevel.MEMORY,
        'threading': DebugLevel.THREADING,
        'full': DebugLevel.FULL
    }
    
    if debug_level.lower() not in debug_level_map:
        print(f"❌ Invalid debug level: {debug_level}")
        print("Valid levels: basic, detailed, profiling, memory, threading, full")
        sys.exit(1)
    
    return debug_level_map[debug_level.lower()]


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Advanced Error Handling and Debugging System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with basic debugging (production)
  python launch_advanced_debugging.py --debug-level basic

  # Launch with detailed debugging (development)
  python launch_advanced_debugging.py --debug-level detailed --port 7862

  # Launch with full profiling
  python launch_advanced_debugging.py --debug-level profiling --log-level DEBUG

  # Launch with custom configuration
  python launch_advanced_debugging.py --debug-level full --port 7863 --share --log-file logs/debug.log
        """
    )
    
    parser.add_argument(
        '--debug-level',
        type=str,
        default='detailed',
        choices=['basic', 'detailed', 'profiling', 'memory', 'threading', 'full'],
        help='Debug level (default: detailed)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7862,
        help='Port to run the server on (default: 7862)'
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
        help='Create a public link for the interface'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: logs/advanced_debug.log)'
    )
    
    parser.add_argument(
        '--no-banner',
        action='store_true',
        help='Skip welcome banner'
    )
    
    parser.add_argument(
        '--show-features',
        action='store_true',
        help='Show feature showcase'
    )
    
    parser.add_argument(
        '--show-levels',
        action='store_true',
        help='Show debug levels information'
    )
    
    args = parser.parse_args()
    
    # Show information if requested
    if args.show_features:
        print_feature_showcase()
        return
    
    if args.show_levels:
        print_debug_levels()
        return
    
    # Print welcome banner
    if not args.no_banner:
        print_welcome_banner()
    
    # Validate debug level
    debug_level = validate_debug_level(args.debug_level)
    
    # Setup logging
    log_file = args.log_file or "logs/advanced_debug.log"
    setup_logging(args.log_level, log_file)
    
    # Print launch configuration
    print(f"\n🚀 **Launch Configuration:**")
    print(f"   • Debug Level: {debug_level.value.upper()}")
    print(f"   • Host: {args.host}")
    print(f"   • Port: {args.port}")
    print(f"   • Share: {args.share}")
    print(f"   • Log Level: {args.log_level}")
    print(f"   • Log File: {log_file}")
    
    # Performance impact warning
    if debug_level in [DebugLevel.PROFILING, DebugLevel.FULL]:
        print(f"\n⚠️  **Performance Warning:**")
        print(f"   • Debug level '{debug_level.value}' has significant performance impact")
        print(f"   • Use only for debugging and development")
        print(f"   • Consider using 'basic' or 'detailed' for production")
    
    # Memory usage warning
    if debug_level == DebugLevel.FULL:
        print(f"\n💾 **Memory Usage Warning:**")
        print(f"   • Full debugging mode uses significant memory")
        print(f"   • Monitor memory usage and clear logs periodically")
        print(f"   • Consider using 'detailed' for regular development")
    
    print(f"\n🔧 **Starting Advanced Error Handling & Debugging System...**")
    print(f"   • Creating interface with {debug_level.value} debugging...")
    
    try:
        # Create and launch the advanced debugging app
        app = create_advanced_debugging_app()
        
        print(f"   • Interface created successfully!")
        print(f"   • Launching Gradio server...")
        print(f"   • Access the interface at: http://{args.host}:{args.port}")
        
        if args.share:
            print(f"   • Public link will be generated...")
        
        # Launch the app
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=True
        )
        
    except KeyboardInterrupt:
        print(f"\n🛑 **System stopped by user**")
        print(f"   • Shutting down gracefully...")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ **Error launching system:**")
        print(f"   • Error: {str(e)}")
        print(f"   • Check configuration and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()



