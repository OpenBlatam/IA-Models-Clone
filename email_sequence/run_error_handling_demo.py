from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import sys
import os
from pathlib import Path
        from examples.error_handling_demo import ErrorHandlingDemo
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Error Handling System Launcher

Simple launcher script to run the error handling demonstration
and test the comprehensive error handling system.
"""


# Add the current directory to the path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def main():
    """Main function to run the error handling demonstration"""
    
    print("🚀 Email Sequence AI System - Error Handling Demonstration")
    print("=" * 70)
    
    try:
        # Import and run the demo
        
        print("✅ Successfully imported error handling demo")
        
        # Create and run the demo
        demo = ErrorHandlingDemo()
        demo.run_full_demo()
        
        print("\n🎉 Error handling demonstration completed successfully!")
        print("The system demonstrated robust error handling for:")
        print("  • Input validation")
        print("  • Data loading operations")
        print("  • Model inference")
        print("  • Gradio integration")
        print("  • Comprehensive error scenarios")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required modules are available.")
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("The error handling system itself encountered an error.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 