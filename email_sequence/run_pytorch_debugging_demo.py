from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import sys
import os
from pathlib import Path
        from examples.pytorch_debugging_demo import main as run_demo
        import traceback
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
PyTorch Debugging Tools Demo Launcher

Launcher script for demonstrating PyTorch's built-in debugging tools
including autograd.detect_anomaly(), profiler, and other debugging utilities.
"""


# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add the features directory to the Python path
features_dir = Path(__file__).parent
sys.path.insert(0, str(features_dir))

def main():
    """Run the PyTorch debugging demonstration"""
    
    print("Starting PyTorch Debugging Tools Demonstration...")
    print("="*60)
    
    try:
        # Import and run the demo
        run_demo()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required dependencies are installed:")
        print("pip install torch torchvision torchaudio")
        print("pip install psutil numpy matplotlib plotly")
        return 1
        
    except Exception as e:
        print(f"Error running demo: {e}")
        traceback.print_exc()
        return 1
    
    print("\n" + "="*60)
    print("PyTorch Debugging Tools Demo completed successfully!")
    print("="*60)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 