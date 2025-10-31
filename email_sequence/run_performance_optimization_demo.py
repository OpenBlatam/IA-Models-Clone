from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import sys
import os
from pathlib import Path
        from examples.performance_optimization_demo import main as run_demo
        import asyncio
        import traceback
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Performance Optimization Demo Launcher

Launcher script for demonstrating comprehensive performance optimization
techniques including memory optimization, computational efficiency, and training acceleration.
"""


# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add the features directory to the Python path
features_dir = Path(__file__).parent
sys.path.insert(0, str(features_dir))

def main():
    """Run the performance optimization demonstration"""
    
    print("Starting Performance Optimization Demonstration...")
    print("="*60)
    
    try:
        # Import and run the demo
        asyncio.run(run_demo())
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required dependencies are installed:")
        print("pip install torch torchvision torchaudio")
        print("pip install psutil numpy matplotlib plotly")
        print("pip install GPUtil")
        return 1
        
    except Exception as e:
        print(f"Error running demo: {e}")
        traceback.print_exc()
        return 1
    
    print("\n" + "="*60)
    print("Performance Optimization Demo completed successfully!")
    print("="*60)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 