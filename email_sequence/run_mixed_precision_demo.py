from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import sys
import os
import asyncio
from pathlib import Path
        from examples.mixed_precision_demo import main as demo_main
        import traceback
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Mixed Precision Training Demo Launcher

Launcher script for the comprehensive mixed precision training demonstration.
"""


# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main launcher function"""
    
    print("Mixed Precision Training Demo Launcher")
    print("="*50)
    
    try:
        # Import the demo
        
        # Run the demo
        asyncio.run(demo_main())
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this from the correct directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Demo failed: {e}")
        traceback.print_exc()
        sys.exit(1)

match __name__:
    case "__main__":
    main() 