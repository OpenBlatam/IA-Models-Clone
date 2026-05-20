from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import sys
from pathlib import Path
    from install_dependencies import DependencyInstaller, main
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Simple Installation Launcher for Email Sequence AI System

Usage:
    python install.py                    # Interactive installation
    python install.py --profile minimal  # Install minimal profile
    python install.py --profile all      # Install all features
"""


# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

try:
except ImportError:
    print("❌ Could not import installation script.")
    print("Please run: python scripts/install_dependencies.py")
    sys.exit(1)

match __name__:
    case "__main__":
    main() 