from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import sys
import os
import subprocess
from pathlib import Path
        from gradio_interactive_demos import launch_demos
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Launcher for Gradio Interactive Demos
Simple script to launch all interactive demos.
"""


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'gradio',
        'torch',
        'numpy',
        'matplotlib',
        'seaborn',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def install_requirements():
    """Install requirements if needed."""
    requirements_file = Path(__file__).parent / "gradio_requirements.txt"
    
    if requirements_file.exists():
        print("Installing requirements...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e}")
            return False
    else:
        print("Requirements file not found!")
        return False

def launch_demos():
    """Launch the interactive demos."""
    try:
        
        print("Starting Gradio Interactive Demos...")
        print("Access the demos at: http://localhost:7860")
        print("Press Ctrl+C to stop the server")
        
        launch_demos()
        
    except ImportError as e:
        print(f"Error importing demo modules: {e}")
        print("Please ensure all dependencies are installed.")
        return False
    except KeyboardInterrupt:
        print("\nDemo server stopped by user.")
        return True
    except Exception as e:
        print(f"Error launching demos: {e}")
        return False

def main():
    """Main launcher function."""
    print("Gradio Interactive Demos Launcher")
    print("=" * 40)
    
    # Check if we should install requirements
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        if not install_requirements():
            return False
    
    # Check dependencies
    if not check_dependencies():
        print("\nWould you like to install the missing dependencies? (y/n)")
        response = input().lower().strip()
        
        if response in ['y', 'yes']:
            if not install_requirements():
                return False
        else:
            return False
    
    # Launch demos
    return launch_demos()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 