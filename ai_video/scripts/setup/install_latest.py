from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import subprocess
import sys
import os
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Install Latest Library Versions
==============================

Automated installation script for the latest stable versions of all required libraries.
"""


def install_libraries():
    """Install all required libraries with latest versions."""
    
    libraries = [
        "torch>=2.1.0",
        "torchvision>=0.16.0", 
        "torchaudio>=2.1.0",
        "transformers>=4.36.0",
        "diffusers>=0.25.0",
        "accelerate>=0.25.0",
        "gradio>=4.0.0",
        "xformers>=0.0.23",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "watchdog>=3.0.0"
    ]
    
    print("Installing latest library versions...")
    
    for lib in libraries:
        try:
            print(f"Installing {lib}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", lib, "--upgrade"
            ])
            print(f"✅ {lib} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {lib}: {e}")
    
    print("Installation completed!")

match __name__:
    case "__main__":
    install_libraries() 