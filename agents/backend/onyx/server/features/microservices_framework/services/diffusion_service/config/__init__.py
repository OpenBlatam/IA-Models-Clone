"""
Diffusion Service Configuration
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml

def get_config() -> Dict[str, Any]:
    """Load service configuration."""
    config_path = os.getenv(
        "DIFFUSION_SERVICE_CONFIG",
        str(Path(__file__).parent.parent.parent.parent / "configs" / "diffusion_config.yaml")
    )
    
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    
    # Default configuration
    return {
        "device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
        "use_fp16": True,
    }



