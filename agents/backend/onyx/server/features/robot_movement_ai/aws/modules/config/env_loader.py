"""
Environment Loader
==================

Environment variable loader.
"""

import os
from typing import Dict, Any, Optional, List

class EnvLoader:
    """Environment variable loader."""
    
    @staticmethod
    def load(prefix: str = "", required: Optional[List[str]] = None) -> Dict[str, Any]:
        """Load environment variables."""
        config = {}
        missing = []
        
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
            
            config_key = key[len(prefix):] if prefix else key
            config[config_key] = value
        
        if required:
            for key in required:
                full_key = f"{prefix}{key}" if prefix else key
                if full_key not in os.environ:
                    missing.append(full_key)
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        return config

