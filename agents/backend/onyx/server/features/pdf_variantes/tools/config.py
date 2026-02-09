"""
Tool Configuration
==================
Centralized configuration for all tools.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ToolConfig:
    """Tool configuration."""
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    retries: int = 3
    auth_token: Optional[str] = None
    output_dir: str = "output"
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "ToolConfig":
        """Create config from environment variables."""
        return cls(
            base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
            timeout=int(os.getenv("API_TIMEOUT", "30")),
            retries=int(os.getenv("API_RETRIES", "3")),
            auth_token=os.getenv("API_AUTH_TOKEN"),
            output_dir=os.getenv("TOOLS_OUTPUT_DIR", "output"),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    @classmethod
    def from_file(cls, file_path: Path) -> "ToolConfig":
        """Load config from file."""
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
                return cls(**data)
        return cls()
    
    def save(self, file_path: Path):
        """Save config to file."""
        with open(file_path, "w") as f:
            json.dump(asdict(self), f, indent=2)


_config: Optional[ToolConfig] = None


def get_config() -> ToolConfig:
    """Get global configuration."""
    global _config
    if _config is None:
        config_file = Path("tools_config.json")
        if config_file.exists():
            _config = ToolConfig.from_file(config_file)
        else:
            _config = ToolConfig.from_env()
    return _config


def set_config(config: ToolConfig):
    """Set global configuration."""
    global _config
    _config = config



