#!/usr/bin/env python3
"""
API Configuration Manager
=========================
Manage and validate API configurations.

⚠️ DEPRECATED: This file is deprecated. Use `tools.config` instead.

For new code, use:
    from tools.config import ToolConfig, get_config
    config = get_config()
"""
import warnings

warnings.warn(
    "api_config.py is deprecated. Use 'tools.config' instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class APIConfig:
    """API configuration."""
    base_url: str
    timeout: int = 30
    retries: int = 3
    headers: Dict[str, str] = None
    auth_token: Optional[str] = None
    verify_ssl: bool = True
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        
        if self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"


class APIConfigManager:
    """API configuration manager."""
    
    def __init__(self, config_file: Path = Path("api_config.json")):
        self.config_file = config_file
        self.configs: Dict[str, APIConfig] = {}
        self.load_configs()
    
    def load_configs(self):
        """Load configurations from file."""
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                data = json.load(f)
                
                for name, config_data in data.get("configs", {}).items():
                    self.configs[name] = APIConfig(**config_data)
        else:
            # Create default config
            self.configs["default"] = APIConfig(
                base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
                auth_token=os.getenv("API_AUTH_TOKEN")
            )
            self.save_configs()
    
    def save_configs(self):
        """Save configurations to file."""
        data = {
            "configs": {
                name: asdict(config)
                for name, config in self.configs.items()
            }
        }
        
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Configurations saved to {self.config_file}")
    
    def get_config(self, name: str = "default") -> Optional[APIConfig]:
        """Get configuration by name."""
        return self.configs.get(name)
    
    def add_config(self, name: str, config: APIConfig):
        """Add or update configuration."""
        self.configs[name] = config
        self.save_configs()
    
    def list_configs(self):
        """List all configurations."""
        print("\n" + "=" * 60)
        print("📋 API Configurations")
        print("=" * 60)
        
        for name, config in self.configs.items():
            print(f"\n{name}:")
            print(f"  Base URL: {config.base_url}")
            print(f"  Timeout: {config.timeout}s")
            print(f"  Retries: {config.retries}")
            if config.auth_token:
                print(f"  Auth: Configured")
            print(f"  Verify SSL: {config.verify_ssl}")
        
        print("\n" + "=" * 60)
    
    def validate_config(self, name: str = "default") -> bool:
        """Validate configuration."""
        config = self.get_config(name)
        if not config:
            print(f"❌ Configuration '{name}' not found")
            return False
        
        errors = []
        
        if not config.base_url:
            errors.append("Base URL is required")
        
        if config.timeout <= 0:
            errors.append("Timeout must be positive")
        
        if config.retries < 0:
            errors.append("Retries must be non-negative")
        
        if errors:
            print(f"❌ Configuration '{name}' has errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print(f"✅ Configuration '{name}' is valid")
        return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Configuration Manager")
    parser.add_argument("--list", action="store_true", help="List all configurations")
    parser.add_argument("--validate", help="Validate configuration")
    parser.add_argument("--add", help="Add new configuration (requires --url)")
    parser.add_argument("--url", help="Base URL for new configuration")
    parser.add_argument("--token", help="Auth token for new configuration")
    
    args = parser.parse_args()
    
    manager = APIConfigManager()
    
    if args.list:
        manager.list_configs()
    elif args.validate:
        manager.validate_config(args.validate)
    elif args.add and args.url:
        config = APIConfig(
            base_url=args.url,
            auth_token=args.token
        )
        manager.add_config(args.add, config)
        print(f"✅ Configuration '{args.add}' added")
    else:
        manager.list_configs()


if __name__ == "__main__":
    main()



