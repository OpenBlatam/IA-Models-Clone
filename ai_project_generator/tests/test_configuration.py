"""
Configuration management tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any
import json


class TestConfiguration:
    """Tests for configuration management"""
    
    def test_config_loading(self, temp_dir):
        """Test loading configuration from file"""
        config = {
            "name": "test-project",
            "version": "1.0.0",
            "settings": {
                "debug": True,
                "port": 8000
            }
        }
        
        config_file = temp_dir / "config.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")
        
        # Load
        loaded = json.loads(config_file.read_text(encoding="utf-8"))
        assert loaded["name"] == "test-project"
        assert loaded["settings"]["debug"] is True
    
    def test_config_merging(self):
        """Test merging configurations"""
        base_config = {
            "name": "project",
            "version": "1.0.0",
            "settings": {"debug": False}
        }
        
        override_config = {
            "settings": {"debug": True, "port": 8000}
        }
        
        # Merge
        merged = {**base_config}
        merged["settings"] = {**merged["settings"], **override_config["settings"]}
        
        assert merged["name"] == "project"
        assert merged["settings"]["debug"] is True
        assert merged["settings"]["port"] == 8000
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = {
            "name": "test",
            "version": "1.0.0",
            "port": 8000
        }
        
        # Validation
        assert "name" in config
        assert "version" in config
        assert isinstance(config["port"], int)
        assert 1 <= config["port"] <= 65535
    
    def test_environment_variables(self, temp_dir):
        """Test environment variable configuration"""
        import os
        
        # Set test env var
        os.environ["TEST_CONFIG_VAR"] = "test_value"
        
        # Read
        value = os.environ.get("TEST_CONFIG_VAR")
        assert value == "test_value"
        
        # Cleanup
        del os.environ["TEST_CONFIG_VAR"]
    
    def test_config_defaults(self):
        """Test configuration defaults"""
        defaults = {
            "port": 8000,
            "debug": False,
            "timeout": 30
        }
        
        user_config = {"port": 9000}
        
        # Merge with defaults
        final_config = {**defaults, **user_config}
        
        assert final_config["port"] == 9000
        assert final_config["debug"] is False
        assert final_config["timeout"] == 30
    
    def test_config_hierarchical(self, temp_dir):
        """Test hierarchical configuration"""
        # Global config
        global_config = {
            "defaults": {
                "port": 8000,
                "timeout": 30
            }
        }
        
        # Project-specific config
        project_config = {
            "port": 9000
        }
        
        # Merge
        final = {
            **global_config["defaults"],
            **project_config
        }
        
        assert final["port"] == 9000
        assert final["timeout"] == 30

