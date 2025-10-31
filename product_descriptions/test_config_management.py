from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import tempfile
import os
import yaml
import json
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from dependencies.config_helpers import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
# Import our configuration manager
    ConfigManager, SecurityConfig, DatabaseConfig, LoggingConfig
)

class TestConfigManager:
    """Test cases for ConfigManager"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Sample valid configuration
        self.valid_config = {
            "security": {
                "max_scan_duration": 300,
                "rate_limit_per_minute": 60,
                "allowed_ports": [22, 80, 443, 8080, 8443],
                "blocked_ips": []
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "security_tools",
                "username": "admin",
                "password": "secret123",
                "pool_size": 10,
                "max_overflow": 20
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_path": "logs/app.log",
                "max_file_size": 10485760,
                "backup_count": 5
            }
        }
    
    def teardown_method(self) -> Any:
        """Cleanup test environment"""
        if os.path.exists(self.config_path):
            os.unlink(self.config_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_config_manager_initialization(self) -> Any:
        """Test ConfigManager initialization"""
        config_manager = ConfigManager(self.config_path)
        assert config_manager.config_path == self.config_path
        assert config_manager._config_cache == {}
        assert "security" in config_manager.schemas
        assert "database" in config_manager.schemas
        assert "logging" in config_manager.schemas
    
    def test_load_yaml_config_success(self) -> Any:
        """Test successful YAML config loading"""
        # Create test config file
        with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_path)
        loaded_config = config_manager.load_yaml_config(self.config_path)
        
        assert loaded_config == self.valid_config
    
    def test_load_yaml_config_file_not_found(self) -> Any:
        """Test YAML config loading with non-existent file"""
        config_manager = ConfigManager(self.config_path)
        
        with pytest.raises(FileNotFoundError):
            config_manager.load_yaml_config(self.config_path)
    
    def test_load_yaml_config_invalid_format(self) -> Any:
        """Test YAML config loading with invalid format"""
        # Create invalid YAML file
        with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("invalid: yaml: content: [")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        config_manager = ConfigManager(self.config_path)
        
        with pytest.raises(ValueError):
            config_manager.load_yaml_config(self.config_path)
    
    def test_save_yaml_config_success(self) -> Any:
        """Test successful YAML config saving"""
        config_manager = ConfigManager(self.config_path)
        config_manager.save_yaml_config(self.valid_config, self.config_path)
        
        # Verify file was created and contains correct content
        assert os.path.exists(self.config_path)
        with open(self.config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            saved_config = yaml.safe_load(f)
        assert saved_config == self.valid_config
    
    def test_validate_config_section_success(self) -> bool:
        """Test successful config section validation"""
        config_manager = ConfigManager(self.config_path)
        
        # Test valid security config
        security_config = {
            "max_scan_duration": 300,
            "rate_limit_per_minute": 60,
            "allowed_ports": [22, 80, 443],
            "blocked_ips": []
        }
        
        result = config_manager.validate_config_section(security_config, "security")
        assert result is True
    
    def test_validate_config_section_invalid(self) -> bool:
        """Test config section validation with invalid data"""
        config_manager = ConfigManager(self.config_path)
        
        # Test invalid security config (negative duration)
        invalid_security_config = {
            "max_scan_duration": -1,  # Invalid: negative value
            "rate_limit_per_minute": 60,
            "allowed_ports": [22, 80, 443],
            "blocked_ips": []
        }
        
        with pytest.raises(ValueError):
            config_manager.validate_config_section(invalid_security_config, "security")
    
    def test_validate_config_section_unknown(self) -> bool:
        """Test config section validation with unknown section"""
        config_manager = ConfigManager(self.config_path)
        
        # Test unknown section (should pass validation)
        unknown_config = {"some_key": "some_value"}
        result = config_manager.validate_config_section(unknown_config, "unknown_section")
        assert result is True
    
    def test_validate_full_config_success(self) -> bool:
        """Test successful full config validation"""
        config_manager = ConfigManager(self.config_path)
        
        result = config_manager.validate_full_config(self.valid_config)
        assert result is True
    
    def test_validate_full_config_invalid(self) -> bool:
        """Test full config validation with invalid data"""
        config_manager = ConfigManager(self.config_path)
        
        # Create config with invalid security section
        invalid_config = self.valid_config.copy()
        invalid_config["security"]["max_scan_duration"] = -1
        
        with pytest.raises(ValueError):
            config_manager.validate_full_config(invalid_config)
    
    def test_get_config_with_cache(self) -> Optional[Dict[str, Any]]:
        """Test config retrieval with caching"""
        # Create test config file
        with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_path)
        
        # First call should load from file
        config1 = config_manager.get_config()
        assert config1 == self.valid_config
        
        # Second call should use cache
        config2 = config_manager.get_config()
        assert config2 == self.valid_config
        assert config1 is config2  # Same object (cached)
    
    def test_get_section_success(self) -> Optional[Dict[str, Any]]:
        """Test successful section retrieval"""
        # Create test config file
        with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_path)
        
        security_section = config_manager.get_section("security")
        assert security_section == self.valid_config["security"]
        
        database_section = config_manager.get_section("database")
        assert database_section == self.valid_config["database"]
    
    def test_get_section_not_found(self) -> Optional[Dict[str, Any]]:
        """Test section retrieval for non-existent section"""
        # Create test config file
        with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_path)
        
        unknown_section = config_manager.get_section("unknown_section")
        assert unknown_section == {}
    
    def test_update_section_success(self) -> Any:
        """Test successful section update"""
        # Create test config file
        with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_path)
        
        # Update security section
        new_security_config = {
            "max_scan_duration": 600,
            "rate_limit_per_minute": 120,
            "allowed_ports": [22, 80, 443, 3306],
            "blocked_ips": ["192.168.1.100"]
        }
        
        config_manager.update_section("security", new_security_config)
        
        # Verify update
        updated_config = config_manager.get_config()
        assert updated_config["security"] == new_security_config
        
        # Verify cache was cleared
        assert config_manager._config_cache == {}
    
    def test_update_section_invalid(self) -> Any:
        """Test section update with invalid data"""
        # Create test config file
        with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_path)
        
        # Try to update with invalid security config
        invalid_security_config = {
            "max_scan_duration": -1,  # Invalid: negative value
            "rate_limit_per_minute": 60,
            "allowed_ports": [22, 80, 443],
            "blocked_ips": []
        }
        
        with pytest.raises(ValueError):
            config_manager.update_section("security", invalid_security_config)
    
    def test_create_default_config(self) -> Any:
        """Test default configuration creation"""
        config_manager = ConfigManager(self.config_path)
        default_config = config_manager.create_default_config()
        
        # Check required sections exist
        assert "security" in default_config
        assert "database" in default_config
        assert "logging" in default_config
        assert "scanning" in default_config
        assert "reporting" in default_config
        
        # Check security section has required fields
        security = default_config["security"]
        assert "max_scan_duration" in security
        assert "rate_limit_per_minute" in security
        assert "allowed_ports" in security
        assert "blocked_ips" in security
    
    def test_initialize_config_new_file(self) -> Any:
        """Test configuration initialization for new file"""
        config_manager = ConfigManager(self.config_path)
        
        # Initialize config
        config_manager.initialize_config()
        
        # Verify file was created
        assert os.path.exists(self.config_path)
        
        # Verify content is valid
        loaded_config = config_manager.load_yaml_config(self.config_path)
        assert "security" in loaded_config
        assert "database" in loaded_config
        assert "logging" in loaded_config
    
    def test_initialize_config_existing_file(self) -> Any:
        """Test configuration initialization with existing file"""
        # Create existing config file
        existing_config = {"existing": "data"}
        with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(existing_config, f)
        
        config_manager = ConfigManager(self.config_path)
        
        # Initialize config (should not overwrite existing)
        config_manager.initialize_config()
        
        # Verify existing content was preserved
        loaded_config = config_manager.load_yaml_config(self.config_path)
        assert loaded_config == existing_config
    
    def test_reload_config(self) -> Any:
        """Test configuration reload"""
        # Create test config file
        with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_path)
        
        # Load config (populates cache)
        config1 = config_manager.get_config()
        
        # Modify file directly
        modified_config = self.valid_config.copy()
        modified_config["security"]["max_scan_duration"] = 900
        
        with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(modified_config, f)
        
        # Reload config
        config2 = config_manager.reload_config()
        
        # Verify reloaded config has updated values
        assert config2["security"]["max_scan_duration"] == 900
        assert config1 != config2
    
    @patch.dict(os.environ, {
        "SECURITY_MAX_SCAN_DURATION": "900",
        "DB_HOST": "production-db.example.com",
        "LOG_LEVEL": "DEBUG"
    })
    def test_get_env_overrides(self) -> Optional[Dict[str, Any]]:
        """Test environment variable overrides retrieval"""
        config_manager = ConfigManager(self.config_path)
        overrides = config_manager.get_env_overrides()
        
        assert "security" in overrides
        assert "database" in overrides
        assert "logging" in overrides
        
        assert overrides["security"]["max_scan_duration"] == 900
        assert overrides["database"]["host"] == "production-db.example.com"
        assert overrides["logging"]["level"] == "DEBUG"
    
    def test_apply_env_overrides(self) -> Any:
        """Test environment variable overrides application"""
        # Create test config file
        with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_path)
        
        # Set environment variables
        with patch.dict(os.environ, {
            "SECURITY_MAX_SCAN_DURATION": "900",
            "DB_HOST": "production-db.example.com"
        }):
            config_manager.apply_env_overrides()
        
        # Verify overrides were applied
        updated_config = config_manager.get_config()
        assert updated_config["security"]["max_scan_duration"] == 900
        assert updated_config["database"]["host"] == "production-db.example.com"

class TestConfigClasses:
    """Test cases for configuration dataclasses"""
    
    def test_security_config_defaults(self) -> Any:
        """Test SecurityConfig default values"""
        config = SecurityConfig()
        
        assert config.max_scan_duration == 300
        assert config.rate_limit_per_minute == 60
        assert config.allowed_ports == [22, 80, 443, 8080, 8443]
        assert config.blocked_ips == []
    
    def test_security_config_custom_values(self) -> Any:
        """Test SecurityConfig with custom values"""
        config = SecurityConfig(
            max_scan_duration=600,
            rate_limit_per_minute=120,
            allowed_ports=[22, 80, 443, 3306],
            blocked_ips=["192.168.1.100"]
        )
        
        assert config.max_scan_duration == 600
        assert config.rate_limit_per_minute == 120
        assert config.allowed_ports == [22, 80, 443, 3306]
        assert config.blocked_ips == ["192.168.1.100"]
    
    def test_database_config_defaults(self) -> Any:
        """Test DatabaseConfig default values"""
        config = DatabaseConfig()
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "security_tools"
        assert config.username == ""
        assert config.password == ""
        assert config.pool_size == 10
        assert config.max_overflow == 20
    
    def test_database_config_custom_values(self) -> Any:
        """Test DatabaseConfig with custom values"""
        config = DatabaseConfig(
            host="production-db.example.com",
            port=5433,
            database="prod_security",
            username="admin",
            password="secret123",
            pool_size=20,
            max_overflow=30
        )
        
        assert config.host == "production-db.example.com"
        assert config.port == 5433
        assert config.database == "prod_security"
        assert config.username == "admin"
        assert config.password == "secret123"
        assert config.pool_size == 20
        assert config.max_overflow == 30
    
    def test_logging_config_defaults(self) -> Any:
        """Test LoggingConfig default values"""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.file_path is None
        assert config.max_file_size == 10 * 1024 * 1024
        assert config.backup_count == 5
    
    def test_logging_config_custom_values(self) -> Any:
        """Test LoggingConfig with custom values"""
        config = LoggingConfig(
            level="DEBUG",
            format="%(levelname)s: %(message)s",
            file_path="logs/debug.log",
            max_file_size=5 * 1024 * 1024,
            backup_count=3
        )
        
        assert config.level == "DEBUG"
        assert config.format == "%(levelname)s: %(message)s"
        assert config.file_path == "logs/debug.log"
        assert config.max_file_size == 5 * 1024 * 1024
        assert config.backup_count == 3

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 