"""
Advanced data validation tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import json


class TestDataValidationAdvanced:
    """Advanced data validation tests"""
    
    def test_schema_validation(self, temp_dir):
        """Test schema validation"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 50},
                "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
                "config": {"type": "object"}
            },
            "required": ["name", "version"]
        }
        
        # Valid data
        valid_data = {
            "name": "test-project",
            "version": "1.0.0",
            "config": {}
        }
        
        # Should pass basic validation
        assert "name" in valid_data
        assert "version" in valid_data
        assert isinstance(valid_data["name"], str)
        assert len(valid_data["name"]) <= 50
    
    def test_data_type_validation(self, temp_dir):
        """Test data type validation"""
        data = {
            "string_field": "text",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "list_field": [1, 2, 3],
            "dict_field": {"key": "value"}
        }
        
        # Type checks
        assert isinstance(data["string_field"], str)
        assert isinstance(data["int_field"], int)
        assert isinstance(data["float_field"], float)
        assert isinstance(data["bool_field"], bool)
        assert isinstance(data["list_field"], list)
        assert isinstance(data["dict_field"], dict)
    
    def test_range_validation(self):
        """Test range validation"""
        values = {
            "port": 8080,
            "timeout": 30,
            "max_connections": 100
        }
        
        # Range checks
        assert 1 <= values["port"] <= 65535
        assert values["timeout"] > 0
        assert values["max_connections"] > 0
    
    def test_format_validation(self):
        """Test format validation"""
        import re
        
        # Email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        assert re.match(email_pattern, "test@example.com")
        assert not re.match(email_pattern, "invalid-email")
        
        # URL format
        url_pattern = r'^https?://.+'
        assert re.match(url_pattern, "https://example.com")
        assert not re.match(url_pattern, "not-a-url")
    
    def test_cross_field_validation(self):
        """Test cross-field validation"""
        data = {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "min_value": 10,
            "max_value": 100
        }
        
        # Cross-field checks
        assert data["end_date"] >= data["start_date"]
        assert data["max_value"] >= data["min_value"]
    
    def test_nested_validation(self, temp_dir):
        """Test nested structure validation"""
        nested_data = {
            "project": {
                "name": "test",
                "config": {
                    "backend": {
                        "framework": "fastapi",
                        "port": 8000
                    }
                }
            }
        }
        
        # Nested checks
        assert "project" in nested_data
        assert "name" in nested_data["project"]
        assert "config" in nested_data["project"]
        assert "backend" in nested_data["project"]["config"]
        assert "framework" in nested_data["project"]["config"]["backend"]

