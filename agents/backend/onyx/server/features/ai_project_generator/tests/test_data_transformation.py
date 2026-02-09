"""
Data transformation tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import json


class TestDataTransformation:
    """Tests for data transformation"""
    
    def test_json_transformation(self, temp_dir):
        """Test JSON data transformation"""
        data = {
            "name": "test",
            "items": [1, 2, 3]
        }
        
        # Transform to JSON string
        json_str = json.dumps(data)
        
        # Transform back
        transformed = json.loads(json_str)
        
        assert transformed["name"] == "test"
        assert transformed["items"] == [1, 2, 3]
    
    def test_data_normalization(self):
        """Test data normalization"""
        data = {
            "NAME": "TEST",
            "Version": "1.0.0",
            "config": {"Key": "Value"}
        }
        
        # Normalize keys to lowercase
        normalized = {k.lower(): v for k, v in data.items()}
        
        assert "name" in normalized
        assert normalized["name"] == "TEST"
    
    def test_data_filtering(self):
        """Test data filtering"""
        data = {
            "name": "test",
            "version": "1.0.0",
            "debug": True,
            "temp": "remove"
        }
        
        # Filter out temp fields
        filtered = {k: v for k, v in data.items() if k != "temp"}
        
        assert "temp" not in filtered
        assert "name" in filtered
        assert "version" in filtered
    
    def test_data_aggregation(self):
        """Test data aggregation"""
        items = [
            {"value": 10},
            {"value": 20},
            {"value": 30}
        ]
        
        # Aggregate
        total = sum(item["value"] for item in items)
        avg = total / len(items)
        
        assert total == 60
        assert avg == 20.0
    
    def test_data_mapping(self):
        """Test data mapping"""
        items = ["item1", "item2", "item3"]
        
        # Map to objects
        mapped = [{"name": item, "id": i} for i, item in enumerate(items)]
        
        assert len(mapped) == 3
        assert mapped[0]["name"] == "item1"
        assert mapped[0]["id"] == 0
    
    def test_data_validation_transformation(self):
        """Test data validation during transformation"""
        raw_data = {
            "name": "  TEST  ",
            "version": "1.0.0",
            "items": [1, 2, 3, None, 5]
        }
        
        # Transform and validate
        transformed = {
            "name": raw_data["name"].strip().lower(),
            "version": raw_data["version"],
            "items": [i for i in raw_data["items"] if i is not None]
        }
        
        assert transformed["name"] == "test"
        assert len(transformed["items"]) == 4
        assert None not in transformed["items"]

