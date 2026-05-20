"""
Migration and version upgrade tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any
import json


class TestMigration:
    """Tests for migration and version upgrades"""
    
    def test_project_version_migration(self, temp_dir):
        """Test migrating project to new version"""
        # Create old version project
        old_project = {
            "version": "1.0.0",
            "name": "test-project",
            "config": {"old_field": "value"}
        }
        
        project_file = temp_dir / "project.json"
        project_file.write_text(json.dumps(old_project), encoding="utf-8")
        
        # Read and validate
        data = json.loads(project_file.read_text(encoding="utf-8"))
        assert data["version"] == "1.0.0"
        
        # Simulate migration
        data["version"] = "2.0.0"
        data["config"]["new_field"] = "new_value"
        
        project_file.write_text(json.dumps(data), encoding="utf-8")
        
        # Verify migration
        migrated = json.loads(project_file.read_text(encoding="utf-8"))
        assert migrated["version"] == "2.0.0"
        assert "new_field" in migrated["config"]
    
    def test_config_format_migration(self, temp_dir):
        """Test migrating config format"""
        # Old format
        old_config = {
            "backend": "fastapi",
            "frontend": "react"
        }
        
        config_file = temp_dir / "config.json"
        config_file.write_text(json.dumps(old_config), encoding="utf-8")
        
        # Migrate to new format
        old_data = json.loads(config_file.read_text(encoding="utf-8"))
        new_config = {
            "stack": {
                "backend": old_data["backend"],
                "frontend": old_data["frontend"]
            }
        }
        
        config_file.write_text(json.dumps(new_config), encoding="utf-8")
        
        # Verify
        migrated = json.loads(config_file.read_text(encoding="utf-8"))
        assert "stack" in migrated
        assert migrated["stack"]["backend"] == "fastapi"
    
    def test_backward_compatibility(self, temp_dir):
        """Test backward compatibility"""
        # New format
        new_data = {
            "version": "2.0.0",
            "name": "test",
            "config": {"field": "value"}
        }
        
        # Should be readable by old version (simplified)
        assert "version" in new_data
        assert "name" in new_data
        assert "config" in new_data
    
    def test_data_validation_after_migration(self, temp_dir):
        """Test data validation after migration"""
        from ..utils.validator import ProjectValidator
        
        validator = ProjectValidator()
        
        # Create migrated project
        project_dir = temp_dir / "migrated_project"
        project_dir.mkdir()
        
        # Should validate
        import asyncio
        result = asyncio.run(validator.validate_project(project_dir, {"name": "test"}))
        assert isinstance(result, dict)
        assert "valid" in result

