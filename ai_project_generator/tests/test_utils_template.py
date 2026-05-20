"""
Tests for TemplateManager utility
"""

import pytest
import asyncio
import json
from pathlib import Path

from ..utils.template_manager import TemplateManager


class TestTemplateManager:
    """Test suite for TemplateManager"""

    def test_init(self, temp_dir):
        """Test TemplateManager initialization"""
        templates_dir = temp_dir / "templates"
        manager = TemplateManager(templates_dir=templates_dir)
        assert manager.templates_dir == templates_dir
        assert templates_dir.exists()

    def test_init_default_dir(self):
        """Test TemplateManager with default directory"""
        manager = TemplateManager()
        assert manager.templates_dir.exists()

    @pytest.mark.asyncio
    async def test_save_template(self, temp_dir):
        """Test saving a template"""
        manager = TemplateManager(templates_dir=temp_dir / "templates")
        
        template_config = {
            "backend_framework": "fastapi",
            "frontend_framework": "react",
            "features": ["auth", "database"]
        }
        
        result = await manager.save_template(
            template_name="test_template",
            template_config=template_config,
            description="Test template description"
        )
        
        assert result["success"] is True
        assert result["template_name"] == "test_template"
        
        # Verify file was created
        template_file = manager.templates_dir / "test_template.json"
        assert template_file.exists()

    @pytest.mark.asyncio
    async def test_load_template(self, temp_dir):
        """Test loading a template"""
        manager = TemplateManager(templates_dir=temp_dir / "templates")
        
        template_config = {"framework": "fastapi"}
        await manager.save_template("test_load", template_config)
        
        loaded = await manager.load_template("test_load")
        assert loaded is not None
        assert loaded["name"] == "test_load"
        assert loaded["config"] == template_config

    @pytest.mark.asyncio
    async def test_load_template_not_found(self, temp_dir):
        """Test loading non-existent template"""
        manager = TemplateManager(templates_dir=temp_dir / "templates")
        
        loaded = await manager.load_template("non_existent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_templates(self, temp_dir):
        """Test listing templates"""
        manager = TemplateManager(templates_dir=temp_dir / "templates")
        
        await manager.save_template("template1", {"f": "fastapi"})
        await manager.save_template("template2", {"f": "flask"})
        
        templates = await manager.list_templates()
        assert len(templates) == 2
        template_names = [t["name"] for t in templates]
        assert "template1" in template_names
        assert "template2" in template_names

    @pytest.mark.asyncio
    async def test_list_templates_empty(self, temp_dir):
        """Test listing templates when none exist"""
        manager = TemplateManager(templates_dir=temp_dir / "templates")
        
        templates = await manager.list_templates()
        assert templates == []

    @pytest.mark.asyncio
    async def test_delete_template(self, temp_dir):
        """Test deleting a template"""
        manager = TemplateManager(templates_dir=temp_dir / "templates")
        
        await manager.save_template("to_delete", {"f": "fastapi"})
        template_file = manager.templates_dir / "to_delete.json"
        assert template_file.exists()
        
        result = await manager.delete_template("to_delete")
        assert result is True
        assert not template_file.exists()

    @pytest.mark.asyncio
    async def test_delete_template_not_found(self, temp_dir):
        """Test deleting non-existent template"""
        manager = TemplateManager(templates_dir=temp_dir / "templates")
        
        result = await manager.delete_template("non_existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_template_persistence(self, temp_dir):
        """Test that templates persist across instances"""
        templates_dir = temp_dir / "templates"
        
        # Save template with first instance
        manager1 = TemplateManager(templates_dir=templates_dir)
        await manager1.save_template("persistent", {"test": "data"})
        
        # Load with second instance
        manager2 = TemplateManager(templates_dir=templates_dir)
        loaded = await manager2.load_template("persistent")
        
        assert loaded is not None
        assert loaded["config"]["test"] == "data"

