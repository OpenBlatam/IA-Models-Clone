"""
Tests for Template Manager
===========================
"""

import pytest
from ..core.templates import TemplateManager


@pytest.fixture
def template_manager():
    """Create template manager for testing."""
    return TemplateManager()


def test_create_template(template_manager):
    """Test creating a template."""
    template_id = template_manager.create_template(
        template_id="welcome_template",
        content="Hello {{name}}, welcome to {{service}}!",
        variables=["name", "service"]
    )
    
    assert template_id == "welcome_template"
    assert template_id in template_manager.templates


def test_render_template(template_manager):
    """Test rendering a template."""
    template_manager.create_template(
        "greeting",
        "Hello {{name}}!",
        ["name"]
    )
    
    result = template_manager.render_template(
        "greeting",
        {"name": "World"}
    )
    
    assert result == "Hello World!"


def test_render_template_missing_variable(template_manager):
    """Test rendering template with missing variable."""
    template_manager.create_template(
        "greeting",
        "Hello {{name}}!",
        ["name"]
    )
    
    # Should handle missing variable gracefully
    result = template_manager.render_template(
        "greeting",
        {}
    )
    
    # Should replace with empty string or raise error
    assert "{{name}}" in result or result == "Hello !"


def test_delete_template(template_manager):
    """Test deleting a template."""
    template_manager.create_template("test_template", "Test", [])
    
    assert "test_template" in template_manager.templates
    
    template_manager.delete_template("test_template")
    
    assert "test_template" not in template_manager.templates


def test_get_template(template_manager):
    """Test getting a template."""
    template_manager.create_template(
        "test_template",
        "Hello {{name}}!",
        ["name"]
    )
    
    template = template_manager.get_template("test_template")
    
    assert template is not None
    assert template.template_id == "test_template"
    assert template.content == "Hello {{name}}!"


def test_list_templates(template_manager):
    """Test listing templates."""
    template_manager.create_template("template1", "Content 1", [])
    template_manager.create_template("template2", "Content 2", [])
    
    templates = template_manager.list_templates()
    
    assert len(templates) >= 2
    assert any(t.template_id == "template1" for t in templates)
    assert any(t.template_id == "template2" for t in templates)


