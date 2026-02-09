"""
Integration tests for Color Grading AI.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from ..core.color_grading_agent import ColorGradingAgent
from ..config.color_grading_config import ColorGradingConfig


@pytest.fixture
async def agent():
    """Create test agent."""
    config = ColorGradingConfig()
    config.openrouter.api_key = "test-key"  # Mock key for testing
    config.enable_cache = False  # Disable cache for tests
    
    test_agent = ColorGradingAgent(config=config, output_dir="test_output")
    yield test_agent
    await test_agent.close()
    # Cleanup
    if Path("test_output").exists():
        shutil.rmtree("test_output")


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization."""
    assert agent is not None
    assert agent.video_processor is not None
    assert agent.image_processor is not None
    assert agent.template_manager is not None


@pytest.mark.asyncio
async def test_template_listing(agent):
    """Test template listing."""
    templates = await agent.list_templates()
    assert isinstance(templates, list)
    assert len(templates) > 0


@pytest.mark.asyncio
async def test_preset_creation(agent):
    """Test preset creation."""
    preset_id = agent.create_preset(
        name="Test Preset",
        description="Test description",
        color_params={"brightness": 0.1, "contrast": 1.2},
        category="test"
    )
    assert preset_id is not None
    
    presets = agent.list_presets()
    assert len(presets) > 0


@pytest.mark.asyncio
async def test_metrics_collection(agent):
    """Test metrics collection."""
    metrics = agent.get_metrics()
    assert isinstance(metrics, dict)


@pytest.mark.asyncio
async def test_resource_stats(agent):
    """Test resource statistics."""
    stats = agent.get_resource_stats()
    assert isinstance(stats, dict)


@pytest.mark.asyncio
async def test_history_management(agent):
    """Test history management."""
    history = agent.get_history(limit=10)
    assert isinstance(history, list)


@pytest.mark.asyncio
async def test_backup_creation(agent):
    """Test backup creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_path = agent.create_backup([tmpdir], "test_backup")
        assert Path(backup_path).exists()


@pytest.mark.asyncio
async def test_plugin_listing(agent):
    """Test plugin listing."""
    plugins = agent.list_plugins()
    assert isinstance(plugins, list)




