"""
Integration tests for Piel Mejorador AI SAM3.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from piel_mejorador_ai_sam3.core.piel_mejorador_agent import PielMejoradorAgent
from piel_mejorador_ai_sam3.config.piel_mejorador_config import PielMejoradorConfig


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_output_dir):
    """Create test configuration."""
    config = PielMejoradorConfig()
    config.openrouter.api_key = "test-key"
    config.output_dir = str(temp_output_dir)
    config.debug = True
    return config


@pytest.mark.asyncio
class TestPielMejoradorAgentIntegration:
    """Integration tests for PielMejoradorAgent."""
    
    async def test_agent_initialization(self, test_config):
        """Test agent initialization."""
        agent = PielMejoradorAgent(config=test_config)
        
        assert agent is not None
        assert agent.config == test_config
        assert agent.task_manager is not None
        assert agent.service_handler is not None
    
    async def test_agent_start_stop(self, test_config):
        """Test agent start and stop."""
        agent = PielMejoradorAgent(config=test_config)
        
        # Start agent
        await agent.start()
        assert agent.running is True
        
        # Stop agent
        await agent.stop()
        assert agent.running is False
    
    async def test_task_creation(self, test_config, temp_output_dir):
        """Test task creation."""
        agent = PielMejoradorAgent(config=test_config)
        
        # Create a test file
        test_file = temp_output_dir / "test.jpg"
        test_file.write_bytes(b"fake image data")
        
        # Create task
        task_id = await agent.mejorar_imagen(
            file_path=str(test_file),
            enhancement_level="medium"
        )
        
        assert task_id is not None
        
        # Check task status
        status = await agent.get_task_status(task_id)
        assert status is not None




