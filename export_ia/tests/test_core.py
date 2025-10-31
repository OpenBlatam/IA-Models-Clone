"""
Tests for core Export IA components.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.engine import ExportIAEngine
from src.core.models import ExportConfig, ExportFormat, DocumentType, QualityLevel
from src.core.config import ConfigManager
from src.core.task_manager import TaskManager
from src.core.quality_manager import QualityManager


class TestExportIAEngine:
    """Test the main ExportIAEngine."""
    
    @pytest.fixture
    async def engine(self):
        """Create a test engine instance."""
        engine = ExportIAEngine()
        await engine.initialize()
        yield engine
        await engine.shutdown()
    
    @pytest.fixture
    def sample_content(self):
        """Sample content for testing."""
        return {
            "title": "Test Document",
            "sections": [
                {
                    "heading": "Introduction",
                    "content": "This is a test document for unit testing."
                },
                {
                    "heading": "Conclusion",
                    "content": "This concludes the test document."
                }
            ]
        }
    
    @pytest.fixture
    def sample_config(self):
        """Sample export configuration."""
        return ExportConfig(
            format=ExportFormat.PDF,
            document_type=DocumentType.REPORT,
            quality_level=QualityLevel.PROFESSIONAL
        )
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert engine._initialized is True
    
    @pytest.mark.asyncio
    async def test_export_document(self, engine, sample_content, sample_config):
        """Test document export."""
        task_id = await engine.export_document(sample_content, sample_config)
        assert task_id is not None
        assert len(task_id) > 0
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, engine, sample_content, sample_config):
        """Test task status retrieval."""
        task_id = await engine.export_document(sample_content, sample_config)
        
        # Wait a bit for processing to start
        await asyncio.sleep(0.1)
        
        status = await engine.get_task_status(task_id)
        assert status is not None
        assert "status" in status
        assert "task_id" in status or "id" in status
    
    @pytest.mark.asyncio
    async def test_export_statistics(self, engine):
        """Test export statistics."""
        stats = engine.get_export_statistics()
        assert stats is not None
        assert hasattr(stats, 'total_tasks')
        assert hasattr(stats, 'active_tasks')
        assert hasattr(stats, 'completed_tasks')
    
    @pytest.mark.asyncio
    async def test_supported_formats(self, engine):
        """Test supported formats listing."""
        formats = engine.list_supported_formats()
        assert formats is not None
        assert len(formats) > 0
        
        # Check that all expected formats are present
        format_names = [fmt["format"] for fmt in formats]
        expected_formats = ["pdf", "docx", "html", "markdown", "rtf", "txt", "json", "xml"]
        for expected in expected_formats:
            assert expected in format_names
    
    @pytest.mark.asyncio
    async def test_quality_config(self, engine):
        """Test quality configuration retrieval."""
        config = engine.get_quality_config(QualityLevel.PROFESSIONAL)
        assert config is not None
        assert hasattr(config, 'font_family')
        assert hasattr(config, 'font_size')
    
    @pytest.mark.asyncio
    async def test_document_template(self, engine):
        """Test document template retrieval."""
        template = engine.get_document_template(DocumentType.REPORT)
        assert template is not None
        assert isinstance(template, dict)
    
    @pytest.mark.asyncio
    async def test_content_validation(self, engine, sample_content, sample_config):
        """Test content validation."""
        metrics = engine.validate_content(sample_content, sample_config)
        assert metrics is not None
        assert hasattr(metrics, 'overall_score')
        assert hasattr(metrics, 'formatting_score')
        assert hasattr(metrics, 'content_score')


class TestConfigManager:
    """Test the configuration manager."""
    
    def test_config_initialization(self):
        """Test configuration manager initialization."""
        config_manager = ConfigManager()
        assert config_manager is not None
        assert config_manager.system_config is not None
    
    def test_quality_config_retrieval(self):
        """Test quality configuration retrieval."""
        config_manager = ConfigManager()
        config = config_manager.get_quality_config(QualityLevel.PROFESSIONAL)
        assert config is not None
        assert config.font_family is not None
        assert config.font_size > 0
    
    def test_template_retrieval(self):
        """Test template retrieval."""
        config_manager = ConfigManager()
        template = config_manager.get_template(DocumentType.REPORT)
        assert template is not None
        assert isinstance(template, dict)
    
    def test_format_features(self):
        """Test format features retrieval."""
        config_manager = ConfigManager()
        features = config_manager.get_format_features(ExportFormat.PDF)
        assert features is not None
        assert isinstance(features, list)
        assert len(features) > 0


class TestTaskManager:
    """Test the task manager."""
    
    @pytest.fixture
    async def task_manager(self):
        """Create a test task manager."""
        from src.core.config import SystemConfig
        config = SystemConfig()
        manager = TaskManager(config)
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_task_submission(self, task_manager, sample_content, sample_config):
        """Test task submission."""
        task_id = await task_manager.submit_task(sample_content, sample_config)
        assert task_id is not None
        assert task_id in task_manager.active_tasks
    
    @pytest.mark.asyncio
    async def test_task_status(self, task_manager, sample_content, sample_config):
        """Test task status retrieval."""
        task_id = await task_manager.submit_task(sample_content, sample_config)
        status = await task_manager.get_task_status(task_id)
        assert status is not None
        assert "status" in status
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self, task_manager, sample_content, sample_config):
        """Test task cancellation."""
        task_id = await task_manager.submit_task(sample_content, sample_config)
        success = await task_manager.cancel_task(task_id)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_statistics(self, task_manager):
        """Test statistics generation."""
        stats = task_manager.get_statistics()
        assert stats is not None
        assert hasattr(stats, 'total_tasks')


class TestQualityManager:
    """Test the quality manager."""
    
    @pytest.fixture
    def quality_manager(self):
        """Create a test quality manager."""
        return QualityManager()
    
    def test_quality_initialization(self, quality_manager):
        """Test quality manager initialization."""
        assert quality_manager is not None
        assert quality_manager.quality_rules is not None
    
    @pytest.mark.asyncio
    async def test_content_processing(self, quality_manager, sample_content, sample_config):
        """Test content processing for quality."""
        processed = await quality_manager.process_content_for_quality(sample_content, sample_config)
        assert processed is not None
        assert "structure" in processed
        assert "formatting" in processed
    
    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, quality_manager, sample_config):
        """Test quality score calculation."""
        result = {"format": "pdf", "pages": 1}
        score = await quality_manager.calculate_quality_score(result, sample_config)
        assert score is not None
        assert 0.0 <= score <= 1.0
    
    def test_quality_metrics(self, quality_manager, sample_content, sample_config):
        """Test quality metrics generation."""
        metrics = quality_manager.get_quality_metrics(sample_content, sample_config)
        assert metrics is not None
        assert hasattr(metrics, 'overall_score')
        assert hasattr(metrics, 'formatting_score')
        assert hasattr(metrics, 'content_score')
        assert hasattr(metrics, 'accessibility_score')
        assert hasattr(metrics, 'professional_score')


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_complete_export_workflow(self):
        """Test complete export workflow."""
        content = {
            "title": "Integration Test Document",
            "sections": [
                {
                    "heading": "Test Section",
                    "content": "This is a test for the complete workflow."
                }
            ]
        }
        
        config = ExportConfig(
            format=ExportFormat.PDF,
            document_type=DocumentType.REPORT,
            quality_level=QualityLevel.PROFESSIONAL
        )
        
        async with ExportIAEngine() as engine:
            # Submit export task
            task_id = await engine.export_document(content, config)
            assert task_id is not None
            
            # Wait for completion
            max_wait = 30  # seconds
            wait_time = 0
            while wait_time < max_wait:
                status = await engine.get_task_status(task_id)
                if status["status"] == "completed":
                    assert status["file_path"] is not None
                    assert os.path.exists(status["file_path"])
                    break
                elif status["status"] == "failed":
                    pytest.fail(f"Export failed: {status.get('error', 'Unknown error')}")
                
                await asyncio.sleep(1)
                wait_time += 1
            
            if wait_time >= max_wait:
                pytest.fail("Export did not complete within timeout")
    
    @pytest.mark.asyncio
    async def test_multiple_format_export(self):
        """Test exporting to multiple formats."""
        content = {
            "title": "Multi-Format Test",
            "sections": [
                {
                    "heading": "Test",
                    "content": "Multi-format export test."
                }
            ]
        }
        
        formats = [ExportFormat.PDF, ExportFormat.DOCX, ExportFormat.HTML]
        
        async with ExportIAEngine() as engine:
            tasks = []
            
            # Create tasks for each format
            for fmt in formats:
                config = ExportConfig(
                    format=fmt,
                    document_type=DocumentType.REPORT,
                    quality_level=QualityLevel.PROFESSIONAL
                )
                task_id = await engine.export_document(content, config)
                tasks.append((task_id, fmt))
            
            # Wait for all tasks to complete
            for task_id, fmt in tasks:
                max_wait = 30
                wait_time = 0
                while wait_time < max_wait:
                    status = await engine.get_task_status(task_id)
                    if status["status"] == "completed":
                        assert status["file_path"] is not None
                        break
                    elif status["status"] == "failed":
                        pytest.fail(f"{fmt.value} export failed: {status.get('error', 'Unknown error')}")
                    
                    await asyncio.sleep(1)
                    wait_time += 1
                
                if wait_time >= max_wait:
                    pytest.fail(f"{fmt.value} export did not complete within timeout")


if __name__ == "__main__":
    pytest.main([__file__])




