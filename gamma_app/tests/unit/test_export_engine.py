"""
Gamma App - Export Engine Unit Tests
"""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from engines.export_engine import AdvancedExportEngine, ExportConfig, ExportFormat

class TestAdvancedExportEngine:
    """Unit tests for Advanced Export Engine"""
    
    @pytest.fixture
    def export_engine(self):
        """Create export engine instance for testing"""
        return AdvancedExportEngine()
    
    @pytest.fixture
    def export_config(self):
        """Create export configuration for testing"""
        return ExportConfig(
            format=ExportFormat.PDF,
            quality="high",
            include_metadata=True,
            watermark="Gamma App",
            template="professional"
        )
    
    @pytest.fixture
    def sample_content(self):
        """Create sample content for testing"""
        return {
            "title": "Test Document",
            "content": [
                {"type": "heading", "text": "Introduction", "level": 1},
                {"type": "paragraph", "text": "This is a test document."},
                {"type": "list", "items": ["Item 1", "Item 2", "Item 3"]},
                {"type": "image", "url": "https://example.com/image.jpg", "alt": "Test Image"}
            ],
            "metadata": {
                "author": "Test Author",
                "created_at": "2024-01-01T00:00:00Z",
                "tags": ["test", "document"]
            }
        }
    
    def test_export_engine_initialization(self, export_engine):
        """Test export engine initialization"""
        assert export_engine is not None
        assert export_engine.supported_formats is not None
        assert len(export_engine.supported_formats) > 0
    
    def test_get_supported_formats(self, export_engine):
        """Test getting supported export formats"""
        formats = export_engine.get_supported_formats()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        
        # Check that all formats have required fields
        for format_info in formats:
            assert "format" in format_info
            assert "name" in format_info
            assert "description" in format_info
            assert "supported_quality" in format_info
    
    def test_validate_export_config(self, export_engine, export_config):
        """Test export configuration validation"""
        result = export_engine.validate_export_config(export_config)
        assert result is True
    
    def test_validate_invalid_export_config(self, export_engine):
        """Test validation of invalid export configuration"""
        invalid_config = ExportConfig(
            format="invalid_format",
            quality="invalid_quality"
        )
        
        result = export_engine.validate_export_config(invalid_config)
        assert result is False
    
    @patch('engines.export_engine.weasyprint.HTML')
    def test_export_to_pdf(self, mock_html, export_engine, sample_content, export_config):
        """Test PDF export"""
        # Mock weasyprint
        mock_html_instance = MagicMock()
        mock_html_instance.write_pdf.return_value = b"PDF content"
        mock_html.return_value = mock_html_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.pdf")
            
            result = export_engine.export_to_pdf(
                content=sample_content,
                output_path=output_path,
                config=export_config
            )
            
            assert result is True
            assert os.path.exists(output_path)
            mock_html.assert_called_once()
            mock_html_instance.write_pdf.assert_called_once()
    
    @patch('engines.export_engine.Presentation')
    def test_export_to_pptx(self, mock_presentation, export_engine, sample_content, export_config):
        """Test PPTX export"""
        # Mock python-pptx
        mock_presentation_instance = MagicMock()
        mock_slide = MagicMock()
        mock_presentation_instance.slides.add_slide.return_value = mock_slide
        mock_presentation.return_value = mock_presentation_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.pptx")
            
            result = export_engine.export_to_pptx(
                content=sample_content,
                output_path=output_path,
                config=export_config
            )
            
            assert result is True
            assert os.path.exists(output_path)
            mock_presentation.assert_called_once()
    
    def test_export_to_html(self, export_engine, sample_content, export_config):
        """Test HTML export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.html")
            
            result = export_engine.export_to_html(
                content=sample_content,
                output_path=output_path,
                config=export_config
            )
            
            assert result is True
            assert os.path.exists(output_path)
            
            # Check that HTML content was written
            with open(output_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                assert "Test Document" in html_content
                assert "Introduction" in html_content
    
    def test_export_to_markdown(self, export_engine, sample_content, export_config):
        """Test Markdown export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.md")
            
            result = export_engine.export_to_markdown(
                content=sample_content,
                output_path=output_path,
                config=export_config
            )
            
            assert result is True
            assert os.path.exists(output_path)
            
            # Check that Markdown content was written
            with open(output_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
                assert "# Test Document" in markdown_content
                assert "## Introduction" in markdown_content
    
    def test_export_to_json(self, export_engine, sample_content, export_config):
        """Test JSON export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.json")
            
            result = export_engine.export_to_json(
                content=sample_content,
                output_path=output_path,
                config=export_config
            )
            
            assert result is True
            assert os.path.exists(output_path)
            
            # Check that JSON content was written
            import json
            with open(output_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
                assert json_content["title"] == "Test Document"
                assert "content" in json_content
                assert "metadata" in json_content
    
    @patch('engines.export_engine.PIL.Image')
    def test_export_to_png(self, mock_image, export_engine, sample_content, export_config):
        """Test PNG export"""
        # Mock PIL Image
        mock_image_instance = MagicMock()
        mock_image.new.return_value = mock_image_instance
        mock_image.return_value = mock_image_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.png")
            
            result = export_engine.export_to_png(
                content=sample_content,
                output_path=output_path,
                config=export_config
            )
            
            assert result is True
            assert os.path.exists(output_path)
            mock_image.new.assert_called_once()
            mock_image_instance.save.assert_called_once()
    
    def test_export_to_zip(self, export_engine, sample_content, export_config):
        """Test ZIP export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.zip")
            
            result = export_engine.export_to_zip(
                content=sample_content,
                output_path=output_path,
                config=export_config
            )
            
            assert result is True
            assert os.path.exists(output_path)
            
            # Check that ZIP file was created and contains files
            import zipfile
            with zipfile.ZipFile(output_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                assert len(file_list) > 0
    
    def test_batch_export(self, export_engine, sample_content):
        """Test batch export functionality"""
        export_configs = [
            ExportConfig(format=ExportFormat.PDF, quality="high"),
            ExportConfig(format=ExportFormat.HTML, quality="medium"),
            ExportConfig(format=ExportFormat.MARKDOWN, quality="low")
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = export_engine.batch_export(
                content=sample_content,
                output_dir=temp_dir,
                configs=export_configs
            )
            
            assert len(results) == 3
            assert all(result["success"] for result in results)
            
            # Check that all files were created
            for result in results:
                assert os.path.exists(result["output_path"])
    
    def test_export_with_template(self, export_engine, sample_content, export_config):
        """Test export with custom template"""
        export_config.template = "custom_template"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.html")
            
            result = export_engine.export_to_html(
                content=sample_content,
                output_path=output_path,
                config=export_config
            )
            
            assert result is True
            assert os.path.exists(output_path)
    
    def test_export_with_watermark(self, export_engine, sample_content, export_config):
        """Test export with watermark"""
        export_config.watermark = "Confidential"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.pdf")
            
            result = export_engine.export_to_pdf(
                content=sample_content,
                output_path=output_path,
                config=export_config
            )
            
            assert result is True
            assert os.path.exists(output_path)
    
    def test_export_quality_settings(self, export_engine, sample_content):
        """Test different quality settings"""
        quality_configs = [
            ExportConfig(format=ExportFormat.PDF, quality="low"),
            ExportConfig(format=ExportConfig(format=ExportFormat.PDF, quality="medium"),
            ExportConfig(format=ExportFormat.PDF, quality="high"),
            ExportConfig(format=ExportFormat.PDF, quality="ultra")
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, config in enumerate(quality_configs):
                output_path = os.path.join(temp_dir, f"test_{i}.pdf")
                
                result = export_engine.export_to_pdf(
                    content=sample_content,
                    output_path=output_path,
                    config=config
                )
                
                assert result is True
                assert os.path.exists(output_path)
    
    def test_export_error_handling(self, export_engine, sample_content, export_config):
        """Test export error handling"""
        # Test with invalid output path
        invalid_path = "/invalid/path/test.pdf"
        
        result = export_engine.export_to_pdf(
            content=sample_content,
            output_path=invalid_path,
            config=export_config
        )
        
        assert result is False
    
    def test_get_export_statistics(self, export_engine):
        """Test getting export statistics"""
        stats = export_engine.get_export_statistics()
        
        assert isinstance(stats, dict)
        assert "total_exports" in stats
        assert "formats_used" in stats
        assert "average_export_time" in stats
        assert "success_rate" in stats
    
    def test_cleanup_temp_files(self, export_engine):
        """Test cleanup of temporary files"""
        # Create some temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "temp_file.txt")
            with open(temp_file, 'w') as f:
                f.write("temp content")
            
            # Cleanup should not affect existing files
            export_engine.cleanup_temp_files()
            
            # File should still exist
            assert os.path.exists(temp_file)

























