"""
Tests for PDF Variant Generator
================================
Comprehensive tests for variant generation functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import io
from datetime import datetime

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from variant_generator import (
    PDFVariantGenerator,
    VariantType,
    VariantOptions
)


@pytest.fixture
def generator():
    """Create variant generator instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield PDFVariantGenerator(Path(temp_dir))


@pytest.fixture
def sample_pdf_file():
    """Create sample PDF file object."""
    pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"
    return io.BytesIO(pdf_content)


@pytest.fixture
def default_options():
    """Default variant options."""
    return VariantOptions()


class TestVariantGeneratorInitialization:
    """Tests for generator initialization."""
    
    def test_init_with_custom_dir(self):
        """Test initialization with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = PDFVariantGenerator(Path(temp_dir))
            assert gen.upload_dir == Path(temp_dir)
            assert gen.upload_dir.exists()
    
    def test_init_with_default_dir(self):
        """Test initialization with default directory."""
        gen = PDFVariantGenerator()
        assert gen.upload_dir.exists()


class TestVariantOptions:
    """Tests for VariantOptions."""
    
    def test_default_options(self):
        """Test default options."""
        options = VariantOptions()
        assert options.max_length is None
        assert options.style == "academic"
        assert options.include_images is True
        assert options.include_tables is True
        assert options.language == "en"
        assert options.tone == "neutral"
    
    def test_custom_options(self):
        """Test custom options."""
        options = VariantOptions(
            max_length=500,
            style="casual",
            include_images=False,
            language="es"
        )
        assert options.max_length == 500
        assert options.style == "casual"
        assert options.include_images is False
        assert options.language == "es"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        options = VariantOptions(max_length=100, style="formal")
        result = options.to_dict()
        assert isinstance(result, dict)
        assert result["max_length"] == 100
        assert result["style"] == "formal"
        assert "include_images" in result
        assert "language" in result


class TestVariantGeneration:
    """Tests for variant generation."""
    
    @pytest.mark.asyncio
    async def test_generate_summary(self, generator, sample_pdf_file, default_options):
        """Test summary generation."""
        result = await generator.generate(
            file=sample_pdf_file,
            variant_type=VariantType.SUMMARY,
            options=default_options
        )
        
        assert isinstance(result, dict)
        assert result["variant_type"] == "summary"
        assert "generated_at" in result
        assert "options" in result
    
    @pytest.mark.asyncio
    async def test_generate_outline(self, generator, sample_pdf_file, default_options):
        """Test outline generation."""
        result = await generator.generate(
            file=sample_pdf_file,
            variant_type=VariantType.OUTLINE,
            options=default_options
        )
        
        assert isinstance(result, dict)
        assert result["variant_type"] == "outline"
    
    @pytest.mark.asyncio
    async def test_generate_highlights(self, generator, sample_pdf_file, default_options):
        """Test highlights generation."""
        result = await generator.generate(
            file=sample_pdf_file,
            variant_type=VariantType.HIGHLIGHTS,
            options=default_options
        )
        
        assert isinstance(result, dict)
        assert result["variant_type"] == "highlights"
    
    @pytest.mark.asyncio
    async def test_generate_notes(self, generator, sample_pdf_file, default_options):
        """Test notes generation."""
        result = await generator.generate(
            file=sample_pdf_file,
            variant_type=VariantType.NOTES,
            options=default_options
        )
        
        assert isinstance(result, dict)
        assert result["variant_type"] == "notes"
    
    @pytest.mark.asyncio
    async def test_generate_quiz(self, generator, sample_pdf_file, default_options):
        """Test quiz generation."""
        result = await generator.generate(
            file=sample_pdf_file,
            variant_type=VariantType.QUIZ,
            options=default_options
        )
        
        assert isinstance(result, dict)
        assert result["variant_type"] == "quiz"
    
    @pytest.mark.asyncio
    async def test_generate_presentation(self, generator, sample_pdf_file, default_options):
        """Test presentation generation."""
        result = await generator.generate(
            file=sample_pdf_file,
            variant_type=VariantType.PRESENTATION,
            options=default_options
        )
        
        assert isinstance(result, dict)
        assert result["variant_type"] == "presentation"
    
    @pytest.mark.asyncio
    async def test_generate_with_default_options(self, generator, sample_pdf_file):
        """Test generation with default options (None)."""
        result = await generator.generate(
            file=sample_pdf_file,
            variant_type=VariantType.SUMMARY,
            options=None
        )
        
        assert isinstance(result, dict)
        assert "options" in result
    
    @pytest.mark.asyncio
    async def test_generate_unknown_variant_type(self, generator, sample_pdf_file):
        """Test generation with unknown variant type."""
        with pytest.raises(ValueError, match="Unknown variant type"):
            await generator.generate(
                file=sample_pdf_file,
                variant_type="unknown_type",
                options=None
            )
    
    @pytest.mark.asyncio
    async def test_generate_includes_timestamp(self, generator, sample_pdf_file, default_options):
        """Test that generated variant includes timestamp."""
        result = await generator.generate(
            file=sample_pdf_file,
            variant_type=VariantType.SUMMARY,
            options=default_options
        )
        
        assert "generated_at" in result
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(result["generated_at"])


class TestVariantTypeEnum:
    """Tests for VariantType enum."""
    
    def test_variant_type_values(self):
        """Test all variant type values."""
        assert VariantType.SUMMARY.value == "summary"
        assert VariantType.OUTLINE.value == "outline"
        assert VariantType.HIGHLIGHTS.value == "highlights"
        assert VariantType.NOTES.value == "notes"
        assert VariantType.QUIZ.value == "quiz"
        assert VariantType.PRESENTATION.value == "presentation"
        assert VariantType.TRANSLATED.value == "translated"
        assert VariantType.ABRIDGED.value == "abridged"
        assert VariantType.EXPANDED.value == "expanded"


class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_generate_with_invalid_file(self, generator, default_options):
        """Test generation with invalid file."""
        invalid_file = io.BytesIO(b"not a pdf")
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            await generator.generate(
                file=invalid_file,
                variant_type=VariantType.SUMMARY,
                options=default_options
            )
    
    @pytest.mark.asyncio
    async def test_generate_with_empty_file(self, generator, default_options):
        """Test generation with empty file."""
        empty_file = io.BytesIO(b"")
        
        with pytest.raises(Exception):
            await generator.generate(
                file=empty_file,
                variant_type=VariantType.SUMMARY,
                options=default_options
            )


class TestVariantOptionsIntegration:
    """Tests for options integration in generation."""
    
    @pytest.mark.asyncio
    async def test_generate_with_custom_max_length(self, generator, sample_pdf_file):
        """Test generation with custom max_length."""
        options = VariantOptions(max_length=200)
        result = await generator.generate(
            file=sample_pdf_file,
            variant_type=VariantType.SUMMARY,
            options=options
        )
        
        assert result["options"]["max_length"] == 200
    
    @pytest.mark.asyncio
    async def test_generate_with_custom_style(self, generator, sample_pdf_file):
        """Test generation with custom style."""
        options = VariantOptions(style="casual")
        result = await generator.generate(
            file=sample_pdf_file,
            variant_type=VariantType.SUMMARY,
            options=options
        )
        
        assert result["options"]["style"] == "casual"
    
    @pytest.mark.asyncio
    async def test_generate_with_custom_language(self, generator, sample_pdf_file):
        """Test generation with custom language."""
        options = VariantOptions(language="es")
        result = await generator.generate(
            file=sample_pdf_file,
            variant_type=VariantType.SUMMARY,
            options=options
        )
        
        assert result["options"]["language"] == "es"

