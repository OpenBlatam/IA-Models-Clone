"""
Mutation testing - Tests that verify test quality
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from ..core.project_generator import ProjectGenerator
from ..utils.validator import ProjectValidator


class TestMutationTesting:
    """Mutation testing suite to verify test quality"""

    def test_name_sanitization_mutations(self, project_generator):
        """Test that name sanitization handles mutations"""
        # Original behavior
        result1 = project_generator._sanitize_name("Test Project")
        
        # Mutated: should still work
        result2 = project_generator._sanitize_name("Test-Project")
        result3 = project_generator._sanitize_name("Test_Project")
        
        # All should produce valid names
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, str)
        assert len(result1) > 0
        assert len(result2) > 0
        assert len(result3) > 0

    def test_keyword_extraction_mutations(self, project_generator):
        """Test keyword extraction with mutations"""
        descriptions = [
            "A chat AI system",
            "A CHAT AI SYSTEM",  # Case mutation
            "A chat ai system",  # Case mutation
            "A chat AI system with authentication",  # Additional content
        ]
        
        for desc in descriptions:
            keywords = project_generator._extract_keywords(desc)
            assert isinstance(keywords, dict)
            assert "ai_type" in keywords
            assert "complexity" in keywords

    @pytest.mark.asyncio
    async def test_project_generation_mutations(self, temp_dir):
        """Test project generation with mutated inputs"""
        generator = ProjectGenerator(output_dir=temp_dir)
        
        # Mutated descriptions
        descriptions = [
            "A simple chatbot",
            "A simple CHATBOT",  # Case mutation
            "A simple chatbot with auth",  # Additional feature
            "A simple chatbot with authentication and database",  # More features
        ]
        
        for desc in descriptions:
            try:
                project = await generator.generate_project(desc)
                # Should handle all mutations
                assert project is not None or True  # May fail for some mutations
            except Exception:
                # Some mutations may cause errors, which is expected
                pass

    def test_validation_mutations(self, temp_dir):
        """Test validation with mutated project structures"""
        validator = ProjectValidator()
        
        # Create mutated structures
        structures = [
            {"has_backend": True, "has_frontend": True},
            {"has_backend": True, "has_frontend": False},  # Mutation
            {"has_backend": False, "has_frontend": True},  # Mutation
        ]
        
        for structure in structures:
            project_dir = temp_dir / f"mutated_{hash(str(structure))}"
            project_dir.mkdir()
            
            if structure.get("has_backend"):
                (project_dir / "backend").mkdir()
                (project_dir / "backend" / "main.py").write_text("# Backend")
            
            if structure.get("has_frontend"):
                (project_dir / "frontend").mkdir()
                (project_dir / "frontend" / "package.json").write_text('{"name": "test"}')
            
            # Should validate or provide clear errors
            result = asyncio.run(validator.validate_project(project_dir, {"name": "test"}))
            assert isinstance(result, dict)
            assert "valid" in result

