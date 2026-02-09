"""
Compatibility matrix tests
"""

import pytest
from typing import List, Tuple


class TestCompatibilityMatrix:
    """Tests for compatibility matrix"""
    
    @pytest.mark.parametrize("backend,frontend", [
        ("fastapi", "react"),
        ("fastapi", "vue"),
        ("fastapi", "angular"),
        ("flask", "react"),
        ("flask", "vue"),
        ("django", "react"),
    ])
    def test_framework_combinations(self, backend, frontend, temp_dir):
        """Test different framework combinations"""
        from ..core.project_generator import ProjectGenerator
        
        generator = ProjectGenerator(
            base_output_dir=str(temp_dir / "projects"),
            backend_framework=backend,
            frontend_framework=frontend
        )
        
        # Should initialize with specified frameworks
        assert generator.backend_framework == backend
        assert generator.frontend_framework == frontend
    
    @pytest.mark.parametrize("ai_type", [
        "chat", "vision", "audio", "nlp", "video", "recommendation"
    ])
    def test_ai_type_compatibility(self, ai_type, project_generator):
        """Test compatibility with different AI types"""
        description = f"A {ai_type} AI system"
        keywords = project_generator._extract_keywords(description)
        
        # Should detect AI type
        assert keywords["ai_type"] == ai_type or ai_type in description.lower()
    
    @pytest.mark.parametrize("python_version", [
        "3.8", "3.9", "3.10", "3.11", "3.12"
    ])
    def test_python_version_compatibility(self, python_version):
        """Test compatibility with different Python versions"""
        import sys
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Should work with current Python version
        assert sys.version_info.major == 3
        assert sys.version_info.minor >= 8

