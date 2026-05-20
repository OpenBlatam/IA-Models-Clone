"""
Internationalization (i18n) tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any


class TestI18n:
    """Tests for internationalization support"""
    
    def test_multilingual_descriptions(self, project_generator):
        """Test handling of multilingual project descriptions"""
        descriptions = [
            "Un proyecto de IA en español",
            "Un projet d'IA en français",
            "Ein KI-Projekt auf Deutsch",
            "Un progetto IA in italiano",
            "Um projeto de IA em português",
            "AIプロジェクト（日本語）",
            "AI项目（中文）",
        ]
        
        for desc in descriptions:
            project = project_generator.generate_project(desc)
            assert project is not None
            assert "project_id" in project
    
    def test_special_characters_handling(self, project_generator):
        """Test handling of special characters in names"""
        special_names = [
            "Proyecto-Ñoño",
            "Projet-Élève",
            "Projekt-Über",
            "Progetto-Àà",
            "プロジェクト-テスト",
            "项目-测试",
        ]
        
        for name in special_names:
            sanitized = project_generator._sanitize_name(name)
            assert isinstance(sanitized, str)
            assert len(sanitized) > 0
    
    def test_unicode_support(self, temp_dir):
        """Test Unicode support in file operations"""
        unicode_content = "测试内容\nTest content\nContenu de test\nテスト内容"
        
        test_file = temp_dir / "unicode_test.txt"
        test_file.write_text(unicode_content, encoding="utf-8")
        
        # Should read correctly
        content = test_file.read_text(encoding="utf-8")
        assert content == unicode_content
    
    def test_locale_aware_formatting(self, temp_dir):
        """Test locale-aware formatting"""
        from datetime import datetime
        
        # Test date formatting
        date = datetime(2024, 1, 15, 10, 30, 0)
        
        # Should format without errors
        formatted = date.strftime("%Y-%m-%d %H:%M:%S")
        assert "2024" in formatted
    
    def test_rtl_language_support(self, project_generator):
        """Test right-to-left language support"""
        rtl_descriptions = [
            "مشروع ذكاء اصطناعي",  # Arabic
            "פרויקט בינה מלאכותית",  # Hebrew
        ]
        
        for desc in rtl_descriptions:
            project = project_generator.generate_project(desc)
            assert project is not None

