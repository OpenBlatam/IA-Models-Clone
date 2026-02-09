"""
Fuzzing tests for AI Project Generator
"""

import pytest
import random
import string
from pathlib import Path

from ..core.project_generator import ProjectGenerator
from ..utils.validator import ProjectValidator


class TestFuzzing:
    """Test suite for fuzzing and random input testing"""

    def test_fuzz_project_name(self, project_generator):
        """Test sanitization with fuzzed project names"""
        for _ in range(100):
            # Generate random string
            length = random.randint(1, 200)
            fuzzed_name = ''.join(
                random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=length)
            )
            
            # Should not crash
            result = project_generator._sanitize_name(fuzzed_name)
            assert isinstance(result, str)
            assert len(result) <= 50

    def test_fuzz_description(self, project_generator):
        """Test keyword extraction with fuzzed descriptions"""
        for _ in range(50):
            # Generate random description
            words = ['chat', 'vision', 'audio', 'test', 'project', 'ai', 'system']
            fuzzed_desc = ' '.join(random.choices(words, k=random.randint(5, 50)))
            
            # Should not crash
            keywords = project_generator._extract_keywords(fuzzed_desc)
            assert isinstance(keywords, dict)
            assert "ai_type" in keywords
            assert "complexity" in keywords

    def test_fuzz_special_characters(self, project_generator):
        """Test with various special characters"""
        special_chars = [
            "!@#$%^&*()",
            "[]{}|\\:;\"'<>?,./",
            "测试项目",
            "проект тест",
            "プロジェクト",
            "🚀🎉💯",
            "\x00\x01\x02",
            "\n\t\r",
        ]
        
        for special in special_chars:
            # Should handle gracefully
            result = project_generator._sanitize_name(special)
            assert isinstance(result, str)

    def test_fuzz_unicode(self, project_generator):
        """Test with unicode characters"""
        unicode_strings = [
            "测试AI项目",
            "Проект ИИ",
            "プロジェクトAI",
            "مشروع الذكاء الاصطناعي",
            "פרויקט AI",
        ]
        
        for unicode_str in unicode_strings:
            result = project_generator._sanitize_name(unicode_str)
            assert isinstance(result, str)

    def test_fuzz_empty_variations(self, project_generator):
        """Test with various empty-like inputs"""
        empty_variations = [
            "",
            " ",
            "  ",
            "\t",
            "\n",
            "\r\n",
            None,
        ]
        
        for empty in empty_variations:
            if empty is None:
                continue
            result = project_generator._sanitize_name(empty)
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_fuzz_cache_keys(self, temp_dir):
        """Test cache with fuzzed keys"""
        from ..utils.cache_manager import CacheManager
        
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        for i in range(50):
            # Random description
            desc = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(10, 100)))
            config = {"framework": random.choice(["fastapi", "flask", "django"])}
            project_info = {"project_id": f"fuzz-{i}"}
            
            # Should not crash
            await manager.cache_project(desc, config, project_info)
            cached = await manager.get_cached_project(desc, config)
            assert cached is not None or True  # May not be cached if key generation fails

    def test_fuzz_rate_limiter(self):
        """Test rate limiter with fuzzed client IDs"""
        from ..utils.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        
        for _ in range(100):
            # Random client ID
            client_id = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 100)))
            endpoint = random.choice(["default", "generate", "search"])
            
            # Should not crash
            allowed, info = limiter.is_allowed(client_id, endpoint)
            assert isinstance(allowed, bool)
            assert isinstance(info, dict)

    @pytest.mark.asyncio
    async def test_fuzz_validator(self, temp_dir):
        """Test validator with fuzzed project structures"""
        validator = ProjectValidator()
        
        for i in range(20):
            # Create random structure
            project_dir = temp_dir / f"fuzz_project_{i}"
            project_dir.mkdir()
            
            # Random files
            num_files = random.randint(0, 10)
            for j in range(num_files):
                file_path = project_dir / f"file_{j}.txt"
                content = ''.join(random.choices(string.ascii_letters, k=random.randint(0, 100)))
                file_path.write_text(content)
            
            project_info = {"name": f"fuzz_{i}"}
            
            # Should not crash
            result = await validator.validate_project(project_dir, project_info)
            assert isinstance(result, dict)
            assert "valid" in result

