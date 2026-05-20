"""
Scalability tests
"""

import pytest
import asyncio
from pathlib import Path
from typing import List


class TestScalability:
    """Tests for scalability"""
    
    @pytest.mark.slow
    def test_scale_to_many_projects(self, project_generator, temp_dir):
        """Test scaling to many projects"""
        num_projects = 50
        
        projects = []
        for i in range(num_projects):
            project = project_generator.generate_project(f"Project {i}")
            projects.append(project)
        
        assert len(projects) == num_projects
        # All should have unique IDs
        project_ids = [p["project_id"] for p in projects if p]
        assert len(set(project_ids)) == len(project_ids)
    
    @pytest.mark.async
    @pytest.mark.slow
    async def test_scale_concurrent_operations(self, temp_dir):
        """Test scaling with concurrent operations"""
        from ..utils.cache_manager import CacheManager
        
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        num_operations = 200
        
        async def cache_operation(i):
            await cache.cache_project(f"Project {i}", {}, {"id": f"proj-{i}"})
            return await cache.get_cached_project(f"Project {i}", {})
        
        # Run many concurrent operations
        tasks = [cache_operation(i) for i in range(num_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle scale
        assert len(results) == num_operations
        successes = [r for r in results if r is not None and not isinstance(r, Exception)]
        assert len(successes) >= num_operations * 0.9  # 90% success rate
    
    def test_scale_large_files(self, temp_dir):
        """Test handling large files"""
        large_content = "x" * (10 * 1024 * 1024)  # 10MB
        
        large_file = temp_dir / "large_file.txt"
        large_file.write_text(large_content)
        
        # Should handle large file
        assert large_file.exists()
        assert large_file.stat().st_size == len(large_content)
        
        # Should be able to read
        content = large_file.read_text(encoding="utf-8")
        assert len(content) == len(large_content)
    
    def test_scale_many_files(self, temp_dir):
        """Test handling many files"""
        num_files = 1000
        
        for i in range(num_files):
            file_path = temp_dir / f"file_{i}.txt"
            file_path.write_text(f"Content {i}")
        
        # Should handle many files
        files = list(temp_dir.glob("file_*.txt"))
        assert len(files) == num_files

