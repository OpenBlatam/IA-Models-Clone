"""
Benchmark tests for performance comparison
"""

import pytest
import time
from pathlib import Path
from typing import Dict, Any


class TestBenchmarks:
    """Benchmark tests for performance comparison"""
    
    @pytest.mark.benchmark
    def test_name_sanitization_benchmark(self, project_generator):
        """Benchmark name sanitization performance"""
        names = ["Test Project Name"] * 1000
        
        start = time.time()
        for name in names:
            project_generator._sanitize_name(name)
        elapsed = time.time() - start
        
        # Should be very fast
        assert elapsed < 1.0  # Less than 1 second for 1000 operations
        print(f"Name sanitization: {elapsed:.4f}s for 1000 operations")
    
    @pytest.mark.benchmark
    def test_keyword_extraction_benchmark(self, project_generator):
        """Benchmark keyword extraction performance"""
        descriptions = [
            "A chat AI system with authentication and database",
            "A vision AI system with object detection",
            "An audio AI system with real-time processing"
        ] * 100
        
        start = time.time()
        for desc in descriptions:
            project_generator._extract_keywords(desc)
        elapsed = time.time() - start
        
        # Should be reasonably fast
        assert elapsed < 5.0  # Less than 5 seconds for 300 operations
        print(f"Keyword extraction: {elapsed:.4f}s for 300 operations")
    
    @pytest.mark.benchmark
    def test_file_operations_benchmark(self, temp_dir):
        """Benchmark file operations"""
        num_files = 100
        
        start = time.time()
        for i in range(num_files):
            file_path = temp_dir / f"test_{i}.txt"
            file_path.write_text(f"Content {i}")
        write_time = time.time() - start
        
        start = time.time()
        for i in range(num_files):
            file_path = temp_dir / f"test_{i}.txt"
            content = file_path.read_text(encoding="utf-8")
        read_time = time.time() - start
        
        # Should be fast
        assert write_time < 2.0
        assert read_time < 2.0
        print(f"File operations: write={write_time:.4f}s, read={read_time:.4f}s")

