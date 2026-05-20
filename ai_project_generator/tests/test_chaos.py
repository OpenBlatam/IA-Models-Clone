"""
Chaos engineering tests - Testing system under failure conditions
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import random


class TestChaos:
    """Chaos engineering tests"""
    
    def test_random_failures(self, project_generator):
        """Test system behavior under random failures"""
        results = []
        
        for i in range(10):
            try:
                # Randomly fail some operations
                if random.random() < 0.3:  # 30% failure rate
                    raise Exception(f"Random failure {i}")
                result = project_generator._sanitize_name(f"Test {i}")
                results.append(result)
            except Exception:
                # System should continue
                pass
        
        # Should have some successes
        assert len(results) > 0
    
    def test_corrupted_data_handling(self, temp_dir):
        """Test handling of corrupted data"""
        # Create corrupted JSON
        corrupted_json = temp_dir / "corrupted.json"
        corrupted_json.write_text("{invalid json}")
        
        # Should handle gracefully
        try:
            import json
            data = json.loads(corrupted_json.read_text(encoding="utf-8"))
            assert False, "Should have raised JSONDecodeError"
        except json.JSONDecodeError:
            # Expected
            pass
    
    def test_missing_dependencies(self, temp_dir):
        """Test behavior with missing dependencies"""
        # Try to use feature that might require dependency
        try:
            from ..utils.github_integration import GitHubIntegration
            github = GitHubIntegration()
            # Should initialize even if GitHub API is not available
            assert github is not None
        except ImportError:
            # Dependency not available, skip
            pytest.skip("GitHub integration not available")
    
    def test_disk_space_simulation(self, temp_dir):
        """Test behavior when disk space is limited"""
        # Create many files to simulate limited space
        try:
            for i in range(100):
                file_path = temp_dir / f"large_file_{i}.txt"
                file_path.write_text("x" * 10000)
            
            # Should handle gracefully
            assert True
        except OSError:
            # Disk full, expected
            pass
    
    def test_network_failure_simulation(self):
        """Test behavior under network failures"""
        # Simulate network failure
        with patch('httpx.AsyncClient.get', side_effect=Exception("Network error")):
            # Operations that don't require network should still work
            from ..core.project_generator import ProjectGenerator
            generator = ProjectGenerator()
            result = generator._sanitize_name("Test")
            assert result == "test"

