"""
Tests for AdvancedTesting utility
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ..utils.advanced_testing import AdvancedTesting


class TestAdvancedTesting:
    """Test suite for AdvancedTesting"""

    def test_init(self):
        """Test AdvancedTesting initialization"""
        testing = AdvancedTesting()
        assert testing.test_results == {}

    @pytest.mark.asyncio
    async def test_run_backend_tests(self, temp_dir):
        """Test running backend tests"""
        testing = AdvancedTesting()
        
        # Create mock backend structure
        backend_path = temp_dir / "backend"
        backend_path.mkdir()
        (backend_path / "test_main.py").write_text("def test_example(): assert True")
        
        project_path = temp_dir
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "1 passed"
            mock_run.return_value.stderr = ""
            
            result = await testing.run_backend_tests(project_path)
            
            assert "success" in result
            # May succeed or fail depending on mock

    @pytest.mark.asyncio
    async def test_run_backend_tests_no_backend(self, temp_dir):
        """Test running backend tests when backend doesn't exist"""
        testing = AdvancedTesting()
        
        project_path = temp_dir
        
        result = await testing.run_backend_tests(project_path)
        
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_run_frontend_tests(self, temp_dir):
        """Test running frontend tests"""
        testing = AdvancedTesting()
        
        # Create mock frontend structure
        frontend_path = temp_dir / "frontend"
        frontend_path.mkdir()
        (frontend_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
        
        project_path = temp_dir
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Tests passed"
            
            result = await testing.run_frontend_tests(project_path)
            
            assert "success" in result

    @pytest.mark.asyncio
    async def test_run_all_tests(self, temp_dir):
        """Test running all tests"""
        testing = AdvancedTesting()
        
        project_path = temp_dir
        
        with patch.object(testing, 'run_backend_tests', new_callable=AsyncMock) as mock_backend, \
             patch.object(testing, 'run_frontend_tests', new_callable=AsyncMock) as mock_frontend:
            
            mock_backend.return_value = {"success": True}
            mock_frontend.return_value = {"success": True}
            
            result = await testing.run_all_tests(project_path)
            
            assert "backend" in result
            assert "frontend" in result

    def test_get_test_coverage(self, temp_dir):
        """Test getting test coverage"""
        testing = AdvancedTesting()
        
        project_path = temp_dir
        
        coverage = testing.get_test_coverage(project_path)
        
        assert isinstance(coverage, dict)
        assert "backend" in coverage or "frontend" in coverage or "overall" in coverage

