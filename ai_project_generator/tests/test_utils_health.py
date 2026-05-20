"""
Tests for AdvancedHealthChecker utility
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ..utils.health_checker import AdvancedHealthChecker


class TestAdvancedHealthChecker:
    """Test suite for AdvancedHealthChecker"""

    def test_init(self):
        """Test AdvancedHealthChecker initialization"""
        checker = AdvancedHealthChecker()
        assert checker is not None

    @pytest.mark.asyncio
    async def test_check_health(self, temp_dir):
        """Test complete health check"""
        checker = AdvancedHealthChecker()
        
        with patch.object(checker, '_check_filesystem', new_callable=AsyncMock) as mock_fs, \
             patch.object(checker, '_check_memory', new_callable=AsyncMock) as mock_mem, \
             patch.object(checker, '_check_disk', new_callable=AsyncMock) as mock_disk:
            
            mock_fs.return_value = {"status": "healthy", "message": "OK"}
            mock_mem.return_value = {"status": "healthy", "percent_used": 50}
            mock_disk.return_value = {"status": "healthy", "percent_used": 60}
            
            health = await checker.check_health()
            
            assert "status" in health
            assert "timestamp" in health
            assert "checks" in health
            assert health["status"] == "healthy"
            assert "filesystem" in health["checks"]
            assert "memory" in health["checks"]
            assert "disk" in health["checks"]

    @pytest.mark.asyncio
    async def test_check_health_degraded(self, temp_dir):
        """Test health check with degraded status"""
        checker = AdvancedHealthChecker()
        
        with patch.object(checker, '_check_filesystem', new_callable=AsyncMock) as mock_fs, \
             patch.object(checker, '_check_memory', new_callable=AsyncMock) as mock_mem, \
             patch.object(checker, '_check_disk', new_callable=AsyncMock) as mock_disk:
            
            mock_fs.return_value = {"status": "healthy", "message": "OK"}
            mock_mem.return_value = {"status": "unhealthy", "message": "High memory"}
            mock_disk.return_value = {"status": "healthy", "percent_used": 60}
            
            health = await checker.check_health()
            
            assert health["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_check_filesystem(self, temp_dir):
        """Test filesystem check"""
        checker = AdvancedHealthChecker()
        
        # Use temp_dir instead of /tmp for Windows compatibility
        with patch('pathlib.Path') as mock_path:
            mock_file = Mock()
            mock_file.write_text = Mock()
            mock_file.unlink = Mock()
            mock_path.return_value = mock_file
            
            result = await checker._check_filesystem()
            
            assert "status" in result
            assert "message" in result

    @pytest.mark.asyncio
    async def test_check_filesystem_error(self, temp_dir):
        """Test filesystem check with error"""
        checker = AdvancedHealthChecker()
        
        with patch('pathlib.Path') as mock_path:
            mock_file = Mock()
            mock_file.write_text.side_effect = Exception("Permission denied")
            mock_path.return_value = mock_file
            
            result = await checker._check_filesystem()
            
            assert result["status"] == "unhealthy"
            assert "error" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_check_memory(self):
        """Test memory check"""
        checker = AdvancedHealthChecker()
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_mem = Mock()
            mock_mem.percent = 50.0
            mock_mem.available = 4 * 1024 * 1024 * 1024  # 4GB
            mock_memory.return_value = mock_mem
            
            result = await checker._check_memory()
            
            assert "status" in result
            assert "percent_used" in result
            assert result["percent_used"] == 50.0

    @pytest.mark.asyncio
    async def test_check_memory_high_usage(self):
        """Test memory check with high usage"""
        checker = AdvancedHealthChecker()
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_mem = Mock()
            mock_mem.percent = 95.0
            mock_mem.available = 512 * 1024 * 1024  # 512MB
            mock_memory.return_value = mock_mem
            
            result = await checker._check_memory()
            
            assert result["status"] == "warning"

    @pytest.mark.asyncio
    async def test_check_memory_no_psutil(self):
        """Test memory check without psutil"""
        checker = AdvancedHealthChecker()
        
        with patch('builtins.__import__', side_effect=ImportError):
            result = await checker._check_memory()
            
            assert result["status"] == "unknown"
            assert "psutil" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_check_disk(self):
        """Test disk check"""
        checker = AdvancedHealthChecker()
        
        with patch('shutil.disk_usage') as mock_disk:
            mock_disk.return_value = (1000, 400, 600)  # total, used, free (GB)
            
            result = await checker._check_disk()
            
            assert "status" in result
            assert "percent_used" in result
            assert result["percent_used"] == 40.0

    @pytest.mark.asyncio
    async def test_check_disk_high_usage(self):
        """Test disk check with high usage"""
        checker = AdvancedHealthChecker()
        
        with patch('shutil.disk_usage') as mock_disk:
            mock_disk.return_value = (1000, 950, 50)  # 95% used
            
            result = await checker._check_disk()
            
            assert result["status"] == "warning"

    @pytest.mark.asyncio
    async def test_check_disk_error(self):
        """Test disk check with error"""
        checker = AdvancedHealthChecker()
        
        with patch('shutil.disk_usage', side_effect=Exception("Disk error")):
            result = await checker._check_disk()
            
            assert result["status"] == "unhealthy"

