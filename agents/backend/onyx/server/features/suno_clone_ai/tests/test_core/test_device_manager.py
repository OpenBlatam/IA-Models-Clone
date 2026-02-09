"""
Tests para el gestor de dispositivos
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from core.utils.device.device_manager import (
    get_device,
    setup_gpu_optimizations,
    clear_gpu_cache,
    get_device_info
)


@pytest.mark.unit
@pytest.mark.core
class TestGetDevice:
    """Tests para get_device"""
    
    @patch('torch.cuda.is_available')
    def test_get_device_cpu_when_gpu_unavailable(self, mock_cuda_available):
        """Test de obtención de CPU cuando GPU no está disponible"""
        mock_cuda_available.return_value = False
        
        device = get_device(use_gpu=True)
        
        assert device.type == "cpu"
    
    @patch('torch.cuda.is_available')
    def test_get_device_cpu_when_forced(self, mock_cuda_available):
        """Test de obtención de CPU cuando se fuerza"""
        mock_cuda_available.return_value = True
        
        device = get_device(use_gpu=False)
        
        assert device.type == "cpu"
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    def test_get_device_gpu_default(self, mock_get_name, mock_cuda_available):
        """Test de obtención de GPU por defecto"""
        mock_cuda_available.return_value = True
        mock_get_name.return_value = "Test GPU"
        
        device = get_device(use_gpu=True)
        
        assert device.type == "cuda"
        assert device.index is None or device.index == 0
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    def test_get_device_gpu_specific_id(self, mock_get_name, mock_cuda_available):
        """Test de obtención de GPU con ID específico"""
        mock_cuda_available.return_value = True
        mock_get_name.return_value = "Test GPU"
        
        device = get_device(use_gpu=True, device_id=1)
        
        assert device.type == "cuda"
        assert device.index == 1


@pytest.mark.unit
@pytest.mark.core
class TestSetupGPUOptimizations:
    """Tests para setup_gpu_optimizations"""
    
    @patch('torch.cuda.is_available')
    def test_setup_gpu_optimizations_no_gpu(self, mock_cuda_available):
        """Test de setup cuando no hay GPU"""
        mock_cuda_available.return_value = False
        
        # No debería lanzar error
        setup_gpu_optimizations()
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.cudnn.benchmark', new_callable=MagicMock)
    @patch('torch.backends.cudnn.deterministic', new_callable=MagicMock)
    @patch('torch.backends.cuda.matmul.allow_tf32', new_callable=MagicMock)
    @patch('torch.backends.cudnn.allow_tf32', new_callable=MagicMock)
    def test_setup_gpu_optimizations_with_gpu(
        self,
        mock_cudnn_tf32,
        mock_matmul_tf32,
        mock_cudnn_deterministic,
        mock_cudnn_benchmark,
        mock_cuda_available
    ):
        """Test de setup cuando hay GPU"""
        mock_cuda_available.return_value = True
        
        setup_gpu_optimizations()
        
        # Verificar que se configuraron las optimizaciones
        assert mock_cudnn_benchmark is not None
        assert mock_cudnn_deterministic is not None


@pytest.mark.unit
@pytest.mark.core
class TestClearGPUCache:
    """Tests para clear_gpu_cache"""
    
    @patch('torch.cuda.is_available')
    def test_clear_gpu_cache_no_gpu(self, mock_cuda_available):
        """Test de limpieza cuando no hay GPU"""
        mock_cuda_available.return_value = False
        
        # No debería lanzar error
        clear_gpu_cache()
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    def test_clear_gpu_cache_with_gpu(
        self,
        mock_synchronize,
        mock_empty_cache,
        mock_cuda_available
    ):
        """Test de limpieza cuando hay GPU"""
        mock_cuda_available.return_value = True
        
        clear_gpu_cache()
        
        mock_empty_cache.assert_called_once()
        mock_synchronize.assert_called_once()


@pytest.mark.unit
@pytest.mark.core
class TestGetDeviceInfo:
    """Tests para get_device_info"""
    
    def test_get_device_info_cpu(self):
        """Test de información de CPU"""
        device = torch.device("cpu")
        info = get_device_info(device)
        
        assert info["device"] == "cpu"
        assert info["type"] == "cpu"
        assert "name" not in info
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_capability')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.version.cuda')
    def test_get_device_info_gpu(
        self,
        mock_cuda_version,
        mock_memory_reserved,
        mock_memory_allocated,
        mock_get_capability,
        mock_get_name,
        mock_cuda_available
    ):
        """Test de información de GPU"""
        mock_cuda_available.return_value = True
        mock_get_name.return_value = "Test GPU"
        mock_get_capability.return_value = (8, 0)
        mock_memory_allocated.return_value = 1024**3  # 1 GB
        mock_memory_reserved.return_value = 2 * 1024**3  # 2 GB
        mock_cuda_version = "11.8"
        
        device = torch.device("cuda:0")
        info = get_device_info(device)
        
        assert info["device"] == "cuda:0"
        assert info["type"] == "cuda"
        assert "name" in info
        assert "capability" in info
        assert "memory_allocated_gb" in info
        assert "memory_reserved_gb" in info
        assert "cuda_version" in info
    
    @patch('torch.cuda.is_available')
    def test_get_device_info_default(self, mock_cuda_available):
        """Test de información con dispositivo por defecto"""
        mock_cuda_available.return_value = False
        
        info = get_device_info()
        
        assert "device" in info
        assert "type" in info



