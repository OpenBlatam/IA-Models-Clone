from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import pytest
import asyncio
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List
from lazy_loading_exploit_modules import (
        import yaml
from typing import Any, List, Dict, Optional
import logging
"""
Tests for Lazy Loading System
Comprehensive test suite for lazy loading of heavy modules
"""


# Import lazy loading components
    LazyModuleManager, ModuleConfig, ModuleType, LoadingStrategy,
    ExploitDatabaseModule, VulnerabilityDatabaseModule, MachineLearningModelModule,
    LazyModule, ModuleMetrics
)

class TestLazyModule:
    """Test cases for base LazyModule class"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path="test/path",
            loading_strategy=LoadingStrategy.ON_DEMAND,
            cache_size=10,
            cache_ttl=3600
        )
    
    def test_lazy_module_initialization(self) -> Any:
        """Test LazyModule initialization"""
        module = LazyModule(self.config)
        
        assert module.config == self.config
        assert module._module is None
        assert module._loaded is False
        assert module._loading is False
        assert module._error is None
        assert len(module._cache) == 0
        assert len(module._cache_timestamps) == 0
    
    def test_lazy_module_metrics_initialization(self) -> Any:
        """Test ModuleMetrics initialization"""
        module = LazyModule(self.config)
        metrics = module.get_metrics()
        
        assert isinstance(metrics, ModuleMetrics)
        assert metrics.load_time == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.access_count == 0
        assert metrics.error_count == 0
        assert metrics.last_accessed is None
        assert metrics.last_loaded is None
    
    @pytest.mark.asyncio
    async def test_lazy_module_load_module_not_implemented(self) -> Any:
        """Test that _load_module raises NotImplementedError"""
        module = LazyModule(self.config)
        
        with pytest.raises(NotImplementedError):
            await module._load_module()
    
    @pytest.mark.asyncio
    async def test_lazy_module_unload_module(self) -> Any:
        """Test module unloading"""
        module = LazyModule(self.config)
        
        # Add some cache data
        module._cache["test_key"] = "test_value"
        module._cache_timestamps["test_key"] = time.time()
        module._loaded = True
        module._module = {"test": "data"}
        
        await module._unload_module()
        
        assert module._module is None
        assert module._loaded is False
        assert len(module._cache) == 0
        assert len(module._cache_timestamps) == 0
        assert module._metrics.memory_usage_mb == 0.0
        assert module._metrics.last_loaded is None

class TestExploitDatabaseModule:
    """Test cases for ExploitDatabaseModule"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path="test/exploits.json",
            loading_strategy=LoadingStrategy.ON_DEMAND,
            cache_size=10,
            cache_ttl=3600
        )
        self.module = ExploitDatabaseModule(self.config)
    
    @pytest.mark.asyncio
    async def test_load_from_file_json(self) -> Any:
        """Test loading exploit database from JSON file"""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "exploits": [
                    {
                        "id": "EXP-001",
                        "name": "Test Exploit",
                        "description": "Test description"
                    }
                ]
            }
            json.dump(test_data, f)
            file_path = f.name
        
        try:
            # Update module path
            self.module.config.module_path = file_path
            
            # Load module
            result = await self.module._load_from_file()
            
            assert isinstance(result, dict)
            assert "exploits" in result
            assert len(result["exploits"]) == 1
            assert result["exploits"][0]["id"] == "EXP-001"
            
        finally:
            # Cleanup
            Path(file_path).unlink()
    
    @pytest.mark.asyncio
    async def test_load_from_file_yaml(self) -> Any:
        """Test loading exploit database from YAML file"""
        
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_data = {
                "exploits": [
                    {
                        "id": "EXP-002",
                        "name": "YAML Test Exploit",
                        "description": "YAML test description"
                    }
                ]
            }
            yaml.dump(test_data, f)
            file_path = f.name
        
        try:
            # Update module path
            self.module.config.module_path = file_path
            
            # Load module
            result = await self.module._load_from_file()
            
            assert isinstance(result, dict)
            assert "exploits" in result
            assert len(result["exploits"]) == 1
            assert result["exploits"][0]["id"] == "EXP-002"
            
        finally:
            # Cleanup
            Path(file_path).unlink()
    
    @pytest.mark.asyncio
    async def test_load_from_file_not_found(self) -> Any:
        """Test loading from non-existent file"""
        self.module.config.module_path = "nonexistent_file.json"
        
        with pytest.raises(FileNotFoundError):
            await self.module._load_from_file()
    
    @pytest.mark.asyncio
    async def test_load_from_file_unsupported_format(self) -> Any:
        """Test loading from unsupported file format"""
        # Create temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test data")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            file_path = f.name
        
        try:
            self.module.config.module_path = file_path
            
            with pytest.raises(ValueError):
                await self.module._load_from_file()
                
        finally:
            Path(file_path).unlink()
    
    @pytest.mark.asyncio
    async def test_search_exploits(self) -> Any:
        """Test exploit search functionality"""
        # Mock module data
        self.module._module = {
            "exploits": [
                {
                    "id": "EXP-001",
                    "name": "Buffer Overflow",
                    "description": "Buffer overflow vulnerability"
                },
                {
                    "id": "EXP-002",
                    "name": "SQL Injection",
                    "description": "SQL injection vulnerability"
                }
            ]
        }
        self.module._loaded = True
        
        # Test search
        results = await self.module.search_exploits("buffer")
        
        assert len(results) == 1
        assert results[0]["id"] == "EXP-001"
        
        # Test search with no results
        results = await self.module.search_exploits("nonexistent")
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_get_exploit_by_id(self) -> Optional[Dict[str, Any]]:
        """Test getting exploit by ID"""
        # Mock module data
        self.module._module = {
            "exploits": [
                {
                    "id": "EXP-001",
                    "name": "Test Exploit",
                    "description": "Test description"
                }
            ]
        }
        self.module._loaded = True
        
        # Test getting existing exploit
        exploit = await self.module.get_exploit_by_id("EXP-001")
        assert exploit is not None
        assert exploit["id"] == "EXP-001"
        
        # Test getting non-existent exploit
        exploit = await self.module.get_exploit_by_id("EXP-999")
        assert exploit is None

class TestVulnerabilityDatabaseModule:
    """Test cases for VulnerabilityDatabaseModule"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.config = ModuleConfig(
            module_type=ModuleType.VULNERABILITY_DATABASE,
            module_path="test/vulnerabilities.json",
            loading_strategy=LoadingStrategy.ON_DEMAND,
            cache_size=10,
            cache_ttl=3600
        )
        self.module = VulnerabilityDatabaseModule(self.config)
    
    @pytest.mark.asyncio
    async def test_get_vulnerability(self) -> Optional[Dict[str, Any]]:
        """Test getting vulnerability by CVE ID"""
        # Mock module data
        self.module._module = {
            "vulnerabilities": [
                {
                    "cve_id": "CVE-2024-1234",
                    "title": "Test Vulnerability",
                    "description": "Test description",
                    "severity": "high"
                }
            ]
        }
        self.module._loaded = True
        
        # Test getting existing vulnerability
        vuln = await self.module.get_vulnerability("CVE-2024-1234")
        assert vuln is not None
        assert vuln["cve_id"] == "CVE-2024-1234"
        
        # Test getting non-existent vulnerability
        vuln = await self.module.get_vulnerability("CVE-2024-9999")
        assert vuln is None
    
    @pytest.mark.asyncio
    async def test_search_vulnerabilities(self) -> Any:
        """Test vulnerability search functionality"""
        # Mock module data
        self.module._module = {
            "vulnerabilities": [
                {
                    "cve_id": "CVE-2024-1234",
                    "title": "Buffer Overflow",
                    "description": "Buffer overflow vulnerability"
                },
                {
                    "cve_id": "CVE-2024-5678",
                    "title": "SQL Injection",
                    "description": "SQL injection vulnerability"
                }
            ]
        }
        self.module._loaded = True
        
        # Test search by description
        results = await self.module.search_vulnerabilities("buffer")
        assert len(results) == 1
        assert results[0]["cve_id"] == "CVE-2024-1234"
        
        # Test search by CVE ID
        results = await self.module.search_vulnerabilities("CVE-2024-5678")
        assert len(results) == 1
        assert results[0]["cve_id"] == "CVE-2024-5678"
        
        # Test search with no results
        results = await self.module.search_vulnerabilities("nonexistent")
        assert len(results) == 0

class TestMachineLearningModelModule:
    """Test cases for MachineLearningModelModule"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.config = ModuleConfig(
            module_type=ModuleType.MACHINE_LEARNING_MODEL,
            module_path="test/model.pkl",
            loading_strategy=LoadingStrategy.ON_DEMAND,
            cache_size=5,
            cache_ttl=7200
        )
        self.module = MachineLearningModelModule(self.config)
    
    @pytest.mark.asyncio
    async def test_load_module(self) -> Any:
        """Test ML model loading"""
        result = await self.module._load_module()
        
        assert isinstance(result, dict)
        assert result["model_type"] == "transformer"
        assert result["model_path"] == self.config.module_path
        assert result["loaded"] is True
    
    @pytest.mark.asyncio
    async def test_predict(self) -> Any:
        """Test ML model prediction"""
        # Mock module data
        self.module._module = {
            "model_type": "transformer",
            "model_path": "test/model.pkl",
            "loaded": True
        }
        self.module._loaded = True
        
        # Test prediction
        prediction = await self.module.predict("test_input")
        
        assert isinstance(prediction, dict)
        assert "prediction" in prediction
        assert "confidence" in prediction
        assert "model_type" in prediction
        assert prediction["model_type"] == "transformer"

class TestLazyModuleManager:
    """Test cases for LazyModuleManager"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.manager = LazyModuleManager()
    
    @pytest.mark.asyncio
    async def test_register_module(self) -> Any:
        """Test module registration"""
        config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path="test/path",
            loading_strategy=LoadingStrategy.ON_DEMAND
        )
        
        await self.manager.register_module("test_module", config)
        
        assert "test_module" in self.manager.modules
        assert isinstance(self.manager.modules["test_module"], ExploitDatabaseModule)
    
    @pytest.mark.asyncio
    async def test_register_duplicate_module(self) -> Any:
        """Test registering duplicate module"""
        config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path="test/path",
            loading_strategy=LoadingStrategy.ON_DEMAND
        )
        
        await self.manager.register_module("test_module", config)
        
        with pytest.raises(ValueError):
            await self.manager.register_module("test_module", config)
    
    @pytest.mark.asyncio
    async def test_get_module(self) -> Optional[Dict[str, Any]]:
        """Test getting registered module"""
        config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path="test/path",
            loading_strategy=LoadingStrategy.ON_DEMAND
        )
        
        await self.manager.register_module("test_module", config)
        module = await self.manager.get_module("test_module")
        
        assert isinstance(module, ExploitDatabaseModule)
        assert module.config == config
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_module(self) -> Optional[Dict[str, Any]]:
        """Test getting non-existent module"""
        with pytest.raises(ValueError):
            await self.manager.get_module("nonexistent_module")
    
    @pytest.mark.asyncio
    async def test_unload_module(self) -> Any:
        """Test module unloading"""
        config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path="test/path",
            loading_strategy=LoadingStrategy.ON_DEMAND
        )
        
        await self.manager.register_module("test_module", config)
        
        # Load the module first
        module = await self.manager.get_module("test_module")
        await module.get_module()
        
        # Unload the module
        await self.manager.unload_module("test_module")
        
        # Check that module is unloaded
        assert not module._loaded
        assert module._module is None
    
    @pytest.mark.asyncio
    async def test_get_all_metrics(self) -> Optional[Dict[str, Any]]:
        """Test getting metrics for all modules"""
        config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path="test/path",
            loading_strategy=LoadingStrategy.ON_DEMAND
        )
        
        await self.manager.register_module("test_module", config)
        metrics = await self.manager.get_all_metrics()
        
        assert "test_module" in metrics
        assert isinstance(metrics["test_module"], ModuleMetrics)
    
    @pytest.mark.asyncio
    async def test_cleanup_all(self) -> Any:
        """Test cleaning up all modules"""
        config1 = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path="test/path1",
            loading_strategy=LoadingStrategy.ON_DEMAND
        )
        config2 = ModuleConfig(
            module_type=ModuleType.VULNERABILITY_DATABASE,
            module_path="test/path2",
            loading_strategy=LoadingStrategy.ON_DEMAND
        )
        
        await self.manager.register_module("module1", config1)
        await self.manager.register_module("module2", config2)
        
        # Load modules
        await self.manager.get_module("module1")
        await self.manager.get_module("module2")
        
        # Cleanup all
        await self.manager.cleanup_all()
        
        # Check that all modules are unloaded
        for module in self.manager.modules.values():
            assert not module._loaded
            assert module._module is None

class TestCaching:
    """Test cases for caching functionality"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path="test/path",
            loading_strategy=LoadingStrategy.ON_DEMAND,
            cache_size=5,
            cache_ttl=10
        )
        self.module = ExploitDatabaseModule(self.config)
    
    @pytest.mark.asyncio
    async def test_get_cached_cache_hit(self) -> Optional[Dict[str, Any]]:
        """Test cache hit behavior"""
        # Add item to cache
        self.module._cache["test_key"] = "test_value"
        self.module._cache_timestamps["test_key"] = time.time()
        
        # Mock fetch function
        fetch_func = AsyncMock(return_value="new_value")
        
        # Get cached value
        result = await self.module.get_cached("test_key", fetch_func)
        
        assert result == "test_value"
        assert self.module._metrics.cache_hits == 1
        assert self.module._metrics.cache_misses == 0
        fetch_func.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_cached_cache_miss(self) -> Optional[Dict[str, Any]]:
        """Test cache miss behavior"""
        # Mock fetch function
        fetch_func = AsyncMock(return_value="new_value")
        
        # Get value (cache miss)
        result = await self.module.get_cached("test_key", fetch_func)
        
        assert result == "new_value"
        assert self.module._metrics.cache_hits == 0
        assert self.module._metrics.cache_misses == 1
        assert "test_key" in self.module._cache
        fetch_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_cached_expired(self) -> Optional[Dict[str, Any]]:
        """Test cache expiration behavior"""
        # Add expired item to cache
        self.module._cache["test_key"] = "old_value"
        self.module._cache_timestamps["test_key"] = time.time() - 20  # Expired
        
        # Mock fetch function
        fetch_func = AsyncMock(return_value="new_value")
        
        # Get value (expired cache)
        result = await self.module.get_cached("test_key", fetch_func)
        
        assert result == "new_value"
        assert self.module._metrics.cache_hits == 0
        assert self.module._metrics.cache_misses == 1
        fetch_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self) -> Any:
        """Test cache eviction when cache is full"""
        # Fill cache
        for i in range(6):  # More than cache_size (5)
            self.module._cache[f"key_{i}"] = f"value_{i}"
            self.module._cache_timestamps[f"key_{i}"] = time.time()
        
        # Mock fetch function
        fetch_func = AsyncMock(return_value="new_value")
        
        # Get new value (should trigger eviction)
        result = await self.module.get_cached("new_key", fetch_func)
        
        assert result == "new_value"
        assert len(self.module._cache) == 5  # Should be at cache_size
        assert len(self.module._cache_timestamps) == 5

class TestLoadingStrategies:
    """Test cases for different loading strategies"""
    
    @pytest.mark.asyncio
    async def test_on_demand_strategy(self) -> Any:
        """Test on-demand loading strategy"""
        config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path="test/path",
            loading_strategy=LoadingStrategy.ON_DEMAND
        )
        
        manager = LazyModuleManager()
        await manager.register_module("test_module", config)
        
        module = await manager.get_module("test_module")
        assert not module._loaded
        
        # Access module (should trigger loading)
        await module.get_module()
        assert module._loaded
    
    @pytest.mark.asyncio
    async def test_preload_strategy(self) -> Any:
        """Test preload loading strategy"""
        config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path="test/path",
            loading_strategy=LoadingStrategy.PRELOAD
        )
        
        manager = LazyModuleManager()
        await manager.register_module("test_module", config)
        
        # Wait for preload to complete
        await asyncio.sleep(0.1)
        
        module = await manager.get_module("test_module")
        assert module._loaded

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 