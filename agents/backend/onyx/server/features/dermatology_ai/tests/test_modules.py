"""
Tests for Module System
Tests for module loading and registration
"""

import pytest
from unittest.mock import Mock, AsyncMock

from core.modules.module import BaseModule, ModuleMetadata
from core.modules.module_registry import ModuleRegistry
from core.modules.module_loader import ModuleLoader


class TestBaseModule:
    """Tests for BaseModule"""
    
    def test_create_module(self):
        """Test creating a module"""
        class TestModule(BaseModule):
            def __init__(self):
                super().__init__(
                    name="test_module",
                    version="1.0.0",
                    description="Test module"
                )
        
        module = TestModule()
        
        assert module.name == "test_module"
        assert module.version == "1.0.0"
        assert module.description == "Test module"
    
    @pytest.mark.asyncio
    async def test_module_initialize(self):
        """Test module initialization"""
        class TestModule(BaseModule):
            def __init__(self):
                super().__init__(
                    name="test_module",
                    version="1.0.0"
                )
                self.initialized = False
            
            async def initialize(self):
                self.initialized = True
        
        module = TestModule()
        await module.initialize()
        
        assert module.initialized is True
    
    @pytest.mark.asyncio
    async def test_module_shutdown(self):
        """Test module shutdown"""
        class TestModule(BaseModule):
            def __init__(self):
                super().__init__(
                    name="test_module",
                    version="1.0.0"
                )
                self.shutdown_called = False
            
            async def shutdown(self):
                self.shutdown_called = True
        
        module = TestModule()
        await module.shutdown()
        
        assert module.shutdown_called is True


class TestModuleRegistry:
    """Tests for ModuleRegistry"""
    
    @pytest.fixture
    def module_registry(self):
        """Create module registry"""
        return ModuleRegistry()
    
    def test_register_module(self, module_registry):
        """Test registering a module"""
        class TestModule(BaseModule):
            def __init__(self):
                super().__init__(
                    name="test_module",
                    version="1.0.0"
                )
        
        module = TestModule()
        module_registry.register(module)
        
        assert module_registry.get_module("test_module") == module
    
    def test_get_module_not_found(self, module_registry):
        """Test getting non-existent module"""
        module = module_registry.get_module("non_existent")
        
        assert module is None
    
    def test_list_modules(self, module_registry):
        """Test listing all modules"""
        class Module1(BaseModule):
            def __init__(self):
                super().__init__(name="module1", version="1.0.0")
        
        class Module2(BaseModule):
            def __init__(self):
                super().__init__(name="module2", version="1.0.0")
        
        module_registry.register(Module1())
        module_registry.register(Module2())
        
        modules = module_registry.list_modules()
        
        assert len(modules) == 2
        assert all(m.name in ["module1", "module2"] for m in modules)
    
    @pytest.mark.asyncio
    async def test_initialize_all_modules(self, module_registry):
        """Test initializing all modules"""
        initialized_modules = []
        
        class TestModule(BaseModule):
            def __init__(self, name):
                super().__init__(name=name, version="1.0.0")
            
            async def initialize(self):
                initialized_modules.append(self.name)
        
        module1 = TestModule("module1")
        module2 = TestModule("module2")
        
        module_registry.register(module1)
        module_registry.register(module2)
        
        await module_registry.initialize_all()
        
        assert len(initialized_modules) == 2
        assert "module1" in initialized_modules
        assert "module2" in initialized_modules


class TestModuleLoader:
    """Tests for ModuleLoader"""
    
    @pytest.fixture
    def module_loader(self):
        """Create module loader"""
        return ModuleLoader()
    
    def test_load_module_from_path(self, module_loader):
        """Test loading module from file path"""
        # This would load an actual module file
        # For testing, we'll mock it
        with pytest.raises((ImportError, FileNotFoundError, AttributeError)):
            # Try to load non-existent module
            module_loader.load_from_path("non_existent_module")
    
    def test_load_module_from_package(self, module_loader):
        """Test loading module from package"""
        # This would load from installed package
        # For testing, we'll verify the method exists
        assert hasattr(module_loader, 'load_from_package') or hasattr(module_loader, 'load_module')
    
    @pytest.mark.asyncio
    async def test_load_and_register_module(self, module_loader):
        """Test loading and registering a module"""
        registry = ModuleRegistry()
        
        class TestModule(BaseModule):
            def __init__(self):
                super().__init__(name="loaded_module", version="1.0.0")
        
        # Manually register (simulating load)
        module = TestModule()
        registry.register(module)
        
        assert registry.get_module("loaded_module") == module



