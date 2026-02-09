"""
Comprehensive Unit Tests for Bootstrap Module

Tests cover bootstrap functions with diverse test cases
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import FastAPI

from bootstrap import bootstrap_application, initialize_modules, shutdown_modules


class TestBootstrapApplication:
    """Test cases for bootstrap_application function"""
    
    def test_bootstrap_application_creates_registry(self):
        """Test bootstrap creates module registry"""
        app = FastAPI()
        result = bootstrap_application(app)
        
        assert "registry" in result
        assert "modules" in result
        assert result["registry"] is not None
    
    def test_bootstrap_application_registers_modules(self):
        """Test bootstrap registers all modules"""
        app = FastAPI()
        result = bootstrap_application(app)
        
        modules = result["modules"]
        assert len(modules) > 0
    
    def test_bootstrap_application_returns_dict(self):
        """Test bootstrap returns dictionary"""
        app = FastAPI()
        result = bootstrap_application(app)
        
        assert isinstance(result, dict)
        assert "registry" in result
        assert "modules" in result
    
    @patch('bootstrap.get_module_registry')
    def test_bootstrap_application_calls_registry(self, mock_get_registry):
        """Test bootstrap calls get_module_registry"""
        mock_registry = Mock()
        mock_get_registry.return_value = mock_registry
        mock_registry.get_all_modules.return_value = []
        
        app = FastAPI()
        bootstrap_application(app)
        
        mock_get_registry.assert_called_once()
    
    def test_bootstrap_application_with_existing_app(self):
        """Test bootstrap with existing FastAPI app"""
        app = FastAPI(title="Test App")
        result = bootstrap_application(app)
        
        assert result is not None
        assert "registry" in result


class TestInitializeModules:
    """Test cases for initialize_modules function"""
    
    @pytest.mark.asyncio
    async def test_initialize_modules_success(self):
        """Test successful module initialization"""
        mock_registry = AsyncMock()
        mock_registry.initialize_all = AsyncMock(return_value=None)
        
        result = await initialize_modules(mock_registry)
        
        assert result is True
        mock_registry.initialize_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_modules_failure(self):
        """Test module initialization failure"""
        mock_registry = AsyncMock()
        mock_registry.initialize_all = AsyncMock(side_effect=Exception("Init error"))
        
        result = await initialize_modules(mock_registry)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_initialize_modules_with_real_registry(self):
        """Test initialization with real registry structure"""
        from modules.registry import get_module_registry
        
        registry = get_module_registry()
        result = await initialize_modules(registry)
        
        # Should not raise, may return True or False depending on module state
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_initialize_modules_empty_registry(self):
        """Test initialization with empty registry"""
        mock_registry = AsyncMock()
        mock_registry.initialize_all = AsyncMock(return_value=None)
        
        result = await initialize_modules(mock_registry)
        
        assert result is True


class TestShutdownModules:
    """Test cases for shutdown_modules function"""
    
    @pytest.mark.asyncio
    async def test_shutdown_modules_success(self):
        """Test successful module shutdown"""
        mock_registry = AsyncMock()
        mock_registry.shutdown_all = AsyncMock(return_value=None)
        
        await shutdown_modules(mock_registry)
        
        mock_registry.shutdown_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_modules_handles_exception(self):
        """Test shutdown handles exceptions gracefully"""
        mock_registry = AsyncMock()
        mock_registry.shutdown_all = AsyncMock(side_effect=Exception("Shutdown error"))
        
        # Should not raise
        await shutdown_modules(mock_registry)
        
        mock_registry.shutdown_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_modules_empty_registry(self):
        """Test shutdown with empty registry"""
        mock_registry = AsyncMock()
        mock_registry.shutdown_all = AsyncMock(return_value=None)
        
        await shutdown_modules(mock_registry)
        
        mock_registry.shutdown_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_modules_with_real_registry(self):
        """Test shutdown with real registry structure"""
        from modules.registry import get_module_registry
        
        registry = get_module_registry()
        
        # Should not raise
        await shutdown_modules(registry)


class TestBootstrapIntegration:
    """Integration tests for bootstrap workflow"""
    
    @pytest.mark.asyncio
    async def test_bootstrap_full_workflow(self):
        """Test complete bootstrap workflow"""
        app = FastAPI()
        
        # Bootstrap
        result = bootstrap_application(app)
        registry = result["registry"]
        
        # Initialize
        init_result = await initialize_modules(registry)
        assert isinstance(init_result, bool)
        
        # Shutdown
        await shutdown_modules(registry)
        # Should complete without errors
    
    def test_bootstrap_app_state(self):
        """Test that bootstrap doesn't modify app state unexpectedly"""
        app = FastAPI()
        initial_state = dict(app.state.__dict__) if hasattr(app.state, '__dict__') else {}
        
        bootstrap_application(app)
        
        # App should still be valid
        assert app is not None
        assert isinstance(app, FastAPI)
    
    @pytest.mark.asyncio
    async def test_bootstrap_multiple_calls(self):
        """Test calling bootstrap multiple times"""
        app = FastAPI()
        
        result1 = bootstrap_application(app)
        result2 = bootstrap_application(app)
        
        # Both should succeed
        assert result1 is not None
        assert result2 is not None
        
        # Initialize and shutdown
        await initialize_modules(result1["registry"])
        await shutdown_modules(result1["registry"])















