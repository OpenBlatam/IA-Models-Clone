"""
Tests para Dependency Injection - Arquitectura Mejorada
=======================================================
"""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock

from core.architecture.di_setup import (
    DISetup,
    setup_dependency_injection,
    get_robot_repository,
    get_movement_repository,
    get_move_robot_use_case
)
from core.architecture.dependency_injection import Lifecycle, Container
from core.architecture.application_layer import (
    IRobotRepository,
    IMovementRepository,
    MoveRobotUseCase
)
from core.architecture.infrastructure_repositories import (
    InMemoryRobotRepository,
    InMemoryMovementRepository
)


class TestDISetup:
    """Tests para DISetup."""
    
    @pytest.fixture
    def config(self):
        """Configuración para tests."""
        return {
            'repository_type': 'in_memory',
            'enable_event_bus': False
        }
    
    @pytest.mark.asyncio
    async def test_setup_repositories(self, config):
        """Test setup de repositorios."""
        setup = DISetup(config)
        await setup.setup()
        
        container = setup.get_container()
        
        # Verificar que repositorios están registrados
        robot_repo = await container.resolve_async(IRobotRepository)
        assert isinstance(robot_repo, InMemoryRobotRepository)
        
        movement_repo = await container.resolve_async(IMovementRepository)
        assert isinstance(movement_repo, InMemoryMovementRepository)
    
    @pytest.mark.asyncio
    async def test_setup_use_cases(self, config):
        """Test setup de use cases."""
        setup = DISetup(config)
        await setup.setup()
        
        container = setup.get_container()
        
        # Verificar que use cases están registrados
        use_case = await container.resolve_async(MoveRobotUseCase)
        assert use_case is not None
    
    @pytest.mark.asyncio
    async def test_setup_twice_does_nothing(self, config):
        """Test que setup dos veces no hace nada."""
        setup = DISetup(config)
        await setup.setup()
        
        # Segunda vez debería ser no-op
        await setup.setup()
        
        # Debería funcionar igual
        container = setup.get_container()
        robot_repo = await container.resolve_async(IRobotRepository)
        assert robot_repo is not None
    
    @pytest.mark.asyncio
    async def test_get_container_before_setup_fails(self, config):
        """Test que obtener container antes de setup falla."""
        setup = DISetup(config)
        
        with pytest.raises(RuntimeError):
            setup.get_container()


class TestHelperFunctions:
    """Tests para helper functions."""
    
    @pytest.mark.asyncio
    async def test_get_robot_repository(self):
        """Test obtener repositorio de robots."""
        await setup_dependency_injection({
            'repository_type': 'in_memory'
        })
        
        repo = await get_robot_repository()
        assert repo is not None
        assert isinstance(repo, InMemoryRobotRepository)
    
    @pytest.mark.asyncio
    async def test_get_movement_repository(self):
        """Test obtener repositorio de movimientos."""
        await setup_dependency_injection({
            'repository_type': 'in_memory'
        })
        
        repo = await get_movement_repository()
        assert repo is not None
        assert isinstance(repo, InMemoryMovementRepository)
    
    @pytest.mark.asyncio
    async def test_get_move_robot_use_case(self):
        """Test obtener use case."""
        await setup_dependency_injection({
            'repository_type': 'in_memory'
        })
        
        use_case = await get_move_robot_use_case()
        assert use_case is not None
        assert isinstance(use_case, MoveRobotUseCase)


class TestContainerLifecycle:
    """Tests para gestión de ciclo de vida."""
    
    @pytest.mark.asyncio
    async def test_singleton_lifecycle(self):
        """Test singleton lifecycle."""
        container = Container()
        
        class TestService:
            pass
        
        instance1 = TestService()
        container.register(
            TestService,
            implementation=instance1,
            lifecycle=Lifecycle.SINGLETON
        )
        
        resolved1 = await container.resolve_async(TestService)
        resolved2 = await container.resolve_async(TestService)
        
        assert resolved1 is resolved2
        assert resolved1 is instance1
    
    @pytest.mark.asyncio
    async def test_scoped_lifecycle(self):
        """Test scoped lifecycle."""
        container = Container()
        
        class TestService:
            pass
        
        container.register(
            TestService,
            factory=lambda: TestService(),
            lifecycle=Lifecycle.SCOPED
        )
        
        # Crear scope
        scope_id = container.create_scope()
        container.enter_scope(scope_id)
        
        try:
            instance1 = await container.resolve_async(TestService)
            instance2 = await container.resolve_async(TestService)
            
            # Misma instancia en mismo scope
            assert instance1 is instance2
            
            # Nuevo scope = nueva instancia
            scope_id2 = container.create_scope()
            container.enter_scope(scope_id2)
            
            instance3 = await container.resolve_async(TestService)
            assert instance3 is not instance1
        finally:
            container.exit_scope()
    
    @pytest.mark.asyncio
    async def test_transient_lifecycle(self):
        """Test transient lifecycle."""
        container = Container()
        
        class TestService:
            pass
        
        container.register(
            TestService,
            factory=lambda: TestService(),
            lifecycle=Lifecycle.TRANSIENT
        )
        
        instance1 = await container.resolve_async(TestService)
        instance2 = await container.resolve_async(TestService)
        
        # Diferentes instancias
        assert instance1 is not instance2




