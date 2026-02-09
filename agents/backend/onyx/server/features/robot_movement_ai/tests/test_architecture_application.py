"""
Tests para Application Layer - Arquitectura Mejorada
====================================================
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.architecture.application_layer import (
    MoveRobotUseCase,
    GetRobotStatusUseCase,
    GetMovementHistoryUseCase,
    MoveRobotCommand,
    GetRobotStatusQuery,
    GetMovementHistoryQuery,
    ApplicationError,
    ErrorCode
)
from core.architecture.domain_improved import (
    Robot,
    RobotMovement,
    Position,
    MovementStatus
)


class TestMoveRobotUseCase:
    """Tests para MoveRobotUseCase."""
    
    @pytest.fixture
    def mock_robot_repo(self):
        """Mock de repositorio de robots."""
        repo = AsyncMock()
        return repo
    
    @pytest.fixture
    def mock_movement_repo(self):
        """Mock de repositorio de movimientos."""
        repo = AsyncMock()
        return repo
    
    @pytest.fixture
    def use_case(self, mock_robot_repo, mock_movement_repo):
        """Crear use case con mocks."""
        return MoveRobotUseCase(
            robot_repository=mock_robot_repo,
            movement_repository=mock_movement_repo
        )
    
    @pytest.mark.asyncio
    async def test_move_robot_success(self, use_case, mock_robot_repo, mock_movement_repo):
        """Test mover robot exitosamente."""
        # Setup mocks
        robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        robot.connect()
        mock_robot_repo.find_by_id.return_value = robot
        mock_robot_repo.save = AsyncMock()
        mock_movement_repo.save = AsyncMock()
        
        # Ejecutar
        command = MoveRobotCommand(
            robot_id="robot-1",
            target_x=1.0,
            target_y=2.0,
            target_z=3.0
        )
        
        result = await use_case.execute(command)
        
        # Verificar
        assert result.movement_id is not None
        assert result.robot_id == "robot-1"
        assert result.status == "executing"
        mock_robot_repo.save.assert_called_once()
        mock_movement_repo.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_move_robot_not_found(self, use_case, mock_robot_repo):
        """Test mover robot que no existe."""
        mock_robot_repo.find_by_id.return_value = None
        
        command = MoveRobotCommand(
            robot_id="robot-999",
            target_x=1.0,
            target_y=2.0,
            target_z=3.0
        )
        
        with pytest.raises(ApplicationError) as exc_info:
            await use_case.execute(command)
        
        assert exc_info.value.code == ErrorCode.APPLICATION_NOT_FOUND
    
    @pytest.mark.asyncio
    async def test_move_robot_not_connected(self, use_case, mock_robot_repo):
        """Test mover robot desconectado."""
        robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        # No conectado
        mock_robot_repo.find_by_id.return_value = robot
        
        command = MoveRobotCommand(
            robot_id="robot-1",
            target_x=1.0,
            target_y=2.0,
            target_z=3.0
        )
        
        with pytest.raises(ApplicationError) as exc_info:
            await use_case.execute(command)
        
        assert exc_info.value.code == ErrorCode.ROBOT_NOT_CONNECTED


class TestGetRobotStatusUseCase:
    """Tests para GetRobotStatusUseCase."""
    
    @pytest.fixture
    def mock_robot_repo(self):
        """Mock de repositorio de robots."""
        repo = AsyncMock()
        return repo
    
    @pytest.fixture
    def use_case(self, mock_robot_repo):
        """Crear use case con mock."""
        return GetRobotStatusUseCase(robot_repository=mock_robot_repo)
    
    @pytest.mark.asyncio
    async def test_get_robot_status_success(self, use_case, mock_robot_repo):
        """Test obtener estado exitosamente."""
        robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        robot.connect()
        robot.update_position(Position(x=1.0, y=2.0, z=3.0))
        
        mock_robot_repo.find_by_id.return_value = robot
        
        query = GetRobotStatusQuery(robot_id="robot-1")
        result = await use_case.execute(query)
        
        assert result.robot_id == "robot-1"
        assert result.brand == "KUKA"
        assert result.is_connected is True
        assert result.current_position is not None
    
    @pytest.mark.asyncio
    async def test_get_robot_status_not_found(self, use_case, mock_robot_repo):
        """Test obtener estado de robot que no existe."""
        mock_robot_repo.find_by_id.return_value = None
        
        query = GetRobotStatusQuery(robot_id="robot-999")
        
        with pytest.raises(ApplicationError) as exc_info:
            await use_case.execute(query)
        
        assert exc_info.value.code == ErrorCode.APPLICATION_NOT_FOUND


class TestGetMovementHistoryUseCase:
    """Tests para GetMovementHistoryUseCase."""
    
    @pytest.fixture
    def mock_movement_repo(self):
        """Mock de repositorio de movimientos."""
        repo = AsyncMock()
        return repo
    
    @pytest.fixture
    def use_case(self, mock_movement_repo):
        """Crear use case con mock."""
        return GetMovementHistoryUseCase(movement_repository=mock_movement_repo)
    
    @pytest.mark.asyncio
    async def test_get_movement_history_success(self, use_case, mock_movement_repo):
        """Test obtener historial exitosamente."""
        movements = [
            RobotMovement(
                robot_id="robot-1",
                target_position=Position(x=1.0, y=2.0, z=3.0)
            ),
            RobotMovement(
                robot_id="robot-1",
                target_position=Position(x=2.0, y=3.0, z=4.0)
            )
        ]
        
        mock_movement_repo.find_by_robot_id.return_value = movements
        
        query = GetMovementHistoryQuery(robot_id="robot-1", limit=10)
        results = await use_case.execute(query)
        
        assert len(results) == 2
        assert all(r.robot_id == "robot-1" for r in results)
    
    @pytest.mark.asyncio
    async def test_get_movement_history_with_status_filter(self, use_case, mock_movement_repo):
        """Test obtener historial filtrado por estado."""
        movements = [
            RobotMovement(
                robot_id="robot-1",
                target_position=Position(x=1.0, y=2.0, z=3.0)
            )
        ]
        movements[0].start()
        movements[0].complete(Position(x=1.0, y=2.0, z=3.0))
        
        mock_movement_repo.find_by_robot_id.return_value = movements
        
        query = GetMovementHistoryQuery(
            robot_id="robot-1",
            limit=10,
            status=MovementStatus.COMPLETED
        )
        results = await use_case.execute(query)
        
        assert len(results) == 1
        assert results[0].status == "completed"




