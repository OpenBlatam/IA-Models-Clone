"""
Tests para Infrastructure Repositories - Arquitectura Mejorada
==============================================================
"""

import pytest
from datetime import datetime

from core.architecture.infrastructure_repositories import (
    InMemoryRobotRepository,
    InMemoryMovementRepository
)
from core.architecture.domain_improved import (
    Robot,
    RobotMovement,
    Position,
    Orientation,
    MovementStatus
)


class TestInMemoryRobotRepository:
    """Tests para InMemoryRobotRepository."""
    
    @pytest.fixture
    async def repo(self):
        """Crear repositorio."""
        repo = InMemoryRobotRepository()
        await repo.initialize()
        return repo
    
    @pytest.mark.asyncio
    async def test_save_and_find_robot(self, repo):
        """Test guardar y encontrar robot."""
        robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        
        await repo.save(robot)
        found = await repo.find_by_id("robot-1")
        
        assert found is not None
        assert found.id == "robot-1"
        assert found.brand == "KUKA"
    
    @pytest.mark.asyncio
    async def test_find_nonexistent_robot(self, repo):
        """Test encontrar robot que no existe."""
        found = await repo.find_by_id("robot-999")
        assert found is None
    
    @pytest.mark.asyncio
    async def test_find_all_robots(self, repo):
        """Test encontrar todos los robots."""
        robot1 = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        robot2 = Robot(robot_id="robot-2", brand="ABB", model="IRB120")
        
        await repo.save(robot1)
        await repo.save(robot2)
        
        all_robots = await repo.find_all()
        assert len(all_robots) == 2
    
    @pytest.mark.asyncio
    async def test_find_by_brand(self, repo):
        """Test encontrar robots por marca."""
        robot1 = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        robot2 = Robot(robot_id="robot-2", brand="KUKA", model="KR150")
        robot3 = Robot(robot_id="robot-3", brand="ABB", model="IRB120")
        
        await repo.save(robot1)
        await repo.save(robot2)
        await repo.save(robot3)
        
        kuka_robots = await repo.find_by_brand("KUKA")
        assert len(kuka_robots) == 2
        assert all(r.brand == "KUKA" for r in kuka_robots)
    
    @pytest.mark.asyncio
    async def test_delete_robot(self, repo):
        """Test eliminar robot."""
        robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        await repo.save(robot)
        
        deleted = await repo.delete("robot-1")
        assert deleted is True
        
        found = await repo.find_by_id("robot-1")
        assert found is None
    
    @pytest.mark.asyncio
    async def test_count_robots(self, repo):
        """Test contar robots."""
        await repo.save(Robot(robot_id="robot-1", brand="KUKA", model="KR210"))
        await repo.save(Robot(robot_id="robot-2", brand="ABB", model="IRB120"))
        
        count = await repo.count()
        assert count == 2


class TestInMemoryMovementRepository:
    """Tests para InMemoryMovementRepository."""
    
    @pytest.fixture
    async def repo(self):
        """Crear repositorio."""
        repo = InMemoryMovementRepository()
        await repo.initialize()
        return repo
    
    @pytest.mark.asyncio
    async def test_save_and_find_movement(self, repo):
        """Test guardar y encontrar movimiento."""
        movement = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=1.0, y=2.0, z=3.0)
        )
        
        await repo.save(movement)
        found = await repo.find_by_id(movement.id)
        
        assert found is not None
        assert found.id == movement.id
        assert found.robot_id == "robot-1"
    
    @pytest.mark.asyncio
    async def test_find_movements_by_robot_id(self, repo):
        """Test encontrar movimientos por robot ID."""
        movement1 = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=1.0, y=2.0, z=3.0)
        )
        movement2 = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=2.0, y=3.0, z=4.0)
        )
        movement3 = RobotMovement(
            robot_id="robot-2",
            target_position=Position(x=3.0, y=4.0, z=5.0)
        )
        
        await repo.save(movement1)
        await repo.save(movement2)
        await repo.save(movement3)
        
        robot1_movements = await repo.find_by_robot_id("robot-1")
        assert len(robot1_movements) == 2
        assert all(m.robot_id == "robot-1" for m in robot1_movements)
    
    @pytest.mark.asyncio
    async def test_find_movements_by_status(self, repo):
        """Test encontrar movimientos por estado."""
        movement1 = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=1.0, y=2.0, z=3.0)
        )
        movement1.start()
        movement1.complete(Position(x=1.0, y=2.0, z=3.0))
        
        movement2 = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=2.0, y=3.0, z=4.0)
        )
        movement2.start()
        movement2.fail("Error")
        
        await repo.save(movement1)
        await repo.save(movement2)
        
        completed = await repo.find_by_status(MovementStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].status == MovementStatus.COMPLETED
        
        failed = await repo.find_by_status(MovementStatus.FAILED)
        assert len(failed) == 1
        assert failed[0].status == MovementStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_delete_movement(self, repo):
        """Test eliminar movimiento."""
        movement = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=1.0, y=2.0, z=3.0)
        )
        await repo.save(movement)
        
        deleted = await repo.delete(movement.id)
        assert deleted is True
        
        found = await repo.find_by_id(movement.id)
        assert found is None
    
    @pytest.mark.asyncio
    async def test_movement_indexing(self, repo):
        """Test que los índices se mantienen correctamente."""
        movement = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=1.0, y=2.0, z=3.0)
        )
        movement.start()
        await repo.save(movement)
        
        # Cambiar estado
        movement.complete(Position(x=1.0, y=2.0, z=3.0))
        await repo.save(movement)
        
        # Verificar que está en el índice correcto
        completed = await repo.find_by_status(MovementStatus.COMPLETED)
        assert len(completed) == 1
        
        executing = await repo.find_by_status(MovementStatus.EXECUTING)
        assert len(executing) == 0




