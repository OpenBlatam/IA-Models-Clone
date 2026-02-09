"""
Tests para Domain Layer - Arquitectura Mejorada
===============================================
"""

import pytest
from datetime import datetime

from core.architecture.domain_improved import (
    Robot,
    RobotMovement,
    Position,
    Orientation,
    MovementStatus,
    DomainError
)


class TestPosition:
    """Tests para Value Object Position."""
    
    def test_create_valid_position(self):
        """Test crear posición válida."""
        pos = Position(x=1.0, y=2.0, z=3.0)
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.z == 3.0
    
    def test_position_validation(self):
        """Test validación de posición."""
        # X fuera de rango
        with pytest.raises(DomainError):
            Position(x=2000.0, y=0.0, z=0.0)
        
        # Y fuera de rango
        with pytest.raises(DomainError):
            Position(x=0.0, y=-2000.0, z=0.0)
    
    def test_position_distance(self):
        """Test cálculo de distancia."""
        pos1 = Position(x=0.0, y=0.0, z=0.0)
        pos2 = Position(x=3.0, y=4.0, z=0.0)
        
        distance = pos1.distance_to(pos2)
        assert distance == 5.0  # 3-4-5 triangle
    
    def test_position_immutability(self):
        """Test que Position es inmutable."""
        pos = Position(x=1.0, y=2.0, z=3.0)
        
        # Intentar modificar debería fallar (frozen dataclass)
        with pytest.raises(Exception):
            pos.x = 5.0


class TestOrientation:
    """Tests para Value Object Orientation."""
    
    def test_create_valid_orientation(self):
        """Test crear orientación válida."""
        # Quaternion normalizado
        ori = Orientation(qx=0.0, qy=0.0, qz=0.0, qw=1.0)
        assert ori.qx == 0.0
        assert ori.qw == 1.0
    
    def test_orientation_validation(self):
        """Test validación de quaternion."""
        # Quaternion no normalizado
        with pytest.raises(DomainError):
            Orientation(qx=1.0, qy=1.0, qz=1.0, qw=1.0)


class TestRobot:
    """Tests para entidad Robot."""
    
    def test_create_robot(self):
        """Test crear robot."""
        robot = Robot(
            robot_id="robot-1",
            brand="KUKA",
            model="KR210"
        )
        
        assert robot.id == "robot-1"
        assert robot.brand == "KUKA"
        assert robot.model == "KR210"
        assert not robot.is_connected
    
    def test_robot_connect(self):
        """Test conectar robot."""
        robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        
        robot.connect()
        assert robot.is_connected
    
    def test_robot_connect_twice_fails(self):
        """Test que conectar dos veces falla."""
        robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        robot.connect()
        
        with pytest.raises(DomainError):
            robot.connect()
    
    def test_robot_disconnect_with_active_movements_fails(self):
        """Test que desconectar con movimientos activos falla."""
        robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        robot.connect()
        robot.register_movement("movement-1")
        
        with pytest.raises(DomainError):
            robot.disconnect()
    
    def test_robot_update_position(self):
        """Test actualizar posición."""
        robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        robot.connect()
        
        new_pos = Position(x=1.0, y=2.0, z=3.0)
        robot.update_position(new_pos)
        
        assert robot.current_position == new_pos
    
    def test_robot_update_position_when_disconnected_fails(self):
        """Test que actualizar posición desconectado falla."""
        robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
        
        new_pos = Position(x=1.0, y=2.0, z=3.0)
        with pytest.raises(DomainError):
            robot.update_position(new_pos)


class TestRobotMovement:
    """Tests para entidad RobotMovement."""
    
    def test_create_movement(self):
        """Test crear movimiento."""
        target_pos = Position(x=1.0, y=2.0, z=3.0)
        movement = RobotMovement(
            robot_id="robot-1",
            target_position=target_pos
        )
        
        assert movement.robot_id == "robot-1"
        assert movement.target_position == target_pos
        assert movement.status == MovementStatus.PENDING
    
    def test_movement_start(self):
        """Test iniciar movimiento."""
        movement = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=1.0, y=2.0, z=3.0)
        )
        
        movement.start()
        assert movement.status == MovementStatus.EXECUTING
        assert movement._started_at is not None
    
    def test_movement_start_twice_fails(self):
        """Test que iniciar dos veces falla."""
        movement = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=1.0, y=2.0, z=3.0)
        )
        movement.start()
        
        with pytest.raises(DomainError):
            movement.start()
    
    def test_movement_complete(self):
        """Test completar movimiento."""
        movement = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=1.0, y=2.0, z=3.0)
        )
        movement.start()
        
        final_pos = Position(x=1.0, y=2.0, z=3.0)
        movement.complete(final_pos)
        
        assert movement.status == MovementStatus.COMPLETED
        assert movement.current_position == final_pos
        assert movement._completed_at is not None
    
    def test_movement_fail(self):
        """Test marcar movimiento como fallido."""
        movement = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=1.0, y=2.0, z=3.0)
        )
        movement.start()
        
        movement.fail("Error de conexión")
        
        assert movement.status == MovementStatus.FAILED
        assert movement._error_message == "Error de conexión"
    
    def test_movement_domain_events(self):
        """Test que se emiten domain events."""
        movement = RobotMovement(
            robot_id="robot-1",
            target_position=Position(x=1.0, y=2.0, z=3.0)
        )
        
        # Iniciar movimiento emite evento
        movement.start()
        events = movement.get_domain_events()
        assert len(events) == 1
        assert events[0].movement_id == movement.id
        
        # Completar movimiento emite evento
        movement.complete(Position(x=1.0, y=2.0, z=3.0))
        events = movement.get_domain_events()
        assert len(events) == 2




