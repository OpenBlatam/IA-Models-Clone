"""
Tests para validaciones del Humanoid Devin Robot
=================================================

Tests unitarios para validaciones de parámetros.
"""

import pytest
import numpy as np

from ..drivers.humanoid_devin_driver import HumanoidDevinDriver, RobotType
from ..exceptions import ValidationError, HumanoidRobotError


class TestDriverValidation:
    """Tests para validaciones del driver."""
    
    def test_invalid_robot_ip(self):
        """Test que IP vacía lanza error."""
        with pytest.raises(ValueError):
            HumanoidDevinDriver(robot_ip="", dof=32)
    
    def test_invalid_robot_port(self):
        """Test que puerto inválido lanza error."""
        with pytest.raises(ValueError):
            HumanoidDevinDriver(robot_ip="192.168.1.100", robot_port=0, dof=32)
        
        with pytest.raises(ValueError):
            HumanoidDevinDriver(robot_ip="192.168.1.100", robot_port=70000, dof=32)
    
    def test_invalid_dof(self):
        """Test que DOF inválido lanza error."""
        with pytest.raises(ValueError):
            HumanoidDevinDriver(robot_ip="192.168.1.100", dof=0)
        
        with pytest.raises(ValueError):
            HumanoidDevinDriver(robot_ip="192.168.1.100", dof=200)
    
    def test_invalid_robot_type(self):
        """Test que tipo de robot inválido lanza error."""
        with pytest.raises(ValueError):
            HumanoidDevinDriver(robot_ip="192.168.1.100", robot_type="invalid")
    
    def test_valid_robot_type_enum(self):
        """Test que tipo de robot enum es válido."""
        robot = HumanoidDevinDriver(
            robot_ip="192.168.1.100",
            robot_type=RobotType.GENERIC
        )
        assert robot.robot_type == RobotType.GENERIC
    
    def test_valid_robot_type_string(self):
        """Test que tipo de robot string es válido."""
        robot = HumanoidDevinDriver(
            robot_ip="192.168.1.100",
            robot_type="generic"
        )
        assert robot.robot_type == RobotType.GENERIC


class TestIntegrationValidation:
    """Tests para validaciones de integraciones."""
    
    @pytest.mark.asyncio
    async def test_set_joint_positions_wrong_length(self):
        """Test que posiciones con longitud incorrecta lanzan error."""
        robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
        await robot.connect()
        
        with pytest.raises(ValueError):
            await robot.set_joint_positions([0.0] * 20)  # Solo 20 en lugar de 32
        
        await robot.disconnect()
    
    @pytest.mark.asyncio
    async def test_set_joint_positions_invalid_type(self):
        """Test que posiciones con tipo incorrecto lanzan error."""
        robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
        await robot.connect()
        
        with pytest.raises(ValueError):
            await robot.set_joint_positions("invalid")  # String en lugar de lista
        
        await robot.disconnect()
    
    @pytest.mark.asyncio
    async def test_move_to_pose_invalid_position(self):
        """Test que posición inválida lanza error."""
        robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
        await robot.connect()
        
        with pytest.raises(ValueError):
            await robot.move_to_pose(
                position=[0.0, 0.0],  # Solo 2 elementos
                orientation=[0.0, 0.0, 0.0, 1.0]
            )
        
        await robot.disconnect()
    
    @pytest.mark.asyncio
    async def test_move_to_pose_invalid_orientation(self):
        """Test que orientación inválida lanza error."""
        robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
        await robot.connect()
        
        with pytest.raises(ValueError):
            await robot.move_to_pose(
                position=[0.0, 0.0, 0.0],
                orientation=[0.0, 0.0, 0.0, 0.0]  # Quaternion con norma cero
            )
        
        await robot.disconnect()
    
    @pytest.mark.asyncio
    async def test_walk_invalid_direction(self):
        """Test que dirección inválida lanza error."""
        robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
        await robot.connect()
        
        with pytest.raises(ValueError):
            await robot.walk(direction="invalid_direction")
        
        await robot.disconnect()
    
    @pytest.mark.asyncio
    async def test_walk_invalid_distance(self):
        """Test que distancia negativa lanza error."""
        robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
        await robot.connect()
        
        with pytest.raises(ValueError):
            await robot.walk(direction="forward", distance=-1.0)
        
        await robot.disconnect()
    
    @pytest.mark.asyncio
    async def test_walk_invalid_speed(self):
        """Test que velocidad fuera de rango lanza error."""
        robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
        await robot.connect()
        
        with pytest.raises(ValueError):
            await robot.walk(direction="forward", speed=2.0)  # > 1.0
        
        await robot.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

