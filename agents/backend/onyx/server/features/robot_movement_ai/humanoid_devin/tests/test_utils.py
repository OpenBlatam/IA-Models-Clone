"""
Tests para utilidades del Humanoid Devin Robot
==============================================

Tests unitarios para las funciones de utilidad.
"""

import pytest
import numpy as np
from typing import List, Tuple

from ..utils import (
    normalize_quaternion,
    quaternion_to_euler,
    euler_to_quaternion,
    clamp,
    normalize_angle,
    validate_joint_positions,
    interpolate_joint_positions,
    calculate_distance,
    smooth_trajectory,
    validate_pose,
    get_joint_velocity
)


class TestQuaternionUtils:
    """Tests para utilidades de quaterniones."""
    
    def test_normalize_quaternion(self):
        """Test normalización de quaternion."""
        # Quaternion no normalizado
        q = np.array([2.0, 0.0, 0.0, 2.0])
        normalized = normalize_quaternion(q)
        
        assert np.allclose(np.linalg.norm(normalized), 1.0)
        assert len(normalized) == 4
    
    def test_normalize_quaternion_invalid(self):
        """Test que quaternion inválido lanza error."""
        with pytest.raises(ValueError):
            normalize_quaternion([0.0, 0.0, 0.0, 0.0])  # Norma cero
    
    def test_quaternion_to_euler(self):
        """Test conversión quaternion a Euler."""
        q = normalize_quaternion([0.0, 0.0, 0.0, 1.0])
        roll, pitch, yaw = quaternion_to_euler(q)
        
        assert isinstance(roll, float)
        assert isinstance(pitch, float)
        assert isinstance(yaw, float)
        assert -np.pi <= roll <= np.pi
        assert -np.pi/2 <= pitch <= np.pi/2
        assert -np.pi <= yaw <= np.pi
    
    def test_euler_to_quaternion(self):
        """Test conversión Euler a quaternion."""
        quaternion = euler_to_quaternion(0.0, 0.0, 0.0)
        
        assert len(quaternion) == 4
        assert np.allclose(np.linalg.norm(quaternion), 1.0)
    
    def test_quaternion_round_trip(self):
        """Test ida y vuelta: Euler -> Quaternion -> Euler."""
        original = (0.5, 0.3, 0.2)
        quaternion = euler_to_quaternion(*original)
        result = quaternion_to_euler(quaternion)
        
        # Verificar que son aproximadamente iguales
        assert np.allclose(original, result, atol=1e-6)


class TestMathUtils:
    """Tests para utilidades matemáticas."""
    
    def test_clamp(self):
        """Test función clamp."""
        assert clamp(5.0, 0.0, 10.0) == 5.0
        assert clamp(-5.0, 0.0, 10.0) == 0.0
        assert clamp(15.0, 0.0, 10.0) == 10.0
    
    def test_clamp_invalid_range(self):
        """Test que clamp lanza error con rango inválido."""
        with pytest.raises(ValueError):
            clamp(5.0, 10.0, 0.0)  # min > max
    
    def test_normalize_angle(self):
        """Test normalización de ángulo."""
        assert abs(normalize_angle(3 * np.pi)) < np.pi
        assert abs(normalize_angle(-3 * np.pi)) < np.pi
        assert normalize_angle(0.0) == 0.0
        assert abs(normalize_angle(np.pi)) <= np.pi
    
    def test_calculate_distance(self):
        """Test cálculo de distancia."""
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [1.0, 0.0, 0.0]
        
        distance = calculate_distance(pos1, pos2)
        assert abs(distance - 1.0) < 1e-6
    
    def test_calculate_distance_3d(self):
        """Test cálculo de distancia en 3D."""
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [1.0, 1.0, 1.0]
        
        distance = calculate_distance(pos1, pos2)
        expected = np.sqrt(3)
        assert abs(distance - expected) < 1e-6


class TestJointUtils:
    """Tests para utilidades de articulaciones."""
    
    def test_validate_joint_positions(self):
        """Test validación de posiciones de articulaciones."""
        positions = [0.0] * 32
        validated = validate_joint_positions(positions, dof=32)
        
        assert len(validated) == 32
        assert isinstance(validated, np.ndarray)
    
    def test_validate_joint_positions_wrong_length(self):
        """Test que validación falla con longitud incorrecta."""
        with pytest.raises(ValueError):
            validate_joint_positions([0.0] * 20, dof=32)
    
    def test_validate_joint_positions_with_limits(self):
        """Test validación con límites de articulaciones."""
        positions = [0.0] * 32
        limits = [(-np.pi, np.pi)] * 32
        
        validated = validate_joint_positions(positions, dof=32, joint_limits=limits)
        assert len(validated) == 32
    
    def test_interpolate_joint_positions(self):
        """Test interpolación de posiciones."""
        start = [0.0] * 32
        end = [1.0] * 32
        
        trajectory = interpolate_joint_positions(start, end, num_steps=10)
        
        assert trajectory.shape == (10, 32)
        assert np.allclose(trajectory[0], start)
        assert np.allclose(trajectory[-1], end)
    
    def test_smooth_trajectory(self):
        """Test suavizado de trayectoria."""
        trajectory = np.array([[i] * 32 for i in range(10)])
        smoothed = smooth_trajectory(trajectory, window_size=5)
        
        assert smoothed.shape == trajectory.shape
        assert len(smoothed) == len(trajectory)
    
    def test_get_joint_velocity(self):
        """Test cálculo de velocidad de articulaciones."""
        current = [0.0] * 32
        previous = [-0.1] * 32
        dt = 0.1
        
        velocities = get_joint_velocity(current, previous, dt)
        
        assert len(velocities) == 32
        assert np.allclose(velocities, [1.0] * 32)


class TestPoseUtils:
    """Tests para utilidades de pose."""
    
    def test_validate_pose(self):
        """Test validación de pose."""
        position = [0.3, -0.2, 1.0]
        orientation = [0.0, 0.0, 0.0, 1.0]
        
        valid_pos, valid_ori = validate_pose(position, orientation)
        
        assert len(valid_pos) == 3
        assert len(valid_ori) == 4
        assert np.allclose(np.linalg.norm(valid_ori), 1.0)
    
    def test_validate_pose_invalid_position(self):
        """Test que validación falla con posición inválida."""
        with pytest.raises(ValueError):
            validate_pose([0.0, 0.0], [0.0, 0.0, 0.0, 1.0])  # Posición con 2 elementos
    
    def test_validate_pose_invalid_orientation(self):
        """Test que validación falla con orientación inválida."""
        with pytest.raises(ValueError):
            validate_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])  # Quaternion con norma cero


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

