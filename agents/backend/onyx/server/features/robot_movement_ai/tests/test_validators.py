"""
Validator Tests - Tests para validadores
=========================================

Tests para módulo de validación de entrada.
"""

import pytest
import numpy as np
from ..utils.input_validators import (
    validate_position,
    validate_quaternion,
    validate_waypoints,
    validate_message,
    validate_obstacles
)
from ..core.exceptions import ValidationError


class TestPositionValidator:
    """Tests para validación de posición."""
    
    def test_valid_position(self):
        """Test de posición válida."""
        result = validate_position(0.5, 0.3, 0.2)
        assert result == (0.5, 0.3, 0.2)
    
    def test_position_out_of_range_x(self):
        """Test de posición X fuera de rango."""
        with pytest.raises(ValidationError) as exc_info:
            validate_position(15.0, 0.3, 0.2)
        assert "X coordinate" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_X_COORDINATE"
    
    def test_position_out_of_range_y(self):
        """Test de posición Y fuera de rango."""
        with pytest.raises(ValidationError) as exc_info:
            validate_position(0.5, -15.0, 0.2)
        assert "Y coordinate" in str(exc_info.value)
    
    def test_position_custom_bounds(self):
        """Test con límites personalizados."""
        result = validate_position(5.0, 5.0, 5.0, min_value=-5.0, max_value=5.0)
        assert result == (5.0, 5.0, 5.0)


class TestQuaternionValidator:
    """Tests para validación de quaternion."""
    
    def test_valid_quaternion(self):
        """Test de quaternion válido."""
        quat = [0.0, 0.0, 0.0, 1.0]
        result = validate_quaternion(quat)
        assert np.allclose(result, np.array(quat) / np.linalg.norm(quat))
    
    def test_quaternion_wrong_length(self):
        """Test de quaternion con longitud incorrecta."""
        with pytest.raises(ValidationError) as exc_info:
            validate_quaternion([0.0, 0.0, 0.0])
        assert "exactly 4 elements" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_QUATERNION_LENGTH"
    
    def test_quaternion_not_normalized(self):
        """Test de quaternion no normalizado."""
        with pytest.raises(ValidationError) as exc_info:
            validate_quaternion([2.0, 0.0, 0.0, 0.0], tolerance=0.1)
        assert "normalized" in str(exc_info.value)
        assert exc_info.value.error_code == "QUATERNION_NOT_NORMALIZED"
    
    def test_quaternion_normalization(self):
        """Test de normalización automática."""
        quat = [2.0, 0.0, 0.0, 0.0]
        result = validate_quaternion(quat, tolerance=0.5)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01


class TestWaypointValidator:
    """Tests para validación de waypoints."""
    
    def test_valid_waypoints(self):
        """Test de waypoints válidos."""
        waypoints = [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 0.5, "y": 0.3, "z": 0.2},
            {"x": 1.0, "y": 0.5, "z": 0.4}
        ]
        result = validate_waypoints(waypoints)
        assert len(result) == 3
    
    def test_insufficient_waypoints(self):
        """Test con pocos waypoints."""
        with pytest.raises(ValidationError) as exc_info:
            validate_waypoints([{"x": 0.0, "y": 0.0, "z": 0.0}])
        assert "At least" in str(exc_info.value)
        assert exc_info.value.error_code == "INSUFFICIENT_WAYPOINTS"
    
    def test_too_many_waypoints(self):
        """Test con demasiados waypoints."""
        waypoints = [{"x": float(i), "y": 0.0, "z": 0.0} for i in range(101)]
        with pytest.raises(ValidationError) as exc_info:
            validate_waypoints(waypoints)
        assert "Maximum" in str(exc_info.value)
        assert exc_info.value.error_code == "TOO_MANY_WAYPOINTS"
    
    def test_invalid_waypoint_format(self):
        """Test con formato de waypoint inválido."""
        waypoints = [{"invalid": "data"}]
        with pytest.raises(ValidationError):
            validate_waypoints(waypoints, min_waypoints=1)


class TestMessageValidator:
    """Tests para validación de mensajes."""
    
    def test_valid_message(self):
        """Test de mensaje válido."""
        result = validate_message("Hello, robot!")
        assert result == "Hello, robot!"
    
    def test_empty_message(self):
        """Test de mensaje vacío."""
        with pytest.raises(ValidationError) as exc_info:
            validate_message("")
        assert "cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == "EMPTY_MESSAGE"
    
    def test_message_too_long(self):
        """Test de mensaje muy largo."""
        long_message = "x" * 10001
        with pytest.raises(ValidationError) as exc_info:
            validate_message(long_message)
        assert "too long" in str(exc_info.value)
        assert exc_info.value.error_code == "MESSAGE_TOO_LONG"
    
    def test_message_trimming(self):
        """Test de trimming automático."""
        result = validate_message("  Hello  ")
        assert result == "Hello"


class TestObstacleValidator:
    """Tests para validación de obstáculos."""
    
    def test_valid_obstacles(self):
        """Test de obstáculos válidos."""
        obstacles = [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        ]
        result = validate_obstacles(obstacles)
        assert len(result) == 2
    
    def test_obstacle_wrong_format(self):
        """Test con formato de obstáculo incorrecto."""
        with pytest.raises(ValidationError) as exc_info:
            validate_obstacles([[0.0, 0.0, 0.0]])
        assert "6 coordinates" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_OBSTACLE_FORMAT"
    
    def test_obstacle_invalid_bounds(self):
        """Test con límites inválidos."""
        with pytest.raises(ValidationError) as exc_info:
            validate_obstacles([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        assert "max coordinates must be >= min coordinates" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_OBSTACLE_BOUNDS"
    
    def test_too_many_obstacles(self):
        """Test con demasiados obstáculos."""
        obstacles = [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]] * 1001
        with pytest.raises(ValidationError) as exc_info:
            validate_obstacles(obstacles)
        assert "Maximum" in str(exc_info.value)
        assert exc_info.value.error_code == "TOO_MANY_OBSTACLES"

