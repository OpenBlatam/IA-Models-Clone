"""
Exception Tests - Tests para sistema de excepciones
====================================================

Tests para el sistema de excepciones mejorado.
"""

import pytest
from ..core.exceptions import (
    BaseRobotException,
    RobotMovementError,
    TrajectoryError,
    IKError,
    ValidationError,
    SafetyError,
    CollisionDetectedError,
    SafetyLimitExceededError,
    EmergencyStopError
)


class TestBaseRobotException:
    """Tests para BaseRobotException."""
    
    def test_basic_exception(self):
        """Test de excepción básica."""
        exc = BaseRobotException("Test error")
        assert str(exc) == "BaseRobotException: Test error"
        assert exc.error_code == "BaseRobotException"
        assert exc.message == "Test error"
    
    def test_exception_with_error_code(self):
        """Test con código de error personalizado."""
        exc = BaseRobotException("Test error", error_code="TEST_ERROR")
        assert exc.error_code == "TEST_ERROR"
    
    def test_exception_with_details(self):
        """Test con detalles adicionales."""
        exc = BaseRobotException(
            "Test error",
            details={"field": "value", "number": 42}
        )
        assert exc.details == {"field": "value", "number": 42}
        assert "Details:" in str(exc)
    
    def test_exception_to_dict(self):
        """Test de serialización a diccionario."""
        exc = BaseRobotException(
            "Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        data = exc.to_dict()
        assert data["error_type"] == "BaseRobotException"
        assert data["error_code"] == "TEST_ERROR"
        assert data["message"] == "Test error"
        assert data["details"] == {"key": "value"}
    
    def test_exception_with_cause(self):
        """Test con excepción causante."""
        cause = ValueError("Original error")
        exc = BaseRobotException("Wrapped error", cause=cause)
        assert exc.cause == cause
        assert exc.traceback is not None


class TestRobotMovementExceptions:
    """Tests para excepciones de movimiento."""
    
    def test_robot_movement_error(self):
        """Test de RobotMovementError."""
        exc = RobotMovementError("Movement failed")
        assert isinstance(exc, BaseRobotException)
        assert "Movement failed" in str(exc)
    
    def test_trajectory_error(self):
        """Test de TrajectoryError."""
        exc = TrajectoryError("Invalid trajectory")
        assert isinstance(exc, RobotMovementError)
        assert "Invalid trajectory" in str(exc)
    
    def test_ik_error(self):
        """Test de IKError."""
        exc = IKError("IK solution not found")
        assert isinstance(exc, RobotMovementError)
        assert "IK solution not found" in str(exc)


class TestSafetyExceptions:
    """Tests para excepciones de seguridad."""
    
    def test_safety_error(self):
        """Test de SafetyError."""
        exc = SafetyError("Safety violation")
        assert isinstance(exc, RobotMovementError)
        assert "Safety violation" in str(exc)
    
    def test_collision_detected_error(self):
        """Test de CollisionDetectedError."""
        exc = CollisionDetectedError(
            "Collision detected",
            details={"position": [0.5, 0.3, 0.2]}
        )
        assert isinstance(exc, SafetyError)
        assert "Collision detected" in str(exc)
        assert exc.details["position"] == [0.5, 0.3, 0.2]
    
    def test_safety_limit_exceeded_error(self):
        """Test de SafetyLimitExceededError."""
        exc = SafetyLimitExceededError(
            "Velocity limit exceeded",
            details={"current_velocity": 2.0, "max_velocity": 1.0}
        )
        assert isinstance(exc, SafetyError)
        assert "Velocity limit exceeded" in str(exc)
    
    def test_emergency_stop_error(self):
        """Test de EmergencyStopError."""
        exc = EmergencyStopError("Emergency stop activated")
        assert isinstance(exc, SafetyError)
        assert "Emergency stop activated" in str(exc)


class TestValidationException:
    """Tests para ValidationError."""
    
    def test_validation_error(self):
        """Test de ValidationError."""
        exc = ValidationError(
            "Invalid input",
            error_code="INVALID_INPUT",
            details={"field": "x", "value": 15.0}
        )
        assert isinstance(exc, BaseRobotException)
        assert exc.error_code == "INVALID_INPUT"
        assert exc.details["field"] == "x"
    
    def test_validation_error_serialization(self):
        """Test de serialización de ValidationError."""
        exc = ValidationError("Invalid data", details={"errors": ["error1", "error2"]})
        data = exc.to_dict()
        assert data["error_type"] == "ValidationError"
        assert "errors" in data["details"]

