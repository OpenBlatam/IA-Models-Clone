"""
Tests for input validators.
"""

import pytest
from ..utils.validators import (
    validate_question,
    validate_robot_type,
    validate_maintenance_type,
    validate_difficulty_level,
    validate_sensor_data,
    sanitize_text
)


class TestValidators:
    """Test cases for validators."""
    
    def test_validate_question_valid(self):
        """Test valid question."""
        assert validate_question("¿Cómo cambio el aceite de un robot?") is True
        assert validate_question("Explica el procedimiento de mantenimiento preventivo") is True
    
    def test_validate_question_invalid(self):
        """Test invalid questions."""
        assert validate_question("") is False  # Too short
        assert validate_question("a" * 501) is False  # Too long
        assert validate_question("   ") is False  # Only whitespace
    
    def test_validate_robot_type(self):
        """Test robot type validation."""
        supported = ["robots_industriales", "robots_medicos", "robots_servicio"]
        assert validate_robot_type("robots_industriales", supported) is True
        assert validate_robot_type("robots_medicos", supported) is True
        assert validate_robot_type("invalid_type", supported) is False
    
    def test_validate_maintenance_type(self):
        """Test maintenance type validation."""
        supported = ["preventivo", "correctivo", "predictivo"]
        assert validate_maintenance_type("preventivo", supported) is True
        assert validate_maintenance_type("correctivo", supported) is True
        assert validate_maintenance_type("invalid", supported) is False
    
    def test_validate_difficulty_level(self):
        """Test difficulty level validation."""
        supported = ["principiante", "intermedio", "avanzado", "experto"]
        assert validate_difficulty_level("principiante", supported) is True
        assert validate_difficulty_level("experto", supported) is True
        assert validate_difficulty_level("invalid", supported) is False
    
    def test_validate_sensor_data_valid(self):
        """Test valid sensor data."""
        valid_data = {
            "temperature": 25.5,
            "pressure": 100.0,
            "vibration": 0.5,
            "current": 10
        }
        assert validate_sensor_data(valid_data) is True
    
    def test_validate_sensor_data_invalid(self):
        """Test invalid sensor data."""
        invalid_data = {
            "temperature": "not_a_number",
            "pressure": 100.0
        }
        assert validate_sensor_data(invalid_data) is False
    
    def test_validate_sensor_data_empty(self):
        """Test empty sensor data."""
        assert validate_sensor_data({}) is True  # Empty dict is valid
    
    def test_sanitize_text(self):
        """Test text sanitization."""
        assert sanitize_text("  hello world  ") == "hello world"
        assert sanitize_text("") == ""
        long_text = "a" * 2000
        assert len(sanitize_text(long_text)) == 1000  # Should be limited






