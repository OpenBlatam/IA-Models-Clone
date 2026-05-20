"""
Tests for Validators
====================
"""

import pytest
from ...core.exceptions import ValidationException
from ...core.error_codes import ErrorCode
from ...utils.validators import (
    validate_dog_breed,
    validate_dog_age,
    validate_dog_size,
    validate_training_goals,
    validate_experience_level
)


def test_validate_dog_breed_valid():
    """Test valid dog breeds."""
    validate_dog_breed("Golden Retriever")
    validate_dog_breed("German Shepherd")
    validate_dog_breed(None)  # Optional


def test_validate_dog_breed_invalid():
    """Test invalid dog breeds."""
    with pytest.raises(ValidationException) as exc_info:
        validate_dog_breed("")
    assert exc_info.value.error_code == ErrorCode.INVALID_DOG_BREED


def test_validate_dog_age_valid():
    """Test valid dog ages."""
    validate_dog_age("2 years")
    validate_dog_age("6 months")
    validate_dog_age(None)  # Optional


def test_validate_dog_age_invalid():
    """Test invalid dog ages."""
    with pytest.raises(ValidationException) as exc_info:
        validate_dog_age("")
    assert exc_info.value.error_code == ErrorCode.INVALID_DOG_AGE


def test_validate_dog_size_valid():
    """Test valid dog sizes."""
    validate_dog_size("small")
    validate_dog_size("medium")
    validate_dog_size("large")
    validate_dog_size("giant")
    validate_dog_size(None)  # Optional


def test_validate_dog_size_invalid():
    """Test invalid dog sizes."""
    with pytest.raises(ValidationException) as exc_info:
        validate_dog_size("extra-large")
    assert exc_info.value.error_code == ErrorCode.INVALID_DOG_SIZE


def test_validate_training_goals_valid():
    """Test valid training goals."""
    validate_training_goals(["obedience", "agility"])
    validate_training_goals(["behavior"])


def test_validate_training_goals_invalid():
    """Test invalid training goals."""
    with pytest.raises(ValidationException):
        validate_training_goals([])
    
    with pytest.raises(ValidationException):
        validate_training_goals(["invalid_goal"])


def test_validate_experience_level_valid():
    """Test valid experience levels."""
    validate_experience_level("beginner")
    validate_experience_level("intermediate")
    validate_experience_level("advanced")
    validate_experience_level(None)  # Optional


def test_validate_experience_level_invalid():
    """Test invalid experience levels."""
    with pytest.raises(ValidationException) as exc_info:
        validate_experience_level("expert")
    assert exc_info.value.error_code == ErrorCode.INVALID_EXPERIENCE_LEVEL

