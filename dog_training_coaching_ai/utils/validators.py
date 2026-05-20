"""
Validation Utilities
====================
"""

from typing import Optional, List, Callable
from ...core.exceptions import ValidationException
from ...core.error_codes import ErrorCode
from .constants import VALID_TRAINING_GOALS, VALID_EXPERIENCE_LEVELS, VALID_DOG_SIZES


def _validate_enum(
    value: Optional[str],
    valid_values: List[str],
    error_code: ErrorCode,
    field: str,
    error_message: str
) -> None:
    """
    Validar que un valor esté en una lista de valores válidos.
    
    Args:
        value: Valor a validar
        valid_values: Lista de valores válidos
        error_code: Código de error
        field: Nombre del campo
        error_message: Mensaje de error
    """
    if value and value.lower() not in [v.lower() for v in valid_values]:
        raise ValidationException(
            error_message,
            error_code=error_code,
            field=field
        )


def validate_dog_breed(breed: Optional[str]) -> None:
    """Validar raza de perro."""
    if breed and len(breed.strip()) < 2:
        raise ValidationException(
            "Dog breed must be at least 2 characters",
            error_code=ErrorCode.INVALID_DOG_BREED,
            field="dog_breed"
        )


def validate_dog_age(age: Optional[str]) -> None:
    """Validar edad del perro."""
    if age:
        age_lower = age.lower()
        valid_ages = ["puppy", "young", "adult", "senior", "months", "years", "weeks"]
        if not any(valid in age_lower for valid in valid_ages) and not age_lower.replace(" ", "").replace("-", "").isdigit():
            raise ValidationException(
                "Invalid dog age format",
                error_code=ErrorCode.INVALID_DOG_AGE,
                field="dog_age"
            )


def validate_training_goals(goals: List[str]) -> None:
    """Validar objetivos de entrenamiento."""
    if not goals:
        raise ValidationException(
            "At least one training goal is required",
            error_code=ErrorCode.INVALID_TRAINING_GOAL,
            field="training_goals"
        )
    
    for goal in goals:
        if goal.lower() not in VALID_TRAINING_GOALS and len(goal) < 3:
            raise ValidationException(
                f"Invalid training goal: {goal}. Valid goals: {', '.join(VALID_TRAINING_GOALS)}",
                error_code=ErrorCode.INVALID_TRAINING_GOAL,
                field="training_goals"
            )


def validate_experience_level(level: Optional[str]) -> None:
    """Validar nivel de experiencia."""
    _validate_enum(
        level,
        VALID_EXPERIENCE_LEVELS,
        ErrorCode.INVALID_EXPERIENCE_LEVEL,
        "experience_level",
        f"Experience level must be one of: {', '.join(VALID_EXPERIENCE_LEVELS)}"
    )


def validate_dog_size(size: Optional[str]) -> None:
    """Validar tamaño del perro."""
    _validate_enum(
        size,
        VALID_DOG_SIZES,
        ErrorCode.INVALID_DOG_SIZE,
        "dog_size",
        f"Dog size must be one of: {', '.join(VALID_DOG_SIZES)}"
    )

