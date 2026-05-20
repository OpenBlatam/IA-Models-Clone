"""
API Dependencies
================
Dependencias compartidas para endpoints de API.
"""

from ...services.coaching_service import DogTrainingCoach


def get_coaching_service() -> DogTrainingCoach:
    """Dependency para obtener servicio de coaching."""
    return DogTrainingCoach()


