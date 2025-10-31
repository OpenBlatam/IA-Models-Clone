"""
PDF Variantes Database Package
Paquete de base de datos para el sistema PDF Variantes
"""

from .models import (
    Base, User, Document, Variant, Topic, BrainstormIdea,
    Collaboration, Annotation, Export, Analytics, DatabaseManager
)
from .migrations import MigrationManager
from .repository import (
    UserRepository, DocumentRepository, VariantRepository,
    TopicRepository, BrainstormRepository, CollaborationRepository,
    AnalyticsRepository
)

__all__ = [
    # Models
    "Base", "User", "Document", "Variant", "Topic", "BrainstormIdea",
    "Collaboration", "Annotation", "Export", "Analytics", "DatabaseManager",
    
    # Migrations
    "MigrationManager",
    
    # Repositories
    "UserRepository", "DocumentRepository", "VariantRepository",
    "TopicRepository", "BrainstormRepository", "CollaborationRepository",
    "AnalyticsRepository"
]
