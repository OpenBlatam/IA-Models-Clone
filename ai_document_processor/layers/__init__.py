"""
Layered Architecture - AI Document Processor
==========================================

Clean layered architecture with clear separation of concerns and dependency inversion.
"""

from .presentation import PresentationLayer
from .business import BusinessLayer
from .data import DataLayer
from .infrastructure import InfrastructureLayer
from .domain import DomainLayer

__version__ = "5.0.0"
__author__ = "AI Document Processor Team"
__description__ = "Layered architecture for AI document processing"

__all__ = [
    "PresentationLayer",
    "BusinessLayer", 
    "DataLayer",
    "InfrastructureLayer",
    "DomainLayer",
]

















