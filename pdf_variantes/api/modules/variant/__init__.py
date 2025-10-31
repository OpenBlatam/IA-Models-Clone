"""
Variant Module
Complete module for variant generation and management
"""

from .domain import VariantEntity, VariantFactory
from .application import (
    GenerateVariantsUseCase,
    GetVariantUseCase,
    ListVariantsUseCase
)
from .infrastructure import VariantRepository
from .presentation import VariantController, VariantPresenter

__all__ = [
    "VariantEntity",
    "VariantFactory",
    "GenerateVariantsUseCase",
    "GetVariantUseCase",
    "ListVariantsUseCase",
    "VariantRepository",
    "VariantController",
    "VariantPresenter"
]






