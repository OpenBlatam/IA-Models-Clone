"""
Dependency Injection Module

Provides dependency injection container and utilities.
"""

from .container import DIContainer, get_container, register_service, get_service

__all__ = [
    "DIContainer",
    "get_container",
    "register_service",
    "get_service",
]
