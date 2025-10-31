"""
Flask Blueprints
===============

Ultra-advanced Flask blueprints with modular architecture.
"""

from flask import Blueprint
from .auth import auth_bp
from .optimization import optimization_bp
from .performance import performance_bp
from .security import security_bp
from .ml import ml_bp
from .ai import ai_bp
from .quantum import quantum_bp
from .edge import edge_bp

__all__ = [
    'auth_bp',
    'optimization_bp', 
    'performance_bp',
    'security_bp',
    'ml_bp',
    'ai_bp',
    'quantum_bp',
    'edge_bp'
]









