from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .core.models import (
from .core.engine import FacebookPostsEngine, create_facebook_posts_engine
from .application.use_cases import (
from .optimization.base import (
from .main import FacebookPostsSystem, create_facebook_posts_system, quick_generate_post
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
📦 Facebook Posts - Source Package
=================================

Paquete principal que contiene todo el código fuente del sistema
de Facebook Posts refactorizado y consolidado.
"""

# Version info
__version__ = "3.0.0"
__author__ = "Facebook Posts Team"
__description__ = "Sistema de Facebook Posts con optimizaciones avanzadas"

# Core exports
    FacebookPost, PostRequest, PostResponse, PostStatus, 
    ContentType, AudienceType, OptimizationLevel, QualityTier,
    FacebookPostFactory, PostMetrics, ContentIdentifier, PublicationWindow
)


# Application exports
    UseCase, GeneratePostUseCase, AnalyzePostUseCase, 
    ApprovePostUseCase, PublishPostUseCase, GetAnalyticsUseCase,
    UseCaseFactory
)

# Optimization exports
    Optimizer, AsyncOptimizer, SyncOptimizer, OptimizationMetrics,
    OptimizationContext, OptimizationResult, OptimizationPipeline,
    OptimizerFactory, optimizer, require_config
)

# Main system export

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__description__',
    
    # Core models
    'FacebookPost',
    'PostRequest', 
    'PostResponse',
    'PostStatus',
    'ContentType',
    'AudienceType',
    'OptimizationLevel',
    'QualityTier',
    'FacebookPostFactory',
    'PostMetrics',
    'ContentIdentifier',
    'PublicationWindow',
    
    # Core engine
    'FacebookPostsEngine',
    'create_facebook_posts_engine',
    
    # Application
    'UseCase',
    'GeneratePostUseCase',
    'AnalyzePostUseCase',
    'ApprovePostUseCase',
    'PublishPostUseCase',
    'GetAnalyticsUseCase',
    'UseCaseFactory',
    
    # Optimization
    'Optimizer',
    'AsyncOptimizer',
    'SyncOptimizer',
    'OptimizationMetrics',
    'OptimizationContext',
    'OptimizationResult',
    'OptimizationPipeline',
    'OptimizerFactory',
    'optimizer',
    'require_config',
    
    # Main system
    'FacebookPostsSystem',
    'create_facebook_posts_system',
    'quick_generate_post'
] 