"""
Paper-Enhanced Optimizer Utilities
==================================

Utilities for creating optimizers enhanced with research paper techniques.

Single Responsibility: Provide paper-enhanced optimizer creation and recommendations.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from .optimizer_adapter import OptimizationCoreAdapter
from .optimizer_constants import DEFAULT_LEARNING_RATE
from .paper_integration import (
    is_paper_registry_available,
    get_optimizer_papers,
    suggest_papers_for_optimizer,
    get_paper_enhanced_params,
    get_paper_based_recommendations
)

logger = logging.getLogger(__name__)


def create_paper_enhanced_optimizer(
    optimizer_type: str,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    problem_type: str = 'general',
    use_papers: bool = True,
    **kwargs
) -> OptimizationCoreAdapter:
    """
    Create an optimizer enhanced with research paper techniques.
    
    Args:
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        problem_type: Type of problem (general, deep_network, sparse, adversarial)
        use_papers: Whether to use paper-based enhancements
        **kwargs: Additional optimizer parameters
    
    Returns:
        OptimizationCoreAdapter with paper enhancements if available
    """
    # Try to get paper-enhanced parameters
    if use_papers and is_paper_registry_available():
        paper_params = get_paper_enhanced_params(optimizer_type)
        if paper_params:
            # Merge paper params with user params (user params take precedence)
            enhanced_kwargs = {**paper_params, **kwargs}
            logger.info(f"📚 Using paper-enhanced parameters for {optimizer_type}")
            return OptimizationCoreAdapter(
                optimizer_type=optimizer_type,
                learning_rate=learning_rate,
                **enhanced_kwargs
            )
    
    # Fallback to standard creation
    return OptimizationCoreAdapter(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        **kwargs
    )


def get_optimizer_with_paper_recommendations(
    optimizer_type: str,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    problem_type: str = 'general'
) -> Dict[str, Any]:
    """
    Get optimizer with paper-based recommendations.
    
    Args:
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        problem_type: Type of problem
    
    Returns:
        Dictionary with optimizer and paper recommendations
    """
    optimizer = create_paper_enhanced_optimizer(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        problem_type=problem_type
    )
    
    paper_recommendations = get_paper_based_recommendations(
        optimizer_type=optimizer_type,
        problem_type=problem_type
    )
    
    return {
        'optimizer': optimizer,
        'optimizer_config': optimizer.get_config(),
        'paper_recommendations': paper_recommendations,
        'paper_available': is_paper_registry_available()
    }


def get_paper_integration_summary() -> Dict[str, Any]:
    """
    Get comprehensive summary of paper integration with optimizers.
    
    Returns:
        Dictionary with paper integration summary
    """
    return {
        'paper_registry_available': is_paper_registry_available(),
        'total_papers': len(get_optimizer_papers()) if is_paper_registry_available() else 0,
        'integration_features': {
            'automatic_enhancement': 'Optimizers automatically use paper enhancements when available',
            'paper_recommendations': 'Get paper recommendations based on optimizer type and problem',
            'parameter_enhancement': 'Paper-based parameter optimization',
            'research_backed': 'All enhancements based on peer-reviewed research papers'
        },
        'available_functions': [
            'create_paper_enhanced_optimizer',
            'get_optimizer_with_paper_recommendations',
            'get_optimizer_papers',
            'suggest_papers_for_optimizer',
            'get_paper_enhanced_params',
            'get_paper_based_recommendations'
        ],
        'usage_example': {
            'basic': f'optimizer = create_paper_enhanced_optimizer("adam", learning_rate={DEFAULT_LEARNING_RATE})',
            'with_recommendations': f'result = get_optimizer_with_paper_recommendations("adam", learning_rate={DEFAULT_LEARNING_RATE})',
            'get_papers': 'papers = get_optimizer_papers(category="optimization")',
            'suggest_papers': 'suggested = suggest_papers_for_optimizer("adam", problem_type="deep_network")'
        },
        'benefits': [
            'Automatically applies state-of-the-art research to optimizers',
            'Improves convergence and performance based on latest papers',
            'Provides recommendations for specific problem types',
            'Transparent integration - papers are clearly identified',
            'Fallback to standard optimizers if papers unavailable'
        ]
    }

