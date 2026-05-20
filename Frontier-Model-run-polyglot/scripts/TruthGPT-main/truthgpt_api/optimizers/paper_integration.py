"""
Paper Integration for Optimizers
================================

Integrates research papers to enhance optimizer performance and capabilities.
This module bridges the optimization_core papers system with the optimizer adapters.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Try to import paper registry
_PAPER_REGISTRY_AVAILABLE = False
_PaperRegistry = None
_PaperAdapter = None

try:
    from optimization_core.core.papers import (
        PaperRegistry,
        get_paper_registry,
        PaperAdapter,
        ModelEnhancer
    )
    _PAPER_REGISTRY_AVAILABLE = True
    _PaperRegistry = PaperRegistry
    _PaperAdapter = PaperAdapter
    logger.info("✅ Paper registry available for optimizer enhancements")
except ImportError as e:
    logger.debug(f"Paper registry not available: {e}")


def is_paper_registry_available() -> bool:
    """Check if paper registry is available."""
    return _PAPER_REGISTRY_AVAILABLE


def get_optimizer_papers(
    category: Optional[str] = None,
    min_speedup: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Get papers relevant to optimizers.
    
    Args:
        category: Paper category filter
        min_speedup: Minimum speedup requirement
    
    Returns:
        List of paper metadata dictionaries
    """
    if not _PAPER_REGISTRY_AVAILABLE:
        return []
    
    try:
        registry = get_paper_registry()
        papers = registry.search_papers(
            query="optimizer optimization gradient descent adam sgd",
            category=category or "optimization",
            min_speedup=min_speedup
        )
        
        return [
            {
                'paper_id': p.paper_id,
                'paper_name': p.paper_name,
                'category': p.category,
                'speedup': p.speedup,
                'accuracy_improvement': p.accuracy_improvement,
                'memory_impact': p.memory_impact,
                'key_techniques': p.key_techniques,
                'arxiv_id': p.arxiv_id,
                'year': p.year,
                'authors': p.authors
            }
            for p in papers
        ]
    except Exception as e:
        logger.error(f"Error getting optimizer papers: {e}")
        return []


def suggest_papers_for_optimizer(
    optimizer_type: str,
    problem_type: str = "general"
) -> List[Dict[str, Any]]:
    """
    Suggest relevant papers for an optimizer type.
    
    Args:
        optimizer_type: Type of optimizer (adam, sgd, etc.)
        problem_type: Type of problem (general, deep_network, sparse, adversarial)
    
    Returns:
        List of suggested papers
    """
    if not _PAPER_REGISTRY_AVAILABLE:
        return []
    
    try:
        registry = get_paper_registry()
        
        # Search for papers relevant to the optimizer type
        query = f"{optimizer_type} optimizer"
        papers = registry.search_papers(query=query, category="optimization")
        
        # Filter based on problem type
        if problem_type == "deep_network":
            papers = [p for p in papers if p.memory_impact in ["low", "medium"]]
        elif problem_type == "sparse":
            papers = [p for p in papers if "sparse" in " ".join(p.key_techniques).lower()]
        elif problem_type == "adversarial":
            papers = [p for p in papers if "adversarial" in " ".join(p.key_techniques).lower()]
        
        # Sort by relevance (speedup + accuracy)
        papers.sort(
            key=lambda p: (p.speedup or 0) + (p.accuracy_improvement or 0) * 0.1,
            reverse=True
        )
        
        return [
            {
                'paper_id': p.paper_id,
                'paper_name': p.paper_name,
                'relevance_score': (p.speedup or 0) + (p.accuracy_improvement or 0) * 0.1,
                'speedup': p.speedup,
                'accuracy_improvement': p.accuracy_improvement,
                'memory_impact': p.memory_impact,
                'key_techniques': p.key_techniques
            }
            for p in papers[:5]  # Top 5
        ]
    except Exception as e:
        logger.error(f"Error suggesting papers: {e}")
        return []


def get_paper_enhanced_params(
    optimizer_type: str,
    paper_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get enhanced parameters from a research paper.
    
    Args:
        optimizer_type: Type of optimizer
        paper_id: Optional specific paper ID
    
    Returns:
        Dictionary with enhanced parameters or None
    """
    if not _PAPER_REGISTRY_AVAILABLE:
        return None
    
    try:
        registry = get_paper_registry()
        
        if paper_id:
            paper = registry.load_paper(paper_id)
            if paper and paper.is_available():
                # Try to extract optimizer parameters from paper
                if hasattr(paper, 'get_optimizer_params'):
                    return paper.get_optimizer_params(optimizer_type)
        else:
            # Search for best paper for this optimizer
            papers = suggest_papers_for_optimizer(optimizer_type)
            if papers:
                best_paper = registry.load_paper(papers[0]['paper_id'])
                if best_paper and best_paper.is_available():
                    if hasattr(best_paper, 'get_optimizer_params'):
                        return best_paper.get_optimizer_params(optimizer_type)
        
        return None
    except Exception as e:
        logger.error(f"Error getting paper-enhanced params: {e}")
        return None


def get_paper_based_recommendations(
    optimizer_type: str,
    problem_type: str = "general"
) -> Dict[str, Any]:
    """
    Get recommendations based on research papers.
    
    Args:
        optimizer_type: Type of optimizer
        problem_type: Type of problem
    
    Returns:
        Dictionary with paper-based recommendations
    """
    if not _PAPER_REGISTRY_AVAILABLE:
        return {
            'available': False,
            'message': 'Paper registry not available'
        }
    
    try:
        suggested_papers = suggest_papers_for_optimizer(optimizer_type, problem_type)
        all_papers = get_optimizer_papers(category="optimization")
        
        return {
            'available': True,
            'suggested_papers': suggested_papers,
            'total_papers': len(all_papers),
            'recommendations': {
                'top_paper': suggested_papers[0] if suggested_papers else None,
                'key_techniques': list(set(
                    tech
                    for paper in suggested_papers[:3]
                    for tech in paper.get('key_techniques', [])
                )),
                'average_speedup': sum(
                    p.get('speedup', 0) or 0
                    for p in suggested_papers
                ) / len(suggested_papers) if suggested_papers else 0
            }
        }
    except Exception as e:
        logger.error(f"Error getting paper recommendations: {e}")
        return {
            'available': False,
            'error': str(e)
        }
