from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from .core.entities.models import (
from .modular_engine import (
    from .turbo_optimization import get_turbo_optimizer
    from .ultra_turbo_engine import get_ultra_turbo_engine
    from .production_engine import get_production_engine
    import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🚀 NLP ENGINE OPTIMIZED - Clean Public API
==========================================

Sistema NLP ultra-optimizado con arquitectura modular.
API pública limpia siguiendo mejores prácticas.
"""

    TextInput,
    AnalysisResult,
    BatchResult,
    AnalysisType,
    OptimizationTier
)

    ModularNLPEngine,
    create_modular_engine,
    quick_sentiment_analysis,
    quick_quality_analysis
)

# Ultra-fast turbo engine
try:
    TURBO_AVAILABLE = True
except ImportError:
    TURBO_AVAILABLE = False
    get_turbo_optimizer = None
    get_ultra_turbo_engine = None

# Legacy compatibility
try:
except ImportError:
    get_production_engine = None

__version__ = "2.0.0"
__author__ = "NLP Optimization Team"

# Public API exports
__all__ = [
    # Main engine
    'ModularNLPEngine',
    'create_modular_engine',
    
    # Quick functions
    'quick_sentiment_analysis',
    'quick_quality_analysis',
    
    # Turbo engines
    'get_turbo_optimizer',
    'get_ultra_turbo_engine',
    
    # Entities
    'TextInput',
    'AnalysisResult',
    'BatchResult',
    'AnalysisType',
    'OptimizationTier',
    
    # Legacy
    'get_production_engine'
]


def get_engine(tier: OptimizationTier = OptimizationTier.ULTRA) -> ModularNLPEngine:
    """
    Factory function to create optimized NLP engine.
    
    Args:
        tier: Optimization tier to use
        
    Returns:
        ModularNLPEngine: Configured engine instance
        
    Example:
        >>> engine = get_engine(OptimizationTier.EXTREME)
        >>> await engine.initialize()
        >>> result = await engine.analyze_sentiment(["Great product!"])
    """
    return create_modular_engine(tier)


async def analyze_text(
    text: str, 
    analysis_type: str = "sentiment",
    tier: OptimizationTier = OptimizationTier.ULTRA
) -> dict:
    """
    Convenience function for single text analysis.
    
    Args:
        text: Text to analyze
        analysis_type: Type of analysis ("sentiment" or "quality")
        tier: Optimization tier
        
    Returns:
        dict: Analysis result
        
    Example:
        >>> result = await analyze_text("Amazing product!", "sentiment")
        >>> print(f"Score: {result['score']:.2f}")
    """
    engine = create_modular_engine(tier)
    await engine.initialize()
    
    return await engine.analyze_single(text, analysis_type)


async def analyze_batch(
    texts: list, 
    analysis_type: str = "sentiment",
    tier: OptimizationTier = OptimizationTier.ULTRA
) -> dict:
    """
    Convenience function for batch text analysis.
    
    Args:
        texts: List of texts to analyze
        analysis_type: Type of analysis ("sentiment" or "quality")
        tier: Optimization tier
        
    Returns:
        dict: Batch analysis result
        
    Example:
        >>> texts = ["Great!", "Terrible!", "Okay."]
        >>> result = await analyze_batch(texts, "sentiment")
        >>> print(f"Average score: {result['average']:.2f}")
    """
    engine = create_modular_engine(tier)
    await engine.initialize()
    
    if analysis_type == "sentiment":
        return await engine.analyze_sentiment(texts)
    elif analysis_type == "quality":
        return await engine.analyze_quality(texts)
    else:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")


# Performance utilities
async def benchmark_performance(
    num_texts: int = 100,
    tier: OptimizationTier = OptimizationTier.EXTREME
) -> dict:
    """
    Benchmark engine performance.
    
    Args:
        num_texts: Number of texts to test
        tier: Optimization tier
        
    Returns:
        dict: Performance metrics
    """
    
    engine = create_modular_engine(tier)
    await engine.initialize()
    
    # Create test data
    texts = [f"Test text number {i} for performance benchmarking." for i in range(num_texts)]
    
    # Benchmark sentiment analysis
    start = time.perf_counter()
    sentiment_result = await engine.analyze_sentiment(texts)
    sentiment_time = time.perf_counter() - start
    
    # Benchmark quality analysis  
    start = time.perf_counter()
    quality_result = await engine.analyze_quality(texts)
    quality_time = time.perf_counter() - start
    
    return {
        'num_texts': num_texts,
        'tier': tier.value,
        'sentiment': {
            'time_ms': sentiment_time * 1000,
            'throughput_ops_per_sec': num_texts / sentiment_time,
            'avg_time_per_text_ms': (sentiment_time * 1000) / num_texts
        },
        'quality': {
            'time_ms': quality_time * 1000,
            'throughput_ops_per_sec': num_texts / quality_time,
            'avg_time_per_text_ms': (quality_time * 1000) / num_texts
        },
        'overall': {
            'total_time_ms': (sentiment_time + quality_time) * 1000,
            'combined_throughput': (num_texts * 2) / (sentiment_time + quality_time)
        }
    }


# Health check
async def health_check() -> dict:
    """
    Perform system health check.
    
    Returns:
        dict: Health status
    """
    try:
        engine = create_modular_engine(OptimizationTier.ULTRA)
        return await engine.health_check()
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': str(__import__('datetime').datetime.now())
        }


# Turbo performance functions
async def turbo_sentiment_analysis(
    texts: list,
    use_ultra_turbo: bool = True
) -> dict:
    """
    Análisis de sentimiento turbo ultra-rápido.
    
    Args:
        texts: Lista de textos
        use_ultra_turbo: Usar motor ultra-turbo si está disponible
        
    Returns:
        dict: Resultado con velocidades extremas
        
    Example:
        >>> result = await turbo_sentiment_analysis(["Amazing!"])
        >>> print(f"Speed: {result['throughput_ops_per_second']:.0f} ops/s")
    """
    if TURBO_AVAILABLE and use_ultra_turbo:
        engine = get_ultra_turbo_engine()
        await engine.initialize()
        return await engine.ultra_turbo_sentiment(texts)
    elif TURBO_AVAILABLE:
        engine = get_turbo_optimizer()
        await engine.initialize()
        return await engine.turbo_sentiment(texts)
    else:
        # Fallback to regular engine
        engine = create_modular_engine(OptimizationTier.EXTREME)
        await engine.initialize()
        return await engine.analyze_sentiment(texts)


async def turbo_quality_analysis(
    texts: list,
    use_ultra_turbo: bool = True
) -> dict:
    """
    Análisis de calidad turbo ultra-rápido.
    
    Args:
        texts: Lista de textos
        use_ultra_turbo: Usar motor ultra-turbo si está disponible
        
    Returns:
        dict: Resultado con velocidades extremas
    """
    if TURBO_AVAILABLE and use_ultra_turbo:
        engine = get_ultra_turbo_engine()
        await engine.initialize()
        return await engine.ultra_turbo_quality(texts)
    elif TURBO_AVAILABLE:
        engine = get_turbo_optimizer()
        await engine.initialize()
        return await engine.turbo_quality(texts)
    else:
        # Fallback to regular engine
        engine = create_modular_engine(OptimizationTier.EXTREME)
        await engine.initialize()
        return await engine.analyze_quality(texts)


async def ultra_fast_mixed_analysis(texts: list) -> dict:
    """
    Análisis mixto ultra-rápido (sentimiento + calidad en paralelo).
    
    Args:
        texts: Lista de textos
        
    Returns:
        dict: Resultados combinados con velocidad máxima
    """
    if TURBO_AVAILABLE:
        engine = get_ultra_turbo_engine()
        await engine.initialize()
        return await engine.ultra_turbo_mixed(texts)
    else:
        # Fallback to parallel regular analysis
        engine = create_modular_engine(OptimizationTier.EXTREME)
        await engine.initialize()
        
        sentiment_task = engine.analyze_sentiment(texts)
        quality_task = engine.analyze_quality(texts)
        
        sentiment_result, quality_result = await asyncio.gather(sentiment_task, quality_task)
        
        return {
            'sentiment': sentiment_result,
            'quality': quality_result,
            'optimization_level': 'FALLBACK_PARALLEL'
        }


# Add turbo functions to exports
__all__.extend([
    'turbo_sentiment_analysis',
    'turbo_quality_analysis', 
    'ultra_fast_mixed_analysis'
]) 