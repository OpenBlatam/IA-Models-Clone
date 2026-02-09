"""
Learning Strategies Module

This module contains various learning strategies and optimization techniques:
- Active Learning
- Adaptive Learning
- Adversarial Learning
- Ensemble Learning
- Transfer Learning
- Continual Learning
- Self-Supervised Learning
- Federated Learning
- Meta Learning
- Multitask Learning
- Reinforcement Learning
- Bayesian Optimization
- Causal Inference
- Hyperparameter Optimization
- Evolutionary Computing
- Neural Architecture Search
"""

from __future__ import annotations

__all__ = [
    # Active Learning
    'ActiveLearningStrategy',
    'UncertaintyMeasure',
    'ActiveLearner',
    
    # Adaptive Learning
    'AdaptiveLearningStrategy',
    'AdaptiveLearner',
    
    # Adversarial Learning
    'AdversarialLearner',
    
    # Ensemble Learning
    'EnsembleLearner',
    
    # Transfer Learning
    'TransferLearner',
    
    # Continual Learning
    'ContinualLearner',
    
    # Self-Supervised Learning
    'SelfSupervisedLearner',
    
    # Federated Learning
    'FederatedLearner',
    
    # Meta Learning
    'MetaLearner',
    
    # Multitask Learning
    'MultitaskLearner',
    
    # Reinforcement Learning
    'ReinforcementLearner',
    
    # Bayesian Optimization
    'BayesianOptimizer',
    
    # Causal Inference
    'CausalInference',
    
    # Hyperparameter Optimization
    'HyperparameterOptimizer',
    
    # Evolutionary Computing
    'EvolutionaryOptimizer',
    
    # Neural Architecture Search
    'NASOptimizer',
]

# Lazy imports for better startup performance
_LAZY_IMPORTS = {
    'ActiveLearningStrategy': '.active_learning',
    'UncertaintyMeasure': '.active_learning',
    'ActiveLearner': '.active_learning',
    'AdaptiveLearningStrategy': '.adaptive_learning',
    'AdaptiveLearner': '.adaptive_learning',
    'AdversarialLearner': '.adversarial_learning',
    'EnsembleLearner': '.ensemble_learning',
    'TransferLearner': '.transfer_learning',
    'ContinualLearner': '.continual_learning',
    'SelfSupervisedLearner': '.self_supervised_learning',
    'FederatedLearner': '.federated_learning',
    'MetaLearner': '.meta_learning',
    'MultitaskLearner': '.multitask_learning',
    'ReinforcementLearner': '.reinforcement_learning',
    'BayesianOptimizer': '.bayesian_optimization',
    'CausalInference': '.causal_inference',
    'HyperparameterOptimizer': '.hyperparameter_optimization',
    'EvolutionaryOptimizer': '.evolutionary_computing',
    'NASOptimizer': '.nas',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for learning modules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        available = sorted(_LAZY_IMPORTS.keys())[:10]
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Available: {', '.join(available)}..."
        )
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        module = __import__(module_path.lstrip('.'), fromlist=[name], level=1)
        obj = getattr(module, name)
        _import_cache[name] = obj
        return obj
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


