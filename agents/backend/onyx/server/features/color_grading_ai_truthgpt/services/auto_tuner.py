"""
Auto Tuner for Color Grading AI
================================

Automatic parameter tuning using optimization algorithms.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import random

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Tuning result."""
    best_params: Dict[str, float]
    best_score: float
    iterations: int
    convergence: bool
    history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class AutoTuner:
    """
    Automatic parameter tuner.
    
    Features:
    - Multiple optimization algorithms
    - Parameter space exploration
    - Convergence detection
    - History tracking
    - Custom objective functions
    """
    
    def __init__(self):
        """Initialize auto tuner."""
        self._objective_func: Optional[Callable] = None
        self._param_bounds: Dict[str, tuple] = {}
        self._tuning_history: List[TuningResult] = []
        self._max_history = 100
    
    def set_objective(self, objective_func: Callable):
        """
        Set objective function.
        
        Args:
            objective_func: Function that takes params and returns score
        """
        self._objective_func = objective_func
        logger.info("Set objective function")
    
    def set_param_bounds(
        self,
        param_name: str,
        min_val: float,
        max_val: float
    ):
        """
        Set parameter bounds.
        
        Args:
            param_name: Parameter name
            min_val: Minimum value
            max_val: Maximum value
        """
        self._param_bounds[param_name] = (min_val, max_val)
        logger.debug(f"Set bounds for {param_name}: [{min_val}, {max_val}]")
    
    def tune(
        self,
        initial_params: Optional[Dict[str, float]] = None,
        algorithm: str = "random_search",
        max_iterations: int = 100,
        convergence_threshold: float = 0.001
    ) -> TuningResult:
        """
        Tune parameters.
        
        Args:
            initial_params: Initial parameters
            algorithm: Optimization algorithm
            max_iterations: Maximum iterations
            convergence_threshold: Convergence threshold
            
        Returns:
            Tuning result
        """
        if not self._objective_func:
            raise ValueError("Objective function not set")
        
        if algorithm == "random_search":
            return self._random_search(initial_params, max_iterations, convergence_threshold)
        elif algorithm == "grid_search":
            return self._grid_search(initial_params, max_iterations)
        elif algorithm == "gradient_descent":
            return self._gradient_descent(initial_params, max_iterations, convergence_threshold)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _random_search(
        self,
        initial_params: Optional[Dict[str, float]],
        max_iterations: int,
        convergence_threshold: float
    ) -> TuningResult:
        """Random search optimization."""
        best_params = initial_params or self._random_params()
        best_score = self._objective_func(best_params)
        history = [{"iteration": 0, "score": best_score, "params": best_params.copy()}]
        
        for i in range(1, max_iterations):
            # Generate random params
            params = self._random_params()
            score = self._objective_func(params)
            
            history.append({
                "iteration": i,
                "score": score,
                "params": params.copy()
            })
            
            # Update best
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            # Check convergence
            if i > 10:
                recent_scores = [h["score"] for h in history[-10:]]
                if max(recent_scores) - min(recent_scores) < convergence_threshold:
                    logger.info(f"Converged at iteration {i}")
                    return TuningResult(
                        best_params=best_params,
                        best_score=best_score,
                        iterations=i,
                        convergence=True,
                        history=history
                    )
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            iterations=max_iterations,
            convergence=False,
            history=history
        )
    
    def _grid_search(
        self,
        initial_params: Optional[Dict[str, float]],
        max_iterations: int
    ) -> TuningResult:
        """Grid search optimization."""
        # Simplified grid search
        best_params = initial_params or self._random_params()
        best_score = self._objective_func(best_params)
        history = []
        
        # Generate grid points
        param_names = list(self._param_bounds.keys())
        grid_size = int(max_iterations ** (1.0 / len(param_names)))
        
        for i in range(min(max_iterations, grid_size ** len(param_names))):
            params = {}
            for param_name in param_names:
                min_val, max_val = self._param_bounds[param_name]
                step = (max_val - min_val) / grid_size
                idx = (i // (grid_size ** param_names.index(param_name))) % grid_size
                params[param_name] = min_val + idx * step
            
            score = self._objective_func(params)
            history.append({
                "iteration": i,
                "score": score,
                "params": params.copy()
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            iterations=len(history),
            convergence=False,
            history=history
        )
    
    def _gradient_descent(
        self,
        initial_params: Optional[Dict[str, float]],
        max_iterations: int,
        convergence_threshold: float
    ) -> TuningResult:
        """Gradient descent optimization."""
        params = initial_params or self._random_params()
        learning_rate = 0.01
        history = []
        
        for i in range(max_iterations):
            score = self._objective_func(params)
            history.append({
                "iteration": i,
                "score": score,
                "params": params.copy()
            })
            
            # Simple gradient approximation
            gradient = {}
            for param_name in params.keys():
                eps = 0.001
                params_plus = params.copy()
                params_plus[param_name] += eps
                score_plus = self._objective_func(params_plus)
                gradient[param_name] = (score_plus - score) / eps
            
            # Update params
            for param_name in params.keys():
                if param_name in self._param_bounds:
                    min_val, max_val = self._param_bounds[param_name]
                    params[param_name] = max(
                        min_val,
                        min(max_val, params[param_name] + learning_rate * gradient[param_name])
                    )
            
            # Check convergence
            if i > 10:
                recent_scores = [h["score"] for h in history[-10:]]
                if max(recent_scores) - min(recent_scores) < convergence_threshold:
                    logger.info(f"Converged at iteration {i}")
                    return TuningResult(
                        best_params=params,
                        best_score=score,
                        iterations=i,
                        convergence=True,
                        history=history
                    )
        
        return TuningResult(
            best_params=params,
            best_score=history[-1]["score"],
            iterations=max_iterations,
            convergence=False,
            history=history
        )
    
    def _random_params(self) -> Dict[str, float]:
        """Generate random parameters within bounds."""
        params = {}
        for param_name, (min_val, max_val) in self._param_bounds.items():
            params[param_name] = random.uniform(min_val, max_val)
        return params
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get auto tuner statistics."""
        return {
            "tuning_runs": len(self._tuning_history),
            "param_bounds_count": len(self._param_bounds),
            "has_objective": self._objective_func is not None,
        }




