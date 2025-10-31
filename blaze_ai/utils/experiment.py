"""
Experiment tracking and hyperparameter optimization utilities for Blaze AI.

This module provides comprehensive experiment management including:
- Experiment tracking and logging
- Hyperparameter optimization
- A/B testing framework
- Results comparison and analysis
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
import warnings

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available, hyperparameter optimization limited")

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ExperimentResult:
    """Result of an experiment."""
    experiment_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    status: str = "running"  # running, completed, failed
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None

class ExperimentTracker:
    """Experiment tracking and management system."""
    
    def __init__(self, storage_dir: str = "./experiments"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, ExperimentResult] = {}
        
        self._load_existing_experiments()
    
    def _load_existing_experiments(self):
        """Load existing experiments from storage."""
        config_file = self.storage_dir / "experiments.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    for exp_data in data.get("experiments", []):
                        exp = ExperimentConfig(**exp_data)
                        self.experiments[exp.experiment_id] = exp
            except Exception as e:
                warnings.warn(f"Failed to load experiments: {e}")
        
        results_file = self.storage_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    for result_data in data.get("results", []):
                        result = ExperimentResult(**result_data)
                        self.results[result.experiment_id] = result
            except Exception as e:
                warnings.warn(f"Failed to load results: {e}")
    
    def _save_experiments(self):
        """Save experiments to storage."""
        config_file = self.storage_dir / "experiments.json"
        data = {
            "experiments": [asdict(exp) for exp in self.experiments.values()]
        }
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_results(self):
        """Save results to storage."""
        results_file = self.storage_dir / "results.json"
        data = {
            "results": [asdict(result) for result in self.results.values()]
        }
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_experiment(self, name: str, description: str = "", parameters: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> str:
        """Create a new experiment."""
        experiment = ExperimentConfig(
            name=name,
            description=description,
            parameters=parameters or {},
            tags=tags or []
        )
        
        self.experiments[experiment.experiment_id] = experiment
        self._save_experiments()
        
        return experiment.experiment_id
    
    def start_experiment(self, experiment_id: str) -> ExperimentResult:
        """Start an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        result = ExperimentResult(experiment_id=experiment_id)
        self.results[experiment_id] = result
        self._save_results()
        
        return result
    
    def update_metrics(self, experiment_id: str, metrics: Dict[str, float]):
        """Update experiment metrics."""
        if experiment_id not in self.results:
            raise ValueError(f"Experiment result {experiment_id} not found")
        
        self.results[experiment_id].metrics.update(metrics)
        self._save_results()
    
    def complete_experiment(self, experiment_id: str, final_metrics: Optional[Dict[str, float]] = None):
        """Mark experiment as completed."""
        if experiment_id not in self.results:
            raise ValueError(f"Experiment result {experiment_id} not found")
        
        result = self.results[experiment_id]
        result.status = "completed"
        result.completed_at = time.time()
        
        if final_metrics:
            result.metrics.update(final_metrics)
        
        self._save_results()
    
    def fail_experiment(self, experiment_id: str, error: str):
        """Mark experiment as failed."""
        if experiment_id not in self.results:
            raise ValueError(f"Experiment result {experiment_id} not found")
        
        result = self.results[experiment_id]
        result.status = "failed"
        result.completed_at = time.time()
        result.error = error
        
        self._save_results()
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration."""
        return self.experiments.get(experiment_id)
    
    def get_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment result."""
        return self.results.get(experiment_id)
    
    def list_experiments(self, tags: Optional[List[str]] = None) -> List[ExperimentConfig]:
        """List experiments, optionally filtered by tags."""
        experiments = list(self.experiments.values())
        
        if tags:
            experiments = [exp for exp in experiments if any(tag in exp.tags for tag in tags)]
        
        return sorted(experiments, key=lambda x: x.created_at, reverse=True)
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        comparison = {
            "experiments": [],
            "metrics_comparison": {},
            "parameter_comparison": {}
        }
        
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            result = self.get_result(exp_id)
            
            if exp and result:
                comparison["experiments"].append({
                    "id": exp_id,
                    "name": exp.name,
                    "status": result.status,
                    "metrics": result.metrics,
                    "parameters": exp.parameters
                })
        
        # Compare metrics
        if comparison["experiments"]:
            all_metrics = set()
            for exp in comparison["experiments"]:
                all_metrics.update(exp["metrics"].keys())
            
            for metric in all_metrics:
                values = [exp["metrics"].get(metric) for exp in comparison["experiments"]]
                values = [v for v in values if v is not None]
                
                if values:
                    comparison["metrics_comparison"][metric] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "values": values
                    }
        
        return comparison

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, study_name: str = "blaze_ai_optimization", storage: Optional[str] = None):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        self.study_name = study_name
        self.storage = storage
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
    
    def optimize(self, objective_func: Callable, n_trials: int = 100, timeout: Optional[int] = None) -> optuna.Study:
        """Run hyperparameter optimization."""
        self.study.optimize(objective_func, n_trials=n_trials, timeout=timeout)
        return self.study
    
    def suggest_hyperparameters(self, trial: optuna.Trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest hyperparameters based on parameter space."""
        params = {}
        
        for param_name, param_config in param_space.items():
            param_type = param_config.get("type", "float")
            
            if param_type == "float":
                if "log" in param_config and param_config["log"]:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["min"],
                        param_config["max"],
                        log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["min"],
                        param_config["max"]
                    )
            
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["min"],
                    param_config["max"],
                    log=param_config.get("log", False)
                )
            
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
        
        return params
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters found."""
        return self.study.best_params
    
    def get_best_value(self) -> float:
        """Get best objective value found."""
        return self.study.best_value
    
    def get_optimization_history(self) -> List[float]:
        """Get optimization history."""
        return [trial.value for trial in self.study.trials if trial.value is not None]

class ABTestManager:
    """A/B testing framework for experiments."""
    
    def __init__(self):
        self.tests: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_test(self, test_name: str, variants: List[str], traffic_split: Optional[List[float]] = None) -> str:
        """Create a new A/B test."""
        if traffic_split and len(traffic_split) != len(variants):
            raise ValueError("Traffic split must match number of variants")
        
        if not traffic_split:
            # Equal split
            traffic_split = [1.0 / len(variants)] * len(variants)
        
        test_id = str(uuid.uuid4())
        self.tests[test_id] = {
            "name": test_name,
            "variants": variants,
            "traffic_split": traffic_split,
            "created_at": time.time(),
            "status": "active"
        }
        
        self.results[test_id] = []
        return test_id
    
    def assign_variant(self, test_id: str, user_id: str) -> str:
        """Assign a variant to a user."""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        
        # Simple hash-based assignment
        hash_val = hash(user_id + test_id) % 100
        cumulative = 0
        
        for i, split in enumerate(test["traffic_split"]):
            cumulative += split * 100
            if hash_val < cumulative:
                return test["variants"][i]
        
        return test["variants"][-1]
    
    def record_result(self, test_id: str, user_id: str, variant: str, metric: str, value: float):
        """Record a test result."""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        result = {
            "user_id": user_id,
            "variant": variant,
            "metric": metric,
            "value": value,
            "timestamp": time.time()
        }
        
        self.results[test_id].append(result)
    
    def analyze_test(self, test_id: str, metric: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        if test_id not in self.tests or test_id not in self.results:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        results = self.results[test_id]
        
        analysis = {
            "test_name": test["name"],
            "metric": metric,
            "variants": {},
            "statistical_significance": {}
        }
        
        # Calculate statistics for each variant
        for variant in test["variants"]:
            variant_results = [r for r in results if r["variant"] == variant and r["metric"] == metric]
            
            if variant_results:
                values = [r["value"] for r in variant_results]
                analysis["variants"][variant] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "std": (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5,
                    "min": min(values),
                    "max": max(values)
                }
        
        return analysis
    
    def stop_test(self, test_id: str):
        """Stop an A/B test."""
        if test_id in self.tests:
            self.tests[test_id]["status"] = "stopped"

# Utility functions
def create_experiment_tracker(storage_dir: str = "./experiments") -> ExperimentTracker:
    """Create a new experiment tracker."""
    return ExperimentTracker(storage_dir)

def create_hyperparameter_optimizer(study_name: str = "blaze_ai_optimization", storage: Optional[str] = None) -> HyperparameterOptimizer:
    """Create a new hyperparameter optimizer."""
    return HyperparameterOptimizer(study_name, storage)

def create_ab_test_manager() -> ABTestManager:
    """Create a new A/B test manager."""
    return ABTestManager()

# Export main classes
__all__ = [
    "ExperimentTracker",
    "HyperparameterOptimizer",
    "ABTestManager",
    "ExperimentConfig",
    "ExperimentResult",
    "create_experiment_tracker",
    "create_hyperparameter_optimizer",
    "create_ab_test_manager"
]


