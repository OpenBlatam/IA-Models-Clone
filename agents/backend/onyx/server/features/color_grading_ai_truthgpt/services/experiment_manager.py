"""
Experiment Manager for Color Grading AI
========================================

A/B testing and experimentation framework.
"""

import logging
import random
import hashlib
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentVariant:
    """Experiment variant."""
    name: str
    weight: float = 0.5  # 0.0 - 1.0
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Experiment:
    """Experiment definition."""
    experiment_id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    target_audience: Optional[Callable] = None
    metrics: List[str] = field(default_factory=list)


class ExperimentManager:
    """
    Experiment manager for A/B testing.
    
    Features:
    - A/B testing
    - Multi-variant testing
    - Weighted distribution
    - Metrics tracking
    - Statistical analysis
    """
    
    def __init__(self):
        """Initialize experiment manager."""
        self._experiments: Dict[str, Experiment] = {}
        self._assignments: Dict[str, str] = {}  # user_id -> variant_name
        self._metrics: Dict[str, Dict[str, List[float]]] = {}  # experiment_id -> variant -> metrics
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        experiment_id: Optional[str] = None
    ) -> str:
        """
        Create new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            variants: List of variant definitions
            experiment_id: Optional experiment ID
            
        Returns:
            Experiment ID
        """
        import uuid
        exp_id = experiment_id or str(uuid.uuid4())
        
        experiment_variants = [
            ExperimentVariant(
                name=v["name"],
                weight=v.get("weight", 1.0 / len(variants)),
                config=v.get("config", {})
            )
            for v in variants
        ]
        
        # Normalize weights
        total_weight = sum(v.weight for v in experiment_variants)
        for variant in experiment_variants:
            variant.weight /= total_weight
        
        experiment = Experiment(
            experiment_id=exp_id,
            name=name,
            description=description,
            variants=experiment_variants
        )
        
        self._experiments[exp_id] = experiment
        self._metrics[exp_id] = {v.name: [] for v in experiment_variants}
        
        logger.info(f"Created experiment: {exp_id} ({name})")
        
        return exp_id
    
    def start_experiment(self, experiment_id: str):
        """Start experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.now()
        logger.info(f"Started experiment: {experiment_id}")
    
    def pause_experiment(self, experiment_id: str):
        """Pause experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.status = ExperimentStatus.PAUSED
        logger.info(f"Paused experiment: {experiment_id}")
    
    def complete_experiment(self, experiment_id: str):
        """Complete experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.now()
        logger.info(f"Completed experiment: {experiment_id}")
    
    def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
        force_variant: Optional[str] = None
    ) -> str:
        """
        Assign user to experiment variant.
        
        Args:
            experiment_id: Experiment ID
            user_id: User ID
            force_variant: Optional forced variant (for testing)
            
        Returns:
            Variant name
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        if experiment.status != ExperimentStatus.RUNNING:
            # Return control variant if not running
            return experiment.variants[0].name if experiment.variants else "control"
        
        # Check if already assigned
        assignment_key = f"{experiment_id}:{user_id}"
        if assignment_key in self._assignments:
            return self._assignments[assignment_key]
        
        # Force variant (for testing)
        if force_variant:
            self._assignments[assignment_key] = force_variant
            return force_variant
        
        # Check target audience
        if experiment.target_audience:
            if not experiment.target_audience(user_id):
                return experiment.variants[0].name  # Control
        
        # Assign based on weights (deterministic)
        variant = self._select_variant(experiment, user_id)
        self._assignments[assignment_key] = variant.name
        
        return variant.name
    
    def _select_variant(self, experiment: Experiment, user_id: str) -> ExperimentVariant:
        """Select variant for user (deterministic)."""
        # Deterministic selection based on user_id hash
        hash_value = int(hashlib.md5(f"{experiment.experiment_id}:{user_id}".encode()).hexdigest(), 16)
        random_value = (hash_value % 10000) / 10000.0  # 0.0 - 1.0
        
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.weight
            if random_value < cumulative:
                return variant
        
        # Fallback to last variant
        return experiment.variants[-1]
    
    def record_metric(
        self,
        experiment_id: str,
        variant_name: str,
        metric_name: str,
        value: float
    ):
        """
        Record metric for experiment variant.
        
        Args:
            experiment_id: Experiment ID
            variant_name: Variant name
            metric_name: Metric name
            value: Metric value
        """
        if experiment_id not in self._metrics:
            self._metrics[experiment_id] = {}
        
        if variant_name not in self._metrics[experiment_id]:
            self._metrics[experiment_id][variant_name] = {}
        
        if metric_name not in self._metrics[experiment_id][variant_name]:
            self._metrics[experiment_id][variant_name][metric_name] = []
        
        self._metrics[experiment_id][variant_name][metric_name].append(value)
    
    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment results.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Results dictionary
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        results = {}
        
        for variant in experiment.variants:
            variant_metrics = self._metrics.get(experiment_id, {}).get(variant.name, {})
            
            variant_results = {}
            for metric_name, values in variant_metrics.items():
                if values:
                    variant_results[metric_name] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "sum": sum(values),
                    }
            
            results[variant.name] = {
                "weight": variant.weight,
                "metrics": variant_results,
            }
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "variants": results,
        }
    
    def get_statistical_significance(
        self,
        experiment_id: str,
        metric_name: str,
        variant_a: str,
        variant_b: str
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance between variants.
        
        Args:
            experiment_id: Experiment ID
            metric_name: Metric name
            variant_a: First variant
            variant_b: Second variant
            
        Returns:
            Statistical analysis
        """
        # Simplified statistical analysis
        metrics_a = self._metrics.get(experiment_id, {}).get(variant_a, {}).get(metric_name, [])
        metrics_b = self._metrics.get(experiment_id, {}).get(variant_b, {}).get(metric_name, [])
        
        if not metrics_a or not metrics_b:
            return {"error": "Insufficient data"}
        
        mean_a = sum(metrics_a) / len(metrics_a)
        mean_b = sum(metrics_b) / len(metrics_b)
        
        # Simple difference
        difference = mean_b - mean_a
        percent_change = (difference / mean_a * 100) if mean_a != 0 else 0
        
        return {
            "variant_a": {
                "name": variant_a,
                "mean": mean_a,
                "count": len(metrics_a),
            },
            "variant_b": {
                "name": variant_b,
                "mean": mean_b,
                "count": len(metrics_b),
            },
            "difference": difference,
            "percent_change": percent_change,
            "note": "Simplified analysis - use proper statistical tests for production",
        }
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """List experiments."""
        experiments = []
        
        for exp_id, experiment in self._experiments.items():
            if status and experiment.status != status:
                continue
            
            experiments.append({
                "experiment_id": exp_id,
                "name": experiment.name,
                "description": experiment.description,
                "status": experiment.status.value,
                "variants": [v.name for v in experiment.variants],
                "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
                "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
            })
        
        return experiments




