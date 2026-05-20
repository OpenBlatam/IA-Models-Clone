"""
Evaluations Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum


class MetricType(str, Enum):
    """Metric types"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    BLEU = "bleu"
    ROUGE = "rouge"
    CUSTOM = "custom"


@dataclass
class Metric:
    """Metric definition"""
    name: str
    metric_type: MetricType
    value: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Evaluation result"""
    evaluation_id: str
    metrics: List[Metric]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class Evaluation:
    """Evaluation definition"""
    
    def __init__(
        self,
        name: str,
        dataset: List[Dict[str, Any]],
        metrics: List[MetricType]
    ):
        self.id = str(uuid4())
        self.name = name
        self.dataset = dataset
        self.metrics = metrics
        self.created_at = datetime.utcnow()
        self.results: Optional[EvaluationResult] = None


class EvalBase(ABC):
    """Base interface for evaluations"""
    
    @abstractmethod
    async def run_evaluation(self, evaluation: Evaluation) -> EvaluationResult:
        """Run evaluation"""
        pass
    
    @abstractmethod
    async def compute_metric(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        metric_type: MetricType
    ) -> float:
        """Compute metric"""
        pass

