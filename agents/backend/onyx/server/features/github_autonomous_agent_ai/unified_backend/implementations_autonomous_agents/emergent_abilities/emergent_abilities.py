"""
Are Emergent Abilities of Large Language Models a Mirage?
==========================================================

Paper: "Are Emergent Abilities of Large Language Models a Mirage?"

Key concepts:
- Emergent abilities in LLMs
- Scaling laws and emergence
- Evaluation metrics
- Measurement artifacts
- True vs apparent emergence
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class EmergenceType(Enum):
    """Types of emergence."""
    TRUE_EMERGENCE = "true_emergence"
    MEASUREMENT_ARTIFACT = "measurement_artifact"
    SCALING_EFFECT = "scaling_effect"
    UNKNOWN = "unknown"


class EvaluationMetric(Enum):
    """Evaluation metrics."""
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PERPLEXITY = "perplexity"
    BLEU = "bleu"
    ROUGE = "rouge"
    HUMAN_EVAL = "human_eval"


@dataclass
class EmergenceObservation:
    """Observation of emergent ability."""
    ability_name: str
    model_size: int  # parameters
    performance: float
    metric: EvaluationMetric
    emergence_type: EmergenceType
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingCurve:
    """Scaling curve data."""
    model_sizes: List[int]
    performances: List[float]
    metric: EvaluationMetric
    emergence_point: Optional[int] = None
    emergence_type: EmergenceType = EmergenceType.UNKNOWN


class EmergentAbilitiesAnalyzer:
    """
    Analyzer for emergent abilities in LLMs.
    
    Distinguishes between true emergence and measurement artifacts.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize analyzer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.observations: List[EmergenceObservation] = []
        self.scaling_curves: Dict[str, ScalingCurve] = {}
        
        # Parameters
        self.emergence_threshold = config.get("emergence_threshold", 0.5)
        self.min_model_size = config.get("min_model_size", 1e6)
        self.max_model_size = config.get("max_model_size", 1e12)
    
    def analyze_emergence(
        self,
        ability_name: str,
        model_sizes: List[int],
        performances: List[float],
        metric: EvaluationMetric
    ) -> EmergenceObservation:
        """
        Analyze if an ability is truly emergent.
        
        Args:
            ability_name: Name of the ability
            model_sizes: List of model sizes (parameters)
            performances: List of performance scores
            metric: Evaluation metric used
            
        Returns:
            Emergence observation
        """
        # Create scaling curve
        curve = ScalingCurve(
            model_sizes=model_sizes,
            performances=performances,
            metric=metric
        )
        
        # Detect emergence point
        emergence_point = self._detect_emergence_point(model_sizes, performances)
        curve.emergence_point = emergence_point
        
        # Classify emergence type
        emergence_type = self._classify_emergence_type(model_sizes, performances, emergence_point)
        curve.emergence_type = emergence_type
        
        self.scaling_curves[ability_name] = curve
        
        # Create observation
        if emergence_point:
            performance_at_emergence = performances[model_sizes.index(emergence_point)]
        else:
            performance_at_emergence = max(performances) if performances else 0.0
        
        observation = EmergenceObservation(
            ability_name=ability_name,
            model_size=emergence_point if emergence_point else max(model_sizes) if model_sizes else 0,
            performance=performance_at_emergence,
            metric=metric,
            emergence_type=emergence_type,
            threshold=self.emergence_threshold
        )
        
        self.observations.append(observation)
        return observation
    
    def _detect_emergence_point(
        self,
        model_sizes: List[int],
        performances: List[float]
    ) -> Optional[int]:
        """Detect the model size where emergence occurs."""
        if len(model_sizes) < 2 or len(performances) < 2:
            return None
        
        # Find point where performance crosses threshold
        for i, (size, perf) in enumerate(zip(model_sizes, performances)):
            if perf >= self.emergence_threshold:
                return size
        
        # Check for sudden jump (emergence signature)
        for i in range(1, len(performances)):
            jump = performances[i] - performances[i-1]
            if jump > 0.2:  # Significant jump
                return model_sizes[i]
        
        return None
    
    def _classify_emergence_type(
        self,
        model_sizes: List[int],
        performances: List[float],
        emergence_point: Optional[int]
    ) -> EmergenceType:
        """Classify the type of emergence."""
        if not emergence_point or len(performances) < 3:
            return EmergenceType.UNKNOWN
        
        # Check if smooth scaling (likely true emergence)
        if self._is_smooth_scaling(model_sizes, performances):
            return EmergenceType.TRUE_EMERGENCE
        
        # Check for measurement artifacts
        if self._has_measurement_artifacts(model_sizes, performances):
            return EmergenceType.MEASUREMENT_ARTIFACT
        
        # Check scaling effects
        if self._is_scaling_effect(model_sizes, performances):
            return EmergenceType.SCALING_EFFECT
        
        return EmergenceType.UNKNOWN
    
    def _is_smooth_scaling(
        self,
        model_sizes: List[int],
        performances: List[float]
    ) -> bool:
        """Check if performance scales smoothly with model size."""
        if len(performances) < 3:
            return False
        
        # Calculate correlation
        log_sizes = [math.log10(s) for s in model_sizes]
        
        # Simple linear correlation check
        diffs = [performances[i] - performances[i-1] for i in range(1, len(performances))]
        avg_diff = sum(diffs) / len(diffs)
        variance = sum((d - avg_diff) ** 2 for d in diffs) / len(diffs)
        
        # Low variance suggests smooth scaling
        return variance < 0.01
    
    def _has_measurement_artifacts(
        self,
        model_sizes: List[int],
        performances: List[float]
    ) -> bool:
        """Check for measurement artifacts."""
        # Look for sudden jumps that might be artifacts
        jumps = [performances[i] - performances[i-1] for i in range(1, len(performances))]
        large_jumps = [j for j in jumps if abs(j) > 0.3]
        
        # Many large jumps suggest artifacts
        return len(large_jumps) > len(performances) * 0.3
    
    def _is_scaling_effect(
        self,
        model_sizes: List[int],
        performances: List[float]
    ) -> bool:
        """Check if it's just a scaling effect."""
        # If performance increases monotonically, might be scaling
        is_monotonic = all(performances[i] >= performances[i-1] for i in range(1, len(performances)))
        
        # Check if improvement rate decreases (diminishing returns)
        if len(performances) >= 3:
            early_improvement = performances[1] - performances[0]
            late_improvement = performances[-1] - performances[-2]
            if early_improvement > late_improvement * 2:
                return True
        
        return is_monotonic and len(performances) > 2
    
    def get_emergence_statistics(self) -> Dict[str, Any]:
        """Get statistics about observed emergences."""
        if not self.observations:
            return {}
        
        true_emergences = [o for o in self.observations if o.emergence_type == EmergenceType.TRUE_EMERGENCE]
        artifacts = [o for o in self.observations if o.emergence_type == EmergenceType.MEASUREMENT_ARTIFACT]
        scaling_effects = [o for o in self.observations if o.emergence_type == EmergenceType.SCALING_EFFECT]
        
        return {
            "total_observations": len(self.observations),
            "true_emergences": len(true_emergences),
            "measurement_artifacts": len(artifacts),
            "scaling_effects": len(scaling_effects),
            "unknown": len(self.observations) - len(true_emergences) - len(artifacts) - len(scaling_effects),
            "average_emergence_size": sum(o.model_size for o in self.observations) / len(self.observations) if self.observations else 0
        }
    
    def get_scaling_curve(self, ability_name: str) -> Optional[ScalingCurve]:
        """Get scaling curve for an ability."""
        return self.scaling_curves.get(ability_name)



