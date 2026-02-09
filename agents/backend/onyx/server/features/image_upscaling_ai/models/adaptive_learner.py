"""
Adaptive Learner
================

Machine learning-based adaptive system that learns from results.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class LearningRecord:
    """Record of a learning experience."""
    image_type: str
    scale_factor: float
    method_used: str
    quality_score: float
    processing_time: float
    user_satisfaction: Optional[float] = None
    timestamp: float = 0.0


class AdaptiveLearner:
    """
    Adaptive learning system that improves over time.
    
    Features:
    - Learn from results
    - Method recommendation
    - Quality prediction
    - Performance optimization
    - User feedback integration
    """
    
    def __init__(self, learning_file: Optional[str] = None):
        """
        Initialize adaptive learner.
        
        Args:
            learning_file: Path to save learning data
        """
        self.learning_file = learning_file or "./learning_data.json"
        self.learning_path = Path(self.learning_file)
        
        # Learning data
        self.records: List[LearningRecord] = []
        self.method_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "avg_quality": 0.0,
            "avg_time": 0.0,
            "count": 0,
            "success_rate": 0.0
        })
        
        # Load existing data
        self._load_learning_data()
        
        logger.info(f"AdaptiveLearner initialized with {len(self.records)} records")
    
    def record_experience(
        self,
        image_type: str,
        scale_factor: float,
        method_used: str,
        quality_score: float,
        processing_time: float,
        user_satisfaction: Optional[float] = None
    ) -> None:
        """
        Record a learning experience.
        
        Args:
            image_type: Type of image (anime, photo, etc.)
            scale_factor: Scale factor used
            method_used: Method used for upscaling
            quality_score: Quality score (0.0-1.0)
            processing_time: Processing time in seconds
            user_satisfaction: User satisfaction (0.0-1.0, optional)
        """
        record = LearningRecord(
            image_type=image_type,
            scale_factor=scale_factor,
            method_used=method_used,
            quality_score=quality_score,
            processing_time=processing_time,
            user_satisfaction=user_satisfaction,
            timestamp=time.time()
        )
        
        self.records.append(record)
        
        # Update method performance
        method_key = f"{method_used}_{image_type}"
        perf = self.method_performance[method_key]
        
        # Update averages
        count = perf["count"]
        perf["avg_quality"] = (
            (perf["avg_quality"] * count + quality_score) / (count + 1)
        )
        perf["avg_time"] = (
            (perf["avg_time"] * count + processing_time) / (count + 1)
        )
        perf["count"] = count + 1
        
        # Update success rate (quality > 0.7)
        if quality_score > 0.7:
            perf["success_rate"] = (
                (perf["success_rate"] * count + 1.0) / (count + 1)
            )
        else:
            perf["success_rate"] = (
                (perf["success_rate"] * count) / (count + 1)
            )
        
        # Save periodically
        if len(self.records) % 10 == 0:
            self._save_learning_data()
    
    def recommend_method(
        self,
        image_type: str,
        scale_factor: float,
        prioritize_speed: bool = False
    ) -> Tuple[str, float]:
        """
        Recommend best method based on learning.
        
        Args:
            image_type: Type of image
            scale_factor: Desired scale factor
            prioritize_speed: Prioritize speed over quality
            
        Returns:
            Tuple of (recommended_method, confidence)
        """
        if not self.records:
            # No data, return default
            return "realesrgan", 0.5
        
        # Find best method for this combination
        best_method = None
        best_score = -1.0
        confidence = 0.0
        
        for method_key, perf in self.method_performance.items():
            method, mtype = method_key.rsplit("_", 1) if "_" in method_key else (method_key, "")
            
            if mtype != image_type:
                continue
            
            # Calculate score
            if prioritize_speed:
                # Speed score: inverse of time, weighted by success rate
                speed_score = 1.0 / max(0.1, perf["avg_time"])
                score = speed_score * perf["success_rate"]
            else:
                # Quality score: weighted by success rate
                score = perf["avg_quality"] * perf["success_rate"]
            
            if score > best_score:
                best_score = score
                best_method = method
                confidence = min(1.0, perf["count"] / 10.0)  # More data = higher confidence
        
        if best_method:
            return best_method, confidence
        
        # Fallback
        return "realesrgan", 0.3
    
    def predict_quality(
        self,
        image_type: str,
        scale_factor: float,
        method: str
    ) -> Tuple[float, float]:
        """
        Predict quality for given parameters.
        
        Args:
            image_type: Type of image
            scale_factor: Scale factor
            method: Method to use
            
        Returns:
            Tuple of (predicted_quality, confidence)
        """
        method_key = f"{method}_{image_type}"
        
        if method_key in self.method_performance:
            perf = self.method_performance[method_key]
            confidence = min(1.0, perf["count"] / 20.0)
            return perf["avg_quality"], confidence
        
        # No data, return default
        return 0.7, 0.0
    
    def predict_time(
        self,
        image_type: str,
        scale_factor: float,
        method: str
    ) -> Tuple[float, float]:
        """
        Predict processing time.
        
        Args:
            image_type: Type of image
            scale_factor: Scale factor
            method: Method to use
            
        Returns:
            Tuple of (predicted_time, confidence)
        """
        method_key = f"{method}_{image_type}"
        
        if method_key in self.method_performance:
            perf = self.method_performance[method_key]
            confidence = min(1.0, perf["count"] / 20.0)
            return perf["avg_time"], confidence
        
        # Default estimates
        defaults = {
            "lanczos": 0.1,
            "opencv_edsr": 0.3,
            "realesrgan": 2.0,
        }
        return defaults.get(method, 1.0), 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if not self.records:
            return {
                "total_records": 0,
                "methods_tested": 0,
                "avg_quality": 0.0,
                "avg_time": 0.0,
            }
        
        return {
            "total_records": len(self.records),
            "methods_tested": len(self.method_performance),
            "avg_quality": statistics.mean([r.quality_score for r in self.records]),
            "avg_time": statistics.mean([r.processing_time for r in self.records]),
            "method_performance": {
                key: {
                    "avg_quality": perf["avg_quality"],
                    "avg_time": perf["avg_time"],
                    "count": perf["count"],
                    "success_rate": perf["success_rate"]
                }
                for key, perf in self.method_performance.items()
            }
        }
    
    def _load_learning_data(self) -> None:
        """Load learning data from file."""
        if not self.learning_path.exists():
            return
        
        try:
            with open(self.learning_path, 'r') as f:
                data = json.load(f)
            
            # Load records
            self.records = [
                LearningRecord(**record_data)
                for record_data in data.get("records", [])
            ]
            
            # Rebuild method performance
            self.method_performance = defaultdict(lambda: {
                "avg_quality": 0.0,
                "avg_time": 0.0,
                "count": 0,
                "success_rate": 0.0
            })
            
            for record in self.records:
                method_key = f"{record.method_used}_{record.image_type}"
                perf = self.method_performance[method_key]
                
                count = perf["count"]
                perf["avg_quality"] = (
                    (perf["avg_quality"] * count + record.quality_score) / (count + 1)
                )
                perf["avg_time"] = (
                    (perf["avg_time"] * count + record.processing_time) / (count + 1)
                )
                perf["count"] = count + 1
                
                if record.quality_score > 0.7:
                    perf["success_rate"] = (
                        (perf["success_rate"] * count + 1.0) / (count + 1)
                    )
                else:
                    perf["success_rate"] = (
                        (perf["success_rate"] * count) / (count + 1)
                    )
            
            logger.info(f"Loaded {len(self.records)} learning records")
            
        except Exception as e:
            logger.error(f"Error loading learning data: {e}")
    
    def _save_learning_data(self) -> None:
        """Save learning data to file."""
        try:
            data = {
                "records": [asdict(record) for record in self.records],
                "last_updated": time.time()
            }
            
            with open(self.learning_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.records)} learning records")
            
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
    
    def reset(self) -> None:
        """Reset learning data."""
        self.records.clear()
        self.method_performance.clear()
        if self.learning_path.exists():
            self.learning_path.unlink()
        logger.info("Learning data reset")


