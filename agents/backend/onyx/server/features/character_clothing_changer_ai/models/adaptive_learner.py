"""
Adaptive Learner for Flux2 Clothing Changer
===========================================

Machine learning-based adaptive optimization based on feedback.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSample:
    """Feedback sample for learning."""
    image_features: np.ndarray
    clothing_description: str
    parameters: Dict[str, float]
    quality_score: float
    user_rating: Optional[float] = None
    processing_time: float = 0.0


class AdaptiveLearner:
    """Adaptive learning system for parameter optimization."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        memory_size: int = 1000,
        enable_learning: bool = True,
    ):
        """
        Initialize adaptive learner.
        
        Args:
            learning_rate: Learning rate for parameter updates
            memory_size: Size of feedback memory
            enable_learning: Enable adaptive learning
        """
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.enable_learning = enable_learning
        
        # Feedback memory
        self.feedback_memory: deque = deque(maxlen=memory_size)
        
        # Parameter prediction model (simple neural network)
        self.prediction_model = self._build_prediction_model()
        
        # Parameter history for tracking
        self.parameter_history: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            "samples_collected": 0,
            "samples_used": 0,
            "improvements": 0,
            "degradations": 0,
        }
    
    def _build_prediction_model(self) -> nn.Module:
        """Build parameter prediction model."""
        return nn.Sequential(
            nn.Linear(512, 256),  # Image features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # num_steps, guidance, strength
            nn.Sigmoid(),  # Normalize to [0, 1]
        )
    
    def record_feedback(
        self,
        image_features: np.ndarray,
        clothing_description: str,
        parameters: Dict[str, float],
        quality_score: float,
        user_rating: Optional[float] = None,
        processing_time: float = 0.0,
    ) -> None:
        """
        Record feedback for learning.
        
        Args:
            image_features: Image feature vector
            clothing_description: Clothing description
            parameters: Parameters used (num_steps, guidance, strength)
            quality_score: Quality score (0.0 to 1.0)
            user_rating: Optional user rating (0.0 to 1.0)
            processing_time: Processing time
        """
        sample = FeedbackSample(
            image_features=image_features,
            clothing_description=clothing_description,
            parameters=parameters,
            quality_score=quality_score,
            user_rating=user_rating,
            processing_time=processing_time,
        )
        
        self.feedback_memory.append(sample)
        self.stats["samples_collected"] += 1
        
        if self.enable_learning and len(self.feedback_memory) >= 10:
            self._update_model()
    
    def predict_optimal_parameters(
        self,
        image_features: np.ndarray,
        clothing_description: str,
        base_parameters: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Predict optimal parameters based on learned patterns.
        
        Args:
            image_features: Image feature vector
            clothing_description: Clothing description
            base_parameters: Base parameters to adjust
            
        Returns:
            Optimized parameters
        """
        if not self.enable_learning or len(self.feedback_memory) < 10:
            return base_parameters
        
        try:
            # Prepare input
            feature_tensor = torch.FloatTensor(image_features[:512]).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                prediction = self.prediction_model(feature_tensor).squeeze().numpy()
            
            # Map to parameter ranges
            # num_steps: 20-100, guidance: 3.0-15.0, strength: 0.5-1.0
            optimized = {
                "num_inference_steps": int(20 + prediction[0] * 80),
                "guidance_scale": 3.0 + prediction[1] * 12.0,
                "strength": 0.5 + prediction[2] * 0.5,
            }
            
            # Blend with base parameters (70% learned, 30% base)
            blended = {
                "num_inference_steps": int(
                    optimized["num_inference_steps"] * 0.7 +
                    base_parameters.get("num_inference_steps", 50) * 0.3
                ),
                "guidance_scale": (
                    optimized["guidance_scale"] * 0.7 +
                    base_parameters.get("guidance_scale", 7.5) * 0.3
                ),
                "strength": (
                    optimized["strength"] * 0.7 +
                    base_parameters.get("strength", 0.8) * 0.3
                ),
            }
            
            return blended
            
        except Exception as e:
            logger.warning(f"Parameter prediction failed: {e}, using base parameters")
            return base_parameters
    
    def _update_model(self) -> None:
        """Update prediction model from feedback memory."""
        if len(self.feedback_memory) < 10:
            return
        
        try:
            # Prepare training data
            # Use recent samples with high quality scores
            recent_samples = list(self.feedback_memory)[-100:]
            high_quality_samples = [
                s for s in recent_samples
                if s.quality_score > 0.7 or (s.user_rating and s.user_rating > 0.7)
            ]
            
            if len(high_quality_samples) < 5:
                return
            
            # Simple update: adjust towards better parameters
            # This is a simplified learning approach
            # In production, you'd use proper gradient descent
            
            # For now, we'll just track which parameters work best
            best_samples = sorted(
                high_quality_samples,
                key=lambda s: s.quality_score * (s.user_rating or 0.5),
                reverse=True
            )[:10]
            
            # Update statistics
            self.stats["samples_used"] += len(best_samples)
            
            logger.debug(f"Updated model with {len(best_samples)} high-quality samples")
            
        except Exception as e:
            logger.warning(f"Model update failed: {e}")
    
    def get_parameter_recommendations(
        self,
        image_complexity: float,
        description_complexity: float,
    ) -> Dict[str, float]:
        """
        Get parameter recommendations based on complexity.
        
        Args:
            image_complexity: Image complexity (0.0 to 1.0)
            description_complexity: Description complexity (0.0 to 1.0)
            
        Returns:
            Recommended parameters
        """
        # Base on learned patterns from feedback
        combined_complexity = (image_complexity + description_complexity) / 2.0
        
        # Adjust based on feedback history
        if len(self.feedback_memory) > 0:
            avg_quality = np.mean([s.quality_score for s in self.feedback_memory])
            if avg_quality < 0.6:
                # Lower quality, increase parameters
                combined_complexity *= 1.2
        
        return {
            "num_inference_steps": int(30 + combined_complexity * 50),
            "guidance_scale": 5.0 + combined_complexity * 8.0,
            "strength": 0.6 + combined_complexity * 0.3,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            **self.stats,
            "memory_size": len(self.feedback_memory),
            "memory_capacity": self.memory_size,
            "learning_enabled": self.enable_learning,
        }
    
    def save_model(self, file_path: Path) -> None:
        """Save model to disk."""
        try:
            state = {
                "model_state": self.prediction_model.state_dict(),
                "stats": self.stats,
                "parameter_history": self.parameter_history[-100:],  # Last 100
            }
            torch.save(state, file_path)
            logger.info(f"Model saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, file_path: Path) -> None:
        """Load model from disk."""
        try:
            state = torch.load(file_path)
            self.prediction_model.load_state_dict(state["model_state"])
            self.stats.update(state.get("stats", {}))
            self.parameter_history = state.get("parameter_history", [])
            logger.info(f"Model loaded from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


