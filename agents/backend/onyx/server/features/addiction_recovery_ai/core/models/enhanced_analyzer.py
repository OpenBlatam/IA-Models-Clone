"""
Enhanced Addiction Analyzer with Deep Learning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging
import numpy as np

from .sentiment_analyzer import (
    RecoverySentimentAnalyzer,
    RecoveryProgressPredictor,
    RelapseRiskPredictor,
    create_sentiment_analyzer,
    create_progress_predictor,
    create_relapse_predictor
)
from .llm_coach import LLMRecoveryCoach, T5RecoveryCoach, create_llm_coach, create_t5_coach

logger = logging.getLogger(__name__)


class EnhancedAddictionAnalyzer:
    """Enhanced analyzer with deep learning capabilities"""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_gpu: bool = True,
        use_sentiment: bool = True,
        use_predictors: bool = True,
        use_llm: bool = True
    ):
        """
        Initialize enhanced analyzer
        
        Args:
            device: PyTorch device
            use_gpu: Use GPU
            use_sentiment: Enable sentiment analysis
            use_predictors: Enable progress/relapse predictors
            use_llm: Enable LLM coaching
        """
        self.device = device or torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        
        # Initialize models
        self.sentiment_analyzer = None
        self.progress_predictor = None
        self.relapse_predictor = None
        self.llm_coach = None
        self.t5_coach = None
        
        if use_sentiment:
            try:
                self.sentiment_analyzer = create_sentiment_analyzer(device=self.device)
            except Exception as e:
                logger.warning(f"Sentiment analyzer not available: {e}")
        
        if use_predictors:
            try:
                self.progress_predictor = create_progress_predictor(
                    input_features=10, device=self.device
                )
                self.relapse_predictor = create_relapse_predictor(
                    input_size=5, device=self.device
                )
            except Exception as e:
                logger.warning(f"Predictors not available: {e}")
        
        if use_llm:
            try:
                self.llm_coach = create_llm_coach(device=self.device)
                self.t5_coach = create_t5_coach(device=self.device)
            except Exception as e:
                logger.warning(f"LLM coach not available: {e}")
        
        logger.info(f"EnhancedAddictionAnalyzer initialized on {self.device}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            
        Returns:
            Sentiment analysis results
        """
        if self.sentiment_analyzer:
            return self.sentiment_analyzer.analyze(text)
        return {"label": "NEUTRAL", "score": 0.5}
    
    def predict_progress(
        self,
        features: Dict[str, float]
    ) -> float:
        """
        Predict recovery progress
        
        Args:
            features: Feature dictionary
            
        Returns:
            Progress score (0-1)
        """
        if self.progress_predictor:
            # Convert features to tensor
            feature_list = [
                features.get("days_sober", 0) / 365.0,
                features.get("cravings_level", 5) / 10.0,
                features.get("stress_level", 5) / 10.0,
                features.get("support_level", 5) / 10.0,
                features.get("mood_score", 5) / 10.0,
                features.get("sleep_quality", 5) / 10.0,
                features.get("exercise_frequency", 2) / 7.0,
                features.get("therapy_sessions", 0) / 10.0,
                features.get("medication_compliance", 1.0),
                features.get("social_activity", 3) / 7.0
            ]
            
            feature_tensor = torch.tensor([feature_list], dtype=torch.float32).to(self.device)
            progress = self.progress_predictor.predict_progress(feature_tensor)
            return progress
        
        # Fallback calculation
        days_sober = features.get("days_sober", 0)
        return min(1.0, days_sober / 365.0)
    
    def predict_relapse_risk(
        self,
        sequence: List[Dict[str, float]]
    ) -> float:
        """
        Predict relapse risk from sequence
        
        Args:
            sequence: Sequence of daily features
            
        Returns:
            Relapse risk (0-1)
        """
        if self.relapse_predictor and len(sequence) > 0:
            # Convert sequence to tensor
            seq_data = []
            for day in sequence[-30:]:  # Last 30 days
                seq_data.append([
                    day.get("cravings_level", 5) / 10.0,
                    day.get("stress_level", 5) / 10.0,
                    day.get("mood_score", 5) / 10.0,
                    day.get("triggers_count", 0) / 10.0,
                    day.get("consumed", 0.0)
                ])
            
            # Pad or truncate to fixed length
            while len(seq_data) < 30:
                seq_data.insert(0, [0.0] * 5)
            
            seq_tensor = torch.tensor([seq_data], dtype=torch.float32).to(self.device)
            risk = self.relapse_predictor.predict_risk(seq_tensor)
            return risk
        
        # Fallback: simple risk calculation
        if len(sequence) > 0:
            recent_cravings = np.mean([d.get("cravings_level", 5) for d in sequence[-7:]])
            return min(1.0, recent_cravings / 10.0)
        
        return 0.5
    
    def generate_coaching(
        self,
        user_situation: str,
        days_sober: int,
        current_challenge: Optional[str] = None
    ) -> str:
        """
        Generate personalized coaching
        
        Args:
            user_situation: User's situation
            days_sober: Days of sobriety
            current_challenge: Current challenge
            
        Returns:
            Coaching message
        """
        if self.t5_coach:
            context = f"User sober {days_sober} days. Situation: {user_situation}"
            if current_challenge:
                context += f" Challenge: {current_challenge}"
            return self.t5_coach.coach(context)
        elif self.llm_coach:
            return self.llm_coach.generate_coaching_message(
                user_situation, days_sober, current_challenge
            )
        
        return f"Keep going! You've been sober for {days_sober} days. Stay strong!"
    
    def generate_motivation(
        self,
        milestone: str,
        achievement: str
    ) -> str:
        """Generate motivational message"""
        if self.t5_coach:
            return self.t5_coach.motivate(f"{milestone}: {achievement}")
        elif self.llm_coach:
            return self.llm_coach.generate_motivational_message(milestone, achievement)
        
        return f"Congratulations on {milestone}! {achievement}"


def create_enhanced_analyzer(
    device: Optional[torch.device] = None,
    use_gpu: bool = True
) -> EnhancedAddictionAnalyzer:
    """Factory function for enhanced analyzer"""
    return EnhancedAddictionAnalyzer(device=device, use_gpu=use_gpu)

