"""Intelligent Model Recommendation Engine"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime


class ModelCategory(Enum):
    VISION = "vision"
    NLP = "nlp"


@dataclass
class ModelProfile:
    model_id: str
    name: str
    category: ModelCategory
    accuracy: float = 0.0
    cost_per_inference: float = 0.0


@dataclass
class RecommendationRequest:
    task_type: str
    category: ModelCategory
    accuracy_requirement: Optional[float] = None


@dataclass
class ModelRecommendation:
    model_id: str
    model_name: str
    confidence_score: float
    estimated_cost: float


class IntelligentModelRecommendationEngine:
    def __init__(self):
        self.model_profiles: Dict[str, ModelProfile] = {}
    
    async def add_model_profile(self, profile: ModelProfile):
        self.model_profiles[profile.model_id] = profile
    
    async def get_recommendations(self, request: RecommendationRequest) -> List[ModelRecommendation]:
        recommendations = []
        relevant_models = [m for m in self.model_profiles.values() if m.category == request.category]
        
        for model in relevant_models:
            score = 0.8 if model.accuracy >= (request.accuracy_requirement or 0.0) else 0.3
            recommendations.append(ModelRecommendation(
                model_id=model.model_id,
                model_name=model.name,
                confidence_score=score,
                estimated_cost=model.cost_per_inference
            ))
        
        return recommendations[:3]
    
    def get_system_status(self) -> Dict[str, any]:
        return {"total_models": len(self.model_profiles)}


def create_recommendation_engine():
    return IntelligentModelRecommendationEngine()