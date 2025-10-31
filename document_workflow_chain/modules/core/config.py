"""
Configuration for Advanced Workflow Chain Engine
===============================================

This module contains configuration settings for the enhanced workflow chain engine,
including model limits, quality thresholds, and performance settings.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class AnalysisLevel(Enum):
    """Levels of document analysis"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    ADVANCED = "advanced"


class ModelPriority(Enum):
    """Model priority for automatic selection"""
    SPEED = "speed"
    QUALITY = "quality"
    COST = "cost"
    BALANCED = "balanced"


@dataclass
class QualityThresholds:
    """Quality assessment thresholds"""
    excellent: float = 0.9
    good: float = 0.7
    average: float = 0.5
    poor: float = 0.3
    
    def get_quality_level(self, score: float) -> str:
        """Get quality level based on score"""
        if score >= self.excellent:
            return "excellent"
        elif score >= self.good:
            return "good"
        elif score >= self.average:
            return "average"
        elif score >= self.poor:
            return "poor"
        else:
            return "very_poor"


@dataclass
class PerformanceSettings:
    """Performance tracking settings"""
    track_metrics: bool = True
    max_history_size: int = 1000
    session_timeout_minutes: int = 60
    enable_trend_analysis: bool = True
    trend_window_size: int = 10


@dataclass
class ContextSettings:
    """Context management settings"""
    max_context_utilization: float = 0.8  # Use max 80% of context window
    compression_threshold: float = 0.7    # Compress when 70% full
    chunk_size_tokens: int = 1000         # Optimal chunk size
    overlap_tokens: int = 100             # Overlap between chunks
    enable_smart_compression: bool = True


@dataclass
class AnalysisSettings:
    """Document analysis settings"""
    enable_sentiment_analysis: bool = True
    enable_topic_extraction: bool = True
    enable_entity_recognition: bool = True
    enable_readability_analysis: bool = True
    enable_coherence_analysis: bool = True
    enable_structure_analysis: bool = True
    max_topics: int = 10
    min_topic_frequency: int = 2


class WorkflowConfig:
    """Main configuration class for workflow chain engine"""
    
    def __init__(self):
        self.quality_thresholds = QualityThresholds()
        self.performance_settings = PerformanceSettings()
        self.context_settings = ContextSettings()
        self.analysis_settings = AnalysisSettings()
        
        # Model configurations
        self.model_configs = {
            "claude-3-5-sonnet-20241022": {
                "context_limit": 200000,
                "cost_per_1k_tokens": 0.003,
                "speed_rating": 8,
                "quality_rating": 9,
                "priority": ModelPriority.QUALITY
            },
            "claude-3-opus-20240229": {
                "context_limit": 200000,
                "cost_per_1k_tokens": 0.015,
                "speed_rating": 6,
                "quality_rating": 10,
                "priority": ModelPriority.QUALITY
            },
            "claude-3-haiku-20240307": {
                "context_limit": 200000,
                "cost_per_1k_tokens": 0.00025,
                "speed_rating": 10,
                "quality_rating": 7,
                "priority": ModelPriority.SPEED
            },
            "gpt-4-turbo": {
                "context_limit": 128000,
                "cost_per_1k_tokens": 0.01,
                "speed_rating": 7,
                "quality_rating": 9,
                "priority": ModelPriority.BALANCED
            },
            "gpt-4": {
                "context_limit": 8192,
                "cost_per_1k_tokens": 0.03,
                "speed_rating": 5,
                "quality_rating": 9,
                "priority": ModelPriority.QUALITY
            },
            "gpt-3.5-turbo": {
                "context_limit": 16384,
                "cost_per_1k_tokens": 0.002,
                "speed_rating": 9,
                "quality_rating": 7,
                "priority": ModelPriority.SPEED
            },
            "gemini-1.5-pro": {
                "context_limit": 1000000,
                "cost_per_1k_tokens": 0.00125,
                "speed_rating": 6,
                "quality_rating": 8,
                "priority": ModelPriority.BALANCED
            },
            "gemini-1.5-flash": {
                "context_limit": 1000000,
                "cost_per_1k_tokens": 0.000075,
                "speed_rating": 10,
                "quality_rating": 7,
                "priority": ModelPriority.SPEED
            },
            "gemini-pro": {
                "context_limit": 32768,
                "cost_per_1k_tokens": 0.0005,
                "speed_rating": 8,
                "quality_rating": 7,
                "priority": ModelPriority.BALANCED
            }
        }
        
        # Analysis level configurations
        self.analysis_levels = {
            AnalysisLevel.BASIC: {
                "enable_sentiment": False,
                "enable_topics": False,
                "enable_entities": False,
                "enable_readability": True,
                "enable_coherence": False,
                "enable_structure": True
            },
            AnalysisLevel.STANDARD: {
                "enable_sentiment": True,
                "enable_topics": True,
                "enable_entities": False,
                "enable_readability": True,
                "enable_coherence": True,
                "enable_structure": True
            },
            AnalysisLevel.COMPREHENSIVE: {
                "enable_sentiment": True,
                "enable_topics": True,
                "enable_entities": True,
                "enable_readability": True,
                "enable_coherence": True,
                "enable_structure": True
            },
            AnalysisLevel.ADVANCED: {
                "enable_sentiment": True,
                "enable_topics": True,
                "enable_entities": True,
                "enable_readability": True,
                "enable_coherence": True,
                "enable_structure": True,
                "enable_performance_tracking": True,
                "enable_trend_analysis": True
            }
        }
    
    def get_optimal_model(self, content_size: int, priority: ModelPriority = ModelPriority.BALANCED) -> str:
        """Get optimal model based on content size and priority"""
        suitable_models = []
        
        for model_name, config in self.model_configs.items():
            if content_size <= config["context_limit"]:
                suitable_models.append((model_name, config))
        
        if not suitable_models:
            # If no model can handle the content, return the one with largest context
            return max(self.model_configs.keys(), 
                      key=lambda m: self.model_configs[m]["context_limit"])
        
        # Sort by priority
        if priority == ModelPriority.SPEED:
            suitable_models.sort(key=lambda x: x[1]["speed_rating"], reverse=True)
        elif priority == ModelPriority.QUALITY:
            suitable_models.sort(key=lambda x: x[1]["quality_rating"], reverse=True)
        elif priority == ModelPriority.COST:
            suitable_models.sort(key=lambda x: x[1]["cost_per_1k_tokens"])
        else:  # BALANCED
            # Weighted score: 40% quality, 30% speed, 30% cost (inverted)
            suitable_models.sort(key=lambda x: (
                x[1]["quality_rating"] * 0.4 + 
                x[1]["speed_rating"] * 0.3 + 
                (1 / x[1]["cost_per_1k_tokens"]) * 0.3
            ), reverse=True)
        
        return suitable_models[0][0]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        return self.model_configs.get(model_name, {})
    
    def get_analysis_config(self, level: AnalysisLevel) -> Dict[str, Any]:
        """Get analysis configuration for a specific level"""
        return self.analysis_levels.get(level, self.analysis_levels[AnalysisLevel.STANDARD])
    
    def estimate_processing_cost(self, content_size: int, model_name: str) -> float:
        """Estimate processing cost for content"""
        config = self.get_model_info(model_name)
        if not config:
            return 0.0
        
        cost_per_1k = config["cost_per_1k_tokens"]
        return (content_size / 1000) * cost_per_1k
    
    def get_recommended_models(self, content_size: int, max_cost: float = None) -> List[Dict[str, Any]]:
        """Get recommended models for content size and budget"""
        recommendations = []
        
        for model_name, config in self.model_configs.items():
            if content_size <= config["context_limit"]:
                cost = self.estimate_processing_cost(content_size, model_name)
                
                if max_cost is None or cost <= max_cost:
                    recommendations.append({
                        "model": model_name,
                        "context_limit": config["context_limit"],
                        "estimated_cost": cost,
                        "speed_rating": config["speed_rating"],
                        "quality_rating": config["quality_rating"],
                        "utilization": (content_size / config["context_limit"]) * 100
                    })
        
        # Sort by utilization (prefer models that use more of their context efficiently)
        recommendations.sort(key=lambda x: x["utilization"], reverse=True)
        
        return recommendations
    
    def validate_content_size(self, content_size: int, model_name: str) -> Dict[str, Any]:
        """Validate if content can be processed by model"""
        config = self.get_model_info(model_name)
        if not config:
            return {"valid": False, "error": "Model not found"}
        
        context_limit = config["context_limit"]
        utilization = (content_size / context_limit) * 100
        
        if content_size > context_limit:
            return {
                "valid": False,
                "error": f"Content size ({content_size:,} tokens) exceeds model limit ({context_limit:,} tokens)",
                "suggestion": "Consider chunking or using a model with larger context"
            }
        
        return {
            "valid": True,
            "utilization": utilization,
            "context_limit": context_limit,
            "remaining_capacity": context_limit - content_size
        }


# Global configuration instance
workflow_config = WorkflowConfig()


def get_workflow_config() -> WorkflowConfig:
    """Get the global workflow configuration"""
    return workflow_config


def get_model_limits() -> Dict[str, int]:
    """Get context limits for all models"""
    return {model: config["context_limit"] for model, config in workflow_config.model_configs.items()}


def get_quality_thresholds() -> QualityThresholds:
    """Get quality assessment thresholds"""
    return workflow_config.quality_thresholds


def get_performance_settings() -> PerformanceSettings:
    """Get performance tracking settings"""
    return workflow_config.performance_settings


def get_context_settings() -> ContextSettings:
    """Get context management settings"""
    return workflow_config.context_settings


def get_analysis_settings() -> AnalysisSettings:
    """Get document analysis settings"""
    return workflow_config.analysis_settings