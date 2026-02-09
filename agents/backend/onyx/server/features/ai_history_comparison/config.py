"""
Configuration for AI History Analyzer and Model Comparison System
================================================================

This module provides comprehensive configuration management for the AI history
analyzer, including model definitions, metric configurations, and analysis settings.

Features:
- Model configuration and definitions
- Metric configuration and weights
- Analysis parameters and thresholds
- Performance benchmarks
- Alert configurations
- Export and reporting settings
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import timedelta
import json
import os


class ModelProvider(Enum):
    """AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    META = "meta"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class ModelCategory(Enum):
    """Model categories"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"


@dataclass
class ModelDefinition:
    """Definition of an AI model"""
    name: str
    provider: ModelProvider
    category: ModelCategory
    version: str
    context_length: int
    parameters: str  # e.g., "7B", "175B", "70B"
    release_date: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    cost_per_1k_tokens: float = 0.0
    max_requests_per_minute: int = 0
    is_active: bool = True


@dataclass
class MetricConfiguration:
    """Configuration for a performance metric"""
    name: str
    description: str
    unit: str
    min_value: float
    max_value: float
    optimal_range: tuple
    weight: float  # Weight in overall scoring
    higher_is_better: bool = True
    calculation_method: str = "mean"  # mean, median, latest, weighted
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisConfiguration:
    """Configuration for analysis parameters"""
    trend_analysis_window_days: int = 90
    comparison_minimum_samples: int = 10
    anomaly_detection_sigma: float = 2.0
    forecast_days: int = 7
    confidence_threshold: float = 0.7
    cache_ttl_seconds: int = 3600
    max_history_days: int = 365
    cleanup_interval_hours: int = 24


@dataclass
class AlertConfiguration:
    """Configuration for alerts and notifications"""
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["email", "webhook"])
    performance_degradation_threshold: float = 0.1  # 10% degradation
    anomaly_detection_enabled: bool = True
    trend_alert_threshold: float = 0.8  # 80% confidence
    cooldown_period_minutes: int = 60
    max_alerts_per_hour: int = 10


@dataclass
class BenchmarkConfiguration:
    """Configuration for performance benchmarks"""
    enable_benchmarking: bool = True
    benchmark_datasets: List[str] = field(default_factory=list)
    benchmark_frequency_days: int = 7
    benchmark_metrics: List[str] = field(default_factory=list)
    baseline_models: List[str] = field(default_factory=list)
    comparison_models: List[str] = field(default_factory=list)


@dataclass
class ExportConfiguration:
    """Configuration for data export and reporting"""
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv", "excel"])
    report_templates: List[str] = field(default_factory=list)
    scheduled_reports: Dict[str, str] = field(default_factory=dict)
    retention_policy_days: int = 1095  # 3 years
    compression_enabled: bool = True
    encryption_enabled: bool = False


class AIHistoryConfig:
    """Main configuration class for AI history analyzer"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.models: Dict[str, ModelDefinition] = {}
        self.metrics: Dict[str, MetricConfiguration] = {}
        self.analysis = AnalysisConfiguration()
        self.alerts = AlertConfiguration()
        self.benchmarks = BenchmarkConfiguration()
        self.export = ExportConfiguration()
        
        # Load default configurations
        self._load_default_models()
        self._load_default_metrics()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def _load_default_models(self):
        """Load default model definitions"""
        default_models = [
            ModelDefinition(
                name="gpt-4",
                provider=ModelProvider.OPENAI,
                category=ModelCategory.TEXT_GENERATION,
                version="4.0",
                context_length=8192,
                parameters="Unknown",
                release_date="2023-03-14",
                description="Advanced text generation model",
                capabilities=["text_generation", "analysis", "reasoning"],
                limitations=["limited_context", "cost"],
                cost_per_1k_tokens=0.03,
                max_requests_per_minute=500
            ),
            ModelDefinition(
                name="gpt-4-turbo",
                provider=ModelProvider.OPENAI,
                category=ModelCategory.TEXT_GENERATION,
                version="4.0-turbo",
                context_length=128000,
                parameters="Unknown",
                release_date="2023-11-06",
                description="Enhanced GPT-4 with larger context",
                capabilities=["text_generation", "analysis", "long_context"],
                limitations=["cost"],
                cost_per_1k_tokens=0.01,
                max_requests_per_minute=1000
            ),
            ModelDefinition(
                name="claude-3-opus",
                provider=ModelProvider.ANTHROPIC,
                category=ModelCategory.TEXT_GENERATION,
                version="3.0",
                context_length=200000,
                parameters="Unknown",
                release_date="2024-02-29",
                description="Most capable Claude model",
                capabilities=["text_generation", "analysis", "reasoning", "coding"],
                limitations=["cost"],
                cost_per_1k_tokens=0.015,
                max_requests_per_minute=1000
            ),
            ModelDefinition(
                name="claude-3-sonnet",
                provider=ModelProvider.ANTHROPIC,
                category=ModelCategory.TEXT_GENERATION,
                version="3.0",
                context_length=200000,
                parameters="Unknown",
                release_date="2024-02-29",
                description="Balanced Claude model",
                capabilities=["text_generation", "analysis", "coding"],
                limitations=["cost"],
                cost_per_1k_tokens=0.003,
                max_requests_per_minute=2000
            ),
            ModelDefinition(
                name="claude-3-haiku",
                provider=ModelProvider.ANTHROPIC,
                category=ModelCategory.TEXT_GENERATION,
                version="3.0",
                context_length=200000,
                parameters="Unknown",
                release_date="2024-02-29",
                description="Fast and efficient Claude model",
                capabilities=["text_generation", "analysis"],
                limitations=["capability"],
                cost_per_1k_tokens=0.00025,
                max_requests_per_minute=5000
            ),
            ModelDefinition(
                name="gemini-1.5-pro",
                provider=ModelProvider.GOOGLE,
                category=ModelCategory.TEXT_GENERATION,
                version="1.5",
                context_length=1000000,
                parameters="Unknown",
                release_date="2024-02-15",
                description="Google's most capable model",
                capabilities=["text_generation", "multimodal", "long_context"],
                limitations=["availability"],
                cost_per_1k_tokens=0.00125,
                max_requests_per_minute=1000
            ),
            ModelDefinition(
                name="gemini-1.5-flash",
                provider=ModelProvider.GOOGLE,
                category=ModelCategory.TEXT_GENERATION,
                version="1.5",
                context_length=1000000,
                parameters="Unknown",
                release_date="2024-02-15",
                description="Fast and efficient Gemini model",
                capabilities=["text_generation", "multimodal"],
                limitations=["capability"],
                cost_per_1k_tokens=0.000075,
                max_requests_per_minute=2000
            )
        ]
        
        for model in default_models:
            self.models[model.name] = model
    
    def _load_default_metrics(self):
        """Load default metric configurations"""
        default_metrics = [
            MetricConfiguration(
                name="quality_score",
                description="Overall quality of generated content",
                unit="score",
                min_value=0.0,
                max_value=1.0,
                optimal_range=(0.7, 1.0),
                weight=0.3,
                higher_is_better=True,
                calculation_method="mean",
                alert_thresholds={"warning": 0.6, "critical": 0.4}
            ),
            MetricConfiguration(
                name="response_time",
                description="Time taken to generate response",
                unit="seconds",
                min_value=0.0,
                max_value=60.0,
                optimal_range=(0.0, 2.0),
                weight=0.2,
                higher_is_better=False,
                calculation_method="mean",
                alert_thresholds={"warning": 5.0, "critical": 10.0}
            ),
            MetricConfiguration(
                name="token_efficiency",
                description="Efficiency of token usage",
                unit="ratio",
                min_value=0.0,
                max_value=1.0,
                optimal_range=(0.8, 1.0),
                weight=0.2,
                higher_is_better=True,
                calculation_method="mean",
                alert_thresholds={"warning": 0.6, "critical": 0.4}
            ),
            MetricConfiguration(
                name="cost_efficiency",
                description="Cost per unit of quality",
                unit="ratio",
                min_value=0.0,
                max_value=1.0,
                optimal_range=(0.7, 1.0),
                weight=0.15,
                higher_is_better=True,
                calculation_method="mean",
                alert_thresholds={"warning": 0.5, "critical": 0.3}
            ),
            MetricConfiguration(
                name="accuracy",
                description="Accuracy of generated content",
                unit="score",
                min_value=0.0,
                max_value=1.0,
                optimal_range=(0.8, 1.0),
                weight=0.15,
                higher_is_better=True,
                calculation_method="mean",
                alert_thresholds={"warning": 0.7, "critical": 0.5}
            ),
            MetricConfiguration(
                name="coherence",
                description="Coherence and logical flow",
                unit="score",
                min_value=0.0,
                max_value=1.0,
                optimal_range=(0.7, 1.0),
                weight=0.1,
                higher_is_better=True,
                calculation_method="mean",
                alert_thresholds={"warning": 0.6, "critical": 0.4}
            ),
            MetricConfiguration(
                name="relevance",
                description="Relevance to input prompt",
                unit="score",
                min_value=0.0,
                max_value=1.0,
                optimal_range=(0.8, 1.0),
                weight=0.1,
                higher_is_better=True,
                calculation_method="mean",
                alert_thresholds={"warning": 0.7, "critical": 0.5}
            ),
            MetricConfiguration(
                name="creativity",
                description="Creativity and originality",
                unit="score",
                min_value=0.0,
                max_value=1.0,
                optimal_range=(0.6, 1.0),
                weight=0.05,
                higher_is_better=True,
                calculation_method="mean",
                alert_thresholds={"warning": 0.4, "critical": 0.2}
            )
        ]
        
        for metric in default_metrics:
            self.metrics[metric.name] = metric
    
    def get_model(self, model_name: str) -> Optional[ModelDefinition]:
        """Get model definition by name"""
        return self.models.get(model_name)
    
    def get_metric(self, metric_name: str) -> Optional[MetricConfiguration]:
        """Get metric configuration by name"""
        return self.metrics.get(metric_name)
    
    def get_models_by_provider(self, provider: ModelProvider) -> List[ModelDefinition]:
        """Get all models from a specific provider"""
        return [model for model in self.models.values() if model.provider == provider]
    
    def get_models_by_category(self, category: ModelCategory) -> List[ModelDefinition]:
        """Get all models in a specific category"""
        return [model for model in self.models.values() if model.category == category]
    
    def get_active_models(self) -> List[ModelDefinition]:
        """Get all active models"""
        return [model for model in self.models.values() if model.is_active]
    
    def get_metric_weights(self) -> Dict[str, float]:
        """Get weights for all metrics"""
        return {name: metric.weight for name, metric in self.metrics.items()}
    
    def get_alert_thresholds(self, metric_name: str) -> Dict[str, float]:
        """Get alert thresholds for a metric"""
        metric = self.get_metric(metric_name)
        return metric.alert_thresholds if metric else {}
    
    def add_model(self, model: ModelDefinition):
        """Add a new model definition"""
        self.models[model.name] = model
    
    def update_model(self, model_name: str, updates: Dict[str, Any]):
        """Update an existing model definition"""
        if model_name in self.models:
            model = self.models[model_name]
            for key, value in updates.items():
                if hasattr(model, key):
                    setattr(model, key, value)
    
    def add_metric(self, metric: MetricConfiguration):
        """Add a new metric configuration"""
        self.metrics[metric.name] = metric
    
    def update_metric(self, metric_name: str, updates: Dict[str, Any]):
        """Update an existing metric configuration"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            for key, value in updates.items():
                if hasattr(metric, key):
                    setattr(metric, key, value)
    
    def get_benchmark_config(self) -> BenchmarkConfiguration:
        """Get benchmark configuration"""
        return self.benchmarks
    
    def get_analysis_config(self) -> AnalysisConfiguration:
        """Get analysis configuration"""
        return self.analysis
    
    def get_alert_config(self) -> AlertConfiguration:
        """Get alert configuration"""
        return self.alerts
    
    def get_export_config(self) -> ExportConfiguration:
        """Get export configuration"""
        return self.export
    
    def save_to_file(self, file_path: str):
        """Save configuration to file"""
        try:
            config_data = {
                "models": {
                    name: {
                        "name": model.name,
                        "provider": model.provider.value,
                        "category": model.category.value,
                        "version": model.version,
                        "context_length": model.context_length,
                        "parameters": model.parameters,
                        "release_date": model.release_date,
                        "description": model.description,
                        "capabilities": model.capabilities,
                        "limitations": model.limitations,
                        "cost_per_1k_tokens": model.cost_per_1k_tokens,
                        "max_requests_per_minute": model.max_requests_per_minute,
                        "is_active": model.is_active
                    }
                    for name, model in self.models.items()
                },
                "metrics": {
                    name: {
                        "name": metric.name,
                        "description": metric.description,
                        "unit": metric.unit,
                        "min_value": metric.min_value,
                        "max_value": metric.max_value,
                        "optimal_range": metric.optimal_range,
                        "weight": metric.weight,
                        "higher_is_better": metric.higher_is_better,
                        "calculation_method": metric.calculation_method,
                        "alert_thresholds": metric.alert_thresholds
                    }
                    for name, metric in self.metrics.items()
                },
                "analysis": {
                    "trend_analysis_window_days": self.analysis.trend_analysis_window_days,
                    "comparison_minimum_samples": self.analysis.comparison_minimum_samples,
                    "anomaly_detection_sigma": self.analysis.anomaly_detection_sigma,
                    "forecast_days": self.analysis.forecast_days,
                    "confidence_threshold": self.analysis.confidence_threshold,
                    "cache_ttl_seconds": self.analysis.cache_ttl_seconds,
                    "max_history_days": self.analysis.max_history_days,
                    "cleanup_interval_hours": self.analysis.cleanup_interval_hours
                },
                "alerts": {
                    "enable_alerts": self.alerts.enable_alerts,
                    "alert_channels": self.alerts.alert_channels,
                    "performance_degradation_threshold": self.alerts.performance_degradation_threshold,
                    "anomaly_detection_enabled": self.alerts.anomaly_detection_enabled,
                    "trend_alert_threshold": self.alerts.trend_alert_threshold,
                    "cooldown_period_minutes": self.alerts.cooldown_period_minutes,
                    "max_alerts_per_hour": self.alerts.max_alerts_per_hour
                },
                "benchmarks": {
                    "enable_benchmarking": self.benchmarks.enable_benchmarking,
                    "benchmark_datasets": self.benchmarks.benchmark_datasets,
                    "benchmark_frequency_days": self.benchmarks.benchmark_frequency_days,
                    "benchmark_metrics": self.benchmarks.benchmark_metrics,
                    "baseline_models": self.benchmarks.baseline_models,
                    "comparison_models": self.benchmarks.comparison_models
                },
                "export": {
                    "export_formats": self.export.export_formats,
                    "report_templates": self.export.report_templates,
                    "scheduled_reports": self.export.scheduled_reports,
                    "retention_policy_days": self.export.retention_policy_days,
                    "compression_enabled": self.export.compression_enabled,
                    "encryption_enabled": self.export.encryption_enabled
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            raise Exception(f"Error saving configuration: {str(e)}")
    
    def load_from_file(self, file_path: str):
        """Load configuration from file"""
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            # Load models
            if "models" in config_data:
                for name, model_data in config_data["models"].items():
                    model = ModelDefinition(
                        name=model_data["name"],
                        provider=ModelProvider(model_data["provider"]),
                        category=ModelCategory(model_data["category"]),
                        version=model_data["version"],
                        context_length=model_data["context_length"],
                        parameters=model_data["parameters"],
                        release_date=model_data["release_date"],
                        description=model_data["description"],
                        capabilities=model_data.get("capabilities", []),
                        limitations=model_data.get("limitations", []),
                        cost_per_1k_tokens=model_data.get("cost_per_1k_tokens", 0.0),
                        max_requests_per_minute=model_data.get("max_requests_per_minute", 0),
                        is_active=model_data.get("is_active", True)
                    )
                    self.models[name] = model
            
            # Load metrics
            if "metrics" in config_data:
                for name, metric_data in config_data["metrics"].items():
                    metric = MetricConfiguration(
                        name=metric_data["name"],
                        description=metric_data["description"],
                        unit=metric_data["unit"],
                        min_value=metric_data["min_value"],
                        max_value=metric_data["max_value"],
                        optimal_range=tuple(metric_data["optimal_range"]),
                        weight=metric_data["weight"],
                        higher_is_better=metric_data.get("higher_is_better", True),
                        calculation_method=metric_data.get("calculation_method", "mean"),
                        alert_thresholds=metric_data.get("alert_thresholds", {})
                    )
                    self.metrics[name] = metric
            
            # Load other configurations
            if "analysis" in config_data:
                analysis_data = config_data["analysis"]
                self.analysis = AnalysisConfiguration(**analysis_data)
            
            if "alerts" in config_data:
                alert_data = config_data["alerts"]
                self.alerts = AlertConfiguration(**alert_data)
            
            if "benchmarks" in config_data:
                benchmark_data = config_data["benchmarks"]
                self.benchmarks = BenchmarkConfiguration(**benchmark_data)
            
            if "export" in config_data:
                export_data = config_data["export"]
                self.export = ExportConfiguration(**export_data)
                
        except Exception as e:
            raise Exception(f"Error loading configuration: {str(e)}")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            "models": {
                "total": len(self.models),
                "active": len(self.get_active_models()),
                "providers": len(set(model.provider for model in self.models.values())),
                "categories": len(set(model.category for model in self.models.values()))
            },
            "metrics": {
                "total": len(self.metrics),
                "weighted_metrics": len([m for m in self.metrics.values() if m.weight > 0])
            },
            "analysis": {
                "trend_window_days": self.analysis.trend_analysis_window_days,
                "max_history_days": self.analysis.max_history_days,
                "confidence_threshold": self.analysis.confidence_threshold
            },
            "alerts": {
                "enabled": self.alerts.enable_alerts,
                "channels": len(self.alerts.alert_channels),
                "anomaly_detection": self.alerts.anomaly_detection_enabled
            },
            "benchmarks": {
                "enabled": self.benchmarks.enable_benchmarking,
                "datasets": len(self.benchmarks.benchmark_datasets),
                "frequency_days": self.benchmarks.benchmark_frequency_days
            },
            "export": {
                "formats": len(self.export.export_formats),
                "retention_days": self.export.retention_policy_days,
                "compression": self.export.compression_enabled
            }
        }


# Global configuration instance
_config: Optional[AIHistoryConfig] = None


def get_ai_history_config(config_file: Optional[str] = None) -> AIHistoryConfig:
    """Get or create global AI history configuration"""
    global _config
    if _config is None:
        _config = AIHistoryConfig(config_file)
    return _config


def load_config_from_file(file_path: str) -> AIHistoryConfig:
    """Load configuration from file and return instance"""
    return AIHistoryConfig(file_path)


def save_config_to_file(config: AIHistoryConfig, file_path: str):
    """Save configuration to file"""
    config.save_to_file(file_path)


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = get_ai_history_config()
    
    # Print configuration summary
    summary = config.get_configuration_summary()
    print("Configuration Summary:")
    print(json.dumps(summary, indent=2))
    
    # Save to file
    config.save_to_file("ai_history_config.json")
    print("Configuration saved to ai_history_config.json")