#!/usr/bin/env python3
"""
SEO Evaluation Configuration
Configuration files and settings for SEO model evaluation
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json

# =============================================================================
# SEO EVALUATION CONFIGURATIONS
# =============================================================================

@dataclass
class SEORankingConfig:
    """Configuration for SEO ranking evaluation."""
    # NDCG evaluation
    ndcg_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # MAP evaluation
    map_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # MRR evaluation
    mrr_threshold: float = 0.5
    
    # Precision/Recall at K
    precision_recall_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # Relevance thresholds
    relevance_threshold: float = 0.5
    high_relevance_threshold: float = 0.8
    
    # Ranking bias correction
    apply_bias_correction: bool = True
    bias_correction_method: str = "inverse_rank"  # "inverse_rank", "position_based"

@dataclass
class SEOContentQualityConfig:
    """Configuration for SEO content quality evaluation."""
    # Content length thresholds
    min_content_length: int = 300
    optimal_content_length: int = 1500
    max_content_length: int = 5000
    
    # Keyword density
    min_keyword_density: float = 0.005
    optimal_keyword_density: float = 0.015
    max_keyword_density: float = 0.03
    
    # Readability scores
    min_readability_score: float = 60.0
    optimal_readability_score: float = 80.0
    
    # Content structure
    require_headings: bool = True
    require_images: bool = False
    require_internal_links: bool = True
    
    # Quality weights
    content_length_weight: float = 0.3
    keyword_density_weight: float = 0.2
    readability_weight: float = 0.3
    structure_weight: float = 0.2

@dataclass
class SEOUserEngagementConfig:
    """Configuration for SEO user engagement evaluation."""
    # Time on page
    min_time_on_page: float = 30.0
    optimal_time_on_page: float = 120.0
    
    # Click-through rate
    min_click_through_rate: float = 0.01
    optimal_click_through_rate: float = 0.05
    
    # Bounce rate
    max_bounce_rate: float = 0.7
    optimal_bounce_rate: float = 0.4
    
    # Scroll depth
    min_scroll_depth: float = 0.5
    optimal_scroll_depth: float = 0.8
    
    # Engagement weights
    time_weight: float = 0.3
    ctr_weight: float = 0.3
    bounce_weight: float = 0.2
    scroll_weight: float = 0.2

@dataclass
class SEOTechnicalConfig:
    """Configuration for SEO technical evaluation."""
    # Page load speed
    max_load_time: float = 3.0
    optimal_load_time: float = 1.5
    
    # Core Web Vitals
    max_lcp: float = 2.5
    max_fid: float = 100.0
    max_cls: float = 0.1
    
    # Mobile friendliness
    min_mobile_score: float = 80.0
    optimal_mobile_score: float = 95.0
    
    # Technical weights
    load_speed_weight: float = 0.2
    mobile_weight: float = 0.2
    lcp_weight: float = 0.2
    fid_weight: float = 0.2
    cls_weight: float = 0.2

@dataclass
class SEOMetricsConfig:
    """Complete SEO metrics configuration."""
    # Task types
    task_types: List[str] = field(default_factory=lambda: [
        "classification", "ranking", "regression", "multitask"
    ])
    
    # Ranking configuration
    ranking: SEORankingConfig = field(default_factory=SEORankingConfig)
    
    # Content quality configuration
    content_quality: SEOContentQualityConfig = field(default_factory=SEOContentQualityConfig)
    
    # User engagement configuration
    user_engagement: SEOUserEngagementConfig = field(default_factory=SEOUserEngagementConfig)
    
    # Technical configuration
    technical: SEOTechnicalConfig = field(default_factory=SEOTechnicalConfig)
    
    # Evaluation settings
    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_bootstrap: bool = True
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Output settings
    save_detailed_results: bool = True
    create_visualizations: bool = True
    export_format: str = "json"  # "json", "csv", "excel"
    output_directory: str = "seo_evaluation_results"

@dataclass
class ClassificationMetricsConfig:
    """Configuration for classification metrics."""
    # Averaging methods
    average_methods: List[str] = field(default_factory=lambda: [
        "micro", "macro", "weighted", "binary"
    ])
    default_average: str = "weighted"
    
    # Zero division handling
    zero_division: int = 0
    
    # Sample weights
    use_sample_weights: bool = True
    
    # Additional metrics
    enable_roc_auc: bool = True
    enable_precision_recall_curve: bool = True
    enable_confusion_matrix: bool = True
    enable_classification_report: bool = True

@dataclass
class RegressionMetricsConfig:
    """Configuration for regression metrics."""
    # Multi-output handling
    multioutput: str = "uniform_average"  # "raw_values", "uniform_average"
    
    # Sample weights
    use_sample_weights: bool = True
    
    # Additional metrics
    enable_mape: bool = True
    enable_smape: bool = True
    enable_huber_loss: bool = True
    enable_quantile_loss: bool = False
    
    # Error thresholds
    mape_threshold: float = 20.0
    smape_threshold: float = 25.0

@dataclass
class VisualizationConfig:
    """Configuration for evaluation visualizations."""
    # Plot settings
    figure_size: tuple = (15, 10)
    dpi: int = 300
    style: str = "seaborn-v0_8"
    
    # Color schemes
    color_palette: str = "viridis"
    cmap: str = "Blues"
    
    # Save settings
    save_format: str = "png"  # "png", "pdf", "svg"
    save_dpi: int = 300
    save_bbox_inches: str = "tight"
    
    # Display settings
    show_plots: bool = True
    interactive_mode: bool = False

@dataclass
class CompleteSEOEvaluationConfig:
    """Complete configuration for SEO evaluation system."""
    # Main configurations
    seo_metrics: SEOMetricsConfig = field(default_factory=SEOMetricsConfig)
    classification: ClassificationMetricsConfig = field(default_factory=ClassificationMetricsConfig)
    regression: RegressionMetricsConfig = field(default_factory=RegressionMetricsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # System settings
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_progress_bars: bool = True
    enable_async_evaluation: bool = True
    
    # Performance settings
    batch_size: int = 32
    num_workers: int = 4
    enable_gpu: bool = True
    mixed_precision: bool = False

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

class SEOEvaluationConfigManager:
    """Manager for SEO evaluation configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
    
    def create_default_config(self) -> CompleteSEOEvaluationConfig:
        """Create default configuration."""
        return CompleteSEOEvaluationConfig()
    
    def load_config_from_yaml(self, filepath: str) -> CompleteSEOEvaluationConfig:
        """Load configuration from YAML file."""
        try:
            with open(filepath, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Convert to dataclass
            config = self._dict_to_config(config_data)
            self.config = config
            return config
            
        except Exception as e:
            print(f"❌ Error loading config from {filepath}: {e}")
            print("📝 Creating default configuration...")
            return self.create_default_config()
    
    def save_config_to_yaml(self, config: CompleteSEOEvaluationConfig, filepath: str):
        """Save configuration to YAML file."""
        try:
            # Convert dataclass to dict
            config_dict = self._config_to_dict(config)
            
            # Save to YAML
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            print(f"✅ Configuration saved to {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving config to {filepath}: {e}")
    
    def _config_to_dict(self, config: CompleteSEOEvaluationConfig) -> Dict[str, Any]:
        """Convert configuration dataclass to dictionary."""
        return asdict(config)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> CompleteSEOEvaluationConfig:
        """Convert dictionary to configuration dataclass."""
        # This is a simplified conversion - in production, you'd want more robust handling
        return CompleteSEOEvaluationConfig(**config_dict)
    
    def validate_config(self, config: CompleteSEOEvaluationConfig) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Validate SEO metrics
        if config.seo_metrics.cv_folds < 2:
            issues.append("Cross-validation folds must be at least 2")
        
        if not (0 < config.seo_metrics.confidence_level < 1):
            issues.append("Confidence level must be between 0 and 1")
        
        # Validate content quality
        if config.seo_metrics.content_quality.min_content_length <= 0:
            issues.append("Minimum content length must be positive")
        
        if not (0 < config.seo_metrics.content_quality.max_keyword_density < 1):
            issues.append("Maximum keyword density must be between 0 and 1")
        
        # Validate user engagement
        if config.seo_metrics.user_engagement.min_time_on_page <= 0:
            issues.append("Minimum time on page must be positive")
        
        if not (0 < config.seo_metrics.user_engagement.max_bounce_rate < 1):
            issues.append("Maximum bounce rate must be between 0 and 1")
        
        # Validate technical
        if config.seo_metrics.technical.max_load_time <= 0:
            issues.append("Maximum load time must be positive")
        
        if config.seo_metrics.technical.max_lcp <= 0:
            issues.append("Maximum LCP must be positive")
        
        return issues
    
    def get_optimized_config(self, task_type: str, dataset_size: int) -> CompleteSEOEvaluationConfig:
        """Get optimized configuration for specific task and dataset."""
        config = self.create_default_config()
        
        # Optimize based on task type
        if task_type == "classification":
            config.classification.default_average = "weighted"
            config.classification.enable_roc_auc = True
        
        elif task_type == "ranking":
            config.seo_metrics.ranking.ndcg_k_values = [1, 3, 5, 10, 20]
            config.seo_metrics.ranking.map_k_values = [1, 3, 5, 10, 20]
        
        elif task_type == "regression":
            config.regression.enable_mape = True
            config.regression.enable_smape = True
        
        # Optimize based on dataset size
        if dataset_size < 1000:
            config.seo_metrics.cv_folds = 3
            config.seo_metrics.bootstrap_samples = 500
        elif dataset_size > 10000:
            config.seo_metrics.cv_folds = 10
            config.seo_metrics.bootstrap_samples = 2000
        
        # Performance optimizations
        if dataset_size > 5000:
            config.batch_size = 64
            config.num_workers = 8
        else:
            config.batch_size = 32
            config.num_workers = 4
        
        return config

# =============================================================================
# CONFIGURATION TEMPLATES
# =============================================================================

def create_seo_classification_config() -> Dict[str, Any]:
    """Create configuration template for SEO classification tasks."""
    return {
        "seo_metrics": {
            "task_types": ["classification"],
            "enable_cross_validation": True,
            "cv_folds": 5,
            "enable_bootstrap": True,
            "bootstrap_samples": 1000
        },
        "classification": {
            "default_average": "weighted",
            "enable_roc_auc": True,
            "enable_confusion_matrix": True
        },
        "visualization": {
            "create_visualizations": True,
            "save_format": "png"
        }
    }

def create_seo_ranking_config() -> Dict[str, Any]:
    """Create configuration template for SEO ranking tasks."""
    return {
        "seo_metrics": {
            "task_types": ["ranking"],
            "ranking": {
                "ndcg_k_values": [1, 3, 5, 10, 20],
                "map_k_values": [1, 3, 5, 10, 20],
                "apply_bias_correction": True
            }
        },
        "visualization": {
            "create_visualizations": True,
            "save_format": "png"
        }
    }

def create_seo_multitask_config() -> Dict[str, Any]:
    """Create configuration template for SEO multitask evaluation."""
    return {
        "seo_metrics": {
            "task_types": ["classification", "ranking", "regression"],
            "enable_cross_validation": True,
            "cv_folds": 5
        },
        "classification": {
            "default_average": "weighted",
            "enable_roc_auc": True
        },
        "regression": {
            "enable_mape": True,
            "enable_smape": True
        },
        "visualization": {
            "create_visualizations": True,
            "figure_size": [20, 15]
        }
    }

# =============================================================================
# CONFIGURATION EXAMPLES
# =============================================================================

def demonstrate_configuration_management():
    """Demonstrate configuration management functionality."""
    print("🔧 SEO Evaluation Configuration Management")
    print("=" * 50)
    
    # Create config manager
    config_manager = SEOEvaluationConfigManager()
    
    # Create default config
    default_config = config_manager.create_default_config()
    print("✅ Default configuration created")
    
    # Validate config
    issues = config_manager.validate_config(default_config)
    if issues:
        print(f"⚠️ Configuration issues found: {issues}")
    else:
        print("✅ Configuration validation passed")
    
    # Create optimized configs
    classification_config = config_manager.get_optimized_config("classification", 5000)
    ranking_config = config_manager.get_optimized_config("ranking", 10000)
    
    print(f"✅ Optimized classification config created")
    print(f"✅ Optimized ranking config created")
    
    # Save configs
    config_manager.save_config_to_yaml(default_config, "seo_evaluation_default.yaml")
    config_manager.save_config_to_yaml(classification_config, "seo_evaluation_classification.yaml")
    config_manager.save_config_to_yaml(ranking_config, "seo_evaluation_ranking.yaml")
    
    print("✅ Configuration files saved")
    
    # Create template configs
    templates = {
        "classification": create_seo_classification_config(),
        "ranking": create_seo_ranking_config(),
        "multitask": create_seo_multitask_config()
    }
    
    for name, template in templates.items():
        with open(f"seo_evaluation_{name}_template.yaml", 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        print(f"✅ {name} template saved")
    
    print("\n🎯 Configuration management demonstration completed!")

if __name__ == "__main__":
    # Run demonstration
    demonstrate_configuration_management()

