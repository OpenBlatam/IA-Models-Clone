"""
Advanced Upscaling Delegator
=============================

Automatic delegation system for AdvancedUpscaling methods.
This module provides a mapping system to automatically delegate methods
to specialized modules, reducing boilerplate code.
"""

from typing import Dict, Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MethodDelegator:
    """Automatic method delegation system."""
    
    # Mapping of method names to module attributes
    DELEGATION_MAP: Dict[str, tuple] = {
        # Analysis methods
        "analyze_image_characteristics": ("analysis_methods", "analyze_image_characteristics"),
        "compare_methods": ("analysis_methods", "compare_methods"),
        "compare_all_methods_comprehensive": ("analysis_methods", "compare_all_methods_comprehensive"),
        "get_processing_recommendations": ("analysis_methods", "get_processing_recommendations"),
        "get_upscaling_recommendations_advanced": ("analysis_methods", "get_upscaling_recommendations_advanced"),
        "get_optimal_upscaling_strategy": ("analysis_methods", "get_optimal_upscaling_strategy"),
        "export_comparison_report": ("analysis_methods", "export_comparison_report"),
        
        # Pipeline methods
        "upscale_with_pipeline": ("pipeline_methods", "upscale_with_pipeline"),
        "create_custom_upscaling_pipeline": ("pipeline_methods", "create_custom_upscaling_pipeline"),
        "list_custom_pipelines": ("pipeline_methods", "list_custom_pipelines"),
        "get_pipeline_info": ("pipeline_methods", "get_pipeline_info"),
        "create_workflow_preset": ("pipeline_methods", "create_workflow_preset"),
        "upscale_with_workflow": ("pipeline_methods", "upscale_with_workflow"),
        "list_workflows": ("pipeline_methods", "list_workflows"),
        "get_workflow_info": ("pipeline_methods", "get_workflow_info"),
        "export_upscaling_config": ("pipeline_methods", "export_upscaling_config"),
        "load_and_apply_config": ("pipeline_methods", "load_and_apply_config"),
        
        # Ensemble methods
        "upscale_with_ensemble": ("ensemble_methods", "upscale_with_ensemble"),
        "upscale_with_multi_scale_ensemble": ("ensemble_methods", "upscale_with_multi_scale_ensemble"),
        "upscale_with_intelligent_fusion": ("ensemble_methods", "upscale_with_intelligent_fusion"),
        "upscale_with_ensemble_learning": ("ensemble_methods", "upscale_with_ensemble_learning"),
        
        # Adaptive methods
        "upscale_with_progressive_enhancement": ("adaptive_methods", "upscale_with_progressive_enhancement"),
        "upscale_with_adaptive_regions": ("adaptive_methods", "upscale_with_adaptive_regions"),
        "upscale_with_adaptive_quality_loop": ("adaptive_methods", "upscale_with_adaptive_quality_loop"),
        "upscale_with_progressive_quality": ("adaptive_methods", "upscale_with_progressive_quality"),
        "upscale_with_region_adaptive_processing": ("adaptive_methods", "upscale_with_region_adaptive_processing"),
        "upscale_with_adaptive_method_selection": ("adaptive_methods", "upscale_with_adaptive_method_selection"),
        
        # Benchmark methods
        "benchmark_all_methods": ("benchmark_methods", "benchmark_all_methods"),
        "get_performance_benchmark": ("benchmark_methods", "get_performance_benchmark"),
        "profile_upscale": ("benchmark_methods", "profile_upscale"),
        
        # Enhancement methods
        "upscale_with_ai_guided_enhancement": ("enhancement_methods", "upscale_with_ai_guided_enhancement"),
        "upscale_with_quality_assurance": ("enhancement_methods", "upscale_with_quality_assurance"),
        "upscale_with_multi_pass_processing": ("enhancement_methods", "upscale_with_multi_pass_processing"),
        "upscale_with_advanced_processing": ("enhancement_methods", "upscale_with_advanced_processing"),
        
        # Meta-learning methods
        "upscale_with_meta_learning": ("meta_methods", "upscale_with_meta_learning"),
        "upscale_with_neural_style_transfer": ("meta_methods", "upscale_with_neural_style_transfer"),
        "upscale_with_attention_mechanism": ("meta_methods", "upscale_with_attention_mechanism"),
        "upscale_with_gradient_boosting": ("meta_methods", "upscale_with_gradient_boosting"),
        
        # Batch methods
        "batch_upscale_optimized": ("batch_methods", "batch_upscale_optimized"),
        
        # ML methods
        "upscale_with_ml_enhancement": ("ml_methods", "upscale_with_ml_enhancement"),
        "upscale_with_deep_learning": ("ml_methods", "upscale_with_deep_learning"),
        "upscale_with_attention_fusion": ("ml_methods", "upscale_with_attention_fusion"),
        "upscale_with_perceptual_loss": ("ml_methods", "upscale_with_perceptual_loss"),
        "upscale_with_perceptual_optimization": ("ml_methods", "upscale_with_perceptual_optimization"),
    }
    
    @classmethod
    def get_delegation(cls, method_name: str) -> Optional[tuple]:
        """Get delegation info for a method."""
        return cls.DELEGATION_MAP.get(method_name)
    
    @classmethod
    def delegate_method(cls, instance: Any, method_name: str, *args, **kwargs):
        """Delegate a method call to the appropriate module."""
        delegation = cls.get_delegation(method_name)
        if delegation:
            module_attr, method_attr = delegation
            module = getattr(instance, module_attr, None)
            if module:
                method = getattr(module, method_attr, None)
                if method:
                    return method(*args, **kwargs)
                else:
                    logger.warning(f"Method {method_attr} not found in {module_attr}")
            else:
                logger.warning(f"Module {module_attr} not found in instance")
        
        raise AttributeError(f"Method {method_name} not found and no delegation available")


