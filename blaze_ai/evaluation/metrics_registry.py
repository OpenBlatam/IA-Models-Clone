"""
Evaluation metrics registry for Blaze AI.

This module provides a unified interface for accessing and managing
all evaluation metrics across different AI tasks.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .classification import evaluate_classification
from .text_generation import evaluate_text_generation, evaluate_text_generation_batch
from .image_generation import evaluate_diffusion_model, evaluate_image_generation_batch
from .seo_optimization import evaluate_seo_optimization
from .brand_voice import evaluate_brand_voice


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    task_type: str
    model_name: str
    timestamp: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "task_type": self.task_type,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "recommendations": self.recommendations
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save results to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class EvaluationMetricsRegistry:
    """
    Central registry for all evaluation metrics.
    
    Provides a unified interface for evaluating different types of AI models
    and tasks, with automatic result storage and comparison capabilities.
    """
    
    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else Path("evaluation_results")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_history: List[EvaluationResult] = []
        self.logger = logging.getLogger(__name__)
        
        # Task type mappings
        self.task_evaluators = {
            "classification": self._evaluate_classification,
            "text_generation": self._evaluate_text_generation,
            "image_generation": self._evaluate_image_generation,
            "seo_optimization": self._evaluate_seo_optimization,
            "brand_voice": self._evaluate_brand_voice
        }
    
    def evaluate(self, task_type: str, model, **kwargs) -> EvaluationResult:
        """
        Evaluate a model for a specific task type.
        
        Args:
            task_type: Type of task (classification, text_generation, etc.)
            model: The model to evaluate
            **kwargs: Additional arguments for the specific evaluator
        
        Returns:
            EvaluationResult containing all metrics and recommendations
        """
        if task_type not in self.task_evaluators:
            raise ValueError(f"Unknown task type: {task_type}. Available: {list(self.task_evaluators.keys())}")
        
        try:
            evaluator = self.task_evaluators[task_type]
            result = evaluator(model, **kwargs)
            
            # Store result
            self.results_history.append(result)
            
            # Auto-save if storage directory is configured
            if self.storage_dir:
                self._auto_save_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating {task_type}: {e}")
            raise
    
    def _evaluate_classification(self, model, **kwargs) -> EvaluationResult:
        """Evaluate classification models."""
        data_loader = kwargs.get("data_loader")
        device = kwargs.get("device", "cuda")
        model_name = kwargs.get("model_name", "unknown_classification_model")
        
        metrics = evaluate_classification(model, data_loader, device)
        
        return EvaluationResult(
            task_type="classification",
            model_name=model_name,
            timestamp=time.time(),
            metrics=metrics,
            metadata={"device": device, "data_loader_size": len(data_loader) if data_loader else 0}
        )
    
    def _evaluate_text_generation(self, model, **kwargs) -> EvaluationResult:
        """Evaluate text generation models."""
        data_loader = kwargs.get("data_loader")
        device = kwargs.get("device", "cuda")
        model_name = kwargs.get("model_name", "unknown_text_generation_model")
        references = kwargs.get("references")
        candidates = kwargs.get("candidates")
        
        metrics = evaluate_text_generation(
            model, data_loader, device, references, candidates
        )
        
        # Generate recommendations based on metrics
        recommendations = self._generate_text_generation_recommendations(metrics)
        
        return EvaluationResult(
            task_type="text_generation",
            model_name=model_name,
            timestamp=time.time(),
            metrics=metrics,
            metadata={
                "device": device,
                "data_loader_size": len(data_loader) if data_loader else 0,
                "has_references": references is not None,
                "has_candidates": candidates is not None
            },
            recommendations=recommendations
        )
    
    def _evaluate_image_generation(self, model, **kwargs) -> EvaluationResult:
        """Evaluate image generation models."""
        data_loader = kwargs.get("data_loader")
        device = kwargs.get("device", "cuda")
        model_name = kwargs.get("model_name", "unknown_image_generation_model")
        real_images = kwargs.get("real_images")
        generated_images = kwargs.get("generated_images")
        text_prompts = kwargs.get("text_prompts")
        
        metrics = evaluate_diffusion_model(
            model, data_loader, device, real_images, generated_images, text_prompts
        )
        
        # Generate recommendations based on metrics
        recommendations = self._generate_image_generation_recommendations(metrics)
        
        return EvaluationResult(
            task_type="image_generation",
            model_name=model_name,
            timestamp=time.time(),
            metrics=metrics,
            metadata={
                "device": device,
                "data_loader_size": len(data_loader) if data_loader else 0,
                "has_real_images": real_images is not None,
                "has_generated_images": generated_images is not None,
                "has_text_prompts": text_prompts is not None
            },
            recommendations=recommendations
        )
    
    def _evaluate_seo_optimization(self, model, **kwargs) -> EvaluationResult:
        """Evaluate SEO optimization."""
        text = kwargs.get("text", "")
        target_keywords = kwargs.get("target_keywords", [])
        url = kwargs.get("url")
        model_name = kwargs.get("model_name", "seo_optimization_analysis")
        
        evaluation = evaluate_seo_optimization(text, target_keywords, url)
        
        # Extract metrics from evaluation
        metrics = {
            "overall_score": evaluation["quality_score"]["overall_score"],
            "grade": evaluation["quality_score"]["grade"],
            "readability_score": evaluation["quality_score"]["readability_score"],
            "length_score": evaluation["quality_score"]["length_score"],
            "keyword_score": evaluation["quality_score"]["keyword_score"],
            "structure_score": evaluation["quality_score"]["structure_score"],
            "technical_score": evaluation["quality_score"]["technical_score"]
        }
        
        return EvaluationResult(
            task_type="seo_optimization",
            model_name=model_name,
            timestamp=time.time(),
            metrics=metrics,
            metadata={
                "text_length": len(text),
                "target_keywords": target_keywords,
                "url": url
            },
            recommendations=evaluation.get("recommendations", [])
        )
    
    def _evaluate_brand_voice(self, model, **kwargs) -> EvaluationResult:
        """Evaluate brand voice analysis."""
        texts = kwargs.get("texts", [])
        brand_guidelines = kwargs.get("brand_guidelines", {})
        model_name = kwargs.get("model_name", "brand_voice_analysis")
        
        evaluation = evaluate_brand_voice(texts, brand_guidelines)
        
        # Extract metrics from evaluation
        metrics = {
            "overall_brand_alignment": evaluation["brand_alignment"]["overall_brand_alignment"],
            "brand_alignment_grade": evaluation["brand_alignment"]["brand_alignment_grade"],
            "tone_consistency": evaluation["tone_consistency"]["overall_tone_consistency"],
            "vocabulary_consistency": evaluation["vocabulary_consistency"]["brand_term_consistency"],
            "sentiment_consistency": evaluation["sentiment_consistency"]["sentiment_consistency"]
        }
        
        return EvaluationResult(
            task_type="brand_voice",
            model_name=model_name,
            timestamp=time.time(),
            metrics=metrics,
            metadata={
                "num_texts": len(texts),
                "has_brand_guidelines": bool(brand_guidelines)
            },
            recommendations=evaluation.get("recommendations", [])
        )
    
    def _generate_text_generation_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations for text generation based on metrics."""
        recommendations = []
        
        # Perplexity recommendations
        if "perplexity" in metrics:
            perplexity = metrics["perplexity"]
            if perplexity > 100:
                recommendations.append("High perplexity detected - consider fine-tuning on domain-specific data")
            elif perplexity > 50:
                recommendations.append("Moderate perplexity - model could benefit from additional training")
        
        # BLEU score recommendations
        if "bleu" in metrics:
            bleu = metrics["bleu"]
            if bleu < 0.1:
                recommendations.append("Very low BLEU score - generated text may not match reference well")
            elif bleu < 0.3:
                recommendations.append("Low BLEU score - consider improving text generation quality")
        
        # BERTScore recommendations
        if "bert_score" in metrics:
            bert_score = metrics["bert_score"]
            if bert_score < 0.7:
                recommendations.append("Low BERTScore - generated text may lack semantic similarity to reference")
        
        # Content quality recommendations
        if "avg_repetition_score" in metrics:
            repetition = metrics["avg_repetition_score"]
            if repetition > 0.3:
                recommendations.append("High repetition detected - consider diversity-promoting techniques")
        
        return recommendations
    
    def _generate_image_generation_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations for image generation based on metrics."""
        recommendations = []
        
        # FID score recommendations
        if "fid_score" in metrics:
            fid = metrics["fid_score"]
            if fid > 100:
                recommendations.append("High FID score - generated images may not match real image distribution")
            elif fid > 50:
                recommendations.append("Moderate FID score - consider improving image generation quality")
        
        # Inception Score recommendations
        if "inception_score" in metrics:
            is_score = metrics["inception_score"]
            if is_score < 3.0:
                recommendations.append("Low Inception Score - generated images may lack diversity and quality")
            elif is_score < 5.0:
                recommendations.append("Moderate Inception Score - room for improvement in image quality")
        
        # LPIPS recommendations
        if "lpips_score" in metrics:
            lpips = metrics["lpips_score"]
            if lpips > 0.5:
                recommendations.append("High LPIPS score - generated images may differ significantly from references")
        
        # Image quality recommendations
        if "avg_sharpness" in metrics:
            sharpness = metrics["avg_sharpness"]
            if sharpness < 0.1:
                recommendations.append("Low sharpness detected - consider improving image generation resolution")
        
        return recommendations
    
    def _auto_save_result(self, result: EvaluationResult) -> None:
        """Automatically save evaluation result to file."""
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(result.timestamp))
            filename = f"{result.task_type}_{result.model_name}_{timestamp_str}.json"
            filepath = self.storage_dir / filename
            
            result.save_to_file(filepath)
            self.logger.info(f"Auto-saved evaluation result to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to auto-save evaluation result: {e}")
    
    def get_results_by_task(self, task_type: str) -> List[EvaluationResult]:
        """Get all results for a specific task type."""
        return [r for r in self.results_history if r.task_type == task_type]
    
    def get_results_by_model(self, model_name: str) -> List[EvaluationResult]:
        """Get all results for a specific model."""
        return [r for r in self.results_history if r.model_name == model_name]
    
    def get_latest_result(self, task_type: str, model_name: str) -> Optional[EvaluationResult]:
        """Get the latest evaluation result for a specific task and model."""
        results = [r for r in self.results_history 
                  if r.task_type == task_type and r.model_name == model_name]
        
        if results:
            return max(results, key=lambda x: x.timestamp)
        return None
    
    def compare_models(self, task_type: str, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple models for a specific task."""
        comparison = {
            "task_type": task_type,
            "models": {},
            "summary": {}
        }
        
        for model_name in model_names:
            latest_result = self.get_latest_result(task_type, model_name)
            if latest_result:
                comparison["models"][model_name] = latest_result.to_dict()
        
        # Generate summary statistics
        if comparison["models"]:
            metrics_to_compare = set()
            for model_data in comparison["models"].values():
                metrics_to_compare.update(model_data["metrics"].keys())
            
            for metric in metrics_to_compare:
                values = []
                for model_data in comparison["models"].values():
                    if metric in model_data["metrics"]:
                        values.append(model_data["metrics"][metric])
                
                if values:
                    comparison["summary"][metric] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "best_model": min(comparison["models"].keys(), 
                                        key=lambda x: comparison["models"][x]["metrics"].get(metric, float('inf')))
                    }
        
        return comparison
    
    def export_results(self, filepath: Union[str, Path], 
                      task_types: Optional[List[str]] = None,
                      model_names: Optional[List[str]] = None) -> None:
        """Export evaluation results to a JSON file."""
        filepath = Path(filepath)
        
        # Filter results if specified
        filtered_results = self.results_history
        if task_types:
            filtered_results = [r for r in filtered_results if r.task_type in task_types]
        if model_names:
            filtered_results = [r for r in filtered_results if r.model_name in model_names]
        
        # Convert to dictionary format
        export_data = {
            "export_timestamp": time.time(),
            "total_results": len(filtered_results),
            "results": [r.to_dict() for r in filtered_results]
        }
        
        # Save to file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(filtered_results)} results to {filepath}")
    
    def clear_history(self) -> None:
        """Clear the results history."""
        self.results_history.clear()
        self.logger.info("Cleared evaluation results history")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored evaluation results."""
        if not self.results_history:
            return {"total_results": 0}
        
        stats = {
            "total_results": len(self.results_history),
            "task_type_distribution": {},
            "model_distribution": {},
            "time_range": {
                "earliest": min(r.timestamp for r in self.results_history),
                "latest": max(r.timestamp for r in self.results_history)
            }
        }
        
        # Count by task type
        for result in self.results_history:
            stats["task_type_distribution"][result.task_type] = \
                stats["task_type_distribution"].get(result.task_type, 0) + 1
        
        # Count by model
        for result in self.results_history:
            stats["model_distribution"][result.model_name] = \
                stats["model_distribution"].get(result.model_name, 0) + 1
        
        return stats


# Global registry instance
_global_registry: Optional[EvaluationMetricsRegistry] = None

def get_evaluation_registry(storage_dir: Optional[Union[str, Path]] = None) -> EvaluationMetricsRegistry:
    """Get the global evaluation metrics registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = EvaluationMetricsRegistry(storage_dir)
    return _global_registry


def evaluate_model(task_type: str, model, **kwargs) -> EvaluationResult:
    """Convenience function to evaluate a model using the global registry."""
    registry = get_evaluation_registry()
    return registry.evaluate(task_type, model, **kwargs)
