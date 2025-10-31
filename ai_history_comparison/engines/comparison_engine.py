"""
Comparison Engine

This module provides content and model comparison capabilities including
similarity analysis, quality differences, and performance comparisons.
"""

import hashlib
import difflib
from typing import Dict, List, Any, Optional
import logging

from ..core.base import BaseEngine
from ..core.config import SystemConfig
from ..core.interfaces import IComparisonEngine
from ..core.exceptions import ComparisonError, ValidationError

logger = logging.getLogger(__name__)


class ComparisonEngine(BaseEngine[Dict[str, Any]], IComparisonEngine):
    """Advanced comparison engine for content and model comparisons"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self._capabilities = [
            "content_similarity",
            "model_comparison",
            "quality_analysis",
            "performance_comparison",
            "similarity_search"
        ]
    
    async def _process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process comparison data"""
        try:
            comparison_type = kwargs.get("comparison_type", "content_similarity")
            
            if comparison_type == "content_similarity":
                content1 = data.get("content1", "")
                content2 = data.get("content2", "")
                return await self.compare_content(content1, content2)
            elif comparison_type == "model_comparison":
                model1_results = data.get("model1_results", {})
                model2_results = data.get("model2_results", {})
                return await self.compare_models(model1_results, model2_results)
            else:
                raise ValidationError(f"Unknown comparison type: {comparison_type}")
                
        except Exception as e:
            logger.error(f"Comparison processing failed: {e}")
            raise ComparisonError(f"Comparison processing failed: {str(e)}", comparison_type="processing")
    
    async def compare_content(self, content1: str, content2: str) -> Dict[str, Any]:
        """Compare two pieces of content"""
        try:
            if not content1 or not content2:
                raise ValidationError("Both content pieces must be provided")
            
            # Calculate similarity score
            similarity_score = self._calculate_similarity(content1, content2)
            
            # Analyze differences
            differences = self._analyze_differences(content1, content2)
            
            # Calculate quality metrics for both contents
            quality1 = await self._calculate_content_quality(content1)
            quality2 = await self._calculate_content_quality(content2)
            
            # Determine quality difference
            quality_difference = self._calculate_quality_difference(quality1, quality2)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(quality1, quality2, differences)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(similarity_score, quality_difference)
            
            return {
                "similarity_score": similarity_score,
                "quality_difference": quality_difference,
                "differences": differences,
                "content1_quality": quality1,
                "content2_quality": quality2,
                "recommendations": recommendations,
                "confidence_score": confidence_score,
                "comparison_metadata": {
                    "content1_length": len(content1),
                    "content2_length": len(content2),
                    "content1_hash": hashlib.md5(content1.encode()).hexdigest(),
                    "content2_hash": hashlib.md5(content2.encode()).hexdigest()
                }
            }
            
        except Exception as e:
            logger.error(f"Content comparison failed: {e}")
            raise ComparisonError(f"Content comparison failed: {str(e)}", comparison_type="content")
    
    async def compare_models(self, model1_results: Dict[str, Any], model2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two model results"""
        try:
            if not model1_results or not model2_results:
                raise ValidationError("Both model results must be provided")
            
            # Extract metrics for comparison
            metrics = self._extract_comparable_metrics(model1_results, model2_results)
            
            # Calculate performance differences
            performance_differences = self._calculate_performance_differences(metrics)
            
            # Determine winner
            winner, confidence = self._determine_winner(performance_differences)
            
            # Generate detailed comparison
            detailed_comparison = self._generate_detailed_comparison(model1_results, model2_results, metrics)
            
            return {
                "winner": winner,
                "confidence": confidence,
                "performance_differences": performance_differences,
                "detailed_comparison": detailed_comparison,
                "metrics_compared": list(metrics.keys()),
                "model1_name": model1_results.get("model_name", "Model 1"),
                "model2_name": model2_results.get("model_name", "Model 2")
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            raise ComparisonError(f"Model comparison failed: {str(e)}", comparison_type="model")
    
    async def find_similar_content(self, content: str, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar content pieces"""
        try:
            if not content:
                raise ValidationError("Content must be provided")
            
            # In a real implementation, this would search a database
            # For now, return a placeholder response
            similar_content = [
                {
                    "content_id": "similar_1",
                    "similarity_score": 0.85,
                    "content_preview": content[:100] + "...",
                    "metadata": {"source": "database", "created_at": "2024-01-01"}
                },
                {
                    "content_id": "similar_2", 
                    "similarity_score": 0.82,
                    "content_preview": content[:100] + "...",
                    "metadata": {"source": "database", "created_at": "2024-01-02"}
                }
            ]
            
            # Filter by threshold
            filtered_results = [item for item in similar_content if item["similarity_score"] >= threshold]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Similar content search failed: {e}")
            raise ComparisonError(f"Similar content search failed: {str(e)}", comparison_type="similarity")
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity score between two content pieces"""
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, content1, content2)
        return matcher.ratio()
    
    def _analyze_differences(self, content1: str, content2: str) -> Dict[str, Any]:
        """Analyze differences between two content pieces"""
        # Calculate basic differences
        length_diff = abs(len(content1) - len(content2))
        word_diff = abs(len(content1.split()) - len(content2.split()))
        
        # Find common and unique words
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        common_words = words1.intersection(words2)
        unique_to_1 = words1 - words2
        unique_to_2 = words2 - words1
        
        return {
            "length_difference": length_diff,
            "word_count_difference": word_diff,
            "common_words": len(common_words),
            "unique_to_content1": len(unique_to_1),
            "unique_to_content2": len(unique_to_2),
            "word_overlap_ratio": len(common_words) / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0
        }
    
    async def _calculate_content_quality(self, content: str) -> Dict[str, float]:
        """Calculate quality metrics for content"""
        words = content.split()
        sentences = content.split('.')
        
        return {
            "readability": min(1.0, max(0.0, 1.0 - (len(words) / len(sentences)) / 20.0)) if sentences else 0.5,
            "coherence": 0.7,  # Placeholder
            "relevance": 0.8,  # Placeholder
            "accuracy": 0.9,   # Placeholder
            "completeness": min(1.0, len(content) / 1000.0)
        }
    
    def _calculate_quality_difference(self, quality1: Dict[str, float], quality2: Dict[str, float]) -> Dict[str, Any]:
        """Calculate quality differences between two contents"""
        differences = {}
        for metric in quality1.keys():
            if metric in quality2:
                diff = quality2[metric] - quality1[metric]
                differences[metric] = {
                    "difference": diff,
                    "percentage_change": (diff / quality1[metric]) * 100 if quality1[metric] > 0 else 0,
                    "better_content": "content2" if diff > 0 else "content1" if diff < 0 else "equal"
                }
        
        return differences
    
    def _generate_recommendations(self, quality1: Dict[str, float], quality2: Dict[str, float], 
                                differences: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison"""
        recommendations = []
        
        for metric, diff_info in differences.items():
            if abs(diff_info["difference"]) > 0.1:  # Significant difference
                if diff_info["better_content"] == "content2":
                    recommendations.append(f"Content 2 has better {metric}. Consider adopting its approach.")
                elif diff_info["better_content"] == "content1":
                    recommendations.append(f"Content 1 has better {metric}. Consider adopting its approach.")
        
        if not recommendations:
            recommendations.append("Both contents are of similar quality. No specific recommendations.")
        
        return recommendations
    
    def _calculate_confidence_score(self, similarity_score: float, quality_difference: Dict[str, Any]) -> float:
        """Calculate confidence score for the comparison"""
        # Base confidence on similarity score
        base_confidence = similarity_score
        
        # Adjust based on quality differences
        quality_variance = sum(abs(diff["difference"]) for diff in quality_difference.values())
        quality_factor = max(0.5, 1.0 - (quality_variance / len(quality_difference)))
        
        return min(1.0, base_confidence * quality_factor)
    
    def _extract_comparable_metrics(self, model1_results: Dict[str, Any], model2_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract comparable metrics from model results"""
        metrics = {}
        
        # Common metrics to compare
        common_metrics = ["accuracy", "precision", "recall", "f1_score", "response_time", "quality_score"]
        
        for metric in common_metrics:
            if metric in model1_results and metric in model2_results:
                try:
                    metrics[metric] = {
                        "model1": float(model1_results[metric]),
                        "model2": float(model2_results[metric])
                    }
                except (ValueError, TypeError):
                    continue
        
        return metrics
    
    def _calculate_performance_differences(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate performance differences between models"""
        differences = {}
        
        for metric, values in metrics.items():
            model1_value = values["model1"]
            model2_value = values["model2"]
            
            diff = model2_value - model1_value
            percentage_diff = (diff / model1_value) * 100 if model1_value > 0 else 0
            
            differences[metric] = {
                "difference": diff,
                "percentage_difference": percentage_diff,
                "better_model": "model2" if diff > 0 else "model1" if diff < 0 else "equal"
            }
        
        return differences
    
    def _determine_winner(self, performance_differences: Dict[str, Any]) -> tuple[Optional[str], float]:
        """Determine the winning model and confidence"""
        model1_wins = 0
        model2_wins = 0
        total_metrics = len(performance_differences)
        
        for metric, diff_info in performance_differences.items():
            if diff_info["better_model"] == "model1":
                model1_wins += 1
            elif diff_info["better_model"] == "model2":
                model2_wins += 1
        
        if model1_wins > model2_wins:
            winner = "model1"
            confidence = model1_wins / total_metrics
        elif model2_wins > model1_wins:
            winner = "model2"
            confidence = model2_wins / total_metrics
        else:
            winner = None
            confidence = 0.5
        
        return winner, confidence
    
    def _generate_detailed_comparison(self, model1_results: Dict[str, Any], model2_results: Dict[str, Any], 
                                    metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate detailed comparison report"""
        return {
            "model1_results": model1_results,
            "model2_results": model2_results,
            "metrics_comparison": metrics,
            "summary": {
                "total_metrics_compared": len(metrics),
                "model1_advantages": sum(1 for diff in metrics.values() if diff["model1"] > diff["model2"]),
                "model2_advantages": sum(1 for diff in metrics.values() if diff["model2"] > diff["model1"])
            }
        }
    
    def get_capabilities(self) -> List[str]:
        """Get list of engine capabilities"""
        return self._capabilities





















