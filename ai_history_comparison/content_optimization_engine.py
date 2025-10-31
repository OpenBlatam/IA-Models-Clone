"""
AI History Comparison System - Content Optimization Engine

This module provides advanced content optimization, A/B testing,
and performance enhancement capabilities.
"""

import logging
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict, Counter
import statistics
import re

# Advanced NLP libraries
try:
    import spacy
    from spacy import displacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

from .ai_history_analyzer import AIHistoryAnalyzer, HistoryEntry
from .content_quality_assurance import quality_assurance, QualityStandard

logger = logging.getLogger(__name__)

class OptimizationGoal(Enum):
    """Content optimization goals"""
    READABILITY = "readability"
    ENGAGEMENT = "engagement"
    SEO = "seo"
    CONVERSION = "conversion"
    ACCESSIBILITY = "accessibility"
    CLARITY = "clarity"
    PERSUASION = "persuasion"
    COMPREHENSION = "comprehension"

class OptimizationStrategy(Enum):
    """Optimization strategies"""
    SENTENCE_SHORTENING = "sentence_shortening"
    VOCABULARY_SIMPLIFICATION = "vocabulary_simplification"
    STRUCTURE_REORGANIZATION = "structure_reorganization"
    KEYWORD_OPTIMIZATION = "keyword_optimization"
    TONE_ADJUSTMENT = "tone_adjustment"
    LENGTH_OPTIMIZATION = "length_optimization"
    HEADING_OPTIMIZATION = "heading_optimization"
    CALL_TO_ACTION_ENHANCEMENT = "call_to_action_enhancement"

class ABTestStatus(Enum):
    """A/B test status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class OptimizationSuggestion:
    """Content optimization suggestion"""
    strategy: OptimizationStrategy
    original_text: str
    optimized_text: str
    improvement_score: float
    confidence: float
    explanation: str
    impact_areas: List[str]
    created_at: datetime

@dataclass
class OptimizationResult:
    """Result of content optimization"""
    original_score: float
    optimized_score: float
    improvement_percentage: float
    suggestions: List[OptimizationSuggestion]
    optimization_goals: List[OptimizationGoal]
    confidence: float
    optimized_at: datetime

@dataclass
class ABTestVariant:
    """A/B test variant"""
    variant_id: str
    variant_name: str
    content: str
    traffic_percentage: float
    metrics: Dict[str, float]
    created_at: datetime

@dataclass
class ABTest:
    """A/B test configuration"""
    test_id: str
    test_name: str
    original_variant: ABTestVariant
    test_variants: List[ABTestVariant]
    optimization_goal: OptimizationGoal
    status: ABTestStatus
    start_date: datetime
    end_date: Optional[datetime]
    traffic_allocation: Dict[str, float]
    success_metrics: List[str]
    minimum_sample_size: int
    confidence_level: float

@dataclass
class ABTestResult:
    """A/B test result"""
    test_id: str
    winning_variant: str
    statistical_significance: float
    improvement_percentage: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    test_duration_days: int
    metrics_comparison: Dict[str, Dict[str, float]]
    recommendations: List[str]
    completed_at: datetime

class ContentOptimizationEngine:
    """
    Advanced content optimization and A/B testing engine
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize content optimization engine"""
        self.config = config or {}
        self.nlp = None
        self.optimization_history: List[OptimizationResult] = []
        self.ab_tests: Dict[str, ABTest] = {}
        self.ab_test_results: Dict[str, ABTestResult] = {}
        
        # Initialize spaCy if available
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded for content optimization")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Optimization strategies and their implementations
        self.optimization_strategies = {
            OptimizationStrategy.SENTENCE_SHORTENING: self._optimize_sentence_length,
            OptimizationStrategy.VOCABULARY_SIMPLIFICATION: self._optimize_vocabulary,
            OptimizationStrategy.STRUCTURE_REORGANIZATION: self._optimize_structure,
            OptimizationStrategy.KEYWORD_OPTIMIZATION: self._optimize_keywords,
            OptimizationStrategy.TONE_ADJUSTMENT: self._optimize_tone,
            OptimizationStrategy.LENGTH_OPTIMIZATION: self._optimize_length,
            OptimizationStrategy.HEADING_OPTIMIZATION: self._optimize_headings,
            OptimizationStrategy.CALL_TO_ACTION_ENHANCEMENT: self._optimize_cta
        }
        
        logger.info("Content Optimization Engine initialized")

    def optimize_content(self, content: str, 
                        optimization_goals: List[OptimizationGoal],
                        target_improvement: float = 20.0) -> OptimizationResult:
        """Optimize content for specified goals"""
        try:
            # Calculate original score
            original_score = self._calculate_content_score(content, optimization_goals)
            
            # Generate optimization suggestions
            suggestions = []
            for goal in optimization_goals:
                goal_suggestions = self._generate_optimization_suggestions(content, goal)
                suggestions.extend(goal_suggestions)
            
            # Apply optimizations
            optimized_content = self._apply_optimizations(content, suggestions)
            
            # Calculate optimized score
            optimized_score = self._calculate_content_score(optimized_content, optimization_goals)
            
            # Calculate improvement
            improvement_percentage = ((optimized_score - original_score) / original_score) * 100 if original_score > 0 else 0
            
            # Calculate confidence
            confidence = self._calculate_optimization_confidence(suggestions, improvement_percentage)
            
            # Create optimization result
            result = OptimizationResult(
                original_score=original_score,
                optimized_score=optimized_score,
                improvement_percentage=improvement_percentage,
                suggestions=suggestions,
                optimization_goals=optimization_goals,
                confidence=confidence,
                optimized_at=datetime.now()
            )
            
            # Store in history
            self.optimization_history.append(result)
            
            logger.info(f"Content optimization completed. Improvement: {improvement_percentage:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            return OptimizationResult(
                original_score=0.0,
                optimized_score=0.0,
                improvement_percentage=0.0,
                suggestions=[],
                optimization_goals=optimization_goals,
                confidence=0.0,
                optimized_at=datetime.now()
            )

    def create_ab_test(self, test_name: str, original_content: str,
                      optimization_goals: List[OptimizationGoal],
                      test_variants: List[str] = None,
                      traffic_allocation: Dict[str, float] = None) -> ABTest:
        """Create an A/B test for content optimization"""
        try:
            test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create original variant
            original_variant = ABTestVariant(
                variant_id="original",
                variant_name="Original",
                content=original_content,
                traffic_percentage=50.0,
                metrics={},
                created_at=datetime.now()
            )
            
            # Create test variants
            if test_variants is None:
                # Generate variants automatically
                test_variants = self._generate_ab_test_variants(original_content, optimization_goals)
            
            variants = [original_variant]
            for i, variant_content in enumerate(test_variants):
                variant = ABTestVariant(
                    variant_id=f"variant_{i+1}",
                    variant_name=f"Variant {i+1}",
                    content=variant_content,
                    traffic_percentage=50.0 / len(test_variants),
                    metrics={},
                    created_at=datetime.now()
                )
                variants.append(variant)
            
            # Set traffic allocation
            if traffic_allocation is None:
                traffic_allocation = {variant.variant_id: 100.0 / len(variants) for variant in variants}
            
            # Create A/B test
            ab_test = ABTest(
                test_id=test_id,
                test_name=test_name,
                original_variant=original_variant,
                test_variants=variants[1:],  # Exclude original from test variants
                optimization_goal=optimization_goals[0] if optimization_goals else OptimizationGoal.READABILITY,
                status=ABTestStatus.DRAFT,
                start_date=datetime.now(),
                end_date=None,
                traffic_allocation=traffic_allocation,
                success_metrics=["engagement_rate", "conversion_rate", "time_on_page"],
                minimum_sample_size=1000,
                confidence_level=0.95
            )
            
            # Store A/B test
            self.ab_tests[test_id] = ab_test
            
            logger.info(f"A/B test created: {test_id}")
            return ab_test
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            raise

    def run_ab_test(self, test_id: str, duration_days: int = 7) -> ABTestResult:
        """Run an A/B test and return results"""
        try:
            if test_id not in self.ab_tests:
                raise ValueError(f"A/B test {test_id} not found")
            
            ab_test = self.ab_tests[test_id]
            
            # Update test status
            ab_test.status = ABTestStatus.RUNNING
            ab_test.end_date = datetime.now() + timedelta(days=duration_days)
            
            # Simulate A/B test results (in real implementation, this would collect actual metrics)
            test_results = self._simulate_ab_test_results(ab_test, duration_days)
            
            # Analyze results
            result = self._analyze_ab_test_results(ab_test, test_results)
            
            # Update test status
            ab_test.status = ABTestStatus.COMPLETED
            
            # Store results
            self.ab_test_results[test_id] = result
            
            logger.info(f"A/B test completed: {test_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error running A/B test: {e}")
            raise

    def get_optimization_recommendations(self, content: str, 
                                       optimization_goals: List[OptimizationGoal]) -> List[Dict[str, Any]]:
        """Get optimization recommendations without applying them"""
        try:
            recommendations = []
            
            for goal in optimization_goals:
                goal_recommendations = self._generate_goal_specific_recommendations(content, goal)
                recommendations.extend(goal_recommendations)
            
            # Sort by potential impact
            recommendations.sort(key=lambda x: x.get("potential_impact", 0), reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            return []

    def analyze_content_performance(self, content: str, 
                                  performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze content performance and suggest optimizations"""
        try:
            # Analyze current performance
            performance_analysis = {
                "current_metrics": performance_metrics,
                "performance_score": self._calculate_performance_score(performance_metrics),
                "strengths": [],
                "weaknesses": [],
                "optimization_opportunities": []
            }
            
            # Identify strengths and weaknesses
            if performance_metrics.get("engagement_rate", 0) > 0.7:
                performance_analysis["strengths"].append("High engagement rate")
            else:
                performance_analysis["weaknesses"].append("Low engagement rate")
                performance_analysis["optimization_opportunities"].append("Improve engagement")
            
            if performance_metrics.get("conversion_rate", 0) > 0.05:
                performance_analysis["strengths"].append("Good conversion rate")
            else:
                performance_analysis["weaknesses"].append("Low conversion rate")
                performance_analysis["optimization_opportunities"].append("Optimize for conversion")
            
            if performance_metrics.get("time_on_page", 0) > 120:
                performance_analysis["strengths"].append("Good time on page")
            else:
                performance_analysis["weaknesses"].append("Low time on page")
                performance_analysis["optimization_opportunities"].append("Improve content engagement")
            
            # Generate specific recommendations
            recommendations = []
            for opportunity in performance_analysis["optimization_opportunities"]:
                if "engagement" in opportunity.lower():
                    recommendations.append({
                        "type": "engagement_optimization",
                        "description": "Improve content engagement",
                        "suggestions": [
                            "Add more interactive elements",
                            "Use storytelling techniques",
                            "Include relevant examples and case studies"
                        ]
                    })
                elif "conversion" in opportunity.lower():
                    recommendations.append({
                        "type": "conversion_optimization",
                        "description": "Optimize for conversions",
                        "suggestions": [
                            "Strengthen call-to-action buttons",
                            "Add social proof elements",
                            "Simplify the conversion process"
                        ]
                    })
            
            performance_analysis["recommendations"] = recommendations
            
            return performance_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing content performance: {e}")
            return {"error": str(e)}

    def _calculate_content_score(self, content: str, goals: List[OptimizationGoal]) -> float:
        """Calculate overall content score based on optimization goals"""
        try:
            scores = []
            
            for goal in goals:
                if goal == OptimizationGoal.READABILITY:
                    score = self._calculate_readability_score(content)
                elif goal == OptimizationGoal.ENGAGEMENT:
                    score = self._calculate_engagement_score(content)
                elif goal == OptimizationGoal.SEO:
                    score = self._calculate_seo_score(content)
                elif goal == OptimizationGoal.CONVERSION:
                    score = self._calculate_conversion_score(content)
                elif goal == OptimizationGoal.ACCESSIBILITY:
                    score = self._calculate_accessibility_score(content)
                elif goal == OptimizationGoal.CLARITY:
                    score = self._calculate_clarity_score(content)
                else:
                    score = 50.0  # Default score
                
                scores.append(score)
            
            return statistics.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating content score: {e}")
            return 0.0

    def _generate_optimization_suggestions(self, content: str, goal: OptimizationGoal) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions for a specific goal"""
        suggestions = []
        
        try:
            if goal == OptimizationGoal.READABILITY:
                suggestions.extend(self._generate_readability_suggestions(content))
            elif goal == OptimizationGoal.ENGAGEMENT:
                suggestions.extend(self._generate_engagement_suggestions(content))
            elif goal == OptimizationGoal.SEO:
                suggestions.extend(self._generate_seo_suggestions(content))
            elif goal == OptimizationGoal.CONVERSION:
                suggestions.extend(self._generate_conversion_suggestions(content))
            elif goal == OptimizationGoal.ACCESSIBILITY:
                suggestions.extend(self._generate_accessibility_suggestions(content))
            elif goal == OptimizationGoal.CLARITY:
                suggestions.extend(self._generate_clarity_suggestions(content))
            
        except Exception as e:
            logger.warning(f"Error generating suggestions for {goal.value}: {e}")
        
        return suggestions

    def _apply_optimizations(self, content: str, suggestions: List[OptimizationSuggestion]) -> str:
        """Apply optimization suggestions to content"""
        try:
            optimized_content = content
            
            # Sort suggestions by improvement score
            suggestions.sort(key=lambda x: x.improvement_score, reverse=True)
            
            # Apply top suggestions
            for suggestion in suggestions[:5]:  # Apply top 5 suggestions
                optimized_content = optimized_content.replace(
                    suggestion.original_text, 
                    suggestion.optimized_text
                )
            
            return optimized_content
            
        except Exception as e:
            logger.warning(f"Error applying optimizations: {e}")
            return content

    def _calculate_optimization_confidence(self, suggestions: List[OptimizationSuggestion], 
                                         improvement_percentage: float) -> float:
        """Calculate confidence in optimization results"""
        try:
            if not suggestions:
                return 0.0
            
            # Base confidence on number of suggestions and improvement
            suggestion_confidence = min(len(suggestions) / 10.0, 1.0)
            improvement_confidence = min(improvement_percentage / 50.0, 1.0)
            
            # Average confidence
            confidence = (suggestion_confidence + improvement_confidence) / 2
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating optimization confidence: {e}")
            return 0.0

    def _generate_ab_test_variants(self, original_content: str, 
                                 goals: List[OptimizationGoal]) -> List[str]:
        """Generate A/B test variants automatically"""
        variants = []
        
        try:
            # Generate variant for each goal
            for goal in goals:
                if goal == OptimizationGoal.READABILITY:
                    variant = self._create_readability_variant(original_content)
                elif goal == OptimizationGoal.ENGAGEMENT:
                    variant = self._create_engagement_variant(original_content)
                elif goal == OptimizationGoal.SEO:
                    variant = self._create_seo_variant(original_content)
                else:
                    continue
                
                if variant and variant != original_content:
                    variants.append(variant)
            
            # Limit to 3 variants maximum
            return variants[:3]
            
        except Exception as e:
            logger.warning(f"Error generating A/B test variants: {e}")
            return []

    def _simulate_ab_test_results(self, ab_test: ABTest, duration_days: int) -> Dict[str, Dict[str, float]]:
        """Simulate A/B test results (for demonstration)"""
        results = {}
        
        try:
            # Simulate results for original variant
            original_metrics = {
                "engagement_rate": random.uniform(0.3, 0.7),
                "conversion_rate": random.uniform(0.02, 0.08),
                "time_on_page": random.uniform(60, 180),
                "bounce_rate": random.uniform(0.2, 0.6)
            }
            results["original"] = original_metrics
            
            # Simulate results for test variants
            for variant in ab_test.test_variants:
                # Add some variation to simulate different performance
                variant_metrics = {}
                for metric, value in original_metrics.items():
                    variation = random.uniform(-0.2, 0.3)  # -20% to +30% variation
                    variant_metrics[metric] = max(0, value * (1 + variation))
                
                results[variant.variant_id] = variant_metrics
            
            return results
            
        except Exception as e:
            logger.warning(f"Error simulating A/B test results: {e}")
            return {}

    def _analyze_ab_test_results(self, ab_test: ABTest, 
                               test_results: Dict[str, Dict[str, float]]) -> ABTestResult:
        """Analyze A/B test results and determine winner"""
        try:
            # Find winning variant based on primary metric
            primary_metric = ab_test.success_metrics[0] if ab_test.success_metrics else "engagement_rate"
            
            best_variant = "original"
            best_score = test_results.get("original", {}).get(primary_metric, 0)
            
            for variant_id, metrics in test_results.items():
                if variant_id != "original":
                    score = metrics.get(primary_metric, 0)
                    if score > best_score:
                        best_score = score
                        best_variant = variant_id
            
            # Calculate improvement percentage
            original_score = test_results.get("original", {}).get(primary_metric, 0)
            improvement_percentage = ((best_score - original_score) / original_score) * 100 if original_score > 0 else 0
            
            # Calculate statistical significance (simplified)
            statistical_significance = min(0.95, 0.5 + (improvement_percentage / 100))
            
            # Generate recommendations
            recommendations = []
            if improvement_percentage > 10:
                recommendations.append(f"Implement {best_variant} as it shows {improvement_percentage:.1f}% improvement")
            elif improvement_percentage > 5:
                recommendations.append(f"Consider implementing {best_variant} for moderate improvement")
            else:
                recommendations.append("No significant improvement detected, consider other optimization strategies")
            
            # Create result
            result = ABTestResult(
                test_id=ab_test.test_id,
                winning_variant=best_variant,
                statistical_significance=statistical_significance,
                improvement_percentage=improvement_percentage,
                confidence_interval=(best_score * 0.9, best_score * 1.1),
                sample_size=ab_test.minimum_sample_size,
                test_duration_days=7,
                metrics_comparison=test_results,
                recommendations=recommendations,
                completed_at=datetime.now()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing A/B test results: {e}")
            raise

    def _generate_goal_specific_recommendations(self, content: str, goal: OptimizationGoal) -> List[Dict[str, Any]]:
        """Generate goal-specific recommendations"""
        recommendations = []
        
        try:
            if goal == OptimizationGoal.READABILITY:
                readability_score = self._calculate_readability_score(content)
                if readability_score < 60:
                    recommendations.append({
                        "type": "readability_improvement",
                        "description": "Improve content readability",
                        "potential_impact": 0.8,
                        "suggestions": [
                            "Shorten long sentences",
                            "Use simpler vocabulary",
                            "Break up complex paragraphs"
                        ]
                    })
            
            elif goal == OptimizationGoal.ENGAGEMENT:
                engagement_score = self._calculate_engagement_score(content)
                if engagement_score < 70:
                    recommendations.append({
                        "type": "engagement_improvement",
                        "description": "Increase content engagement",
                        "potential_impact": 0.7,
                        "suggestions": [
                            "Add compelling headlines",
                            "Include relevant examples",
                            "Use storytelling techniques"
                        ]
                    })
            
            elif goal == OptimizationGoal.SEO:
                seo_score = self._calculate_seo_score(content)
                if seo_score < 60:
                    recommendations.append({
                        "type": "seo_improvement",
                        "description": "Optimize for search engines",
                        "potential_impact": 0.6,
                        "suggestions": [
                            "Add relevant keywords",
                            "Improve meta descriptions",
                            "Optimize heading structure"
                        ]
                    })
            
        except Exception as e:
            logger.warning(f"Error generating goal-specific recommendations: {e}")
        
        return recommendations

    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score"""
        try:
            if HAS_TEXTSTAT:
                return flesch_reading_ease(content)
            else:
                # Simplified calculation
                sentences = content.split('.')
                words = content.split()
                if len(sentences) == 0 or len(words) == 0:
                    return 0.0
                
                avg_sentence_length = len(words) / len(sentences)
                return max(0, 100 - avg_sentence_length * 2)
        except:
            return 50.0

    def _calculate_engagement_score(self, content: str) -> float:
        """Calculate engagement score"""
        try:
            score = 50.0  # Base score
            
            # Check for engagement elements
            if '?' in content:
                score += 10  # Questions increase engagement
            
            if any(word in content.lower() for word in ['you', 'your', 'we', 'our']):
                score += 10  # Personal pronouns increase engagement
            
            if len(content.split()) > 100:
                score += 10  # Substantial content
            
            if any(word in content.lower() for word in ['story', 'example', 'case study']):
                score += 15  # Storytelling elements
            
            return min(100, score)
        except:
            return 50.0

    def _calculate_seo_score(self, content: str) -> float:
        """Calculate SEO score"""
        try:
            score = 50.0  # Base score
            
            # Check for SEO elements
            if len(content) > 300:
                score += 10  # Minimum content length
            
            if '<h1>' in content or '# ' in content:
                score += 10  # Heading structure
            
            if any(word in content.lower() for word in ['meta', 'description', 'keyword']):
                score += 10  # SEO elements
            
            # Check keyword density (simplified)
            words = content.lower().split()
            if len(words) > 0:
                word_freq = Counter(words)
                max_freq = max(word_freq.values())
                density = max_freq / len(words)
                if 0.01 <= density <= 0.03:
                    score += 15  # Good keyword density
            
            return min(100, score)
        except:
            return 50.0

    def _calculate_conversion_score(self, content: str) -> float:
        """Calculate conversion score"""
        try:
            score = 50.0  # Base score
            
            # Check for conversion elements
            cta_words = ['buy', 'download', 'sign up', 'get started', 'learn more', 'contact']
            if any(word in content.lower() for word in cta_words):
                score += 20  # Call-to-action elements
            
            if any(word in content.lower() for word in ['free', 'limited time', 'exclusive']):
                score += 15  # Urgency/scarcity elements
            
            if any(word in content.lower() for word in ['testimonial', 'review', 'rating']):
                score += 15  # Social proof elements
            
            return min(100, score)
        except:
            return 50.0

    def _calculate_accessibility_score(self, content: str) -> float:
        """Calculate accessibility score"""
        try:
            score = 50.0  # Base score
            
            # Check for accessibility elements
            if 'alt=' in content:
                score += 15  # Alt text for images
            
            if re.search(r'<h[1-6]', content):
                score += 15  # Heading structure
            
            if len(content) > 200:
                score += 10  # Substantial content
            
            # Check for simple language
            complex_words = [word for word in content.split() if len(word) > 8]
            if len(complex_words) / len(content.split()) < 0.1:
                score += 10  # Simple language
            
            return min(100, score)
        except:
            return 50.0

    def _calculate_clarity_score(self, content: str) -> float:
        """Calculate clarity score"""
        try:
            score = 50.0  # Base score
            
            # Check for clarity elements
            sentences = content.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            if avg_sentence_length < 20:
                score += 15  # Short sentences
            
            if any(word in content.lower() for word in ['because', 'therefore', 'however', 'although']):
                score += 10  # Logical connectors
            
            if any(word in content.lower() for word in ['example', 'for instance', 'such as']):
                score += 10  # Examples and illustrations
            
            return min(100, score)
        except:
            return 50.0

    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score"""
        try:
            if not metrics:
                return 0.0
            
            # Weight different metrics
            weights = {
                "engagement_rate": 0.3,
                "conversion_rate": 0.4,
                "time_on_page": 0.2,
                "bounce_rate": 0.1
            }
            
            score = 0.0
            total_weight = 0.0
            
            for metric, value in metrics.items():
                if metric in weights:
                    # Normalize values (simplified)
                    if metric == "bounce_rate":
                        normalized_value = max(0, 100 - value * 100)  # Lower bounce rate is better
                    else:
                        normalized_value = min(100, value * 100)
                    
                    score += normalized_value * weights[metric]
                    total_weight += weights[metric]
            
            return score / total_weight if total_weight > 0 else 0.0
        except:
            return 0.0

    # Optimization strategy implementations
    def _optimize_sentence_length(self, content: str) -> str:
        """Optimize sentence length"""
        sentences = content.split('.')
        optimized_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 25:  # Long sentence
                # Split into shorter sentences (simplified)
                mid_point = len(words) // 2
                first_part = ' '.join(words[:mid_point])
                second_part = ' '.join(words[mid_point:])
                optimized_sentences.extend([first_part, second_part])
            else:
                optimized_sentences.append(sentence)
        
        return '. '.join(optimized_sentences)

    def _optimize_vocabulary(self, content: str) -> str:
        """Optimize vocabulary complexity"""
        # Simple vocabulary replacement (in practice, would use a more sophisticated approach)
        replacements = {
            'utilize': 'use',
            'facilitate': 'help',
            'implement': 'do',
            'comprehensive': 'complete',
            'substantial': 'large'
        }
        
        optimized_content = content
        for complex_word, simple_word in replacements.items():
            optimized_content = optimized_content.replace(complex_word, simple_word)
        
        return optimized_content

    def _optimize_structure(self, content: str) -> str:
        """Optimize content structure"""
        # Add headings and improve structure (simplified)
        paragraphs = content.split('\n\n')
        optimized_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            if i == 0 and not paragraph.startswith('#'):
                optimized_paragraphs.append(f"# {paragraph}")
            else:
                optimized_paragraphs.append(paragraph)
        
        return '\n\n'.join(optimized_paragraphs)

    def _optimize_keywords(self, content: str) -> str:
        """Optimize keywords for SEO"""
        # Add relevant keywords (simplified)
        if 'content' in content.lower() and 'optimization' not in content.lower():
            content = content.replace('content', 'content optimization')
        
        return content

    def _optimize_tone(self, content: str) -> str:
        """Optimize tone for engagement"""
        # Make tone more engaging (simplified)
        replacements = {
            'is': 'is',
            'are': 'are',
            'will': 'will'
        }
        
        # Add more engaging language
        if 'you' not in content.lower():
            content = f"You'll find that {content.lower()}"
        
        return content

    def _optimize_length(self, content: str) -> str:
        """Optimize content length"""
        words = content.split()
        if len(words) > 1000:
            # Truncate to optimal length
            return ' '.join(words[:800]) + "..."
        elif len(words) < 200:
            # Add more content
            return content + " This additional content provides more value and context for readers."
        
        return content

    def _optimize_headings(self, content: str) -> str:
        """Optimize headings"""
        # Ensure proper heading structure
        if not content.startswith('#'):
            content = f"# {content}"
        
        return content

    def _optimize_cta(self, content: str) -> str:
        """Optimize call-to-action"""
        # Add call-to-action if missing
        cta_indicators = ['buy', 'download', 'sign up', 'get started']
        if not any(indicator in content.lower() for indicator in cta_indicators):
            content += "\n\nReady to get started? Contact us today!"
        
        return content

    def _generate_readability_suggestions(self, content: str) -> List[OptimizationSuggestion]:
        """Generate readability optimization suggestions"""
        suggestions = []
        
        try:
            sentences = content.split('.')
            for sentence in sentences:
                words = sentence.split()
                if len(words) > 25:
                    # Suggest shortening long sentence
                    shortened = ' '.join(words[:20]) + "..."
                    suggestions.append(OptimizationSuggestion(
                        strategy=OptimizationStrategy.SENTENCE_SHORTENING,
                        original_text=sentence,
                        optimized_text=shortened,
                        improvement_score=0.8,
                        confidence=0.9,
                        explanation="Long sentences reduce readability",
                        impact_areas=["readability", "comprehension"],
                        created_at=datetime.now()
                    ))
        except Exception as e:
            logger.warning(f"Error generating readability suggestions: {e}")
        
        return suggestions

    def _generate_engagement_suggestions(self, content: str) -> List[OptimizationSuggestion]:
        """Generate engagement optimization suggestions"""
        suggestions = []
        
        try:
            if '?' not in content:
                # Add a question to increase engagement
                question = "What do you think about this topic?"
                suggestions.append(OptimizationSuggestion(
                    strategy=OptimizationStrategy.TONE_ADJUSTMENT,
                    original_text=content[-50:],  # Last 50 characters
                    optimized_text=content[-50:] + f" {question}",
                    improvement_score=0.6,
                    confidence=0.7,
                    explanation="Questions increase reader engagement",
                    impact_areas=["engagement", "interaction"],
                    created_at=datetime.now()
                ))
        except Exception as e:
            logger.warning(f"Error generating engagement suggestions: {e}")
        
        return suggestions

    def _generate_seo_suggestions(self, content: str) -> List[OptimizationSuggestion]:
        """Generate SEO optimization suggestions"""
        suggestions = []
        
        try:
            if len(content) < 300:
                # Suggest adding more content
                additional_content = " This additional content provides more comprehensive information and improves SEO performance."
                suggestions.append(OptimizationSuggestion(
                    strategy=OptimizationStrategy.LENGTH_OPTIMIZATION,
                    original_text=content,
                    optimized_text=content + additional_content,
                    improvement_score=0.7,
                    confidence=0.8,
                    explanation="Longer content performs better in search engines",
                    impact_areas=["seo", "search_ranking"],
                    created_at=datetime.now()
                ))
        except Exception as e:
            logger.warning(f"Error generating SEO suggestions: {e}")
        
        return suggestions

    def _generate_conversion_suggestions(self, content: str) -> List[OptimizationSuggestion]:
        """Generate conversion optimization suggestions"""
        suggestions = []
        
        try:
            cta_words = ['buy', 'download', 'sign up', 'get started']
            if not any(word in content.lower() for word in cta_words):
                # Add call-to-action
                cta = "Ready to get started? Contact us today!"
                suggestions.append(OptimizationSuggestion(
                    strategy=OptimizationStrategy.CALL_TO_ACTION_ENHANCEMENT,
                    original_text=content,
                    optimized_text=content + f"\n\n{cta}",
                    improvement_score=0.9,
                    confidence=0.8,
                    explanation="Clear call-to-action improves conversion rates",
                    impact_areas=["conversion", "engagement"],
                    created_at=datetime.now()
                ))
        except Exception as e:
            logger.warning(f"Error generating conversion suggestions: {e}")
        
        return suggestions

    def _generate_accessibility_suggestions(self, content: str) -> List[OptimizationSuggestion]:
        """Generate accessibility optimization suggestions"""
        suggestions = []
        
        try:
            if not re.search(r'<h[1-6]', content):
                # Add heading structure
                suggestions.append(OptimizationSuggestion(
                    strategy=OptimizationStrategy.HEADING_OPTIMIZATION,
                    original_text=content[:100],
                    optimized_text=f"# {content[:100]}",
                    improvement_score=0.8,
                    confidence=0.9,
                    explanation="Proper heading structure improves accessibility",
                    impact_areas=["accessibility", "seo"],
                    created_at=datetime.now()
                ))
        except Exception as e:
            logger.warning(f"Error generating accessibility suggestions: {e}")
        
        return suggestions

    def _generate_clarity_suggestions(self, content: str) -> List[OptimizationSuggestion]:
        """Generate clarity optimization suggestions"""
        suggestions = []
        
        try:
            # Check for complex sentences
            sentences = content.split('.')
            for sentence in sentences:
                if len(sentence.split()) > 20:
                    # Suggest breaking into simpler sentences
                    words = sentence.split()
                    mid_point = len(words) // 2
                    first_part = ' '.join(words[:mid_point])
                    second_part = ' '.join(words[mid_point:])
                    simplified = f"{first_part}. {second_part}"
                    
                    suggestions.append(OptimizationSuggestion(
                        strategy=OptimizationStrategy.SENTENCE_SHORTENING,
                        original_text=sentence,
                        optimized_text=simplified,
                        improvement_score=0.7,
                        confidence=0.8,
                        explanation="Shorter sentences improve clarity",
                        impact_areas=["clarity", "comprehension"],
                        created_at=datetime.now()
                    ))
        except Exception as e:
            logger.warning(f"Error generating clarity suggestions: {e}")
        
        return suggestions

    def _create_readability_variant(self, content: str) -> str:
        """Create a readability-optimized variant"""
        return self._optimize_sentence_length(content)

    def _create_engagement_variant(self, content: str) -> str:
        """Create an engagement-optimized variant"""
        return self._optimize_tone(content)

    def _create_seo_variant(self, content: str) -> str:
        """Create an SEO-optimized variant"""
        return self._optimize_keywords(content)

    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history"""
        return self.optimization_history

    def get_ab_test_summary(self) -> Dict[str, Any]:
        """Get A/B test summary"""
        return {
            "total_tests": len(self.ab_tests),
            "completed_tests": len(self.ab_test_results),
            "active_tests": sum(1 for test in self.ab_tests.values() if test.status == ABTestStatus.RUNNING),
            "average_improvement": statistics.mean([result.improvement_percentage for result in self.ab_test_results.values()]) if self.ab_test_results else 0.0
        }


# Global optimization engine instance
optimization_engine = ContentOptimizationEngine()

# Convenience functions
def optimize_content(content: str, optimization_goals: List[OptimizationGoal], target_improvement: float = 20.0) -> OptimizationResult:
    """Optimize content for specified goals"""
    return optimization_engine.optimize_content(content, optimization_goals, target_improvement)

def create_ab_test(test_name: str, original_content: str, optimization_goals: List[OptimizationGoal]) -> ABTest:
    """Create an A/B test"""
    return optimization_engine.create_ab_test(test_name, original_content, optimization_goals)

def run_ab_test(test_id: str, duration_days: int = 7) -> ABTestResult:
    """Run an A/B test"""
    return optimization_engine.run_ab_test(test_id, duration_days)

def get_optimization_recommendations(content: str, optimization_goals: List[OptimizationGoal]) -> List[Dict[str, Any]]:
    """Get optimization recommendations"""
    return optimization_engine.get_optimization_recommendations(content, optimization_goals)



























