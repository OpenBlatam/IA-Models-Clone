"""
Model Comparison Engine
=======================

Advanced model comparison system with comprehensive benchmarking,
evaluation metrics, and intelligent model selection capabilities.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, log_loss, confusion_matrix,
    classification_report, mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ComparisonType(str, Enum):
    """Types of model comparisons"""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    SPEED = "speed"
    MEMORY = "memory"
    COST = "cost"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"
    COMPREHENSIVE = "comprehensive"


class EvaluationMetric(str, Enum):
    """Evaluation metrics for model comparison"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    MAPE = "mape"
    LOG_LOSS = "log_loss"
    INFERENCE_TIME = "inference_time"
    TRAINING_TIME = "training_time"
    MEMORY_USAGE = "memory_usage"
    MODEL_SIZE = "model_size"
    THROUGHPUT = "throughput"
    LATENCY = "latency"


class StatisticalTest(str, Enum):
    """Statistical tests for model comparison"""
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    CHI_SQUARE = "chi_square"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    ANDERSON_DARLING = "anderson_darling"


@dataclass
class ModelEvaluation:
    """Model evaluation results"""
    model_name: str
    model_type: str
    evaluation_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, float]
    evaluation_time: float
    data_size: int
    cross_validation_scores: List[float]
    created_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelComparison:
    """Model comparison results"""
    comparison_id: str
    model_a: str
    model_b: str
    comparison_type: ComparisonType
    evaluation_metrics: Dict[str, float]
    statistical_tests: Dict[str, Dict[str, Any]]
    winner: str
    confidence: float
    effect_size: float
    practical_significance: str
    recommendations: List[str]
    created_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BenchmarkResult:
    """Benchmark result for a model"""
    model_name: str
    benchmark_name: str
    scores: Dict[str, float]
    rank: int
    percentile: float
    performance_category: str
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    created_at: datetime


@dataclass
class ModelRanking:
    """Model ranking results"""
    ranking_id: str
    ranking_criteria: str
    models: List[Dict[str, Any]]
    ranking_method: str
    confidence_score: float
    created_at: datetime


class ModelComparisonEngine:
    """Advanced model comparison engine with comprehensive evaluation"""
    
    def __init__(self, max_comparisons: int = 1000):
        self.max_comparisons = max_comparisons
        self.model_evaluations: Dict[str, ModelEvaluation] = {}
        self.model_comparisons: List[ModelComparison] = []
        self.benchmark_results: List[BenchmarkResult] = []
        self.model_rankings: List[ModelRanking] = []
        
        # Evaluation configuration
        self.evaluation_config = {
            "cross_validation_folds": 5,
            "random_state": 42,
            "confidence_level": 0.95,
            "statistical_significance_threshold": 0.05,
            "effect_size_threshold": 0.2,
            "bootstrap_samples": 1000
        }
        
        # Cache for evaluation results
        self.evaluation_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def evaluate_model(self, 
                           model_name: str,
                           model: Any,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           X_train: np.ndarray = None,
                           y_train: np.ndarray = None,
                           model_type: str = "classification",
                           evaluation_metrics: List[EvaluationMetric] = None) -> ModelEvaluation:
        """Comprehensive model evaluation"""
        try:
            if evaluation_metrics is None:
                evaluation_metrics = self._get_default_metrics(model_type)
            
            start_time = datetime.now()
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate evaluation metrics
            eval_metrics = await self._calculate_evaluation_metrics(
                y_test, y_pred, model_type, evaluation_metrics
            )
            
            # Calculate confidence intervals
            confidence_intervals = await self._calculate_confidence_intervals(
                y_test, y_pred, model_type, evaluation_metrics
            )
            
            # Calculate statistical significance
            statistical_significance = await self._calculate_statistical_significance(
                y_test, y_pred, model_type
            )
            
            # Cross-validation scores
            cv_scores = []
            if X_train is not None and y_train is not None:
                cv_scores = await self._calculate_cross_validation_scores(
                    model, X_train, y_train, model_type
                )
            
            end_time = datetime.now()
            evaluation_time = (end_time - start_time).total_seconds()
            
            # Create model evaluation
            evaluation = ModelEvaluation(
                model_name=model_name,
                model_type=model_type,
                evaluation_metrics=eval_metrics,
                confidence_intervals=confidence_intervals,
                statistical_significance=statistical_significance,
                evaluation_time=evaluation_time,
                data_size=len(X_test),
                cross_validation_scores=cv_scores,
                created_at=start_time,
                metadata={
                    "evaluation_metrics_used": [m.value for m in evaluation_metrics],
                    "model_class": model.__class__.__name__,
                    "test_data_shape": X_test.shape
                }
            )
            
            # Store evaluation
            self.model_evaluations[model_name] = evaluation
            
            logger.info(f"Evaluated model {model_name} with {len(eval_metrics)} metrics")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise e
    
    async def compare_models(self, 
                           model_a_name: str,
                           model_b_name: str,
                           comparison_type: ComparisonType = ComparisonType.COMPREHENSIVE,
                           statistical_tests: List[StatisticalTest] = None) -> ModelComparison:
        """Compare two models comprehensively"""
        try:
            if model_a_name not in self.model_evaluations:
                raise ValueError(f"Model {model_a_name} not evaluated")
            if model_b_name not in self.model_evaluations:
                raise ValueError(f"Model {model_b_name} not evaluated")
            
            if statistical_tests is None:
                statistical_tests = [StatisticalTest.T_TEST, StatisticalTest.WILCOXON]
            
            comparison_id = hashlib.md5(f"{model_a_name}_{model_b_name}_{comparison_type}_{datetime.now()}".encode()).hexdigest()
            
            eval_a = self.model_evaluations[model_a_name]
            eval_b = self.model_evaluations[model_b_name]
            
            # Calculate comparison metrics
            comparison_metrics = await self._calculate_comparison_metrics(
                eval_a, eval_b, comparison_type
            )
            
            # Perform statistical tests
            statistical_results = await self._perform_statistical_tests(
                eval_a, eval_b, statistical_tests
            )
            
            # Determine winner
            winner, confidence, effect_size = await self._determine_winner(
                eval_a, eval_b, comparison_metrics, statistical_results
            )
            
            # Assess practical significance
            practical_significance = await self._assess_practical_significance(
                effect_size, comparison_metrics
            )
            
            # Generate recommendations
            recommendations = await self._generate_comparison_recommendations(
                model_a_name, model_b_name, winner, comparison_metrics, practical_significance
            )
            
            # Create comparison result
            comparison = ModelComparison(
                comparison_id=comparison_id,
                model_a=model_a_name,
                model_b=model_b_name,
                comparison_type=comparison_type,
                evaluation_metrics=comparison_metrics,
                statistical_tests=statistical_results,
                winner=winner,
                confidence=confidence,
                effect_size=effect_size,
                practical_significance=practical_significance,
                recommendations=recommendations,
                created_at=datetime.now(),
                metadata={
                    "evaluation_a_id": eval_a.model_name,
                    "evaluation_b_id": eval_b.model_name,
                    "statistical_tests_used": [t.value for t in statistical_tests]
                }
            )
            
            # Store comparison
            self.model_comparisons.append(comparison)
            
            logger.info(f"Compared models {model_a_name} vs {model_b_name}: {winner} wins")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise e
    
    async def benchmark_models(self, 
                             models: Dict[str, Any],
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             benchmark_name: str = "default_benchmark",
                             benchmark_metrics: List[EvaluationMetric] = None) -> List[BenchmarkResult]:
        """Benchmark multiple models"""
        try:
            if benchmark_metrics is None:
                benchmark_metrics = [EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE, EvaluationMetric.R2_SCORE]
            
            benchmark_results = []
            
            # Evaluate all models
            for model_name, model in models.items():
                try:
                    evaluation = await self.evaluate_model(
                        model_name=model_name,
                        model=model,
                        X_test=X_test,
                        y_test=y_test
                    )
                    
                    # Calculate benchmark scores
                    scores = {}
                    for metric in benchmark_metrics:
                        if metric.value in evaluation.evaluation_metrics:
                            scores[metric.value] = evaluation.evaluation_metrics[metric.value]
                    
                    benchmark_results.append({
                        "model_name": model_name,
                        "evaluation": evaluation,
                        "scores": scores
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate model {model_name}: {str(e)}")
                    continue
            
            # Rank models
            ranked_results = await self._rank_benchmark_results(benchmark_results, benchmark_metrics)
            
            # Create benchmark results
            final_results = []
            for i, result in enumerate(ranked_results):
                benchmark_result = BenchmarkResult(
                    model_name=result["model_name"],
                    benchmark_name=benchmark_name,
                    scores=result["scores"],
                    rank=i + 1,
                    percentile=((len(ranked_results) - i) / len(ranked_results)) * 100,
                    performance_category=await self._categorize_performance(result["scores"]),
                    strengths=await self._identify_strengths(result["evaluation"]),
                    weaknesses=await self._identify_weaknesses(result["evaluation"]),
                    recommendations=await self._generate_benchmark_recommendations(result["evaluation"]),
                    created_at=datetime.now()
                )
                final_results.append(benchmark_result)
            
            # Store benchmark results
            self.benchmark_results.extend(final_results)
            
            logger.info(f"Benchmarked {len(final_results)} models")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error benchmarking models: {str(e)}")
            return []
    
    async def rank_models(self, 
                        ranking_criteria: str,
                        models: List[str],
                        weights: Dict[str, float] = None) -> ModelRanking:
        """Rank models based on specified criteria"""
        try:
            if weights is None:
                weights = self._get_default_weights(ranking_criteria)
            
            ranking_id = hashlib.md5(f"{ranking_criteria}_{datetime.now()}".encode()).hexdigest()
            
            # Get evaluations for models
            model_data = []
            for model_name in models:
                if model_name in self.model_evaluations:
                    evaluation = self.model_evaluations[model_name]
                    
                    # Calculate weighted score
                    weighted_score = await self._calculate_weighted_score(
                        evaluation.evaluation_metrics, weights
                    )
                    
                    model_data.append({
                        "model_name": model_name,
                        "evaluation": evaluation,
                        "weighted_score": weighted_score,
                        "individual_scores": evaluation.evaluation_metrics
                    })
            
            # Sort by weighted score
            model_data.sort(key=lambda x: x["weighted_score"], reverse=True)
            
            # Calculate confidence score
            confidence_score = await self._calculate_ranking_confidence(model_data)
            
            # Create ranking
            ranking = ModelRanking(
                ranking_id=ranking_id,
                ranking_criteria=ranking_criteria,
                models=model_data,
                ranking_method="weighted_scoring",
                confidence_score=confidence_score,
                created_at=datetime.now()
            )
            
            # Store ranking
            self.model_rankings.append(ranking)
            
            logger.info(f"Ranked {len(model_data)} models by {ranking_criteria}")
            
            return ranking
            
        except Exception as e:
            logger.error(f"Error ranking models: {str(e)}")
            raise e
    
    async def get_comparison_analytics(self, model_name: str = None) -> Dict[str, Any]:
        """Get comprehensive comparison analytics"""
        try:
            if model_name:
                # Get analytics for specific model
                model_comparisons = [c for c in self.model_comparisons 
                                   if c.model_a == model_name or c.model_b == model_name]
                model_benchmarks = [b for b in self.benchmark_results if b.model_name == model_name]
                model_rankings = [r for r in self.model_rankings 
                                if any(m["model_name"] == model_name for m in r.models)]
                
                analytics = {
                    "model_name": model_name,
                    "total_comparisons": len(model_comparisons),
                    "total_benchmarks": len(model_benchmarks),
                    "total_rankings": len(model_rankings),
                    "win_rate": await self._calculate_win_rate(model_comparisons, model_name),
                    "average_performance": await self._calculate_average_performance(model_benchmarks),
                    "ranking_history": [{"criteria": r.ranking_criteria, "rank": self._get_model_rank(r, model_name)} 
                                      for r in model_rankings],
                    "strengths": await self._identify_model_strengths(model_comparisons, model_name),
                    "weaknesses": await self._identify_model_weaknesses(model_comparisons, model_name),
                    "recommendations": await self._generate_model_recommendations(model_comparisons, model_benchmarks, model_name)
                }
            else:
                # Get global analytics
                analytics = {
                    "total_models": len(self.model_evaluations),
                    "total_comparisons": len(self.model_comparisons),
                    "total_benchmarks": len(self.benchmark_results),
                    "total_rankings": len(self.model_rankings),
                    "model_performance_distribution": await self._analyze_performance_distribution(),
                    "comparison_statistics": await self._analyze_comparison_statistics(),
                    "benchmark_insights": await self._analyze_benchmark_insights(),
                    "ranking_insights": await self._analyze_ranking_insights(),
                    "top_performers": await self._identify_top_performers(),
                    "improvement_opportunities": await self._identify_improvement_opportunities()
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting comparison analytics: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    def _get_default_metrics(self, model_type: str) -> List[EvaluationMetric]:
        """Get default metrics for model type"""
        if model_type == "classification":
            return [EvaluationMetric.ACCURACY, EvaluationMetric.PRECISION, 
                   EvaluationMetric.RECALL, EvaluationMetric.F1_SCORE, EvaluationMetric.ROC_AUC]
        elif model_type == "regression":
            return [EvaluationMetric.MSE, EvaluationMetric.MAE, 
                   EvaluationMetric.R2_SCORE, EvaluationMetric.MAPE]
        else:
            return [EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE, EvaluationMetric.R2_SCORE]
    
    async def _calculate_evaluation_metrics(self, 
                                          y_true: np.ndarray,
                                          y_pred: np.ndarray,
                                          model_type: str,
                                          metrics: List[EvaluationMetric]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            results = {}
            
            for metric in metrics:
                try:
                    if metric == EvaluationMetric.ACCURACY:
                        results[metric.value] = accuracy_score(y_true, y_pred)
                    elif metric == EvaluationMetric.PRECISION:
                        results[metric.value] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    elif metric == EvaluationMetric.RECALL:
                        results[metric.value] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    elif metric == EvaluationMetric.F1_SCORE:
                        results[metric.value] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    elif metric == EvaluationMetric.ROC_AUC:
                        if model_type == "classification" and len(np.unique(y_true)) == 2:
                            results[metric.value] = roc_auc_score(y_true, y_pred)
                        else:
                            results[metric.value] = 0.0
                    elif metric == EvaluationMetric.MSE:
                        results[metric.value] = mean_squared_error(y_true, y_pred)
                    elif metric == EvaluationMetric.MAE:
                        results[metric.value] = mean_absolute_error(y_true, y_pred)
                    elif metric == EvaluationMetric.R2_SCORE:
                        results[metric.value] = r2_score(y_true, y_pred)
                    elif metric == EvaluationMetric.MAPE:
                        results[metric.value] = mean_absolute_percentage_error(y_true, y_pred)
                    elif metric == EvaluationMetric.LOG_LOSS:
                        if model_type == "classification":
                            results[metric.value] = log_loss(y_true, y_pred)
                        else:
                            results[metric.value] = 0.0
                except Exception as e:
                    logger.warning(f"Error calculating {metric.value}: {str(e)}")
                    results[metric.value] = 0.0
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating evaluation metrics: {str(e)}")
            return {}
    
    async def _calculate_confidence_intervals(self, 
                                            y_true: np.ndarray,
                                            y_pred: np.ndarray,
                                            model_type: str,
                                            metrics: List[EvaluationMetric]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for metrics"""
        try:
            confidence_intervals = {}
            confidence_level = self.evaluation_config["confidence_level"]
            
            for metric in metrics:
                try:
                    # Bootstrap sampling for confidence intervals
                    n_samples = self.evaluation_config["bootstrap_samples"]
                    bootstrap_scores = []
                    
                    for _ in range(n_samples):
                        # Sample with replacement
                        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
                        y_true_sample = y_true[indices]
                        y_pred_sample = y_pred[indices]
                        
                        # Calculate metric for this sample
                        if metric == EvaluationMetric.ACCURACY:
                            score = accuracy_score(y_true_sample, y_pred_sample)
                        elif metric == EvaluationMetric.F1_SCORE:
                            score = f1_score(y_true_sample, y_pred_sample, average='weighted', zero_division=0)
                        elif metric == EvaluationMetric.R2_SCORE:
                            score = r2_score(y_true_sample, y_pred_sample)
                        else:
                            continue
                        
                        bootstrap_scores.append(score)
                    
                    if bootstrap_scores:
                        # Calculate confidence interval
                        alpha = 1 - confidence_level
                        lower = np.percentile(bootstrap_scores, (alpha/2) * 100)
                        upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
                        confidence_intervals[metric.value] = (float(lower), float(upper))
                    
                except Exception as e:
                    logger.warning(f"Error calculating confidence interval for {metric.value}: {str(e)}")
                    confidence_intervals[metric.value] = (0.0, 1.0)
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            return {}
    
    async def _calculate_statistical_significance(self, 
                                                y_true: np.ndarray,
                                                y_pred: np.ndarray,
                                                model_type: str) -> Dict[str, float]:
        """Calculate statistical significance measures"""
        try:
            significance = {}
            
            # Calculate p-values for different metrics
            if model_type == "classification":
                # Accuracy significance
                accuracy = accuracy_score(y_true, y_pred)
                n = len(y_true)
                # Binomial test for accuracy
                p_value = stats.binom_test(int(accuracy * n), n, 0.5, alternative='two-sided')
                significance["accuracy_p_value"] = p_value
                
                # F1 score significance (simplified)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                significance["f1_score_p_value"] = 1.0 - f1  # Simplified p-value
            
            elif model_type == "regression":
                # R2 significance
                r2 = r2_score(y_true, y_pred)
                n = len(y_true)
                k = 1  # Number of predictors (simplified)
                f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
                p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
                significance["r2_p_value"] = p_value
            
            return significance
            
        except Exception as e:
            logger.error(f"Error calculating statistical significance: {str(e)}")
            return {}
    
    async def _calculate_cross_validation_scores(self, 
                                               model: Any,
                                               X_train: np.ndarray,
                                               y_train: np.ndarray,
                                               model_type: str) -> List[float]:
        """Calculate cross-validation scores"""
        try:
            cv_folds = self.evaluation_config["cross_validation_folds"]
            random_state = self.evaluation_config["random_state"]
            
            if model_type == "classification":
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'r2'
            
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            
            return scores.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating cross-validation scores: {str(e)}")
            return []
    
    async def _calculate_comparison_metrics(self, 
                                          eval_a: ModelEvaluation,
                                          eval_b: ModelEvaluation,
                                          comparison_type: ComparisonType) -> Dict[str, float]:
        """Calculate comparison metrics between two models"""
        try:
            comparison_metrics = {}
            
            # Compare common metrics
            common_metrics = set(eval_a.evaluation_metrics.keys()) & set(eval_b.evaluation_metrics.keys())
            
            for metric in common_metrics:
                value_a = eval_a.evaluation_metrics[metric]
                value_b = eval_b.evaluation_metrics[metric]
                
                # Calculate difference
                if metric in ['mse', 'mae', 'mape', 'log_loss']:
                    # Lower is better
                    difference = value_b - value_a
                    improvement = difference / value_a if value_a != 0 else 0
                else:
                    # Higher is better
                    difference = value_a - value_b
                    improvement = difference / value_b if value_b != 0 else 0
                
                comparison_metrics[f"{metric}_difference"] = difference
                comparison_metrics[f"{metric}_improvement"] = improvement
            
            # Calculate overall comparison score
            if comparison_type == ComparisonType.COMPREHENSIVE:
                comparison_metrics["overall_score"] = await self._calculate_overall_comparison_score(
                    eval_a, eval_b
                )
            
            return comparison_metrics
            
        except Exception as e:
            logger.error(f"Error calculating comparison metrics: {str(e)}")
            return {}
    
    async def _perform_statistical_tests(self, 
                                       eval_a: ModelEvaluation,
                                       eval_b: ModelEvaluation,
                                       tests: List[StatisticalTest]) -> Dict[str, Dict[str, Any]]:
        """Perform statistical tests between two models"""
        try:
            results = {}
            
            # Get cross-validation scores for both models
            scores_a = eval_a.cross_validation_scores
            scores_b = eval_b.cross_validation_scores
            
            if not scores_a or not scores_b:
                return results
            
            for test in tests:
                try:
                    if test == StatisticalTest.T_TEST:
                        statistic, p_value = stats.ttest_ind(scores_a, scores_b)
                        results[test.value] = {
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "significant": p_value < self.evaluation_config["statistical_significance_threshold"]
                        }
                    elif test == StatisticalTest.WILCOXON:
                        statistic, p_value = stats.wilcoxon(scores_a, scores_b)
                        results[test.value] = {
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "significant": p_value < self.evaluation_config["statistical_significance_threshold"]
                        }
                    elif test == StatisticalTest.MANN_WHITNEY:
                        statistic, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
                        results[test.value] = {
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "significant": p_value < self.evaluation_config["statistical_significance_threshold"]
                        }
                except Exception as e:
                    logger.warning(f"Error performing {test.value}: {str(e)}")
                    results[test.value] = {
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "significant": False,
                        "error": str(e)
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing statistical tests: {str(e)}")
            return {}
    
    async def _determine_winner(self, 
                              eval_a: ModelEvaluation,
                              eval_b: ModelEvaluation,
                              comparison_metrics: Dict[str, float],
                              statistical_results: Dict[str, Dict[str, Any]]) -> Tuple[str, float, float]:
        """Determine the winner between two models"""
        try:
            # Count significant wins
            wins_a = 0
            wins_b = 0
            total_comparisons = 0
            
            for test_name, test_result in statistical_results.items():
                if test_result.get("significant", False):
                    total_comparisons += 1
                    if test_result["statistic"] > 0:  # Model A better
                        wins_a += 1
                    else:  # Model B better
                        wins_b += 1
            
            # Calculate confidence
            if total_comparisons > 0:
                confidence = max(wins_a, wins_b) / total_comparisons
            else:
                confidence = 0.5
            
            # Calculate effect size (Cohen's d)
            scores_a = eval_a.cross_validation_scores
            scores_b = eval_b.cross_validation_scores
            
            if scores_a and scores_b:
                mean_a = np.mean(scores_a)
                mean_b = np.mean(scores_b)
                std_a = np.std(scores_a)
                std_b = np.std(scores_b)
                
                pooled_std = np.sqrt(((len(scores_a) - 1) * std_a**2 + (len(scores_b) - 1) * std_b**2) / 
                                   (len(scores_a) + len(scores_b) - 2))
                
                if pooled_std > 0:
                    effect_size = abs(mean_a - mean_b) / pooled_std
                else:
                    effect_size = 0.0
            else:
                effect_size = 0.0
            
            # Determine winner
            if wins_a > wins_b:
                winner = eval_a.model_name
            elif wins_b > wins_a:
                winner = eval_b.model_name
            else:
                # Tie - use overall score if available
                if "overall_score" in comparison_metrics:
                    if comparison_metrics["overall_score"] > 0:
                        winner = eval_a.model_name
                    else:
                        winner = eval_b.model_name
                else:
                    winner = "tie"
            
            return winner, confidence, effect_size
            
        except Exception as e:
            logger.error(f"Error determining winner: {str(e)}")
            return "unknown", 0.0, 0.0
    
    async def _assess_practical_significance(self, 
                                           effect_size: float,
                                           comparison_metrics: Dict[str, float]) -> str:
        """Assess practical significance of the difference"""
        try:
            if effect_size < 0.2:
                return "negligible"
            elif effect_size < 0.5:
                return "small"
            elif effect_size < 0.8:
                return "medium"
            else:
                return "large"
                
        except Exception as e:
            logger.error(f"Error assessing practical significance: {str(e)}")
            return "unknown"
    
    async def _generate_comparison_recommendations(self, 
                                                model_a: str,
                                                model_b: str,
                                                winner: str,
                                                comparison_metrics: Dict[str, float],
                                                practical_significance: str) -> List[str]:
        """Generate recommendations based on comparison"""
        try:
            recommendations = []
            
            if winner != "tie":
                recommendations.append(f"Model {winner} is the recommended choice based on statistical analysis")
                
                if practical_significance in ["medium", "large"]:
                    recommendations.append(f"The performance difference is {practical_significance} and practically significant")
                elif practical_significance == "small":
                    recommendations.append("The performance difference is small - consider other factors like cost, speed, or interpretability")
                else:
                    recommendations.append("The performance difference is negligible - choose based on other criteria")
            else:
                recommendations.append("Models perform similarly - consider other factors like cost, speed, or interpretability")
            
            # Add specific recommendations based on metrics
            for metric, improvement in comparison_metrics.items():
                if metric.endswith("_improvement") and abs(improvement) > 0.1:
                    if improvement > 0:
                        recommendations.append(f"Model {model_a} shows {improvement:.1%} improvement in {metric.replace('_improvement', '')}")
                    else:
                        recommendations.append(f"Model {model_b} shows {abs(improvement):.1%} improvement in {metric.replace('_improvement', '')}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating comparison recommendations: {str(e)}")
            return ["Unable to generate recommendations"]
    
    async def _calculate_overall_comparison_score(self, 
                                                eval_a: ModelEvaluation,
                                                eval_b: ModelEvaluation) -> float:
        """Calculate overall comparison score"""
        try:
            # Weight different metrics
            weights = {
                "accuracy": 0.3,
                "f1_score": 0.25,
                "r2_score": 0.2,
                "precision": 0.15,
                "recall": 0.1
            }
            
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in eval_a.evaluation_metrics and metric in eval_b.evaluation_metrics:
                    value_a = eval_a.evaluation_metrics[metric]
                    value_b = eval_b.evaluation_metrics[metric]
                    
                    if metric in ['mse', 'mae', 'mape', 'log_loss']:
                        # Lower is better
                        if value_b > 0:
                            improvement = (value_b - value_a) / value_b
                        else:
                            improvement = 0
                    else:
                        # Higher is better
                        if value_b > 0:
                            improvement = (value_a - value_b) / value_b
                        else:
                            improvement = 0
                    
                    score += improvement * weight
                    total_weight += weight
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall comparison score: {str(e)}")
            return 0.0
    
    async def _rank_benchmark_results(self, 
                                    results: List[Dict[str, Any]], 
                                    metrics: List[EvaluationMetric]) -> List[Dict[str, Any]]:
        """Rank benchmark results"""
        try:
            # Calculate composite scores
            for result in results:
                scores = result["scores"]
                composite_score = 0.0
                count = 0
                
                for metric in metrics:
                    if metric.value in scores:
                        value = scores[metric.value]
                        
                        # Normalize score (0-1)
                        if metric in [EvaluationMetric.MSE, EvaluationMetric.MAE, EvaluationMetric.MAPE, EvaluationMetric.LOG_LOSS]:
                            # Lower is better - invert
                            normalized = 1.0 / (1.0 + value)
                        else:
                            # Higher is better
                            normalized = min(1.0, max(0.0, value))
                        
                        composite_score += normalized
                        count += 1
                
                result["composite_score"] = composite_score / count if count > 0 else 0.0
            
            # Sort by composite score
            results.sort(key=lambda x: x["composite_score"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error ranking benchmark results: {str(e)}")
            return results
    
    async def _categorize_performance(self, scores: Dict[str, float]) -> str:
        """Categorize model performance"""
        try:
            if not scores:
                return "unknown"
            
            # Calculate average score
            values = list(scores.values())
            avg_score = np.mean(values)
            
            if avg_score >= 0.9:
                return "excellent"
            elif avg_score >= 0.8:
                return "good"
            elif avg_score >= 0.7:
                return "fair"
            elif avg_score >= 0.6:
                return "poor"
            else:
                return "very_poor"
                
        except Exception as e:
            logger.error(f"Error categorizing performance: {str(e)}")
            return "unknown"
    
    async def _identify_strengths(self, evaluation: ModelEvaluation) -> List[str]:
        """Identify model strengths"""
        try:
            strengths = []
            metrics = evaluation.evaluation_metrics
            
            # Define thresholds for strengths
            thresholds = {
                "accuracy": 0.9,
                "f1_score": 0.85,
                "r2_score": 0.8,
                "precision": 0.85,
                "recall": 0.85
            }
            
            for metric, threshold in thresholds.items():
                if metric in metrics and metrics[metric] >= threshold:
                    strengths.append(f"High {metric}: {metrics[metric]:.3f}")
            
            # Check cross-validation consistency
            cv_scores = evaluation.cross_validation_scores
            if cv_scores and len(cv_scores) > 1:
                cv_std = np.std(cv_scores)
                if cv_std < 0.05:
                    strengths.append("Consistent cross-validation performance")
            
            return strengths
            
        except Exception as e:
            logger.error(f"Error identifying strengths: {str(e)}")
            return []
    
    async def _identify_weaknesses(self, evaluation: ModelEvaluation) -> List[str]:
        """Identify model weaknesses"""
        try:
            weaknesses = []
            metrics = evaluation.evaluation_metrics
            
            # Define thresholds for weaknesses
            thresholds = {
                "accuracy": 0.7,
                "f1_score": 0.6,
                "r2_score": 0.5,
                "precision": 0.6,
                "recall": 0.6
            }
            
            for metric, threshold in thresholds.items():
                if metric in metrics and metrics[metric] < threshold:
                    weaknesses.append(f"Low {metric}: {metrics[metric]:.3f}")
            
            # Check cross-validation consistency
            cv_scores = evaluation.cross_validation_scores
            if cv_scores and len(cv_scores) > 1:
                cv_std = np.std(cv_scores)
                if cv_std > 0.1:
                    weaknesses.append("Inconsistent cross-validation performance")
            
            return weaknesses
            
        except Exception as e:
            logger.error(f"Error identifying weaknesses: {str(e)}")
            return []
    
    async def _generate_benchmark_recommendations(self, evaluation: ModelEvaluation) -> List[str]:
        """Generate benchmark recommendations"""
        try:
            recommendations = []
            metrics = evaluation.evaluation_metrics
            
            # Performance-based recommendations
            if "accuracy" in metrics:
                if metrics["accuracy"] < 0.8:
                    recommendations.append("Consider feature engineering or model tuning to improve accuracy")
                elif metrics["accuracy"] > 0.95:
                    recommendations.append("Excellent accuracy - consider deployment in production")
            
            if "f1_score" in metrics:
                if metrics["f1_score"] < 0.7:
                    recommendations.append("Low F1 score - check class imbalance and consider resampling")
            
            if "r2_score" in metrics:
                if metrics["r2_score"] < 0.6:
                    recommendations.append("Low R² score - consider more complex model or feature selection")
            
            # Cross-validation recommendations
            cv_scores = evaluation.cross_validation_scores
            if cv_scores and len(cv_scores) > 1:
                cv_std = np.std(cv_scores)
                if cv_std > 0.1:
                    recommendations.append("High cross-validation variance - consider regularization or more data")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating benchmark recommendations: {str(e)}")
            return []
    
    def _get_default_weights(self, ranking_criteria: str) -> Dict[str, float]:
        """Get default weights for ranking criteria"""
        if ranking_criteria == "accuracy":
            return {"accuracy": 1.0}
        elif ranking_criteria == "balanced":
            return {"accuracy": 0.4, "f1_score": 0.3, "precision": 0.15, "recall": 0.15}
        elif ranking_criteria == "precision":
            return {"precision": 0.6, "accuracy": 0.4}
        elif ranking_criteria == "recall":
            return {"recall": 0.6, "accuracy": 0.4}
        else:
            return {"accuracy": 0.5, "f1_score": 0.5}
    
    async def _calculate_weighted_score(self, 
                                      metrics: Dict[str, float], 
                                      weights: Dict[str, float]) -> float:
        """Calculate weighted score"""
        try:
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    weighted_score += metrics[metric] * weight
                    total_weight += weight
            
            return weighted_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating weighted score: {str(e)}")
            return 0.0
    
    async def _calculate_ranking_confidence(self, model_data: List[Dict[str, Any]]) -> float:
        """Calculate confidence in ranking"""
        try:
            if len(model_data) < 2:
                return 0.0
            
            # Calculate score differences
            scores = [m["weighted_score"] for m in model_data]
            max_score = max(scores)
            min_score = min(scores)
            
            if max_score == min_score:
                return 0.0
            
            # Confidence based on score separation
            score_range = max_score - min_score
            confidence = min(1.0, score_range / max_score)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating ranking confidence: {str(e)}")
            return 0.0
    
    def _get_model_rank(self, ranking: ModelRanking, model_name: str) -> int:
        """Get model rank in a ranking"""
        try:
            for i, model_data in enumerate(ranking.models):
                if model_data["model_name"] == model_name:
                    return i + 1
            return -1
        except Exception as e:
            logger.error(f"Error getting model rank: {str(e)}")
            return -1
    
    async def _calculate_win_rate(self, comparisons: List[ModelComparison], model_name: str) -> float:
        """Calculate win rate for a model"""
        try:
            if not comparisons:
                return 0.0
            
            wins = 0
            total = 0
            
            for comparison in comparisons:
                if comparison.model_a == model_name or comparison.model_b == model_name:
                    total += 1
                    if comparison.winner == model_name:
                        wins += 1
            
            return wins / total if total > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {str(e)}")
            return 0.0
    
    async def _calculate_average_performance(self, benchmarks: List[BenchmarkResult]) -> Dict[str, float]:
        """Calculate average performance across benchmarks"""
        try:
            if not benchmarks:
                return {}
            
            # Aggregate scores across benchmarks
            all_scores = defaultdict(list)
            
            for benchmark in benchmarks:
                for metric, score in benchmark.scores.items():
                    all_scores[metric].append(score)
            
            # Calculate averages
            averages = {}
            for metric, scores in all_scores.items():
                averages[metric] = np.mean(scores)
            
            return averages
            
        except Exception as e:
            logger.error(f"Error calculating average performance: {str(e)}")
            return {}
    
    async def _identify_model_strengths(self, comparisons: List[ModelComparison], model_name: str) -> List[str]:
        """Identify model strengths from comparisons"""
        try:
            strengths = []
            
            for comparison in comparisons:
                if comparison.winner == model_name:
                    for metric, improvement in comparison.evaluation_metrics.items():
                        if metric.endswith("_improvement") and improvement > 0.1:
                            strengths.append(f"Strong in {metric.replace('_improvement', '')}")
            
            return list(set(strengths))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error identifying model strengths: {str(e)}")
            return []
    
    async def _identify_model_weaknesses(self, comparisons: List[ModelComparison], model_name: str) -> List[str]:
        """Identify model weaknesses from comparisons"""
        try:
            weaknesses = []
            
            for comparison in comparisons:
                if comparison.winner != model_name and (comparison.model_a == model_name or comparison.model_b == model_name):
                    for metric, improvement in comparison.evaluation_metrics.items():
                        if metric.endswith("_improvement") and improvement < -0.1:
                            weaknesses.append(f"Weak in {metric.replace('_improvement', '')}")
            
            return list(set(weaknesses))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error identifying model weaknesses: {str(e)}")
            return []
    
    async def _generate_model_recommendations(self, 
                                            comparisons: List[ModelComparison],
                                            benchmarks: List[BenchmarkResult],
                                            model_name: str) -> List[str]:
        """Generate recommendations for a model"""
        try:
            recommendations = []
            
            # Win rate recommendations
            win_rate = await self._calculate_win_rate(comparisons, model_name)
            if win_rate > 0.8:
                recommendations.append("High win rate - excellent model performance")
            elif win_rate < 0.3:
                recommendations.append("Low win rate - consider model improvement or replacement")
            
            # Benchmark recommendations
            model_benchmarks = [b for b in benchmarks if b.model_name == model_name]
            if model_benchmarks:
                avg_rank = np.mean([b.rank for b in model_benchmarks])
                if avg_rank <= 2:
                    recommendations.append("Top performer in benchmarks - consider for production")
                elif avg_rank >= 4:
                    recommendations.append("Below average performance - needs improvement")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating model recommendations: {str(e)}")
            return []
    
    async def _analyze_performance_distribution(self) -> Dict[str, Any]:
        """Analyze performance distribution across all models"""
        try:
            if not self.model_evaluations:
                return {}
            
            # Collect all metrics
            all_metrics = defaultdict(list)
            for evaluation in self.model_evaluations.values():
                for metric, value in evaluation.evaluation_metrics.items():
                    all_metrics[metric].append(value)
            
            # Calculate statistics
            distribution = {}
            for metric, values in all_metrics.items():
                distribution[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "count": len(values)
                }
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error analyzing performance distribution: {str(e)}")
            return {}
    
    async def _analyze_comparison_statistics(self) -> Dict[str, Any]:
        """Analyze comparison statistics"""
        try:
            if not self.model_comparisons:
                return {}
            
            # Analyze win rates
            win_counts = defaultdict(int)
            total_comparisons = defaultdict(int)
            
            for comparison in self.model_comparisons:
                total_comparisons[comparison.model_a] += 1
                total_comparisons[comparison.model_b] += 1
                
                if comparison.winner != "tie":
                    win_counts[comparison.winner] += 1
            
            # Calculate win rates
            win_rates = {}
            for model, total in total_comparisons.items():
                win_rates[model] = win_counts[model] / total if total > 0 else 0.0
            
            return {
                "total_comparisons": len(self.model_comparisons),
                "win_rates": win_rates,
                "most_compared_model": max(total_comparisons.items(), key=lambda x: x[1])[0] if total_comparisons else None,
                "best_performer": max(win_rates.items(), key=lambda x: x[1])[0] if win_rates else None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing comparison statistics: {str(e)}")
            return {}
    
    async def _analyze_benchmark_insights(self) -> Dict[str, Any]:
        """Analyze benchmark insights"""
        try:
            if not self.benchmark_results:
                return {}
            
            # Analyze benchmark performance
            benchmark_names = set(b.benchmark_name for b in self.benchmark_results)
            insights = {}
            
            for benchmark_name in benchmark_names:
                benchmark_results = [b for b in self.benchmark_results if b.benchmark_name == benchmark_name]
                
                insights[benchmark_name] = {
                    "total_models": len(benchmark_results),
                    "average_rank": np.mean([b.rank for b in benchmark_results]),
                    "performance_categories": {
                        cat: len([b for b in benchmark_results if b.performance_category == cat])
                        for cat in ["excellent", "good", "fair", "poor", "very_poor"]
                    }
                }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing benchmark insights: {str(e)}")
            return {}
    
    async def _analyze_ranking_insights(self) -> Dict[str, Any]:
        """Analyze ranking insights"""
        try:
            if not self.model_rankings:
                return {}
            
            # Analyze ranking patterns
            ranking_criteria = set(r.ranking_criteria for r in self.model_rankings)
            insights = {}
            
            for criteria in ranking_criteria:
                rankings = [r for r in self.model_rankings if r.ranking_criteria == criteria]
                
                insights[criteria] = {
                    "total_rankings": len(rankings),
                    "average_confidence": np.mean([r.confidence_score for r in rankings]),
                    "most_common_winner": self._get_most_common_winner(rankings)
                }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing ranking insights: {str(e)}")
            return {}
    
    def _get_most_common_winner(self, rankings: List[ModelRanking]) -> str:
        """Get most common winner in rankings"""
        try:
            if not rankings:
                return None
            
            winner_counts = defaultdict(int)
            for ranking in rankings:
                if ranking.models:
                    winner = ranking.models[0]["model_name"]
                    winner_counts[winner] += 1
            
            return max(winner_counts.items(), key=lambda x: x[1])[0] if winner_counts else None
            
        except Exception as e:
            logger.error(f"Error getting most common winner: {str(e)}")
            return None
    
    async def _identify_top_performers(self) -> List[Dict[str, Any]]:
        """Identify top performing models"""
        try:
            if not self.model_evaluations:
                return []
            
            # Calculate composite scores
            model_scores = []
            for model_name, evaluation in self.model_evaluations.items():
                metrics = evaluation.evaluation_metrics
                composite_score = 0.0
                count = 0
                
                for metric, value in metrics.items():
                    if metric in ["accuracy", "f1_score", "r2_score"]:
                        composite_score += value
                        count += 1
                
                if count > 0:
                    model_scores.append({
                        "model_name": model_name,
                        "composite_score": composite_score / count,
                        "evaluation": evaluation
                    })
            
            # Sort by composite score
            model_scores.sort(key=lambda x: x["composite_score"], reverse=True)
            
            return model_scores[:5]  # Top 5
            
        except Exception as e:
            logger.error(f"Error identifying top performers: {str(e)}")
            return []
    
    async def _identify_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """Identify improvement opportunities"""
        try:
            opportunities = []
            
            # Analyze models with low performance
            for model_name, evaluation in self.model_evaluations.items():
                metrics = evaluation.evaluation_metrics
                
                # Check for low performance metrics
                low_metrics = []
                for metric, value in metrics.items():
                    if metric in ["accuracy", "f1_score", "r2_score"] and value < 0.7:
                        low_metrics.append(metric)
                
                if low_metrics:
                    opportunities.append({
                        "model_name": model_name,
                        "low_metrics": low_metrics,
                        "recommendation": f"Improve {', '.join(low_metrics)}"
                    })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying improvement opportunities: {str(e)}")
            return []


# Global comparison engine instance
_comparison_engine: Optional[ModelComparisonEngine] = None


def get_model_comparison_engine(max_comparisons: int = 1000) -> ModelComparisonEngine:
    """Get or create global model comparison engine instance"""
    global _comparison_engine
    if _comparison_engine is None:
        _comparison_engine = ModelComparisonEngine(max_comparisons)
    return _comparison_engine


# Example usage
async def main():
    """Example usage of the model comparison engine"""
    engine = get_model_comparison_engine()
    
    # Generate sample data
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(random_state=42)
    
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    # Evaluate models
    rf_evaluation = await engine.evaluate_model(
        model_name="RandomForest",
        model=rf_model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train
    )
    
    lr_evaluation = await engine.evaluate_model(
        model_name="LogisticRegression",
        model=lr_model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train
    )
    
    # Compare models
    comparison = await engine.compare_models("RandomForest", "LogisticRegression")
    print(f"Comparison result: {comparison.winner} wins with {comparison.confidence:.3f} confidence")
    
    # Benchmark models
    models = {"RandomForest": rf_model, "LogisticRegression": lr_model}
    benchmark_results = await engine.benchmark_models(models, X_test, y_test)
    print(f"Benchmarked {len(benchmark_results)} models")
    
    # Rank models
    ranking = await engine.rank_models("balanced", ["RandomForest", "LogisticRegression"])
    print(f"Ranking: {ranking.models[0]['model_name']} is ranked #1")
    
    # Get analytics
    analytics = await engine.get_comparison_analytics()
    print(f"Analytics: {analytics.get('total_models', 0)} models analyzed")


if __name__ == "__main__":
    asyncio.run(main())

























