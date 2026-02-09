"""
Domain Services
==============

This module contains domain services that encapsulate business logic that doesn't naturally
belong to a single entity. These services coordinate between entities and enforce business rules.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import statistics
import numpy as np
from dataclasses import dataclass

from .domain import (
    HistoryEntry, ComparisonResult, TrendAnalysis, QualityReport,
    ContentMetrics, PerformanceMetric, TrendDirection, AnalysisStatus
)


class ContentAnalysisService:
    """Service for analyzing content quality and characteristics"""
    
    def analyze_content(self, content: str) -> ContentMetrics:
        """Analyze content and return comprehensive metrics"""
        # Basic text analysis
        words = content.split()
        sentences = content.split('.')
        
        # Calculate basic metrics
        word_count = len(words)
        sentence_count = len([s for s in sentences if s.strip()])
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        # Calculate readability score (simplified Flesch Reading Ease)
        readability_score = self._calculate_readability(word_count, sentence_count, content)
        
        # Calculate sentiment score (simplified)
        sentiment_score = self._calculate_sentiment(content)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(content, words)
        
        # Calculate topic diversity
        topic_diversity = self._calculate_topic_diversity(words)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency(content)
        
        return ContentMetrics(
            readability_score=readability_score,
            sentiment_score=sentiment_score,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            complexity_score=complexity_score,
            topic_diversity=topic_diversity,
            consistency_score=consistency_score,
            quality_score=self._calculate_quality_score(
                readability_score, sentiment_score, complexity_score, consistency_score
            ),
            coherence_score=self._calculate_coherence(content),
            relevance_score=self._calculate_relevance(content),
            creativity_score=self._calculate_creativity(content)
        )
    
    def _calculate_readability(self, word_count: int, sentence_count: int, content: str) -> float:
        """Calculate readability score (0-1 scale)"""
        if sentence_count == 0 or word_count == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease
        avg_sentence_length = word_count / sentence_count
        avg_syllables_per_word = self._count_syllables(content) / word_count
        
        # Normalize to 0-1 scale
        readability = max(0, min(1, (206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)) / 100))
        return readability
    
    def _calculate_sentiment(self, content: str) -> float:
        """Calculate sentiment score (-1 to 1, normalized to 0-1)"""
        # Simplified sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor', 'worst']
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return (sentiment + 1) / 2  # Normalize to 0-1
    
    def _calculate_complexity(self, content: str, words: List[str]) -> float:
        """Calculate content complexity (0-1 scale)"""
        if not words:
            return 0.0
        
        # Factors: average word length, sentence length, vocabulary diversity
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / len(words)
        
        # Normalize complexity score
        complexity = (avg_word_length / 10 + vocabulary_diversity) / 2
        return min(1.0, complexity)
    
    def _calculate_topic_diversity(self, words: List[str]) -> float:
        """Calculate topic diversity (0-1 scale)"""
        if not words:
            return 0.0
        
        # Simple topic diversity based on unique words
        unique_words = len(set(word.lower() for word in words))
        return min(1.0, unique_words / len(words))
    
    def _calculate_consistency(self, content: str) -> float:
        """Calculate content consistency (0-1 scale)"""
        # Simplified consistency check
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) < 2:
            return 1.0
        
        # Check for consistent sentence length
        sentence_lengths = [len(s.split()) for s in sentences]
        if not sentence_lengths:
            return 1.0
        
        # Calculate coefficient of variation
        mean_length = statistics.mean(sentence_lengths)
        if mean_length == 0:
            return 1.0
        
        std_length = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
        cv = std_length / mean_length
        
        # Convert to consistency score (lower CV = higher consistency)
        return max(0.0, 1.0 - cv)
    
    def _calculate_quality_score(self, readability: float, sentiment: float, 
                               complexity: float, consistency: float) -> float:
        """Calculate overall quality score"""
        # Weighted average of quality factors
        weights = {
            'readability': 0.3,
            'sentiment': 0.2,
            'complexity': 0.2,
            'consistency': 0.3
        }
        
        return (readability * weights['readability'] + 
                sentiment * weights['sentiment'] + 
                complexity * weights['complexity'] + 
                consistency * weights['consistency'])
    
    def _calculate_coherence(self, content: str) -> float:
        """Calculate content coherence (0-1 scale)"""
        # Simplified coherence check based on sentence transitions
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) < 2:
            return 1.0
        
        # Check for transition words and logical flow
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'consequently', 'thus']
        transition_count = sum(1 for sentence in sentences 
                             if any(word in sentence.lower() for word in transition_words))
        
        return min(1.0, transition_count / len(sentences) + 0.5)
    
    def _calculate_relevance(self, content: str) -> float:
        """Calculate content relevance (0-1 scale)"""
        # Simplified relevance check
        # In a real implementation, this would compare against a topic or context
        return 0.8  # Placeholder
    
    def _calculate_creativity(self, content: str) -> float:
        """Calculate content creativity (0-1 scale)"""
        # Simplified creativity check based on unique word usage
        words = content.split()
        if not words:
            return 0.0
        
        unique_words = len(set(word.lower() for word in words))
        return min(1.0, unique_words / len(words))
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified)"""
        vowels = 'aeiouy'
        count = 0
        for word in text.lower().split():
            word_count = 0
            prev_was_vowel = False
            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        word_count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            if word.endswith('e'):
                word_count -= 1
            if word_count == 0:
                word_count = 1
            count += word_count
        return count


class ModelComparisonService:
    """Service for comparing AI models and their outputs"""
    
    def compare_entries(self, entry1: HistoryEntry, entry2: HistoryEntry) -> ComparisonResult:
        """Compare two history entries and return detailed comparison"""
        # Calculate similarity score
        similarity_score = self._calculate_similarity(entry1, entry2)
        
        # Calculate quality differences
        quality_difference = self._calculate_quality_difference(entry1, entry2)
        
        # Determine trend direction
        trend_direction = self._determine_trend_direction(entry1, entry2)
        
        # Identify significant changes
        significant_changes = self._identify_significant_changes(entry1, entry2)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(entry1, entry2, quality_difference)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(entry1, entry2)
        
        return ComparisonResult.create(
            entry1_id=entry1.id,
            entry2_id=entry2.id,
            similarity_score=similarity_score,
            quality_difference=quality_difference,
            trend_direction=trend_direction,
            significant_changes=significant_changes,
            recommendations=recommendations,
            confidence_score=confidence_score
        )
    
    def _calculate_similarity(self, entry1: HistoryEntry, entry2: HistoryEntry) -> float:
        """Calculate similarity between two entries"""
        # Content similarity (simplified)
        content_sim = self._text_similarity(entry1.content, entry2.content)
        
        # Metrics similarity
        metrics_sim = self._metrics_similarity(entry1.metrics, entry2.metrics)
        
        # Weighted combination
        return (content_sim * 0.6 + metrics_sim * 0.4)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simplified Jaccard similarity)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _metrics_similarity(self, metrics1: ContentMetrics, metrics2: ContentMetrics) -> float:
        """Calculate similarity between metrics"""
        similarities = []
        
        # Compare each metric
        for attr in ['readability_score', 'sentiment_score', 'complexity_score', 
                    'consistency_score', 'quality_score']:
            val1 = getattr(metrics1, attr)
            val2 = getattr(metrics2, attr)
            
            if val1 is not None and val2 is not None:
                # Calculate similarity (1 - absolute difference)
                similarity = 1.0 - abs(val1 - val2)
                similarities.append(similarity)
        
        return statistics.mean(similarities) if similarities else 0.0
    
    def _calculate_quality_difference(self, entry1: HistoryEntry, entry2: HistoryEntry) -> Dict[str, float]:
        """Calculate quality differences between entries"""
        differences = {}
        
        # Compare metrics
        for attr in ['readability_score', 'sentiment_score', 'complexity_score', 
                    'consistency_score', 'quality_score']:
            val1 = getattr(entry1.metrics, attr)
            val2 = getattr(entry2.metrics, attr)
            
            if val1 is not None and val2 is not None:
                differences[attr] = val2 - val1  # Positive means entry2 is better
        
        return differences
    
    def _determine_trend_direction(self, entry1: HistoryEntry, entry2: HistoryEntry) -> TrendDirection:
        """Determine trend direction between entries"""
        if entry1.timestamp > entry2.timestamp:
            entry1, entry2 = entry2, entry1  # Ensure chronological order
        
        quality1 = entry1.calculate_quality_score()
        quality2 = entry2.calculate_quality_score()
        
        difference = quality2 - quality1
        
        if abs(difference) < 0.05:  # Less than 5% difference
            return TrendDirection.STABLE
        elif difference > 0:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DECLINING
    
    def _identify_significant_changes(self, entry1: HistoryEntry, entry2: HistoryEntry) -> List[str]:
        """Identify significant changes between entries"""
        changes = []
        threshold = 0.1  # 10% change threshold
        
        # Check quality score change
        quality1 = entry1.calculate_quality_score()
        quality2 = entry2.calculate_quality_score()
        if abs(quality2 - quality1) > threshold:
            changes.append(f"Quality score changed by {abs(quality2 - quality1):.2f}")
        
        # Check individual metrics
        for attr in ['readability_score', 'sentiment_score', 'complexity_score']:
            val1 = getattr(entry1.metrics, attr)
            val2 = getattr(entry2.metrics, attr)
            
            if val1 is not None and val2 is not None and abs(val2 - val1) > threshold:
                changes.append(f"{attr.replace('_', ' ').title()} changed by {abs(val2 - val1):.2f}")
        
        return changes
    
    def _generate_recommendations(self, entry1: HistoryEntry, entry2: HistoryEntry, 
                                quality_difference: Dict[str, float]) -> List[str]:
        """Generate recommendations based on comparison"""
        recommendations = []
        
        # Analyze quality differences
        for metric, difference in quality_difference.items():
            if abs(difference) > 0.1:  # Significant difference
                if difference > 0:
                    recommendations.append(f"Consider applying {metric.replace('_', ' ')} improvements from newer version")
                else:
                    recommendations.append(f"Review {metric.replace('_', ' ')} degradation in newer version")
        
        # General recommendations
        if entry2.calculate_quality_score() > entry1.calculate_quality_score():
            recommendations.append("Overall quality has improved - consider adopting newer approach")
        elif entry2.calculate_quality_score() < entry1.calculate_quality_score():
            recommendations.append("Quality has declined - investigate recent changes")
        
        return recommendations
    
    def _calculate_confidence(self, entry1: HistoryEntry, entry2: HistoryEntry) -> float:
        """Calculate confidence in comparison result"""
        # Factors affecting confidence
        factors = []
        
        # Content length similarity
        len1, len2 = len(entry1.content), len(entry2.content)
        length_similarity = 1.0 - abs(len1 - len2) / max(len1, len2)
        factors.append(length_similarity)
        
        # Time difference (closer in time = higher confidence)
        time_diff = abs((entry2.timestamp - entry1.timestamp).total_seconds())
        time_confidence = max(0.0, 1.0 - time_diff / (30 * 24 * 3600))  # 30 days
        factors.append(time_confidence)
        
        # Model version similarity
        model_similarity = 1.0 if entry1.model_version == entry2.model_version else 0.5
        factors.append(model_similarity)
        
        return statistics.mean(factors)


class TrendAnalysisService:
    """Service for analyzing trends in model performance"""
    
    def analyze_trends(self, entries: List[HistoryEntry], metric: PerformanceMetric) -> TrendAnalysis:
        """Analyze trends in a list of entries for a specific metric"""
        if len(entries) < 2:
            raise ValueError("Need at least 2 entries for trend analysis")
        
        # Sort entries by timestamp
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        
        # Extract metric values
        values = self._extract_metric_values(sorted_entries, metric)
        timestamps = [e.timestamp for e in sorted_entries]
        
        # Calculate trend direction and strength
        trend_direction, trend_strength = self._calculate_trend(values)
        
        # Calculate confidence
        confidence = self._calculate_trend_confidence(values, timestamps)
        
        # Generate forecast
        forecast = self._generate_forecast(values, timestamps)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(values, timestamps)
        
        return TrendAnalysis.create(
            model_name=sorted_entries[0].model_version,
            metric=metric,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            confidence=confidence,
            forecast=forecast,
            anomalies=anomalies
        )
    
    def _extract_metric_values(self, entries: List[HistoryEntry], metric: PerformanceMetric) -> List[float]:
        """Extract metric values from entries"""
        values = []
        for entry in entries:
            if metric == PerformanceMetric.QUALITY_SCORE:
                values.append(entry.calculate_quality_score())
            elif metric == PerformanceMetric.READABILITY:
                values.append(entry.metrics.readability_score or 0.0)
            elif metric == PerformanceMetric.SENTIMENT:
                values.append(entry.metrics.sentiment_score or 0.0)
            elif metric == PerformanceMetric.COMPLEXITY:
                values.append(entry.metrics.complexity_score or 0.0)
            else:
                # Default to quality score
                values.append(entry.calculate_quality_score())
        
        return values
    
    def _calculate_trend(self, values: List[float]) -> Tuple[TrendDirection, float]:
        """Calculate trend direction and strength"""
        if len(values) < 2:
            return TrendDirection.STABLE, 0.0
        
        # Simple linear regression
        x = list(range(len(values)))
        n = len(values)
        
        # Calculate slope
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Very small slope
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.IMPROVING
        else:
            direction = TrendDirection.DECLINING
        
        # Calculate trend strength (R-squared)
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in values)
        ss_res = sum((values[i] - (slope * x[i] + (sum_y - slope * sum_x) / n)) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        strength = max(0.0, min(1.0, r_squared))
        
        return direction, strength
    
    def _calculate_trend_confidence(self, values: List[float], timestamps: List[datetime]) -> float:
        """Calculate confidence in trend analysis"""
        factors = []
        
        # Sample size factor
        sample_factor = min(1.0, len(values) / 10)  # More samples = higher confidence
        factors.append(sample_factor)
        
        # Time span factor
        if len(timestamps) > 1:
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            time_factor = min(1.0, time_span / (30 * 24 * 3600))  # 30 days
            factors.append(time_factor)
        
        # Variance factor (lower variance = higher confidence)
        if len(values) > 1:
            variance = statistics.variance(values)
            variance_factor = max(0.0, 1.0 - variance)
            factors.append(variance_factor)
        
        return statistics.mean(factors) if factors else 0.0
    
    def _generate_forecast(self, values: List[float], timestamps: List[datetime]) -> List[tuple]:
        """Generate simple forecast"""
        if len(values) < 2:
            return []
        
        # Simple linear extrapolation
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Generate forecast for next 7 days
        forecast = []
        last_timestamp = timestamps[-1]
        
        for i in range(1, 8):  # Next 7 days
            forecast_x = len(values) + i - 1
            forecast_value = slope * forecast_x + intercept
            forecast_timestamp = last_timestamp + timedelta(days=i)
            forecast.append((forecast_timestamp, max(0.0, min(1.0, forecast_value))))
        
        return forecast
    
    def _detect_anomalies(self, values: List[float], timestamps: List[datetime]) -> List[tuple]:
        """Detect anomalies in the data"""
        if len(values) < 3:
            return []
        
        anomalies = []
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_val == 0:
            return []
        
        # Z-score based anomaly detection
        for i, (value, timestamp) in enumerate(zip(values, timestamps)):
            z_score = abs(value - mean_val) / std_val
            if z_score > 2.0:  # 2 standard deviations
                anomalies.append((timestamp, value))
        
        return anomalies


class QualityAssessmentService:
    """Service for assessing and reporting on content quality"""
    
    def generate_quality_report(self, entries: List[HistoryEntry], 
                              report_type: str = "comprehensive") -> QualityReport:
        """Generate a comprehensive quality report"""
        if not entries:
            raise ValueError("No entries provided for quality report")
        
        # Calculate summary statistics
        summary = self._calculate_summary(entries)
        
        # Calculate average metrics
        average_metrics = self._calculate_average_metrics(entries)
        
        # Analyze trends
        trends = self._analyze_quality_trends(entries)
        
        # Identify outliers
        outliers = self._identify_outliers(entries)
        
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(entries, average_metrics, trends)
        
        # Determine time window
        timestamps = [e.timestamp for e in entries]
        time_window_start = min(timestamps)
        time_window_end = max(timestamps)
        
        return QualityReport.create(
            report_type=report_type,
            summary=summary,
            average_metrics=average_metrics,
            trends=trends,
            outliers=outliers,
            recommendations=recommendations,
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            total_entries=len(entries)
        )
    
    def _calculate_summary(self, entries: List[HistoryEntry]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        quality_scores = [e.calculate_quality_score() for e in entries]
        
        return {
            "total_entries": len(entries),
            "average_quality": statistics.mean(quality_scores),
            "quality_std": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "high_quality_count": sum(1 for score in quality_scores if score >= 0.7),
            "low_quality_count": sum(1 for score in quality_scores if score < 0.5),
            "model_versions": list(set(e.model_version for e in entries))
        }
    
    def _calculate_average_metrics(self, entries: List[HistoryEntry]) -> Dict[str, float]:
        """Calculate average metrics across all entries"""
        metrics = {}
        
        # Collect all metric values
        metric_values = {
            'readability_score': [],
            'sentiment_score': [],
            'complexity_score': [],
            'consistency_score': [],
            'quality_score': [],
            'coherence_score': [],
            'relevance_score': [],
            'creativity_score': []
        }
        
        for entry in entries:
            for metric_name in metric_values:
                value = getattr(entry.metrics, metric_name)
                if value is not None:
                    metric_values[metric_name].append(value)
        
        # Calculate averages
        for metric_name, values in metric_values.items():
            if values:
                metrics[metric_name] = statistics.mean(values)
        
        return metrics
    
    def _analyze_quality_trends(self, entries: List[HistoryEntry]) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        if len(entries) < 2:
            return {"trend": "insufficient_data"}
        
        # Sort by timestamp
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        
        # Calculate quality trend
        quality_scores = [e.calculate_quality_score() for e in sorted_entries]
        
        # Simple trend analysis
        first_half = quality_scores[:len(quality_scores)//2]
        second_half = quality_scores[len(quality_scores)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        trend_direction = "improving" if second_avg > first_avg else "declining" if second_avg < first_avg else "stable"
        trend_magnitude = abs(second_avg - first_avg)
        
        return {
            "trend": trend_direction,
            "magnitude": trend_magnitude,
            "first_half_avg": first_avg,
            "second_half_avg": second_avg,
            "overall_change": second_avg - first_avg
        }
    
    def _identify_outliers(self, entries: List[HistoryEntry]) -> List[Dict[str, Any]]:
        """Identify quality outliers"""
        quality_scores = [e.calculate_quality_score() for e in entries]
        
        if len(quality_scores) < 3:
            return []
        
        mean_quality = statistics.mean(quality_scores)
        std_quality = statistics.stdev(quality_scores)
        
        outliers = []
        for entry, score in zip(entries, quality_scores):
            z_score = abs(score - mean_quality) / std_quality
            if z_score > 2.0:  # 2 standard deviations
                outliers.append({
                    "entry_id": entry.id,
                    "quality_score": score,
                    "z_score": z_score,
                    "timestamp": entry.timestamp.isoformat(),
                    "model_version": entry.model_version
                })
        
        return outliers
    
    def _generate_quality_recommendations(self, entries: List[HistoryEntry], 
                                        average_metrics: Dict[str, float],
                                        trends: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        # Analyze average metrics
        for metric, value in average_metrics.items():
            if value < 0.6:  # Low performance
                recommendations.append(f"Improve {metric.replace('_', ' ')} - current average: {value:.2f}")
        
        # Analyze trends
        if trends.get("trend") == "declining":
            recommendations.append("Quality is declining over time - investigate recent changes")
        elif trends.get("trend") == "improving":
            recommendations.append("Quality is improving - continue current practices")
        
        # General recommendations
        high_quality_count = sum(1 for e in entries if e.calculate_quality_score() >= 0.7)
        if high_quality_count / len(entries) < 0.5:
            recommendations.append("Less than 50% of content meets high quality standards")
        
        return recommendations




