"""
Tests for AI History Analyzer

This module contains comprehensive tests for the core AI History Analyzer functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from ai_history_comparison.ai_history_analyzer import (
    AIHistoryAnalyzer, ComparisonType, MetricType,
    ContentMetrics, ComparisonResult, TrendAnalysis
)


class TestAIHistoryAnalyzer:
    """Test cases for AIHistoryAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer instance for each test"""
        return AIHistoryAnalyzer()
    
    @pytest.fixture
    def sample_content(self):
        """Sample content for testing"""
        return "This is a sample content for testing the AI history analyzer. It contains multiple sentences to test various metrics."
    
    @pytest.fixture
    def sample_content_2(self):
        """Second sample content for comparison testing"""
        return "This is another sample content for testing comparisons. It has different characteristics and metrics."
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.history_entries == []
        assert analyzer.metrics_cache == {}
        assert analyzer.sentiment_analyzer is not None
        assert analyzer.tfidf_vectorizer is not None
    
    def test_add_history_entry(self, analyzer, sample_content):
        """Test adding a history entry"""
        entry_id = analyzer.add_history_entry(
            content=sample_content,
            model_version="test-model-v1",
            metadata={"test": "data"}
        )
        
        assert entry_id is not None
        assert len(analyzer.history_entries) == 1
        
        entry = analyzer.history_entries[0]
        assert entry.content == sample_content
        assert entry.model_version == "test-model-v1"
        assert entry.metadata == {"test": "data"}
        assert entry.metrics is not None
        assert entry_id in analyzer.metrics_cache
    
    def test_compute_content_metrics(self, analyzer, sample_content):
        """Test content metrics computation"""
        metrics = analyzer._compute_content_metrics(sample_content)
        
        assert isinstance(metrics, ContentMetrics)
        assert metrics.readability_score is not None
        assert metrics.sentiment_score is not None
        assert metrics.word_count > 0
        assert metrics.sentence_count > 0
        assert metrics.avg_word_length > 0
        assert metrics.complexity_score >= 0
        assert metrics.topic_diversity >= 0
        assert metrics.consistency_score >= 0
        assert metrics.content_hash is not None
    
    def test_compute_similarity(self, analyzer, sample_content, sample_content_2):
        """Test content similarity computation"""
        similarity = analyzer._compute_similarity(sample_content, sample_content_2)
        
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
    
    def test_compare_entries(self, analyzer, sample_content, sample_content_2):
        """Test comparing two entries"""
        # Add two entries
        entry_id_1 = analyzer.add_history_entry(sample_content, "test-model-v1")
        entry_id_2 = analyzer.add_history_entry(sample_content_2, "test-model-v1")
        
        # Compare entries
        result = analyzer.compare_entries(
            entry_id_1, entry_id_2, [ComparisonType.CONTENT_SIMILARITY]
        )
        
        assert isinstance(result, ComparisonResult)
        assert result.similarity_score is not None
        assert result.quality_difference is not None
        assert result.trend_direction in ["improving", "declining", "stable"]
        assert isinstance(result.significant_changes, list)
        assert isinstance(result.recommendations, list)
        assert result.confidence_score is not None
    
    def test_analyze_trends(self, analyzer, sample_content):
        """Test trend analysis"""
        # Add multiple entries with different timestamps
        base_time = datetime.now()
        for i in range(5):
            entry_id = analyzer.add_history_entry(
                f"{sample_content} Entry {i}",
                "test-model-v1"
            )
            # Manually set timestamp to create a trend
            entry = analyzer._get_entry_by_id(entry_id)
            entry.timestamp = base_time + timedelta(hours=i)
        
        # Analyze trends
        trend = analyzer.analyze_trends(MetricType.READABILITY)
        
        assert isinstance(trend, TrendAnalysis)
        assert trend.metric_name == "readability"
        assert trend.trend_direction in ["increasing", "decreasing", "stable"]
        assert isinstance(trend.change_percentage, float)
        assert isinstance(trend.significance_level, float)
        assert len(trend.data_points) == 5
    
    def test_cluster_similar_content(self, analyzer):
        """Test content clustering"""
        # Add multiple entries
        contents = [
            "This is about technology and AI",
            "This is about artificial intelligence and machine learning",
            "This is about cooking and recipes",
            "This is about food and cooking techniques",
            "This is about sports and athletics",
            "This is about football and basketball"
        ]
        
        for content in contents:
            analyzer.add_history_entry(content, "test-model-v1")
        
        # Cluster content
        clusters = analyzer.cluster_similar_content(n_clusters=3)
        
        assert isinstance(clusters, dict)
        assert len(clusters) == 3
        assert all(isinstance(cluster_id, int) for cluster_id in clusters.keys())
        assert all(isinstance(entry_ids, list) for entry_ids in clusters.values())
    
    def test_generate_quality_report(self, analyzer, sample_content):
        """Test quality report generation"""
        # Add multiple entries
        for i in range(10):
            analyzer.add_history_entry(f"{sample_content} {i}", "test-model-v1")
        
        # Generate report
        report = analyzer.generate_quality_report()
        
        assert isinstance(report, dict)
        assert "summary" in report
        assert "average_metrics" in report
        assert "trends" in report
        assert "outliers" in report
        assert "recommendations" in report
        
        assert report["summary"]["total_entries"] == 10
        assert isinstance(report["average_metrics"], dict)
        assert isinstance(report["trends"], dict)
        assert isinstance(report["outliers"], list)
        assert isinstance(report["recommendations"], list)
    
    def test_get_entry_by_id(self, analyzer, sample_content):
        """Test getting entry by ID"""
        entry_id = analyzer.add_history_entry(sample_content, "test-model-v1")
        
        entry = analyzer._get_entry_by_id(entry_id)
        assert entry is not None
        assert entry.id == entry_id
        assert entry.content == sample_content
        
        # Test non-existent entry
        non_existent = analyzer._get_entry_by_id("non-existent-id")
        assert non_existent is None
    
    def test_get_entries_by_time_range(self, analyzer, sample_content):
        """Test getting entries by time range"""
        base_time = datetime.now()
        
        # Add entries with different timestamps
        entry_ids = []
        for i in range(5):
            entry_id = analyzer.add_history_entry(f"{sample_content} {i}", "test-model-v1")
            entry = analyzer._get_entry_by_id(entry_id)
            entry.timestamp = base_time + timedelta(hours=i)
            entry_ids.append(entry_id)
        
        # Get entries in time range
        start_time = base_time + timedelta(hours=1)
        end_time = base_time + timedelta(hours=3)
        
        entries = analyzer.get_entries_by_time_range(start_time, end_time)
        assert len(entries) == 3  # Should include entries at hours 1, 2, 3
    
    def test_get_entries_by_model_version(self, analyzer, sample_content):
        """Test getting entries by model version"""
        # Add entries with different model versions
        analyzer.add_history_entry(sample_content, "model-v1")
        analyzer.add_history_entry(sample_content, "model-v2")
        analyzer.add_history_entry(sample_content, "model-v1")
        
        # Get entries for specific model version
        v1_entries = analyzer.get_entries_by_model_version("model-v1")
        v2_entries = analyzer.get_entries_by_model_version("model-v2")
        
        assert len(v1_entries) == 2
        assert len(v2_entries) == 1
        assert all(entry.model_version == "model-v1" for entry in v1_entries)
        assert all(entry.model_version == "model-v2" for entry in v2_entries)
    
    def test_clear_history(self, analyzer, sample_content):
        """Test clearing history"""
        # Add some entries
        analyzer.add_history_entry(sample_content, "test-model-v1")
        analyzer.add_history_entry(sample_content, "test-model-v1")
        
        assert len(analyzer.history_entries) == 2
        assert len(analyzer.metrics_cache) == 2
        
        # Clear history
        analyzer.clear_history()
        
        assert len(analyzer.history_entries) == 0
        assert len(analyzer.metrics_cache) == 0
    
    def test_export_import_history(self, analyzer, sample_content):
        """Test exporting and importing history"""
        # Add some entries
        entry_id_1 = analyzer.add_history_entry(sample_content, "test-model-v1")
        entry_id_2 = analyzer.add_history_entry(sample_content, "test-model-v2")
        
        # Export history
        export_data = analyzer.export_history("json")
        assert isinstance(export_data, str)
        
        # Clear history
        analyzer.clear_history()
        assert len(analyzer.history_entries) == 0
        
        # Import history
        analyzer.import_history(export_data, "json")
        assert len(analyzer.history_entries) == 2
        
        # Verify entries were restored
        restored_entry_1 = analyzer._get_entry_by_id(entry_id_1)
        restored_entry_2 = analyzer._get_entry_by_id(entry_id_2)
        assert restored_entry_1 is not None
        assert restored_entry_2 is not None
    
    def test_compute_quality_difference(self, analyzer):
        """Test quality difference computation"""
        metrics1 = ContentMetrics(
            readability_score=50.0,
            sentiment_score=0.2,
            word_count=100,
            sentence_count=5,
            avg_word_length=5.0,
            complexity_score=0.5,
            topic_diversity=0.3,
            consistency_score=0.7,
            timestamp=datetime.now(),
            content_hash="hash1"
        )
        
        metrics2 = ContentMetrics(
            readability_score=60.0,
            sentiment_score=0.4,
            word_count=120,
            sentence_count=6,
            avg_word_length=5.2,
            complexity_score=0.6,
            topic_diversity=0.4,
            consistency_score=0.8,
            timestamp=datetime.now(),
            content_hash="hash2"
        )
        
        diff = analyzer._compute_quality_difference(metrics1, metrics2)
        
        assert isinstance(diff, dict)
        assert "readability" in diff
        assert "sentiment" in diff
        assert "word_count" in diff
        assert "complexity" in diff
        assert "topic_diversity" in diff
        
        assert diff["readability"] == 10.0  # 60 - 50
        assert diff["sentiment"] == 0.2     # 0.4 - 0.2
        assert diff["word_count"] == 20     # 120 - 100
    
    def test_determine_trend_direction(self, analyzer):
        """Test trend direction determination"""
        base_time = datetime.now()
        
        metrics1 = ContentMetrics(
            readability_score=50.0,
            sentiment_score=0.2,
            word_count=100,
            sentence_count=5,
            avg_word_length=5.0,
            complexity_score=0.5,
            topic_diversity=0.3,
            consistency_score=0.7,
            timestamp=base_time,
            content_hash="hash1"
        )
        
        metrics2 = ContentMetrics(
            readability_score=70.0,  # Improved readability
            sentiment_score=0.4,     # Improved sentiment
            word_count=100,
            sentence_count=5,
            avg_word_length=5.0,
            complexity_score=0.5,
            topic_diversity=0.3,
            consistency_score=0.7,
            timestamp=base_time + timedelta(hours=1),
            content_hash="hash2"
        )
        
        direction = analyzer._determine_trend_direction(
            metrics1, metrics2, metrics1.timestamp, metrics2.timestamp
        )
        
        assert direction in ["improving", "declining", "stable"]
    
    def test_identify_significant_changes(self, analyzer):
        """Test significant changes identification"""
        metrics1 = ContentMetrics(
            readability_score=50.0,
            sentiment_score=0.2,
            word_count=100,
            sentence_count=5,
            avg_word_length=5.0,
            complexity_score=0.5,
            topic_diversity=0.3,
            consistency_score=0.7,
            timestamp=datetime.now(),
            content_hash="hash1"
        )
        
        metrics2 = ContentMetrics(
            readability_score=30.0,  # Significant decrease
            sentiment_score=0.2,
            word_count=100,
            sentence_count=5,
            avg_word_length=5.0,
            complexity_score=0.5,
            topic_diversity=0.3,
            consistency_score=0.7,
            timestamp=datetime.now(),
            content_hash="hash2"
        )
        
        changes = analyzer._identify_significant_changes(
            metrics1, metrics2, [ComparisonType.QUALITY_METRICS]
        )
        
        assert isinstance(changes, list)
        # Should detect significant readability change
        assert any("readability" in change.lower() for change in changes)
    
    def test_generate_recommendations(self, analyzer):
        """Test recommendations generation"""
        metrics1 = ContentMetrics(
            readability_score=50.0,
            sentiment_score=0.2,
            word_count=100,
            sentence_count=5,
            avg_word_length=5.0,
            complexity_score=0.5,
            topic_diversity=0.3,
            consistency_score=0.7,
            timestamp=datetime.now(),
            content_hash="hash1"
        )
        
        metrics2 = ContentMetrics(
            readability_score=20.0,  # Very low readability
            sentiment_score=-0.6,    # Very negative sentiment
            word_count=50,           # Very short
            sentence_count=2,
            avg_word_length=5.0,
            complexity_score=0.5,
            topic_diversity=0.3,
            consistency_score=0.7,
            timestamp=datetime.now(),
            content_hash="hash2"
        )
        
        changes = ["Significant readability change detected"]
        recommendations = analyzer._generate_recommendations(metrics1, metrics2, changes)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should have recommendations for low readability, negative sentiment, and short length
        assert any("readability" in rec.lower() for rec in recommendations)
        assert any("negative" in rec.lower() for rec in recommendations)
        assert any("brief" in rec.lower() for rec in recommendations)
    
    def test_compute_confidence_score(self, analyzer, sample_content):
        """Test confidence score computation"""
        entry1 = analyzer.add_history_entry(sample_content, "test-model-v1")
        entry2 = analyzer.add_history_entry(sample_content, "test-model-v1")
        
        entry_obj_1 = analyzer._get_entry_by_id(entry1)
        entry_obj_2 = analyzer._get_entry_by_id(entry2)
        
        confidence = analyzer._compute_confidence_score(
            entry_obj_1, entry_obj_2, 0.8
        )
        
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_compute_trend_statistics(self, analyzer):
        """Test trend statistics computation"""
        base_time = datetime.now()
        data_points = [
            (base_time, 10.0),
            (base_time + timedelta(hours=1), 15.0),
            (base_time + timedelta(hours=2), 20.0),
            (base_time + timedelta(hours=3), 25.0),
            (base_time + timedelta(hours=4), 30.0)
        ]
        
        direction, change_percentage = analyzer._compute_trend_statistics(data_points)
        
        assert direction == "increasing"
        assert change_percentage == 200.0  # (30-10)/10 * 100
    
    def test_compute_significance_level(self, analyzer):
        """Test significance level computation"""
        # Test with consistent trend
        data_points = [
            (datetime.now(), 10.0),
            (datetime.now(), 12.0),
            (datetime.now(), 14.0),
            (datetime.now(), 16.0),
            (datetime.now(), 18.0)
        ]
        
        significance = analyzer._compute_significance_level(data_points)
        assert isinstance(significance, float)
        assert 0 <= significance <= 1
    
    def test_predict_future_value(self, analyzer):
        """Test future value prediction"""
        base_time = datetime.now()
        data_points = [
            (base_time, 10.0),
            (base_time + timedelta(hours=1), 12.0),
            (base_time + timedelta(hours=2), 14.0),
            (base_time + timedelta(hours=3), 16.0),
            (base_time + timedelta(hours=4), 18.0)
        ]
        
        prediction = analyzer._predict_future_value(data_points)
        assert isinstance(prediction, float)
        assert prediction > 18.0  # Should predict higher value based on trend
    
    def test_extract_metric_value(self, analyzer):
        """Test metric value extraction"""
        metrics = ContentMetrics(
            readability_score=50.0,
            sentiment_score=0.2,
            word_count=100,
            sentence_count=5,
            avg_word_length=5.0,
            complexity_score=0.5,
            topic_diversity=0.3,
            consistency_score=0.7,
            timestamp=datetime.now(),
            content_hash="hash1"
        )
        
        readability = analyzer._extract_metric_value(metrics, MetricType.READABILITY)
        sentiment = analyzer._extract_metric_value(metrics, MetricType.SENTIMENT)
        length = analyzer._extract_metric_value(metrics, MetricType.LENGTH)
        
        assert readability == 50.0
        assert sentiment == 0.2
        assert length == 100.0
    
    def test_compute_complexity_score(self, analyzer):
        """Test complexity score computation"""
        content = "This is a simple sentence. This is another simple sentence."
        words = content.split()
        sentences = content.split('.')
        
        complexity = analyzer._compute_complexity_score(content, words, sentences)
        
        assert isinstance(complexity, float)
        assert 0 <= complexity <= 1
    
    def test_compute_topic_diversity(self, analyzer):
        """Test topic diversity computation"""
        content = "This is about technology and artificial intelligence. It discusses machine learning and AI applications."
        
        diversity = analyzer._compute_topic_diversity(content)
        
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1
    
    def test_identify_outliers(self, analyzer, sample_content):
        """Test outlier identification"""
        # Add entries with varying metrics
        for i in range(10):
            content = f"{sample_content} {i}"
            entry_id = analyzer.add_history_entry(content, "test-model-v1")
            entry = analyzer._get_entry_by_id(entry_id)
            
            # Manually set some extreme values to create outliers
            if i == 0:
                entry.metrics.readability_score = 10.0  # Very low
            elif i == 1:
                entry.metrics.sentiment_score = -0.9   # Very negative
            else:
                entry.metrics.readability_score = 50.0 + i
                entry.metrics.sentiment_score = 0.1 + i * 0.05
        
        outliers = analyzer._identify_outliers(analyzer.history_entries)
        
        assert isinstance(outliers, list)
        # Should identify the extreme values as outliers
        assert len(outliers) >= 2
    
    def test_generate_quality_recommendations(self, analyzer, sample_content):
        """Test quality recommendations generation"""
        # Add entries with varying quality
        for i in range(5):
            content = f"{sample_content} {i}"
            entry_id = analyzer.add_history_entry(content, "test-model-v1")
            entry = analyzer._get_entry_by_id(entry_id)
            
            # Set some poor quality metrics
            entry.metrics.readability_score = 20.0  # Low readability
            entry.metrics.sentiment_score = -0.5    # Negative sentiment
            entry.metrics.word_count = 50           # Short content
        
        outliers = []  # No outliers for this test
        recommendations = analyzer._generate_quality_recommendations(
            analyzer.history_entries, outliers
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should have recommendations for low readability, negative sentiment, and short content
        assert any("readability" in rec.lower() for rec in recommendations)
        assert any("negative" in rec.lower() for rec in recommendations)
        assert any("brief" in rec.lower() for rec in recommendations)
    
    def test_generate_entry_id(self, analyzer, sample_content):
        """Test entry ID generation"""
        entry_id = analyzer._generate_entry_id(sample_content, "test-model-v1")
        
        assert isinstance(entry_id, str)
        assert "test-model-v1" in entry_id
        assert len(entry_id) > 0
    
    def test_get_entry_count(self, analyzer, sample_content):
        """Test getting entry count"""
        assert analyzer.get_entry_count() == 0
        
        analyzer.add_history_entry(sample_content, "test-model-v1")
        assert analyzer.get_entry_count() == 1
        
        analyzer.add_history_entry(sample_content, "test-model-v1")
        assert analyzer.get_entry_count() == 2


class TestComparisonTypes:
    """Test cases for ComparisonType enum"""
    
    def test_comparison_type_values(self):
        """Test ComparisonType enum values"""
        assert ComparisonType.CONTENT_SIMILARITY.value == "content_similarity"
        assert ComparisonType.QUALITY_METRICS.value == "quality_metrics"
        assert ComparisonType.PERFORMANCE_TRENDS.value == "performance_trends"
        assert ComparisonType.SENTIMENT_ANALYSIS.value == "sentiment_analysis"
        assert ComparisonType.TOPIC_EVOLUTION.value == "topic_evolution"
        assert ComparisonType.STYLE_CONSISTENCY.value == "style_consistency"
        assert ComparisonType.ERROR_PATTERNS.value == "error_patterns"
        assert ComparisonType.RESPONSE_TIME.value == "response_time"


class TestMetricTypes:
    """Test cases for MetricType enum"""
    
    def test_metric_type_values(self):
        """Test MetricType enum values"""
        assert MetricType.READABILITY.value == "readability"
        assert MetricType.SENTIMENT.value == "sentiment"
        assert MetricType.LENGTH.value == "length"
        assert MetricType.COMPLEXITY.value == "complexity"
        assert MetricType.TOPIC_DIVERSITY.value == "topic_diversity"
        assert MetricType.CONSISTENCY.value == "consistency"
        assert MetricType.ACCURACY.value == "accuracy"
        assert MetricType.RELEVANCE.value == "relevance"


class TestContentMetrics:
    """Test cases for ContentMetrics dataclass"""
    
    def test_content_metrics_creation(self):
        """Test ContentMetrics creation"""
        metrics = ContentMetrics(
            readability_score=50.0,
            sentiment_score=0.2,
            word_count=100,
            sentence_count=5,
            avg_word_length=5.0,
            complexity_score=0.5,
            topic_diversity=0.3,
            consistency_score=0.7,
            timestamp=datetime.now(),
            content_hash="test_hash"
        )
        
        assert metrics.readability_score == 50.0
        assert metrics.sentiment_score == 0.2
        assert metrics.word_count == 100
        assert metrics.sentence_count == 5
        assert metrics.avg_word_length == 5.0
        assert metrics.complexity_score == 0.5
        assert metrics.topic_diversity == 0.3
        assert metrics.consistency_score == 0.7
        assert metrics.content_hash == "test_hash"


class TestComparisonResult:
    """Test cases for ComparisonResult dataclass"""
    
    def test_comparison_result_creation(self):
        """Test ComparisonResult creation"""
        result = ComparisonResult(
            similarity_score=0.8,
            quality_difference={"readability": 10.0, "sentiment": 0.2},
            trend_direction="improving",
            significant_changes=["Readability improved"],
            recommendations=["Continue current approach"],
            confidence_score=0.9
        )
        
        assert result.similarity_score == 0.8
        assert result.quality_difference == {"readability": 10.0, "sentiment": 0.2}
        assert result.trend_direction == "improving"
        assert result.significant_changes == ["Readability improved"]
        assert result.recommendations == ["Continue current approach"]
        assert result.confidence_score == 0.9


class TestTrendAnalysis:
    """Test cases for TrendAnalysis dataclass"""
    
    def test_trend_analysis_creation(self):
        """Test TrendAnalysis creation"""
        data_points = [(datetime.now(), 10.0), (datetime.now(), 12.0)]
        
        trend = TrendAnalysis(
            metric_name="readability",
            trend_direction="increasing",
            change_percentage=20.0,
            significance_level=0.8,
            data_points=data_points,
            prediction=14.0
        )
        
        assert trend.metric_name == "readability"
        assert trend.trend_direction == "increasing"
        assert trend.change_percentage == 20.0
        assert trend.significance_level == 0.8
        assert trend.data_points == data_points
        assert trend.prediction == 14.0


# Integration tests
class TestAnalyzerIntegration:
    """Integration tests for the analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer for integration tests"""
        return AIHistoryAnalyzer()
    
    def test_full_workflow(self, analyzer):
        """Test complete workflow from content analysis to reporting"""
        # Step 1: Add multiple content pieces
        contents = [
            "This is a simple, clear sentence about technology.",
            "This is a more complex sentence with advanced vocabulary and technical terminology.",
            "This is another simple sentence about artificial intelligence.",
            "This is a very complex sentence with multiple clauses, technical jargon, and sophisticated language structures."
        ]
        
        entry_ids = []
        for i, content in enumerate(contents):
            entry_id = analyzer.add_history_entry(content, f"model-v{i+1}")
            entry_ids.append(entry_id)
        
        assert len(analyzer.history_entries) == 4
        
        # Step 2: Compare entries
        comparison = analyzer.compare_entries(
            entry_ids[0], entry_ids[1], [ComparisonType.CONTENT_SIMILARITY]
        )
        assert comparison.similarity_score is not None
        
        # Step 3: Analyze trends
        trend = analyzer.analyze_trends(MetricType.READABILITY)
        assert trend.metric_name == "readability"
        
        # Step 4: Cluster content
        clusters = analyzer.cluster_similar_content(n_clusters=2)
        assert len(clusters) == 2
        
        # Step 5: Generate quality report
        report = analyzer.generate_quality_report()
        assert "summary" in report
        assert report["summary"]["total_entries"] == 4
        
        # Step 6: Export and import
        export_data = analyzer.export_history("json")
        analyzer.clear_history()
        analyzer.import_history(export_data, "json")
        assert len(analyzer.history_entries) == 4
    
    def test_error_handling(self, analyzer):
        """Test error handling in various scenarios"""
        # Test comparing non-existent entries
        with pytest.raises(ValueError):
            analyzer.compare_entries("non-existent-1", "non-existent-2", [])
        
        # Test trend analysis with insufficient data
        with pytest.raises(ValueError):
            analyzer.analyze_trends(MetricType.READABILITY)
        
        # Test clustering with insufficient data
        with pytest.raises(ValueError):
            analyzer.cluster_similar_content(n_clusters=5)
        
        # Test quality report with no data
        report = analyzer.generate_quality_report()
        assert "error" in report



























