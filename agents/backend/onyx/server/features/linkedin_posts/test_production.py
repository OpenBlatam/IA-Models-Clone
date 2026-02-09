"""
🧪 LinkedIn Posts Optimization - Production Test Suite
====================================================

Comprehensive testing for production-ready LinkedIn optimizer
"""

import pytest
import asyncio
import torch
from unittest.mock import Mock, patch, AsyncMock
from linkedin_optimizer_production import (
    ContentType,
    OptimizationStrategy,
    ContentData,
    ContentMetrics,
    OptimizationResult,
    TransformerContentAnalyzer,
    MLContentOptimizer,
    MLEngagementPredictor,
    LinkedInOptimizationService,
    create_linkedin_optimization_service
)

# Test data
SAMPLE_CONTENT = """
Just finished an amazing project using React and TypeScript! 
The development experience was incredible, and the final result exceeded expectations.
#react #typescript #webdevelopment
"""

@pytest.fixture
def sample_content_data():
    """Create sample content data for testing."""
    return ContentData(
        id="test123",
        content=SAMPLE_CONTENT,
        content_type=ContentType.POST,
        hashtags=["#react", "#typescript"],
        mentions=["@user1"],
        links=["https://example.com"],
        media_urls=["image1.jpg"]
    )

@pytest.fixture
def mock_analyzer():
    """Create mock content analyzer."""
    analyzer = Mock(spec=TransformerContentAnalyzer)
    analyzer.analyze_content = AsyncMock(return_value={
        "content_length": 150,
        "hashtag_count": 3,
        "mention_count": 1,
        "link_count": 1,
        "media_count": 1,
        "sentiment_score": 0.8,
        "sentiment_label": "LABEL_2",
        "sentiment_confidence": 0.9,
        "industry": "technology and software",
        "readability_score": 75.0,
        "complexity_score": 60.0,
        "engagement_potential": 25,
        "optimal_posting_time": "Tuesday-Thursday, 9-11 AM or 2-4 PM",
        "recommended_hashtags": ["#technology", "#innovation", "#ai"],
        "content_improvements": ["Add engaging elements"]
    })
    return analyzer

@pytest.fixture
def mock_optimizer(mock_analyzer):
    """Create mock content optimizer."""
    optimizer = Mock(spec=MLContentOptimizer)
    optimizer.analyzer = mock_analyzer
    optimizer.optimize_content = AsyncMock(return_value=OptimizationResult(
        original_content=sample_content_data(),
        optimized_content=sample_content_data(),
        optimization_score=85.0,
        improvements=["Added hashtags", "Enhanced content"],
        predicted_engagement_increase=1.2,
        confidence_score=0.9
    ))
    return optimizer

@pytest.fixture
def mock_predictor(mock_analyzer):
    """Create mock engagement predictor."""
    predictor = Mock(spec=MLEngagementPredictor)
    predictor.analyzer = mock_analyzer
    predictor.predict_engagement = AsyncMock(return_value=8.5)
    return predictor

class TestContentData:
    """Test ContentData class."""
    
    def test_content_data_creation(self):
        """Test ContentData creation with default values."""
        content = ContentData(
            id="test",
            content="Test content",
            content_type=ContentType.POST
        )
        
        assert content.id == "test"
        assert content.content == "Test content"
        assert content.content_type == ContentType.POST
        assert content.hashtags == []
        assert content.metrics is not None
    
    def test_content_data_with_hashtags(self):
        """Test ContentData with hashtags."""
        content = ContentData(
            id="test",
            content="Test content",
            content_type=ContentType.POST,
            hashtags=["#test", "#example"]
        )
        
        assert len(content.hashtags) == 2
        assert "#test" in content.hashtags

class TestContentMetrics:
    """Test ContentMetrics class."""
    
    def test_engagement_rate_calculation(self):
        """Test engagement rate calculation."""
        metrics = ContentMetrics(
            views=1000,
            likes=50,
            shares=20,
            comments=10,
            clicks=15
        )
        
        engagement_rate = metrics.calculate_engagement_rate()
        expected_rate = (50 + 20 + 10 + 15) / 1000 * 100
        
        assert engagement_rate == expected_rate
        assert metrics.engagement_rate == expected_rate
    
    def test_engagement_rate_zero_views(self):
        """Test engagement rate with zero views."""
        metrics = ContentMetrics(
            views=0,
            likes=10,
            shares=5
        )
        
        engagement_rate = metrics.calculate_engagement_rate()
        assert engagement_rate == 0.0

class TestOptimizationStrategy:
    """Test OptimizationStrategy enum."""
    
    def test_strategy_values(self):
        """Test optimization strategy values."""
        assert OptimizationStrategy.ENGAGEMENT.value == "engagement"
        assert OptimizationStrategy.REACH.value == "reach"
        assert OptimizationStrategy.CLICKS.value == "clicks"
        assert OptimizationStrategy.SHARES.value == "shares"
        assert OptimizationStrategy.COMMENTS.value == "comments"

@pytest.mark.asyncio
class TestTransformerContentAnalyzer:
    """Test TransformerContentAnalyzer class."""
    
    @patch('linkedin_optimizer_production.AutoTokenizer')
    @patch('linkedin_optimizer_production.AutoModel')
    @patch('linkedin_optimizer_production.pipeline')
    def test_analyzer_initialization(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test analyzer initialization."""
        mock_pipeline.return_value = Mock()
        
        analyzer = TransformerContentAnalyzer()
        
        assert analyzer.tokenizer is not None
        assert analyzer.model is not None
        assert analyzer.sentiment_analyzer is not None
        assert analyzer.text_classifier is not None
    
    def test_extract_hashtags(self):
        """Test hashtag extraction."""
        analyzer = TransformerContentAnalyzer()
        content = "This is a #test post with #hashtags #example"
        
        hashtags = analyzer._extract_hashtags(content)
        
        assert len(hashtags) == 3
        assert "#test" in hashtags
        assert "#hashtags" in hashtags
        assert "#example" in hashtags
    
    def test_normalize_sentiment(self):
        """Test sentiment normalization."""
        analyzer = TransformerContentAnalyzer()
        
        # Test negative sentiment
        assert analyzer._normalize_sentiment("LABEL_0", 0.8) == -0.8
        
        # Test neutral sentiment
        assert analyzer._normalize_sentiment("LABEL_1", 0.9) == 0.0
        
        # Test positive sentiment
        assert analyzer._normalize_sentiment("LABEL_2", 0.7) == 0.7
    
    def test_calculate_content_quality(self):
        """Test content quality calculation."""
        analyzer = TransformerContentAnalyzer()
        content = "This is a test sentence. It has multiple sentences. For testing purposes."
        
        quality = analyzer._calculate_content_quality(content)
        
        assert "readability" in quality
        assert "complexity" in quality
        assert "engagement_potential" in quality
        assert "improvements" in quality
        assert isinstance(quality["readability"], float)
        assert isinstance(quality["complexity"], float)
    
    def test_get_optimal_posting_time(self):
        """Test optimal posting time recommendations."""
        analyzer = TransformerContentAnalyzer()
        
        tech_time = analyzer._get_optimal_posting_time("technology and software")
        business_time = analyzer._get_optimal_posting_time("business and leadership")
        
        assert "Tuesday-Thursday" in tech_time
        assert "Monday-Friday" in business_time
    
    def test_get_recommended_hashtags(self):
        """Test hashtag recommendations."""
        analyzer = TransformerContentAnalyzer()
        
        tech_hashtags = analyzer._get_recommended_hashtags("technology and software", "tech content")
        business_hashtags = analyzer._get_recommended_hashtags("business and leadership", "business content")
        
        assert "#technology" in tech_hashtags
        assert "#business" in business_hashtags
        assert "#linkedin" in tech_hashtags  # Trending hashtags should be included

@pytest.mark.asyncio
class TestMLContentOptimizer:
    """Test MLContentOptimizer class."""
    
    def test_optimizer_initialization(self, mock_analyzer):
        """Test optimizer initialization."""
        optimizer = MLContentOptimizer(mock_analyzer)
        
        assert optimizer.analyzer == mock_analyzer
        assert optimizer.optimization_model is not None
        assert optimizer.scaler is not None
    
    @patch('linkedin_optimizer_production.joblib.load')
    def test_load_existing_model(self, mock_load, mock_analyzer):
        """Test loading existing optimization model."""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        with patch('os.path.exists', return_value=True):
            optimizer = MLContentOptimizer(mock_analyzer)
            assert optimizer.optimization_model == mock_model
    
    def test_calculate_basic_score(self, mock_analyzer):
        """Test basic optimization score calculation."""
        optimizer = MLContentOptimizer(mock_analyzer)
        
        original = ContentData(id="orig", content="Short", content_type=ContentType.POST)
        optimized = ContentData(
            id="opt", 
            content="Longer content with\n\nstructure", 
            content_type=ContentType.POST,
            hashtags=["#test", "#example"]
        )
        
        score = optimizer._calculate_basic_score(original, optimized)
        
        # Should get points for: hashtags (+20), content length (+30), structure (+25), call-to-action (+25)
        assert score == 100.0
    
    def test_predict_engagement_increase(self, mock_analyzer):
        """Test engagement increase prediction."""
        optimizer = MLContentOptimizer(mock_analyzer)
        
        # Test different score ranges
        assert optimizer._predict_engagement_increase(20) == 0.2
        assert optimizer._predict_engagement_increase(50) == 0.7
        assert optimizer._predict_engagement_increase(90) == 1.7
    
    def test_calculate_confidence(self, mock_analyzer):
        """Test confidence score calculation."""
        optimizer = MLContentOptimizer(mock_analyzer)
        
        analysis = {
            "sentiment_confidence": 0.8,
            "readability_score": 70,
            "engagement_potential": 30
        }
        
        confidence = optimizer._calculate_confidence(analysis, 80)
        
        # Base: 0.5 + sentiment: 0.16 + readability: 0.1 + engagement: 0.1 + optimization: 0.1
        expected = 0.5 + 0.8 * 0.2 + 0.1 + 0.1 + 0.1
        assert abs(confidence - expected) < 0.01

@pytest.mark.asyncio
class TestMLEngagementPredictor:
    """Test MLEngagementPredictor class."""
    
    def test_predictor_initialization(self, mock_analyzer):
        """Test predictor initialization."""
        predictor = MLEngagementPredictor(mock_analyzer)
        
        assert predictor.analyzer == mock_analyzer
        assert predictor.prediction_model is not None
    
    def test_create_prediction_model(self, mock_analyzer):
        """Test PyTorch model creation."""
        predictor = MLEngagementPredictor(mock_analyzer)
        
        model = predictor.prediction_model
        assert isinstance(model, torch.nn.Module)
        
        # Test model forward pass
        input_tensor = torch.randn(1, 15)
        output = model(input_tensor)
        
        assert output.shape == (1, 1)
        assert output.item() >= 0 and output.item() <= 15.0
    
    def test_extract_features(self, mock_analyzer):
        """Test feature extraction."""
        predictor = MLEngagementPredictor(mock_analyzer)
        
        content = ContentData(
            id="test",
            content="Test content" * 50,  # 600 characters
            content_type=ContentType.VIDEO,
            hashtags=["#test", "#example"],
            mentions=["@user"],
            links=["https://example.com"],
            media_urls=["image.jpg"]
        )
        
        analysis = {
            "sentiment_score": 0.5,
            "readability_score": 75,
            "complexity_score": 60,
            "engagement_potential": 25
        }
        
        features = predictor._extract_features(content, analysis)
        
        assert len(features) == 15
        assert features[0] == 0.6  # Normalized content length
        assert features[1] == 0.2  # Normalized hashtag count
        assert features[9] == 1.0  # Video content flag
        assert features[10] == 0.0  # Image content flag

@pytest.mark.asyncio
class TestLinkedInOptimizationService:
    """Test LinkedInOptimizationService class."""
    
    def test_service_initialization(self):
        """Test service initialization."""
        service = LinkedInOptimizationService()
        
        assert service.analyzer is not None
        assert service.optimizer is not None
        assert service.predictor is not None
        assert service.request_count == 0
        assert service.avg_response_time == 0.0
    
    async def test_optimize_linkedin_post(self, mock_optimizer, mock_analyzer):
        """Test LinkedIn post optimization."""
        service = LinkedInOptimizationService()
        service.optimizer = mock_optimizer
        
        result = await service.optimize_linkedin_post(SAMPLE_CONTENT, OptimizationStrategy.ENGAGEMENT)
        
        assert isinstance(result, OptimizationResult)
        assert result.optimization_score == 85.0
        assert result.confidence_score == 0.9
        assert service.request_count == 1
        assert service.avg_response_time > 0
    
    async def test_predict_post_engagement(self, mock_predictor, mock_analyzer):
        """Test engagement prediction."""
        service = LinkedInOptimizationService()
        service.predictor = mock_predictor
        
        engagement = await service.predict_post_engagement(SAMPLE_CONTENT)
        
        assert engagement == 8.5
    
    async def test_get_content_insights(self, mock_analyzer):
        """Test content insights retrieval."""
        service = LinkedInOptimizationService()
        service.analyzer = mock_analyzer
        
        insights = await service.get_content_insights(SAMPLE_CONTENT)
        
        assert "industry" in insights
        assert "sentiment_score" in insights
        assert "readability_score" in insights
    
    def test_performance_metrics_update(self):
        """Test performance metrics updating."""
        service = LinkedInOptimizationService()
        
        # First request
        service._update_performance_metrics(1.0)
        assert service.avg_response_time == 1.0
        
        # Second request
        service._update_performance_metrics(2.0)
        assert service.avg_response_time == 1.5
    
    def test_get_performance_stats(self):
        """Test performance statistics retrieval."""
        service = LinkedInOptimizationService()
        service.request_count = 5
        service.avg_response_time = 1.2
        
        stats = service.get_performance_stats()
        
        assert stats["total_requests"] == 5
        assert stats["average_response_time"] == 1.2
        assert "device" in stats
        assert "gpu_available" in stats

class TestFactoryFunction:
    """Test factory function."""
    
    def test_create_service(self):
        """Test service creation via factory function."""
        service = create_linkedin_optimization_service()
        
        assert isinstance(service, LinkedInOptimizationService)
        assert service.analyzer is not None
        assert service.optimizer is not None
        assert service.predictor is not None

@pytest.mark.asyncio
class TestIntegration:
    """Integration tests."""
    
    async def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        service = create_linkedin_optimization_service()
        
        # Test with real content
        result = await service.optimize_linkedin_post(
            "Just finished a great project! #coding #development",
            OptimizationStrategy.ENGAGEMENT
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.original_content is not None
        assert result.optimized_content is not None
        assert result.optimization_score >= 0
        assert result.confidence_score >= 0
    
    async def test_error_handling(self):
        """Test error handling in the service."""
        service = create_linkedin_optimization_service()
        
        # Test with empty content
        with pytest.raises(Exception):
            await service.optimize_linkedin_post("", OptimizationStrategy.ENGAGEMENT)

# Performance tests
@pytest.mark.benchmark
class TestPerformance:
    """Performance tests."""
    
    @pytest.mark.asyncio
    async def test_optimization_performance(self, benchmark):
        """Benchmark optimization performance."""
        service = create_linkedin_optimization_service()
        
        async def optimize_content():
            return await service.optimize_linkedin_post(SAMPLE_CONTENT, OptimizationStrategy.ENGAGEMENT)
        
        result = benchmark(optimize_content)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_engagement_prediction_performance(self, benchmark):
        """Benchmark engagement prediction performance."""
        service = create_linkedin_optimization_service()
        
        async def predict_engagement():
            return await service.predict_post_engagement(SAMPLE_CONTENT)
        
        result = benchmark(predict_engagement)
        assert result >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])






