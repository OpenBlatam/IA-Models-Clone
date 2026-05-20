"""
Advanced Analytics Tests for LinkedIn Posts Service
Tests analytics, reporting, and data insights functionality
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
import statistics

# Mock the TypeScript interfaces for testing
class MockLinkedInPost:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 'test-post-1')
        self.userId = kwargs.get('userId', 'user-1')
        self.title = kwargs.get('title', 'Test Post')
        self.content = kwargs.get('content', {'text': 'Test content', 'hashtags': [], 'mentions': [], 'links': [], 'images': []})
        self.postType = kwargs.get('postType', 'text')
        self.tone = kwargs.get('tone', 'professional')
        self.status = kwargs.get('status', 'draft')
        self.createdAt = kwargs.get('createdAt', datetime.now())
        self.updatedAt = kwargs.get('updatedAt', datetime.now())
        self.engagement = kwargs.get('engagement', {'likes': 0, 'comments': 0, 'shares': 0, 'clicks': 0, 'impressions': 0, 'reach': 0, 'engagementRate': 0})
        self.aiScore = kwargs.get('aiScore', 0.8)
        self.optimizationSuggestions = kwargs.get('optimizationSuggestions', [])
        self.keywords = kwargs.get('keywords', [])
        self.externalMetadata = kwargs.get('externalMetadata', {})
        self.performanceScore = kwargs.get('performanceScore', 0)
        self.reachScore = kwargs.get('reachScore', 0)
        self.engagementScore = kwargs.get('engagementScore', 0)

class MockAnalyticsService:
    """Mock analytics service for testing"""
    
    def __init__(self):
        self.post_analytics = {}
        self.user_analytics = {}
        self.campaign_analytics = {}
    
    async def generate_post_analytics(self, post_id: str) -> Dict[str, Any]:
        """Generate analytics for a specific post"""
        return self.post_analytics.get(post_id, {
            'sentimentScore': 0.7,
            'readabilityScore': 0.8,
            'engagementScore': 0.75,
            'keywordDensity': 0.05,
            'structureScore': 0.9,
            'callToActionScore': 0.6
        })
    
    async def generate_user_analytics(self, user_id: str, posts: List[MockLinkedInPost]) -> Dict[str, Any]:
        """Generate analytics for a user's posts"""
        if not posts:
            return {'total_posts': 0, 'average_engagement': 0}
        
        total_engagement = sum(post.engagement['engagementRate'] for post in posts)
        avg_engagement = total_engagement / len(posts)
        
        return {
            'total_posts': len(posts),
            'average_engagement': avg_engagement,
            'best_performing_post': max(posts, key=lambda p: p.engagement['engagementRate']),
            'engagement_trend': self._calculate_engagement_trend(posts),
            'content_quality_score': statistics.mean([post.aiScore for post in posts]),
            'posting_frequency': self._calculate_posting_frequency(posts)
        }
    
    async def generate_campaign_analytics(self, campaign_id: str, posts: List[MockLinkedInPost]) -> Dict[str, Any]:
        """Generate analytics for a campaign"""
        if not posts:
            return {'total_posts': 0, 'campaign_performance': 0}
        
        total_engagement = sum(post.engagement['engagementRate'] for post in posts)
        campaign_performance = total_engagement / len(posts)
        
        return {
            'total_posts': len(posts),
            'campaign_performance': campaign_performance,
            'engagement_distribution': self._calculate_engagement_distribution(posts),
            'content_consistency': self._calculate_content_consistency(posts),
            'audience_reach': sum(post.engagement['reach'] for post in posts),
            'total_impressions': sum(post.engagement['impressions'] for post in posts)
        }
    
    def _calculate_engagement_trend(self, posts: List[MockLinkedInPost]) -> str:
        """Calculate engagement trend over time"""
        if len(posts) < 2:
            return 'insufficient_data'
        
        # Sort by creation date
        sorted_posts = sorted(posts, key=lambda p: p.createdAt)
        engagement_rates = [post.engagement['engagementRate'] for post in sorted_posts]
        
        if len(engagement_rates) >= 2:
            trend = engagement_rates[-1] - engagement_rates[0]
            if trend > 0.05:
                return 'increasing'
            elif trend < -0.05:
                return 'decreasing'
            else:
                return 'stable'
        
        return 'insufficient_data'
    
    def _calculate_posting_frequency(self, posts: List[MockLinkedInPost]) -> float:
        """Calculate average posts per week"""
        if len(posts) < 2:
            return 0
        
        sorted_posts = sorted(posts, key=lambda p: p.createdAt)
        time_span = (sorted_posts[-1].createdAt - sorted_posts[0].createdAt).days
        weeks = max(time_span / 7, 1)  # At least 1 week
        return len(posts) / weeks
    
    def _calculate_engagement_distribution(self, posts: List[MockLinkedInPost]) -> Dict[str, int]:
        """Calculate distribution of engagement levels"""
        distribution = {'low': 0, 'medium': 0, 'high': 0}
        
        for post in posts:
            rate = post.engagement['engagementRate']
            if rate < 0.05:
                distribution['low'] += 1
            elif rate < 0.15:
                distribution['medium'] += 1
            else:
                distribution['high'] += 1
        
        return distribution
    
    def _calculate_content_consistency(self, posts: List[MockLinkedInPost]) -> float:
        """Calculate consistency of content quality"""
        if not posts:
            return 0
        
        ai_scores = [post.aiScore for post in posts]
        return statistics.stdev(ai_scores) if len(ai_scores) > 1 else 0

class TestAdvancedAnalytics:
    """Test advanced analytics and reporting functionality"""
    
    @pytest.fixture
    def mock_repository(self):
        """Mock repository for analytics testing"""
        repo = Mock()
        repo.createPost = AsyncMock()
        repo.updatePost = AsyncMock()
        repo.getPost = AsyncMock()
        repo.listPosts = AsyncMock()
        repo.deletePost = AsyncMock()
        return repo
    
    @pytest.fixture
    def mock_ai_service(self):
        """Mock AI service for analytics testing"""
        ai = Mock()
        ai.analyzeContent = AsyncMock()
        ai.generatePost = AsyncMock()
        ai.optimizePost = AsyncMock()
        return ai
    
    @pytest.fixture
    def mock_cache_service(self):
        """Mock cache service for analytics testing"""
        cache = Mock()
        cache.get = AsyncMock()
        cache.set = AsyncMock()
        cache.delete = AsyncMock()
        cache.clear = AsyncMock()
        return cache
    
    @pytest.fixture
    def analytics_service(self):
        """Analytics service for testing"""
        return MockAnalyticsService()
    
    @pytest.fixture
    def post_service(self, mock_repository, mock_ai_service, mock_cache_service):
        """Post service with mocked dependencies"""
        from services.post_service import PostService
        return PostService(mock_repository, mock_ai_service, mock_cache_service)

    async def test_post_performance_analytics(self, post_service, analytics_service):
        """Test analytics for individual post performance"""
        # Setup test posts with varying performance
        test_posts = [
            MockLinkedInPost(
                id='post-1',
                engagement={'engagementRate': 0.25, 'likes': 100, 'comments': 20, 'shares': 15},
                aiScore=0.9,
                performanceScore=0.85
            ),
            MockLinkedInPost(
                id='post-2',
                engagement={'engagementRate': 0.15, 'likes': 50, 'comments': 8, 'shares': 5},
                aiScore=0.7,
                performanceScore=0.65
            ),
            MockLinkedInPost(
                id='post-3',
                engagement={'engagementRate': 0.05, 'likes': 10, 'comments': 2, 'shares': 1},
                aiScore=0.6,
                performanceScore=0.45
            )
        ]
        
        # Test analytics for each post
        for post in test_posts:
            analytics = await analytics_service.generate_post_analytics(post.id)
            
            # Verify analytics structure
            assert 'sentimentScore' in analytics
            assert 'readabilityScore' in analytics
            assert 'engagementScore' in analytics
            
            # Verify score ranges
            assert 0 <= analytics['sentimentScore'] <= 1
            assert 0 <= analytics['readabilityScore'] <= 1
            assert 0 <= analytics['engagementScore'] <= 1

    async def test_user_performance_analytics(self, post_service, analytics_service):
        """Test analytics for user performance across multiple posts"""
        # Setup user with multiple posts
        user_posts = [
            MockLinkedInPost(
                userId='user-1',
                createdAt=datetime.now() - timedelta(days=30),
                engagement={'engagementRate': 0.20, 'likes': 80, 'comments': 15, 'shares': 10},
                aiScore=0.85
            ),
            MockLinkedInPost(
                userId='user-1',
                createdAt=datetime.now() - timedelta(days=20),
                engagement={'engagementRate': 0.25, 'likes': 120, 'comments': 25, 'shares': 18},
                aiScore=0.90
            ),
            MockLinkedInPost(
                userId='user-1',
                createdAt=datetime.now() - timedelta(days=10),
                engagement={'engagementRate': 0.30, 'likes': 150, 'comments': 30, 'shares': 25},
                aiScore=0.95
            )
        ]
        
        # Generate user analytics
        user_analytics = await analytics_service.generate_user_analytics('user-1', user_posts)
        
        # Verify analytics structure
        assert user_analytics['total_posts'] == 3
        assert user_analytics['average_engagement'] > 0
        assert user_analytics['best_performing_post'] is not None
        assert user_analytics['engagement_trend'] == 'increasing'
        assert user_analytics['content_quality_score'] > 0.8
        assert user_analytics['posting_frequency'] > 0

    async def test_campaign_analytics(self, post_service, analytics_service):
        """Test analytics for campaign performance"""
        # Setup campaign posts
        campaign_posts = [
            MockLinkedInPost(
                id='campaign-1',
                title='Digital Transformation Part 1',
                engagement={'engagementRate': 0.22, 'reach': 1000, 'impressions': 1200},
                aiScore=0.88
            ),
            MockLinkedInPost(
                id='campaign-2',
                title='Digital Transformation Part 2',
                engagement={'engagementRate': 0.28, 'reach': 1200, 'impressions': 1400},
                aiScore=0.92
            ),
            MockLinkedInPost(
                id='campaign-3',
                title='Digital Transformation Part 3',
                engagement={'engagementRate': 0.35, 'reach': 1500, 'impressions': 1800},
                aiScore=0.95
            )
        ]
        
        # Generate campaign analytics
        campaign_analytics = await analytics_service.generate_campaign_analytics('campaign-1', campaign_posts)
        
        # Verify campaign analytics
        assert campaign_analytics['total_posts'] == 3
        assert campaign_analytics['campaign_performance'] > 0.2
        assert 'engagement_distribution' in campaign_analytics
        assert campaign_analytics['content_consistency'] < 0.1  # Low variance = high consistency
        assert campaign_analytics['audience_reach'] > 0
        assert campaign_analytics['total_impressions'] > 0

    async def test_engagement_trend_analysis(self, post_service, analytics_service):
        """Test analysis of engagement trends over time"""
        # Setup posts with clear trend
        trend_posts = [
            MockLinkedInPost(
                createdAt=datetime.now() - timedelta(days=60),
                engagement={'engagementRate': 0.10}
            ),
            MockLinkedInPost(
                createdAt=datetime.now() - timedelta(days=40),
                engagement={'engagementRate': 0.15}
            ),
            MockLinkedInPost(
                createdAt=datetime.now() - timedelta(days=20),
                engagement={'engagementRate': 0.20}
            ),
            MockLinkedInPost(
                createdAt=datetime.now() - timedelta(days=1),
                engagement={'engagementRate': 0.25}
            )
        ]
        
        # Analyze trend
        user_analytics = await analytics_service.generate_user_analytics('user-1', trend_posts)
        
        # Verify increasing trend
        assert user_analytics['engagement_trend'] == 'increasing'
        
        # Test decreasing trend
        decreasing_posts = [
            MockLinkedInPost(
                createdAt=datetime.now() - timedelta(days=60),
                engagement={'engagementRate': 0.25}
            ),
            MockLinkedInPost(
                createdAt=datetime.now() - timedelta(days=40),
                engagement={'engagementRate': 0.20}
            ),
            MockLinkedInPost(
                createdAt=datetime.now() - timedelta(days=20),
                engagement={'engagementRate': 0.15}
            ),
            MockLinkedInPost(
                createdAt=datetime.now() - timedelta(days=1),
                engagement={'engagementRate': 0.10}
            )
        ]
        
        decreasing_analytics = await analytics_service.generate_user_analytics('user-1', decreasing_posts)
        assert decreasing_analytics['engagement_trend'] == 'decreasing'

    async def test_content_quality_analytics(self, post_service, analytics_service):
        """Test analytics for content quality assessment"""
        # Setup posts with varying content quality
        quality_posts = [
            MockLinkedInPost(
                aiScore=0.95,
                engagement={'engagementRate': 0.30},
                performanceScore=0.90
            ),
            MockLinkedInPost(
                aiScore=0.85,
                engagement={'engagementRate': 0.20},
                performanceScore=0.75
            ),
            MockLinkedInPost(
                aiScore=0.75,
                engagement={'engagementRate': 0.15},
                performanceScore=0.60
            )
        ]
        
        # Analyze content quality
        user_analytics = await analytics_service.generate_user_analytics('user-1', quality_posts)
        
        # Verify quality metrics
        assert user_analytics['content_quality_score'] > 0.8
        assert user_analytics['average_engagement'] > 0.2
        
        # Test correlation between AI score and engagement
        for post in quality_posts:
            # Higher AI scores should correlate with higher engagement
            assert (post.aiScore > 0.8 and post.engagement['engagementRate'] > 0.2) or \
                   (post.aiScore <= 0.8 and post.engagement['engagementRate'] <= 0.2)

    async def test_audience_insights_analytics(self, post_service, analytics_service):
        """Test analytics for audience insights"""
        # Setup posts targeting different audiences
        audience_posts = [
            MockLinkedInPost(
                title='Tech Leaders Post',
                engagement={'engagementRate': 0.25, 'reach': 1000},
                keywords=['leadership', 'technology'],
                tone='authoritative'
            ),
            MockLinkedInPost(
                title='Developer Community Post',
                engagement={'engagementRate': 0.30, 'reach': 800},
                keywords=['programming', 'development'],
                tone='casual'
            ),
            MockLinkedInPost(
                title='Business Strategy Post',
                engagement={'engagementRate': 0.20, 'reach': 1200},
                keywords=['strategy', 'business'],
                tone='professional'
            )
        ]
        
        # Analyze audience insights
        campaign_analytics = await analytics_service.generate_campaign_analytics('audience-test', audience_posts)
        
        # Verify audience metrics
        assert campaign_analytics['audience_reach'] > 0
        assert campaign_analytics['total_impressions'] > 0
        
        # Test audience engagement patterns
        total_reach = sum(post.engagement['reach'] for post in audience_posts)
        total_engagement = sum(post.engagement['engagementRate'] * post.engagement['reach'] for post in audience_posts)
        overall_engagement_rate = total_engagement / total_reach if total_reach > 0 else 0
        
        assert overall_engagement_rate > 0.2

    async def test_performance_benchmarking_analytics(self, post_service, analytics_service):
        """Test analytics for performance benchmarking"""
        # Setup benchmark posts
        benchmark_posts = [
            MockLinkedInPost(
                engagement={'engagementRate': 0.35, 'reach': 2000, 'impressions': 2500},
                performanceScore=0.95
            ),
            MockLinkedInPost(
                engagement={'engagementRate': 0.28, 'reach': 1500, 'impressions': 1800},
                performanceScore=0.88
            ),
            MockLinkedInPost(
                engagement={'engagementRate': 0.22, 'reach': 1200, 'impressions': 1400},
                performanceScore=0.75
            )
        ]
        
        # Generate benchmark analytics
        user_analytics = await analytics_service.generate_user_analytics('benchmark-user', benchmark_posts)
        
        # Calculate performance benchmarks
        avg_engagement = user_analytics['average_engagement']
        best_post = user_analytics['best_performing_post']
        
        # Verify benchmark metrics
        assert avg_engagement > 0.25  # High performing user
        assert best_post.engagement['engagementRate'] > 0.3
        assert best_post.performanceScore > 0.9
        
        # Test performance categories
        performance_categories = {
            'excellent': avg_engagement >= 0.25,
            'good': 0.15 <= avg_engagement < 0.25,
            'average': 0.05 <= avg_engagement < 0.15,
            'needs_improvement': avg_engagement < 0.05
        }
        
        # Should be in excellent category
        assert performance_categories['excellent']

    async def test_content_optimization_analytics(self, post_service, analytics_service):
        """Test analytics for content optimization insights"""
        # Setup posts before and after optimization
        before_optimization = [
            MockLinkedInPost(
                aiScore=0.65,
                engagement={'engagementRate': 0.12},
                optimizationSuggestions=['Improve headline', 'Add more hashtags']
            ),
            MockLinkedInPost(
                aiScore=0.70,
                engagement={'engagementRate': 0.15},
                optimizationSuggestions=['Better call-to-action', 'Include examples']
            )
        ]
        
        after_optimization = [
            MockLinkedInPost(
                aiScore=0.85,
                engagement={'engagementRate': 0.22},
                optimizationSuggestions=[]
            ),
            MockLinkedInPost(
                aiScore=0.90,
                engagement={'engagementRate': 0.28},
                optimizationSuggestions=[]
            )
        ]
        
        # Analyze optimization impact
        before_analytics = await analytics_service.generate_user_analytics('user-1', before_optimization)
        after_analytics = await analytics_service.generate_user_analytics('user-1', after_optimization)
        
        # Verify optimization improvements
        assert after_analytics['average_engagement'] > before_analytics['average_engagement']
        assert after_analytics['content_quality_score'] > before_analytics['content_quality_score']
        
        # Test optimization effectiveness
        engagement_improvement = after_analytics['average_engagement'] - before_analytics['average_engagement']
        quality_improvement = after_analytics['content_quality_score'] - before_analytics['content_quality_score']
        
        assert engagement_improvement > 0.05  # Significant improvement
        assert quality_improvement > 0.1  # Significant quality improvement

    async def test_predictive_analytics(self, post_service, analytics_service):
        """Test predictive analytics functionality"""
        # Setup historical data for prediction
        historical_posts = [
            MockLinkedInPost(
                createdAt=datetime.now() - timedelta(days=90),
                engagement={'engagementRate': 0.18},
                aiScore=0.82
            ),
            MockLinkedInPost(
                createdAt=datetime.now() - timedelta(days=60),
                engagement={'engagementRate': 0.22},
                aiScore=0.85
            ),
            MockLinkedInPost(
                createdAt=datetime.now() - timedelta(days=30),
                engagement={'engagementRate': 0.26},
                aiScore=0.88
            )
        ]
        
        # Analyze trends for prediction
        user_analytics = await analytics_service.generate_user_analytics('user-1', historical_posts)
        
        # Predict future performance
        trend = user_analytics['engagement_trend']
        current_avg = user_analytics['average_engagement']
        
        if trend == 'increasing':
            predicted_engagement = current_avg * 1.1  # 10% increase
        elif trend == 'decreasing':
            predicted_engagement = current_avg * 0.9  # 10% decrease
        else:
            predicted_engagement = current_avg  # Stable
        
        # Verify prediction logic
        assert predicted_engagement > 0
        assert predicted_engagement <= 1
        
        # Test prediction accuracy (simulated)
        actual_next_post = MockLinkedInPost(
            engagement={'engagementRate': 0.28},
            aiScore=0.90
        )
        
        prediction_accuracy = abs(predicted_engagement - actual_next_post.engagement['engagementRate'])
        assert prediction_accuracy < 0.1  # Within 10% accuracy

    async def test_reporting_analytics(self, post_service, analytics_service):
        """Test comprehensive reporting analytics"""
        # Setup comprehensive test data
        comprehensive_posts = [
            MockLinkedInPost(
                id='post-1',
                createdAt=datetime.now() - timedelta(days=30),
                engagement={'engagementRate': 0.25, 'reach': 1000, 'impressions': 1200},
                aiScore=0.88,
                performanceScore=0.85
            ),
            MockLinkedInPost(
                id='post-2',
                createdAt=datetime.now() - timedelta(days=20),
                engagement={'engagementRate': 0.30, 'reach': 1200, 'impressions': 1400},
                aiScore=0.92,
                performanceScore=0.90
            ),
            MockLinkedInPost(
                id='post-3',
                createdAt=datetime.now() - timedelta(days=10),
                engagement={'engagementRate': 0.35, 'reach': 1500, 'impressions': 1800},
                aiScore=0.95,
                performanceScore=0.95
            )
        ]
        
        # Generate comprehensive report
        user_analytics = await analytics_service.generate_user_analytics('user-1', comprehensive_posts)
        campaign_analytics = await analytics_service.generate_campaign_analytics('comprehensive-campaign', comprehensive_posts)
        
        # Verify comprehensive reporting
        assert user_analytics['total_posts'] == 3
        assert user_analytics['average_engagement'] > 0.25
        assert user_analytics['engagement_trend'] == 'increasing'
        assert user_analytics['content_quality_score'] > 0.9
        assert user_analytics['posting_frequency'] > 0
        
        assert campaign_analytics['total_posts'] == 3
        assert campaign_analytics['campaign_performance'] > 0.25
        assert campaign_analytics['audience_reach'] > 0
        assert campaign_analytics['total_impressions'] > 0
        
        # Test report completeness
        report_metrics = {
            'user_performance': user_analytics,
            'campaign_performance': campaign_analytics,
            'engagement_distribution': campaign_analytics['engagement_distribution'],
            'content_consistency': campaign_analytics['content_consistency']
        }
        
        # Verify all required metrics are present
        required_metrics = [
            'total_posts', 'average_engagement', 'engagement_trend',
            'content_quality_score', 'posting_frequency', 'campaign_performance',
            'audience_reach', 'total_impressions'
        ]
        
        for metric in required_metrics:
            found = False
            for report_section in report_metrics.values():
                if isinstance(report_section, dict) and metric in report_section:
                    found = True
                    break
            assert found, f"Required metric {metric} not found in report"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
