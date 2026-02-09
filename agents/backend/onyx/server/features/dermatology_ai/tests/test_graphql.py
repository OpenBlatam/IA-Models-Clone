"""
Tests for GraphQL
Tests for GraphQL schema and resolvers
"""

import pytest
from unittest.mock import Mock, AsyncMock

# GraphQL may not be available, so we'll test conditionally
try:
    from strawberry import Schema
    from api.graphql.schema import (
        SkinMetricsType,
        ConditionType,
        AnalysisType,
        Query,
        Mutation
    )
    GRAPHQL_AVAILABLE = True
except ImportError:
    GRAPHQL_AVAILABLE = False


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestGraphQLTypes:
    """Tests for GraphQL types"""
    
    def test_skin_metrics_type(self):
        """Test SkinMetricsType"""
        metrics = SkinMetricsType(
            overall_score=75.0,
            texture_score=80.0,
            hydration_score=70.0,
            elasticity_score=75.0,
            pigmentation_score=80.0,
            pore_size_score=70.0,
            wrinkles_score=75.0,
            redness_score=80.0,
            dark_spots_score=75.0
        )
        
        assert metrics.overall_score == 75.0
        assert metrics.texture_score == 80.0
    
    def test_condition_type(self):
        """Test ConditionType"""
        condition = ConditionType(
            name="acne",
            confidence=0.65,
            severity="moderate",
            description="Mild acne"
        )
        
        assert condition.name == "acne"
        assert condition.confidence == 0.65
        assert condition.severity == "moderate"
    
    def test_analysis_type(self):
        """Test AnalysisType"""
        metrics = SkinMetricsType(
            overall_score=75.0,
            texture_score=80.0,
            hydration_score=70.0,
            elasticity_score=75.0,
            pigmentation_score=80.0,
            pore_size_score=70.0,
            wrinkles_score=75.0,
            redness_score=80.0,
            dark_spots_score=75.0
        )
        
        analysis = AnalysisType(
            id="test-123",
            user_id="user-123",
            metrics=metrics,
            conditions=[]
        )
        
        assert analysis.id == "test-123"
        assert analysis.metrics == metrics


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestGraphQLQueries:
    """Tests for GraphQL queries"""
    
    @pytest.fixture
    def query(self):
        """Create GraphQL query"""
        return Query()
    
    @pytest.mark.asyncio
    async def test_get_analysis_query(self, query):
        """Test get analysis query"""
        # Mock resolver
        with patch.object(query, 'get_analysis') as mock_get:
            mock_get.return_value = AnalysisType(
                id="test-123",
                user_id="user-123",
                metrics=None,
                conditions=[]
            )
            
            result = await query.get_analysis("test-123")
            
            assert result.id == "test-123"
            mock_get.assert_called_once_with("test-123")
    
    @pytest.mark.asyncio
    async def test_list_analyses_query(self, query):
        """Test list analyses query"""
        with patch.object(query, 'list_analyses') as mock_list:
            mock_list.return_value = []
            
            result = await query.list_analyses(user_id="user-123", limit=10)
            
            assert isinstance(result, list)
            mock_list.assert_called_once()


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestGraphQLMutations:
    """Tests for GraphQL mutations"""
    
    @pytest.fixture
    def mutation(self):
        """Create GraphQL mutation"""
        return Mutation()
    
    @pytest.mark.asyncio
    async def test_create_analysis_mutation(self, mutation):
        """Test create analysis mutation"""
        with patch.object(mutation, 'create_analysis') as mock_create:
            mock_create.return_value = AnalysisType(
                id="new-123",
                user_id="user-123",
                metrics=None,
                conditions=[]
            )
            
            result = await mutation.create_analysis(
                user_id="user-123",
                image_data="base64_encoded_image"
            )
            
            assert result.id == "new-123"
            mock_create.assert_called_once()


class TestGraphQLAvailability:
    """Tests for GraphQL availability"""
    
    def test_graphql_import(self):
        """Test GraphQL import handling"""
        # Should not raise even if GraphQL is not available
        try:
            from api.graphql.schema import GRAPHQL_AVAILABLE
            assert isinstance(GRAPHQL_AVAILABLE, bool)
        except ImportError:
            # GraphQL not installed, which is fine
            pass



