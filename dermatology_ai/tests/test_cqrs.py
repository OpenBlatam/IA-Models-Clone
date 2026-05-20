"""
Tests for CQRS Pattern
Tests for commands, queries, and handlers
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from core.cqrs.commands import CreateAnalysisCommand, UpdateAnalysisCommand
from core.cqrs.queries import GetAnalysisQuery, GetAnalysisHistoryQuery
from core.cqrs.handlers import (
    CreateAnalysisHandler,
    UpdateAnalysisHandler,
    GetAnalysisHandler,
    GetAnalysisHistoryHandler
)
from core.domain.entities import (
    Analysis,
    AnalysisStatus,
    SkinMetrics,
    Condition,
    SkinType
)
from core.domain.interfaces import IAnalysisRepository


class TestCreateAnalysisCommand:
    """Tests for CreateAnalysisCommand"""
    
    def test_create_command(self):
        """Test creating a command"""
        command = CreateAnalysisCommand(
            user_id="user-123",
            image_data=b"image_bytes",
            metadata={"filename": "test.jpg"}
        )
        
        assert command.user_id == "user-123"
        assert command.image_data == b"image_bytes"
        assert command.metadata["filename"] == "test.jpg"


class TestCreateAnalysisHandler:
    """Tests for CreateAnalysisHandler"""
    
    @pytest.mark.asyncio
    async def test_handle_create_analysis(self, mock_analysis_repository):
        """Test handling create analysis command"""
        handler = CreateAnalysisHandler(
            analysis_repository=mock_analysis_repository
        )
        
        command = CreateAnalysisCommand(
            user_id="user-123",
            image_data=b"image_bytes",
            metadata={}
        )
        
        mock_analysis_repository.create = AsyncMock(return_value=Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        ))
        
        result = await handler.handle(command)
        
        assert result is not None
        assert result.user_id == "user-123"
        mock_analysis_repository.create.assert_called_once()


class TestGetAnalysisQuery:
    """Tests for GetAnalysisQuery"""
    
    def test_create_query(self):
        """Test creating a query"""
        query = GetAnalysisQuery(analysis_id="test-123")
        
        assert query.analysis_id == "test-123"


class TestGetAnalysisHandler:
    """Tests for GetAnalysisHandler"""
    
    @pytest.mark.asyncio
    async def test_handle_get_analysis(self, mock_analysis_repository):
        """Test handling get analysis query"""
        handler = GetAnalysisHandler(
            analysis_repository=mock_analysis_repository
        )
        
        query = GetAnalysisQuery(analysis_id="test-123")
        
        mock_analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.COMPLETED
        )
        
        mock_analysis_repository.get_by_id = AsyncMock(return_value=mock_analysis)
        
        result = await handler.handle(query)
        
        assert result is not None
        assert result.id == "test-123"
        mock_analysis_repository.get_by_id.assert_called_once_with("test-123")
    
    @pytest.mark.asyncio
    async def test_handle_get_analysis_not_found(self, mock_analysis_repository):
        """Test handling get analysis query when not found"""
        handler = GetAnalysisHandler(
            analysis_repository=mock_analysis_repository
        )
        
        query = GetAnalysisQuery(analysis_id="non-existent")
        
        mock_analysis_repository.get_by_id = AsyncMock(return_value=None)
        
        result = await handler.handle(query)
        
        assert result is None


class TestGetAnalysisHistoryQuery:
    """Tests for GetAnalysisHistoryQuery"""
    
    def test_create_history_query(self):
        """Test creating a history query"""
        query = GetAnalysisHistoryQuery(
            user_id="user-123",
            limit=10,
            offset=0
        )
        
        assert query.user_id == "user-123"
        assert query.limit == 10
        assert query.offset == 0


class TestGetAnalysisHistoryHandler:
    """Tests for GetAnalysisHistoryHandler"""
    
    @pytest.mark.asyncio
    async def test_handle_get_history(self, mock_analysis_repository):
        """Test handling get history query"""
        handler = GetAnalysisHistoryHandler(
            analysis_repository=mock_analysis_repository
        )
        
        query = GetAnalysisHistoryQuery(
            user_id="user-123",
            limit=10,
            offset=0
        )
        
        analyses = [
            Analysis(
                id=f"test-{i}",
                user_id="user-123",
                status=AnalysisStatus.COMPLETED
            )
            for i in range(3)
        ]
        
        mock_analysis_repository.get_by_user = AsyncMock(return_value=analyses)
        
        result = await handler.handle(query)
        
        assert len(result) == 3
        assert all(a.user_id == "user-123" for a in result)
        mock_analysis_repository.get_by_user.assert_called_once()


class TestUpdateAnalysisCommand:
    """Tests for UpdateAnalysisCommand"""
    
    def test_create_update_command(self):
        """Test creating an update command"""
        metrics = SkinMetrics(
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
        
        command = UpdateAnalysisCommand(
            analysis_id="test-123",
            metrics=metrics,
            status=AnalysisStatus.COMPLETED
        )
        
        assert command.analysis_id == "test-123"
        assert command.metrics == metrics
        assert command.status == AnalysisStatus.COMPLETED


class TestUpdateAnalysisHandler:
    """Tests for UpdateAnalysisHandler"""
    
    @pytest.mark.asyncio
    async def test_handle_update_analysis(self, mock_analysis_repository):
        """Test handling update analysis command"""
        handler = UpdateAnalysisHandler(
            analysis_repository=mock_analysis_repository
        )
        
        metrics = SkinMetrics(
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
        
        command = UpdateAnalysisCommand(
            analysis_id="test-123",
            metrics=metrics,
            status=AnalysisStatus.COMPLETED
        )
        
        existing_analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        )
        
        updated_analysis = Analysis(
            id="test-123",
            user_id="user-123",
            metrics=metrics,
            status=AnalysisStatus.COMPLETED
        )
        
        mock_analysis_repository.get_by_id = AsyncMock(return_value=existing_analysis)
        mock_analysis_repository.update = AsyncMock(return_value=updated_analysis)
        
        result = await handler.handle(command)
        
        assert result is not None
        assert result.status == AnalysisStatus.COMPLETED
        assert result.metrics == metrics
        mock_analysis_repository.update.assert_called_once()



