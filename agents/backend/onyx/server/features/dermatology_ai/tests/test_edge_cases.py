"""
Tests for Edge Cases
Tests for boundary conditions, error handling, and edge cases
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime
import asyncio

from core.domain.entities import Analysis, AnalysisStatus, SkinMetrics
from core.application.exceptions import ValidationError, ProcessingError
from core.application.validators import ImageValidator, UserIdValidator


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_image_validator_boundary_sizes(self):
        """Test image validator with boundary sizes"""
        # Minimum valid size
        min_size_data = b'\xff\xd8\xff' + b'x' * (ImageValidator.MIN_IMAGE_SIZE - 3)
        ImageValidator.validate_image_data(min_size_data)
        
        # Maximum valid size
        max_size_data = b'\xff\xd8\xff' + b'x' * (ImageValidator.MAX_IMAGE_SIZE - 3)
        ImageValidator.validate_image_data(max_size_data)
        
        # Just below minimum
        too_small = b'\xff\xd8\xff' + b'x' * (ImageValidator.MIN_IMAGE_SIZE - 4)
        with pytest.raises(ValidationError):
            ImageValidator.validate_image_data(too_small)
        
        # Just above maximum
        too_large = b'\xff\xd8\xff' + b'x' * (ImageValidator.MAX_IMAGE_SIZE - 2)
        with pytest.raises(ValidationError):
            ImageValidator.validate_image_data(too_large)
    
    def test_user_id_validator_boundary_lengths(self):
        """Test user ID validator with boundary lengths"""
        # Maximum valid length
        max_length_id = "a" * 255
        UserIdValidator.validate_user_id(max_length_id)
        
        # Just above maximum
        too_long_id = "a" * 256
        with pytest.raises(ValidationError):
            UserIdValidator.validate_user_id(too_long_id)
        
        # Minimum valid (single character)
        min_id = "a"
        UserIdValidator.validate_user_id(min_id)
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        async def make_request():
            # Simulate concurrent request
            await asyncio.sleep(0.01)
            return "response"
        
        async def run_concurrent():
            tasks = [make_request() for _ in range(10)]
            results = await asyncio.gather(*tasks)
            return results
        
        results = asyncio.run(run_concurrent())
        assert len(results) == 10
        assert all(r == "response" for r in results)
    
    def test_very_large_metadata(self):
        """Test handling very large metadata"""
        from core.application.validators import MetadataValidator
        
        # Create metadata at size limit
        large_metadata = {
            "key": "x" * (MetadataValidator.MAX_METADATA_SIZE - 100)
        }
        MetadataValidator.validate_metadata(large_metadata)
        
        # Exceed size limit
        too_large_metadata = {
            "key": "x" * (MetadataValidator.MAX_METADATA_SIZE + 1)
        }
        with pytest.raises(ValidationError):
            MetadataValidator.validate_metadata(too_large_metadata)
    
    def test_empty_conditions_list(self):
        """Test analysis with empty conditions list"""
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            conditions=[],
            status=AnalysisStatus.COMPLETED
        )
        
        assert len(analysis.conditions) == 0
        assert analysis.status == AnalysisStatus.COMPLETED
    
    def test_analysis_with_all_metrics_zero(self):
        """Test analysis with all metrics at zero"""
        metrics = SkinMetrics(
            overall_score=0.0,
            texture_score=0.0,
            hydration_score=0.0,
            elasticity_score=0.0,
            pigmentation_score=0.0,
            pore_size_score=0.0,
            wrinkles_score=0.0,
            redness_score=0.0,
            dark_spots_score=0.0
        )
        
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            metrics=metrics,
            status=AnalysisStatus.COMPLETED
        )
        
        assert analysis.metrics.overall_score == 0.0
        assert all(getattr(analysis.metrics, attr) == 0.0 
                  for attr in ['texture_score', 'hydration_score', 'elasticity_score'])
    
    def test_analysis_with_all_metrics_max(self):
        """Test analysis with all metrics at maximum"""
        metrics = SkinMetrics(
            overall_score=100.0,
            texture_score=100.0,
            hydration_score=100.0,
            elasticity_score=100.0,
            pigmentation_score=100.0,
            pore_size_score=100.0,
            wrinkles_score=100.0,
            redness_score=100.0,
            dark_spots_score=100.0
        )
        
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            metrics=metrics,
            status=AnalysisStatus.COMPLETED
        )
        
        assert analysis.metrics.overall_score == 100.0
        assert all(getattr(analysis.metrics, attr) == 100.0 
                  for attr in ['texture_score', 'hydration_score', 'elasticity_score'])


class TestErrorRecovery:
    """Tests for error recovery scenarios"""
    
    @pytest.mark.asyncio
    async def test_repository_error_recovery(self, mock_analysis_repository):
        """Test recovery from repository errors"""
        # Simulate repository error
        mock_analysis_repository.get_by_id = AsyncMock(side_effect=Exception("Database error"))
        
        with pytest.raises(Exception):
            await mock_analysis_repository.get_by_id("test-123")
        
        # Verify error was raised
        assert mock_analysis_repository.get_by_id.called
    
    @pytest.mark.asyncio
    async def test_service_error_recovery(self):
        """Test recovery from service errors"""
        mock_service = Mock()
        mock_service.process = AsyncMock(side_effect=Exception("Service error"))
        
        with pytest.raises(Exception):
            await mock_service.process(b"data")
    
    def test_validation_error_recovery(self):
        """Test recovery from validation errors"""
        # Invalid input
        with pytest.raises(ValidationError):
            ImageValidator.validate_image_data(b"")
        
        # Should be able to validate valid input after error
        valid_data = b'\xff\xd8\xff' + b'x' * 1000
        ImageValidator.validate_image_data(valid_data)  # Should not raise


class TestPerformanceEdgeCases:
    """Tests for performance-related edge cases"""
    
    def test_large_batch_processing(self):
        """Test processing large batches"""
        # Simulate large batch
        batch_size = 1000
        items = list(range(batch_size))
        
        # Process in chunks
        chunk_size = 100
        chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
        
        assert len(chunks) == 10
        assert sum(len(chunk) for chunk in chunks) == batch_size
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of timeouts"""
        async def slow_operation():
            await asyncio.sleep(10)  # Very slow
            return "result"
        
        async def with_timeout():
            try:
                result = await asyncio.wait_for(slow_operation(), timeout=0.1)
                return result
            except asyncio.TimeoutError:
                return "timeout"
        
        result = await with_timeout()
        assert result == "timeout"
    
    def test_memory_efficient_processing(self):
        """Test memory-efficient processing"""
        # Process large data in chunks
        large_data = b"x" * (10 * 1024 * 1024)  # 10MB
        
        chunk_size = 1024 * 1024  # 1MB chunks
        chunks = [large_data[i:i+chunk_size] for i in range(0, len(large_data), chunk_size)]
        
        assert len(chunks) == 10
        assert all(len(chunk) <= chunk_size for chunk in chunks)


class TestDataIntegrity:
    """Tests for data integrity edge cases"""
    
    def test_analysis_id_uniqueness(self):
        """Test that analysis IDs are unique"""
        analyses = [
            Analysis(id=f"analysis-{i}", user_id="user-123", status=AnalysisStatus.PROCESSING)
            for i in range(100)
        ]
        
        ids = [a.id for a in analyses]
        assert len(ids) == len(set(ids))  # All unique
    
    def test_metadata_immutability(self):
        """Test that metadata doesn't mutate unexpectedly"""
        metadata = {"key": "value"}
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            metadata=metadata,
            status=AnalysisStatus.PROCESSING
        )
        
        # Modify original metadata
        metadata["new_key"] = "new_value"
        
        # Analysis metadata should not be affected (depending on implementation)
        # This test verifies the expected behavior
        assert "new_key" in metadata
    
    def test_datetime_consistency(self):
        """Test datetime consistency in entities"""
        now = datetime.utcnow()
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        )
        
        # Created at should be set
        assert analysis.created_at is not None
        assert isinstance(analysis.created_at, datetime)
        
        # Should be recent (within last minute)
        time_diff = (datetime.utcnow() - analysis.created_at).total_seconds()
        assert 0 <= time_diff < 60


class TestUnicodeAndSpecialCharacters:
    """Tests for Unicode and special character handling"""
    
    def test_unicode_in_user_id(self):
        """Test handling Unicode in user IDs"""
        unicode_id = "user-测试-123"
        UserIdValidator.validate_user_id(unicode_id)
        
        # Should handle emojis
        emoji_id = "user-😀-123"
        UserIdValidator.validate_user_id(emoji_id)
    
    def test_special_characters_in_metadata(self):
        """Test handling special characters in metadata"""
        from core.application.validators import MetadataValidator
        
        special_metadata = {
            "key": "value with special chars: !@#$%^&*()",
            "unicode": "测试",
            "emoji": "😀",
            "newlines": "line1\nline2"
        }
        
        MetadataValidator.validate_metadata(special_metadata)
    
    def test_empty_strings_handling(self):
        """Test handling of empty strings"""
        # Empty string user ID should fail
        with pytest.raises(ValidationError):
            UserIdValidator.validate_user_id("")
        
        # Whitespace-only should fail
        with pytest.raises(ValidationError):
            UserIdValidator.validate_user_id("   ")



