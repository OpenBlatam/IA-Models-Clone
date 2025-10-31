from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from ...api.main import (
from ...core.models import VideoRequest, VideoResponse, VideoQuality, VideoFormat
from ...config.config_manager import OnyxAIVideoConfig
        import yaml
        import yaml
        import yaml
from typing import Any, List, Dict, Optional
import logging
"""
Integration tests for the main API module.
"""


    OnyxAIVideoSystem, initialize_system, shutdown_system,
    generate_video, generate_video_with_vision, get_system_status,
    get_metrics, get_active_requests, cancel_request, get_system_info,
    health_check, version_info, create_system_instance
)


class TestOnyxAIVideoSystem:
    """Test the main OnyxAIVideoSystem class."""
    
    @pytest.mark.integration
    async def test_system_initialization(self, temp_dir, sample_config) -> Any:
        """Test complete system initialization."""
        # Create config file
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(sample_config, f)
        
        # Create system instance
        system = OnyxAIVideoSystem(str(config_file))
        
        # Initialize system
        await system.initialize()
        
        assert system.initialized is True
        assert system.config is not None
        assert system.logger is not None
        assert system.performance_monitor is not None
        assert system.security_manager is not None
        assert system.onyx_integration is not None
        assert system.video_workflow is not None
        assert system.plugin_manager is not None
    
    @pytest.mark.integration
    async def test_system_shutdown(self, real_system) -> Any:
        """Test system shutdown."""
        await real_system.initialize()
        
        # Shutdown system
        await real_system.shutdown()
        
        assert real_system.initialized is False
        assert real_system.shutdown_requested is True
    
    @pytest.mark.integration
    async def test_generate_video_basic(self, real_system, sample_video_request) -> Any:
        """Test basic video generation."""
        await real_system.initialize()
        
        # Mock video workflow
        with patch.object(real_system.video_workflow, 'generate_video', 
                         new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = VideoResponse(
                request_id=sample_video_request.request_id,
                status="completed",
                output_url="http://example.com/video.mp4",
                processing_time=5.0
            )
            
            response = await real_system.generate_video(sample_video_request)
            
            assert response.request_id == sample_video_request.request_id
            assert response.status == "completed"
            assert response.output_url == "http://example.com/video.mp4"
            assert response.processing_time == 5.0
    
    @pytest.mark.integration
    async def test_generate_video_with_vision(self, real_system, sample_video_request) -> Any:
        """Test video generation with vision capabilities."""
        await real_system.initialize()
        
        # Mock vision workflow
        with patch.object(real_system.video_workflow, 'generate_video_with_vision', 
                         new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = VideoResponse(
                request_id=sample_video_request.request_id,
                status="completed",
                output_url="http://example.com/video.mp4",
                processing_time=8.0,
                metadata={"vision_used": True}
            )
            
            image_data = b"fake_image_data"
            response = await real_system.generate_video_with_vision(
                sample_video_request, image_data
            )
            
            assert response.request_id == sample_video_request.request_id
            assert response.status == "completed"
            assert response.metadata["vision_used"] is True
    
    @pytest.mark.integration
    async def test_get_system_status(self, real_system) -> Optional[Dict[str, Any]]:
        """Test system status retrieval."""
        await real_system.initialize()
        
        status = await real_system.get_system_status()
        
        assert status["status"] == "running"
        assert status["initialized"] is True
        assert status["version"] == "1.0.0"
        assert "components" in status
        assert "performance" in status
        assert "security" in status
    
    @pytest.mark.integration
    async def test_get_metrics(self, real_system) -> Optional[Dict[str, Any]]:
        """Test metrics retrieval."""
        await real_system.initialize()
        
        metrics = await real_system.get_metrics()
        
        assert "system_metrics" in metrics
        assert "performance_metrics" in metrics
        assert "request_metrics" in metrics
        assert "error_metrics" in metrics
    
    @pytest.mark.integration
    async async def test_request_management(self, real_system, sample_video_request) -> Any:
        """Test request management functionality."""
        await real_system.initialize()
        
        # Add a request to active requests
        real_system.active_requests[sample_video_request.request_id] = {
            "request": sample_video_request,
            "start_time": asyncio.get_event_loop().time(),
            "status": "processing"
        }
        
        # Test get active requests
        active_requests = real_system.get_active_requests()
        assert sample_video_request.request_id in active_requests
        
        # Test cancel request
        result = await real_system.cancel_request(sample_video_request.request_id)
        assert result is True
        
        # Verify request was removed
        active_requests = real_system.get_active_requests()
        assert sample_video_request.request_id not in active_requests
    
    @pytest.mark.integration
    async def test_error_handling(self, real_system, sample_video_request) -> Any:
        """Test error handling in video generation."""
        await real_system.initialize()
        
        # Mock workflow to raise exception
        with patch.object(real_system.video_workflow, 'generate_video', 
                         new_callable=AsyncMock, side_effect=Exception("Test error")):
            
            with pytest.raises(Exception, match="Test error"):
                await real_system.generate_video(sample_video_request)
    
    @pytest.mark.integration
    async async def test_concurrent_requests(self, real_system) -> Any:
        """Test handling of concurrent requests."""
        await real_system.initialize()
        
        # Create multiple requests
        requests = []
        for i in range(3):
            request = VideoRequest(
                input_text=f"Test video {i}",
                user_id=f"user{i}",
                quality=VideoQuality.LOW,
                duration=10,
                output_format=VideoFormat.MP4
            )
            requests.append(request)
        
        # Mock video generation
        with patch.object(real_system.video_workflow, 'generate_video', 
                         new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = VideoResponse(
                request_id="test",
                status="completed",
                output_url="http://example.com/video.mp4"
            )
            
            # Execute requests concurrently
            tasks = [real_system.generate_video(req) for req in requests]
            responses = await asyncio.gather(*tasks)
            
            assert len(responses) == 3
            for response in responses:
                assert response.status == "completed"
    
    @pytest.mark.integration
    async def test_system_info(self, real_system) -> Any:
        """Test system information retrieval."""
        await real_system.initialize()
        
        info = real_system.get_system_info()
        
        assert info["system_name"] == "Test AI Video System"
        assert info["version"] == "1.0.0"
        assert info["environment"] == "testing"
        assert "python_version" in info
        assert "platform" in info
        assert "onyx_integration" in info
    
    @pytest.mark.integration
    async def test_health_check(self, real_system) -> Any:
        """Test health check functionality."""
        await real_system.initialize()
        
        health = await real_system.health_check()
        
        assert health["status"] == "healthy"
        assert health["timestamp"] is not None
        assert "components" in health
        assert "performance" in health
    
    @pytest.mark.integration
    async def test_version_info(self, real_system) -> Any:
        """Test version information."""
        version = real_system.version_info()
        
        assert version["version"] == "1.0.0"
        assert version["build_date"] is not None
        assert version["git_commit"] is not None
        assert "dependencies" in version


class TestAPIFunctions:
    """Test API utility functions."""
    
    @pytest.mark.integration
    async def test_initialize_system(self, temp_dir, sample_config) -> Any:
        """Test initialize_system function."""
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(sample_config, f)
        
        system = await initialize_system(str(config_file))
        
        assert system.initialized is True
        assert system.config is not None
        
        # Cleanup
        await system.shutdown()
    
    @pytest.mark.integration
    async def test_shutdown_system(self, real_system) -> Any:
        """Test shutdown_system function."""
        await real_system.initialize()
        
        await shutdown_system(real_system)
        
        assert real_system.initialized is False
        assert real_system.shutdown_requested is True
    
    @pytest.mark.integration
    async def test_generate_video_function(self, real_system, sample_video_request) -> Any:
        """Test generate_video function."""
        await real_system.initialize()
        
        with patch.object(real_system.video_workflow, 'generate_video', 
                         new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = VideoResponse(
                request_id=sample_video_request.request_id,
                status="completed",
                output_url="http://example.com/video.mp4"
            )
            
            response = await generate_video(real_system, sample_video_request)
            
            assert response.request_id == sample_video_request.request_id
            assert response.status == "completed"
    
    @pytest.mark.integration
    async def test_generate_video_with_vision_function(self, real_system, sample_video_request) -> Any:
        """Test generate_video_with_vision function."""
        await real_system.initialize()
        
        with patch.object(real_system.video_workflow, 'generate_video_with_vision', 
                         new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = VideoResponse(
                request_id=sample_video_request.request_id,
                status="completed",
                output_url="http://example.com/video.mp4"
            )
            
            image_data = b"fake_image_data"
            response = await generate_video_with_vision(
                real_system, sample_video_request, image_data
            )
            
            assert response.request_id == sample_video_request.request_id
            assert response.status == "completed"
    
    @pytest.mark.integration
    async def test_get_system_status_function(self, real_system) -> Optional[Dict[str, Any]]:
        """Test get_system_status function."""
        await real_system.initialize()
        
        status = await get_system_status(real_system)
        
        assert status["status"] == "running"
        assert status["initialized"] is True
    
    @pytest.mark.integration
    async def test_get_metrics_function(self, real_system) -> Optional[Dict[str, Any]]:
        """Test get_metrics function."""
        await real_system.initialize()
        
        metrics = await get_metrics(real_system)
        
        assert "system_metrics" in metrics
        assert "performance_metrics" in metrics
    
    @pytest.mark.integration
    async async def test_get_active_requests_function(self, real_system, sample_video_request) -> Optional[Dict[str, Any]]:
        """Test get_active_requests function."""
        await real_system.initialize()
        
        # Add a request
        real_system.active_requests[sample_video_request.request_id] = {
            "request": sample_video_request,
            "start_time": asyncio.get_event_loop().time(),
            "status": "processing"
        }
        
        active_requests = get_active_requests(real_system)
        assert sample_video_request.request_id in active_requests
    
    @pytest.mark.integration
    async async def test_cancel_request_function(self, real_system, sample_video_request) -> Any:
        """Test cancel_request function."""
        await real_system.initialize()
        
        # Add a request
        real_system.active_requests[sample_video_request.request_id] = {
            "request": sample_video_request,
            "start_time": asyncio.get_event_loop().time(),
            "status": "processing"
        }
        
        result = await cancel_request(real_system, sample_video_request.request_id)
        assert result is True
        
        # Verify request was removed
        active_requests = get_active_requests(real_system)
        assert sample_video_request.request_id not in active_requests
    
    @pytest.mark.integration
    async def test_get_system_info_function(self, real_system) -> Optional[Dict[str, Any]]:
        """Test get_system_info function."""
        await real_system.initialize()
        
        info = get_system_info(real_system)
        
        assert info["system_name"] == "Test AI Video System"
        assert info["version"] == "1.0.0"
    
    @pytest.mark.integration
    async def test_health_check_function(self, real_system) -> Any:
        """Test health_check function."""
        await real_system.initialize()
        
        health = await health_check(real_system)
        
        assert health["status"] == "healthy"
        assert "timestamp" in health
    
    @pytest.mark.integration
    async def test_version_info_function(self, real_system) -> Any:
        """Test version_info function."""
        version = version_info(real_system)
        
        assert version["version"] == "1.0.0"
        assert "build_date" in version
    
    @pytest.mark.integration
    async def test_create_system_instance(self, temp_dir, sample_config) -> Any:
        """Test create_system_instance function."""
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(sample_config, f)
        
        system = create_system_instance(str(config_file))
        
        assert system is not None
        assert system.config_path == str(config_file)
        assert system.config is None  # Not loaded yet


class TestAPIErrorHandling:
    """Test API error handling scenarios."""
    
    @pytest.mark.integration
    async def test_invalid_config_file(self) -> Any:
        """Test handling of invalid config file."""
        with pytest.raises(FileNotFoundError):
            system = OnyxAIVideoSystem("nonexistent.yaml")
            await system.initialize()
    
    @pytest.mark.integration
    async async def test_invalid_video_request(self, real_system) -> Any:
        """Test handling of invalid video request."""
        await real_system.initialize()
        
        # Create invalid request
        invalid_request = VideoRequest(
            input_text="",  # Empty input
            user_id="test_user",
            quality=VideoQuality.LOW,
            duration=10,
            output_format=VideoFormat.MP4
        )
        
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            await real_system.generate_video(invalid_request)
    
    @pytest.mark.integration
    async def test_workflow_failure(self, real_system, sample_video_request) -> Any:
        """Test handling of workflow failure."""
        await real_system.initialize()
        
        # Mock workflow to fail
        with patch.object(real_system.video_workflow, 'generate_video', 
                         new_callable=AsyncMock, side_effect=Exception("Workflow failed")):
            
            with pytest.raises(Exception, match="Workflow failed"):
                await real_system.generate_video(sample_video_request)
    
    @pytest.mark.integration
    async def test_system_not_initialized(self, real_system, sample_video_request) -> Any:
        """Test handling when system is not initialized."""
        # Don't initialize the system
        
        with pytest.raises(RuntimeError, match="System not initialized"):
            await real_system.generate_video(sample_video_request)
    
    @pytest.mark.integration
    async def test_shutdown_during_operation(self, real_system, sample_video_request) -> Any:
        """Test handling when system is shutdown during operation."""
        await real_system.initialize()
        
        # Start shutdown
        real_system.shutdown_requested = True
        
        with pytest.raises(RuntimeError, match="System is shutting down"):
            await real_system.generate_video(sample_video_request)


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    @pytest.mark.integration
    async async def test_request_tracking(self, real_system, sample_video_request) -> Any:
        """Test request tracking and metrics."""
        await real_system.initialize()
        
        # Mock video generation
        with patch.object(real_system.video_workflow, 'generate_video', 
                         new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = VideoResponse(
                request_id=sample_video_request.request_id,
                status="completed",
                output_url="http://example.com/video.mp4",
                processing_time=5.0
            )
            
            response = await real_system.generate_video(sample_video_request)
            
            # Check that request was tracked
            metrics = await real_system.get_metrics()
            assert metrics["request_metrics"]["total_requests"] >= 1
            assert metrics["request_metrics"]["successful_requests"] >= 1
    
    @pytest.mark.integration
    async async def test_concurrent_request_limits(self, real_system) -> Any:
        """Test concurrent request limits."""
        await real_system.initialize()
        
        # Set low concurrent limit
        real_system.config.performance.max_concurrent_requests = 2
        
        # Create multiple requests
        requests = []
        for i in range(5):
            request = VideoRequest(
                input_text=f"Test video {i}",
                user_id=f"user{i}",
                quality=VideoQuality.LOW,
                duration=10,
                output_format=VideoFormat.MP4
            )
            requests.append(request)
        
        # Mock video generation with delay
        async def delayed_generate(request) -> Any:
            await asyncio.sleep(0.1)  # Simulate processing time
            return VideoResponse(
                request_id=request.request_id,
                status="completed",
                output_url="http://example.com/video.mp4"
            )
        
        with patch.object(real_system.video_workflow, 'generate_video', 
                         new_callable=AsyncMock, side_effect=delayed_generate):
            
            # Execute requests concurrently
            tasks = [real_system.generate_video(req) for req in requests]
            responses = await asyncio.gather(*tasks)
            
            assert len(responses) == 5
            for response in responses:
                assert response.status == "completed"
    
    @pytest.mark.integration
    async def test_memory_usage_tracking(self, real_system) -> Any:
        """Test memory usage tracking."""
        await real_system.initialize()
        
        # Generate some load
        for i in range(10):
            request = VideoRequest(
                input_text=f"Test video {i}",
                user_id=f"user{i}",
                quality=VideoQuality.LOW,
                duration=10,
                output_format=VideoFormat.MP4
            )
            
            with patch.object(real_system.video_workflow, 'generate_video', 
                             new_callable=AsyncMock) as mock_generate:
                mock_generate.return_value = VideoResponse(
                    request_id=request.request_id,
                    status="completed",
                    output_url="http://example.com/video.mp4"
                )
                
                await real_system.generate_video(request)
        
        # Check memory metrics
        metrics = await real_system.get_metrics()
        assert "memory_usage" in metrics["system_metrics"]
        assert metrics["system_metrics"]["memory_usage"]["used_memory"] > 0


class TestAPISecurity:
    """Test API security features."""
    
    @pytest.mark.integration
    async def test_input_validation(self, real_system) -> Any:
        """Test input validation security."""
        await real_system.initialize()
        
        # Test malicious input
        malicious_request = VideoRequest(
            input_text="<script>alert('xss')</script>",
            user_id="test_user",
            quality=VideoQuality.LOW,
            duration=10,
            output_format=VideoFormat.MP4
        )
        
        # Should sanitize input
        with patch.object(real_system.video_workflow, 'generate_video', 
                         new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = VideoResponse(
                request_id=malicious_request.request_id,
                status="completed",
                output_url="http://example.com/video.mp4"
            )
            
            response = await real_system.generate_video(malicious_request)
            assert response.status == "completed"
    
    @pytest.mark.integration
    async def test_rate_limiting(self, real_system) -> Any:
        """Test rate limiting functionality."""
        await real_system.initialize()
        
        # Enable rate limiting
        real_system.config.security.rate_limit_enabled = True
        real_system.config.security.rate_limit_requests = 2
        real_system.config.security.rate_limit_window = 60
        
        # Create requests
        requests = []
        for i in range(5):
            request = VideoRequest(
                input_text=f"Test video {i}",
                user_id="same_user",  # Same user for rate limiting
                quality=VideoQuality.LOW,
                duration=10,
                output_format=VideoFormat.MP4
            )
            requests.append(request)
        
        # Mock video generation
        with patch.object(real_system.video_workflow, 'generate_video', 
                         new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = VideoResponse(
                request_id="test",
                status="completed",
                output_url="http://example.com/video.mp4"
            )
            
            # First two requests should succeed
            response1 = await real_system.generate_video(requests[0])
            response2 = await real_system.generate_video(requests[1])
            
            assert response1.status == "completed"
            assert response2.status == "completed"
            
            # Third request should be rate limited
            with pytest.raises(Exception, match="Rate limit exceeded"):
                await real_system.generate_video(requests[2])
    
    @pytest.mark.integration
    async def test_access_control(self, real_system, sample_video_request) -> Any:
        """Test access control functionality."""
        await real_system.initialize()
        
        # Mock access validation to deny
        with patch.object(real_system.security_manager, 'validate_access', 
                         new_callable=AsyncMock, return_value=False):
            
            with pytest.raises(Exception, match="Access denied"):
                await real_system.generate_video(sample_video_request)
    
    @pytest.mark.integration
    async def test_security_event_logging(self, real_system) -> Any:
        """Test security event logging."""
        await real_system.initialize()
        
        # Trigger a security event (rate limiting)
        real_system.config.security.rate_limit_enabled = True
        real_system.config.security.rate_limit_requests = 1
        
        request1 = VideoRequest(
            input_text="Test video 1",
            user_id="test_user",
            quality=VideoQuality.LOW,
            duration=10,
            output_format=VideoFormat.MP4
        )
        
        request2 = VideoRequest(
            input_text="Test video 2",
            user_id="test_user",  # Same user
            quality=VideoQuality.LOW,
            duration=10,
            output_format=VideoFormat.MP4
        )
        
        with patch.object(real_system.video_workflow, 'generate_video', 
                         new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = VideoResponse(
                request_id="test",
                status="completed",
                output_url="http://example.com/video.mp4"
            )
            
            # First request should succeed
            await real_system.generate_video(request1)
            
            # Second request should trigger security event
            with pytest.raises(Exception):
                await real_system.generate_video(request2)
        
        # Check security events
        security_events = real_system.security_manager.security_events
        assert len(security_events) > 0
        assert any(event.event_type == "rate_limit_exceeded" for event in security_events) 