"""
Advanced Testing Framework for OpusClip Improved
==============================================

Comprehensive testing system with unit, integration, and performance tests.
"""

import asyncio
import logging
import pytest
import pytest_asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock, patch
import json
import tempfile
import os
from pathlib import Path

from .schemas import get_settings
from .exceptions import TestingError, create_testing_error

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    status: str  # "passed", "failed", "skipped", "error"
    duration: float
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class TestSuite:
    """Test suite configuration"""
    name: str
    description: str
    tests: List[str]
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    timeout: int = 300


class TestDatabase:
    """Test database utilities"""
    
    @staticmethod
    async def create_test_database():
        """Create test database"""
        # Create in-memory SQLite database for testing
        pass
    
    @staticmethod
    async def cleanup_test_database():
        """Cleanup test database"""
        pass
    
    @staticmethod
    async def seed_test_data():
        """Seed test database with sample data"""
        pass


class TestFixtures:
    """Test fixtures and utilities"""
    
    @staticmethod
    def create_test_video_file() -> str:
        """Create a test video file"""
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.write(b"fake video content")
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def create_test_user_data() -> Dict[str, Any]:
        """Create test user data"""
        return {
            "email": "test@example.com",
            "username": "testuser",
            "full_name": "Test User",
            "password": "testpassword123"
        }
    
    @staticmethod
    def create_test_project_data() -> Dict[str, Any]:
        """Create test project data"""
        return {
            "name": "Test Project",
            "description": "Test project description",
            "is_public": False
        }
    
    @staticmethod
    def create_test_analysis_data() -> Dict[str, Any]:
        """Create test analysis data"""
        return {
            "video_url": "https://example.com/test-video.mp4",
            "video_filename": "test-video.mp4",
            "duration": 120.0,
            "fps": 30.0,
            "resolution": "1920x1080",
            "format": "mp4"
        }
    
    @staticmethod
    def create_test_generation_data() -> Dict[str, Any]:
        """Create test generation data"""
        return {
            "clip_type": "viral",
            "target_duration": 30,
            "max_clips": 5,
            "include_intro": True,
            "include_outro": True,
            "add_captions": True,
            "add_watermark": False
        }


class MockServices:
    """Mock services for testing"""
    
    @staticmethod
    def mock_ai_service():
        """Mock AI service responses"""
        return {
            "transcript": "This is a test transcript of the video content.",
            "sentiment_scores": {
                "positive": 0.7,
                "negative": 0.1,
                "neutral": 0.2
            },
            "key_moments": [
                {"start_time": 10.0, "end_time": 20.0, "confidence": 0.9},
                {"start_time": 45.0, "end_time": 55.0, "confidence": 0.8}
            ],
            "topics": ["technology", "innovation", "future"],
            "emotions": ["excitement", "curiosity", "optimism"],
            "viral_potential": 0.75
        }
    
    @staticmethod
    def mock_video_processor():
        """Mock video processor responses"""
        return {
            "duration": 120.0,
            "fps": 30.0,
            "resolution": "1920x1080",
            "format": "mp4",
            "file_size": 1024000,
            "scene_changes": [15.0, 30.0, 45.0, 60.0],
            "face_detections": [
                {"timestamp": 10.0, "faces": [{"bbox": [100, 100, 200, 200], "confidence": 0.9}]}
            ]
        }
    
    @staticmethod
    def mock_storage_service():
        """Mock storage service responses"""
        return {
            "file_id": "test_file_id",
            "url": "https://storage.example.com/test_file_id",
            "size": 1024000,
            "uploaded_at": datetime.utcnow().isoformat()
        }


class UnitTests:
    """Unit tests for core functionality"""
    
    @pytest.mark.asyncio
    async def test_user_creation(self):
        """Test user creation"""
        from .database import DatabaseOperations, get_database_session
        from .auth import AuthenticationService
        
        user_data = TestFixtures.create_test_user_data()
        auth_service = AuthenticationService()
        
        async with get_database_session() as session:
            db_ops = DatabaseOperations(session)
            user = await db_ops.create_user(user_data)
            
            assert user.email == user_data["email"]
            assert user.username == user_data["username"]
            assert user.full_name == user_data["full_name"]
            assert user.is_active is True
    
    @pytest.mark.asyncio
    async def test_video_analysis(self):
        """Test video analysis functionality"""
        from .services import VideoAnalysisService
        
        analysis_data = TestFixtures.create_test_analysis_data()
        mock_ai_response = MockServices.mock_ai_service()
        
        with patch('improved.ai_engine.ai_engine.analyze_video') as mock_analyze:
            mock_analyze.return_value = mock_ai_response
            
            service = VideoAnalysisService()
            result = await service.analyze_video(analysis_data)
            
            assert result["status"] == "completed"
            assert "transcript" in result
            assert "sentiment_scores" in result
            assert "key_moments" in result
    
    @pytest.mark.asyncio
    async def test_clip_generation(self):
        """Test clip generation functionality"""
        from .services import ClipGenerationService
        
        generation_data = TestFixtures.create_test_generation_data()
        mock_clips = [
            {"start_time": 10.0, "end_time": 40.0, "confidence": 0.9},
            {"start_time": 45.0, "end_time": 75.0, "confidence": 0.8}
        ]
        
        with patch('improved.ai_engine.ai_engine.generate_clips') as mock_generate:
            mock_generate.return_value = mock_clips
            
            service = ClipGenerationService()
            result = await service.generate_clips(generation_data)
            
            assert result["status"] == "completed"
            assert len(result["clips"]) == 2
            assert result["clips"][0]["start_time"] == 10.0
    
    @pytest.mark.asyncio
    async def test_authentication(self):
        """Test authentication functionality"""
        from .auth import AuthenticationService
        
        auth_service = AuthenticationService()
        
        # Test password hashing
        password = "testpassword123"
        hashed = auth_service.get_password_hash(password)
        assert hashed != password
        assert auth_service.verify_password(password, hashed)
        
        # Test token creation
        token_data = {"sub": "test_user_id", "email": "test@example.com"}
        token = auth_service.create_access_token(token_data)
        assert token is not None
        
        # Test token verification
        payload = auth_service.verify_token(token)
        assert payload["sub"] == "test_user_id"
        assert payload["email"] == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        from .rate_limiter import RateLimiter, RateLimitConfig, RateLimitStrategy
        
        rate_limiter = RateLimiter()
        config = RateLimitConfig(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            limit=2,
            window=60
        )
        
        # Test rate limiting
        result1 = await rate_limiter.check_rate_limit("test_user", config)
        assert result1.allowed is True
        
        result2 = await rate_limiter.check_rate_limit("test_user", config)
        assert result2.allowed is True
        
        result3 = await rate_limiter.check_rate_limit("test_user", config)
        assert result3.allowed is False
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test caching functionality"""
        from .cache import CacheManager
        
        cache_manager = CacheManager()
        
        # Test cache set and get
        await cache_manager.set("test_key", {"data": "test_value"}, "default")
        result = await cache_manager.get("test_key", "default")
        
        assert result["data"] == "test_value"
        
        # Test cache delete
        await cache_manager.delete("test_key")
        result = await cache_manager.get("test_key", "default")
        assert result is None


class IntegrationTests:
    """Integration tests for system components"""
    
    @pytest.mark.asyncio
    async def test_video_processing_pipeline(self):
        """Test complete video processing pipeline"""
        from .services import VideoAnalysisService, ClipGenerationService, ClipExportService
        
        # Create test video file
        test_video = TestFixtures.create_test_video_file()
        
        try:
            # Mock external services
            with patch('improved.ai_engine.ai_engine.analyze_video') as mock_analyze, \
                 patch('improved.video_processor.video_processor.extract_segments') as mock_extract, \
                 patch('improved.video_processor.video_processor.create_clip') as mock_create:
                
                mock_analyze.return_value = MockServices.mock_ai_service()
                mock_extract.return_value = [{"start_time": 10.0, "end_time": 40.0}]
                mock_create.return_value = "test_clip.mp4"
                
                # Test analysis
                analysis_service = VideoAnalysisService()
                analysis_data = TestFixtures.create_test_analysis_data()
                analysis_data["video_path"] = test_video
                
                analysis_result = await analysis_service.analyze_video(analysis_data)
                assert analysis_result["status"] == "completed"
                
                # Test generation
                generation_service = ClipGenerationService()
                generation_data = TestFixtures.create_test_generation_data()
                generation_data["analysis_id"] = analysis_result["analysis_id"]
                
                generation_result = await generation_service.generate_clips(generation_data)
                assert generation_result["status"] == "completed"
                
                # Test export
                export_service = ClipExportService()
                export_data = {
                    "generation_id": generation_result["generation_id"],
                    "format": "mp4",
                    "quality": "high",
                    "target_platform": "youtube"
                }
                
                export_result = await export_service.export_clips(export_data)
                assert export_result["status"] == "completed"
        
        finally:
            # Cleanup
            if os.path.exists(test_video):
                os.unlink(test_video)
    
    @pytest.mark.asyncio
    async def test_database_operations(self):
        """Test database operations"""
        from .database import DatabaseOperations, get_database_session
        from .auth import AuthenticationService
        
        auth_service = AuthenticationService()
        
        async with get_database_session() as session:
            db_ops = DatabaseOperations(session)
            
            # Test user operations
            user_data = TestFixtures.create_test_user_data()
            user = await db_ops.create_user(user_data)
            
            retrieved_user = await db_ops.get_user_by_id(user.id)
            assert retrieved_user.id == user.id
            
            # Test project operations
            project_data = TestFixtures.create_test_project_data()
            project_data["owner_id"] = user.id
            project = await db_ops.create_project(project_data)
            
            user_projects = await db_ops.get_user_projects(user.id)
            assert len(user_projects) == 1
            assert user_projects[0].id == project.id
    
    @pytest.mark.asyncio
    async def test_api_endpoints(self):
        """Test API endpoints"""
        from fastapi.testclient import TestClient
        from .app import create_app
        
        app = create_app()
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/api/v2/opus-clip/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        # Test metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_webhook_system(self):
        """Test webhook system"""
        from .webhooks import WebhookManager, WebhookEvent
        
        webhook_manager = WebhookManager()
        await webhook_manager.start()
        
        try:
            # Test webhook registration
            from .webhooks import WebhookConfig
            config = WebhookConfig(
                webhook_id="test_webhook",
                url="https://example.com/webhook",
                events=[WebhookEvent.VIDEO_ANALYSIS_COMPLETED]
            )
            
            webhook_manager.register_webhook(config)
            
            # Test webhook trigger
            await webhook_manager.trigger_event(
                WebhookEvent.VIDEO_ANALYSIS_COMPLETED,
                {"analysis_id": "test_analysis", "status": "completed"}
            )
            
            # Test webhook stats
            stats = webhook_manager.get_webhook_stats()
            assert stats["total_webhooks"] == 1
        
        finally:
            await webhook_manager.stop()


class PerformanceTests:
    """Performance tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_video_processing(self):
        """Test concurrent video processing"""
        from .services import VideoAnalysisService
        
        service = VideoAnalysisService()
        
        # Create multiple test videos
        test_videos = []
        for i in range(5):
            video = TestFixtures.create_test_video_file()
            test_videos.append(video)
        
        try:
            # Mock AI service
            with patch('improved.ai_engine.ai_engine.analyze_video') as mock_analyze:
                mock_analyze.return_value = MockServices.mock_ai_service()
                
                # Process videos concurrently
                start_time = datetime.utcnow()
                
                tasks = []
                for video in test_videos:
                    analysis_data = TestFixtures.create_test_analysis_data()
                    analysis_data["video_path"] = video
                    tasks.append(service.analyze_video(analysis_data))
                
                results = await asyncio.gather(*tasks)
                
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                # Verify all completed successfully
                for result in results:
                    assert result["status"] == "completed"
                
                # Performance assertion (should complete within reasonable time)
                assert duration < 10.0  # 10 seconds for 5 videos
        
        finally:
            # Cleanup
            for video in test_videos:
                if os.path.exists(video):
                    os.unlink(video)
    
    @pytest.mark.asyncio
    async def test_database_performance(self):
        """Test database performance"""
        from .database import DatabaseOperations, get_database_session
        
        async with get_database_session() as session:
            db_ops = DatabaseOperations(session)
            
            # Test bulk user creation
            start_time = datetime.utcnow()
            
            users = []
            for i in range(100):
                user_data = TestFixtures.create_test_user_data()
                user_data["email"] = f"test{i}@example.com"
                user_data["username"] = f"testuser{i}"
                users.append(user_data)
            
            # Create users
            created_users = []
            for user_data in users:
                user = await db_ops.create_user(user_data)
                created_users.append(user)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Performance assertion
            assert len(created_users) == 100
            assert duration < 5.0  # 5 seconds for 100 users
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance"""
        from .cache import CacheManager
        
        cache_manager = CacheManager()
        
        # Test cache performance
        start_time = datetime.utcnow()
        
        # Set many cache entries
        for i in range(1000):
            await cache_manager.set(f"key_{i}", {"data": f"value_{i}"}, "default")
        
        # Get many cache entries
        for i in range(1000):
            result = await cache_manager.get(f"key_{i}", "default")
            assert result["data"] == f"value_{i}"
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Performance assertion
        assert duration < 2.0  # 2 seconds for 2000 operations


class TestRunner:
    """Test runner for executing test suites"""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.test_suites: List[TestSuite] = []
    
    def add_test_suite(self, suite: TestSuite):
        """Add test suite"""
        self.test_suites.append(suite)
    
    async def run_test_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run a test suite"""
        results = []
        
        logger.info(f"Running test suite: {suite.name}")
        
        # Setup
        if suite.setup_func:
            await suite.setup_func()
        
        try:
            # Run tests
            for test_name in suite.tests:
                result = await self._run_single_test(test_name, suite.timeout)
                results.append(result)
                self.test_results.append(result)
        
        finally:
            # Teardown
            if suite.teardown_func:
                await suite.teardown_func()
        
        return results
    
    async def _run_single_test(self, test_name: str, timeout: int) -> TestResult:
        """Run a single test"""
        start_time = datetime.utcnow()
        
        try:
            # Get test function
            test_func = getattr(self, test_name, None)
            if not test_func:
                raise ValueError(f"Test function {test_name} not found")
            
            # Run test with timeout
            await asyncio.wait_for(test_func(), timeout=timeout)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_name,
                status="passed",
                duration=duration
            )
            
        except asyncio.TimeoutError:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_name,
                status="failed",
                duration=duration,
                error_message="Test timed out"
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_name,
                status="failed",
                duration=duration,
                error_message=str(e),
                error_traceback=str(e.__traceback__)
            )
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        all_results = []
        
        for suite in self.test_suites:
            results = await self.run_test_suite(suite)
            all_results.extend(results)
        
        # Calculate statistics
        total_tests = len(all_results)
        passed_tests = len([r for r in all_results if r.status == "passed"])
        failed_tests = len([r for r in all_results if r.status == "failed"])
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "results": all_results
        }
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate test report"""
        report = f"""
# Test Report

## Summary
- Total Tests: {results['total_tests']}
- Passed: {results['passed_tests']}
- Failed: {results['failed_tests']}
- Success Rate: {results['success_rate']:.2f}%

## Test Results
"""
        
        for result in results['results']:
            status_emoji = "✅" if result.status == "passed" else "❌"
            report += f"- {status_emoji} {result.test_name} ({result.duration:.2f}s)\n"
            
            if result.error_message:
                report += f"  Error: {result.error_message}\n"
        
        return report


# Global test runner
test_runner = TestRunner()

# Register test suites
unit_test_suite = TestSuite(
    name="Unit Tests",
    description="Unit tests for core functionality",
    tests=[
        "test_user_creation",
        "test_video_analysis",
        "test_clip_generation",
        "test_authentication",
        "test_rate_limiting",
        "test_caching"
    ]
)

integration_test_suite = TestSuite(
    name="Integration Tests",
    description="Integration tests for system components",
    tests=[
        "test_video_processing_pipeline",
        "test_database_operations",
        "test_api_endpoints",
        "test_webhook_system"
    ]
)

performance_test_suite = TestSuite(
    name="Performance Tests",
    description="Performance tests for system scalability",
    tests=[
        "test_concurrent_video_processing",
        "test_database_performance",
        "test_cache_performance"
    ]
)

test_runner.add_test_suite(unit_test_suite)
test_runner.add_test_suite(integration_test_suite)
test_runner.add_test_suite(performance_test_suite)





























