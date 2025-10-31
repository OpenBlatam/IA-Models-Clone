"""
Comprehensive Test Suite for Refactored Opus Clip

Advanced testing framework with:
- Unit tests
- Integration tests
- Performance tests
- End-to-end tests
- Load testing
- Mock testing
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable
import asyncio
import pytest
import unittest
import time
import tempfile
import os
from pathlib import Path
import json
import structlog
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import cv2
import moviepy.editor as mp

# Import refactored components
from ..core.base_processor import BaseProcessor, ProcessorResult, ProcessorConfig
from ..core.config_manager import ConfigManager, Environment
from ..core.job_manager import JobManager, JobPriority
from ..processors.refactored_analyzer import RefactoredOpusClipAnalyzer
from ..processors.refactored_exporter import RefactoredOpusClipExporter
from ..monitoring.performance_monitor import PerformanceMonitor

logger = structlog.get_logger("test_suite")

class TestBaseProcessor(unittest.TestCase):
    """Test cases for BaseProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ProcessorConfig(
            max_retries=2,
            timeout_seconds=10.0,
            enable_caching=True,
            cache_ttl_seconds=60
        )
        
        class TestProcessor(BaseProcessor):
            async def _process_impl(self, input_data):
                await asyncio.sleep(0.1)  # Simulate processing
                return ProcessorResult(
                    success=True,
                    data={"result": "test"},
                    processing_time=0.1
                )
        
        self.processor = TestProcessor(self.config)
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.config.max_retries, 2)
        self.assertEqual(self.processor.config.timeout_seconds, 10.0)
        self.assertTrue(self.processor.config.enable_caching)
    
    @pytest.mark.asyncio
    async def test_successful_processing(self):
        """Test successful processing."""
        result = await self.processor.process({"test": "data"})
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["result"], "test")
        self.assertGreater(result.processing_time, 0)
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test result caching."""
        input_data = {"test": "data"}
        
        # First call
        result1 = await self.processor.process(input_data)
        
        # Second call should use cache
        result2 = await self.processor.process(input_data)
        
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        # Should be faster due to caching
        self.assertLessEqual(result2.processing_time, result1.processing_time)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling."""
        class ErrorProcessor(BaseProcessor):
            async def _process_impl(self, input_data):
                raise Exception("Test error")
        
        processor = ErrorProcessor(self.config)
        result = await processor.process({"test": "data"})
        
        self.assertFalse(result.success)
        self.assertIn("Test error", result.error)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling."""
        class SlowProcessor(BaseProcessor):
            async def _process_impl(self, input_data):
                await asyncio.sleep(15.0)  # Longer than timeout
                return ProcessorResult(success=True, data={"result": "test"})
        
        config = ProcessorConfig(timeout_seconds=1.0)
        processor = SlowProcessor(config)
        result = await processor.process({"test": "data"})
        
        self.assertFalse(result.success)
        self.assertIn("timeout", result.error.lower())

class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        self.assertIsNotNone(self.config_manager.database)
        self.assertIsNotNone(self.config_manager.redis)
        self.assertIsNotNone(self.config_manager.ai)
        self.assertIsNotNone(self.config_manager.video)
        self.assertIsNotNone(self.config_manager.performance)
        self.assertIsNotNone(self.config_manager.security)
    
    def test_environment_detection(self):
        """Test environment detection."""
        env = self.config_manager.environment
        self.assertIn(env, [Environment.DEVELOPMENT, Environment.PRODUCTION, Environment.STAGING, Environment.TESTING])
    
    def test_database_url_generation(self):
        """Test database URL generation."""
        url = self.config_manager.get_database_url()
        self.assertIn("postgresql://", url)
        self.assertIn(self.config_manager.database.host, url)
        self.assertIn(str(self.config_manager.database.port), url)
    
    def test_redis_url_generation(self):
        """Test Redis URL generation."""
        url = self.config_manager.get_redis_url()
        self.assertIn("redis://", url)
        self.assertIn(self.config_manager.redis.host, url)
        self.assertIn(str(self.config_manager.redis.port), url)
    
    def test_config_summary(self):
        """Test configuration summary."""
        summary = self.config_manager.get_config_summary()
        self.assertIn("environment", summary)
        self.assertIn("database", summary)
        self.assertIn("redis", summary)
        self.assertIn("ai", summary)
        self.assertIn("video", summary)
        self.assertIn("performance", summary)

class TestJobManager(unittest.TestCase):
    """Test cases for JobManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.job_manager = JobManager(max_workers=2, enable_persistence=False)
        
        # Register test processor
        async def test_processor(data, metadata):
            await asyncio.sleep(0.1)
            return {"result": "processed", "data": data}
        
        self.job_manager.register_processor("test", test_processor)
    
    @pytest.mark.asyncio
    async def test_job_submission(self):
        """Test job submission."""
        job_id = await self.job_manager.submit_job(
            "test",
            {"test": "data"},
            JobPriority.NORMAL
        )
        
        self.assertIsNotNone(job_id)
        self.assertIn(job_id, self.job_manager.jobs)
    
    @pytest.mark.asyncio
    async def test_job_processing(self):
        """Test job processing."""
        job_id = await self.job_manager.submit_job(
            "test",
            {"test": "data"},
            JobPriority.NORMAL
        )
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        status = await self.job_manager.get_job_status(job_id)
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "completed")
    
    @pytest.mark.asyncio
    async def test_job_cancellation(self):
        """Test job cancellation."""
        job_id = await self.job_manager.submit_job(
            "test",
            {"test": "data"},
            JobPriority.NORMAL
        )
        
        success = await self.job_manager.cancel_job(job_id)
        self.assertTrue(success)
        
        status = await self.job_manager.get_job_status(job_id)
        self.assertEqual(status["status"], "cancelled")
    
    @pytest.mark.asyncio
    async def test_job_retry(self):
        """Test job retry."""
        # Submit a job that will fail
        job_id = await self.job_manager.submit_job(
            "nonexistent_processor",
            {"test": "data"},
            JobPriority.NORMAL
        )
        
        # Wait for failure
        await asyncio.sleep(0.5)
        
        # Retry the job
        success = await self.job_manager.retry_job(job_id)
        self.assertTrue(success)
    
    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test job statistics."""
        stats = await self.job_manager.get_statistics()
        
        self.assertIn("total_jobs", stats)
        self.assertIn("completed_jobs", stats)
        self.assertIn("failed_jobs", stats)
        self.assertIn("active_jobs", stats)

class TestRefactoredAnalyzer(unittest.TestCase):
    """Test cases for RefactoredOpusClipAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ProcessorConfig(
            max_retries=1,
            timeout_seconds=30.0,
            enable_caching=False
        )
        self.app_config = ConfigManager()
        self.analyzer = RefactoredOpusClipAnalyzer(self.config, self.app_config)
    
    def create_test_video(self) -> str:
        """Create a test video file."""
        # Create a simple test video
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "test_video.mp4")
        
        # Create a simple video using moviepy
        clip = mp.ColorClip(size=(640, 480), color=(255, 0, 0), duration=5)
        clip.write_videofile(video_path, fps=24, verbose=False, logger=None)
        
        return video_path
    
    @pytest.mark.asyncio
    async def test_video_analysis(self):
        """Test video analysis."""
        video_path = self.create_test_video()
        
        try:
            result = await self.analyzer.process({
                "video_path": video_path,
                "max_clips": 5,
                "min_duration": 1.0,
                "max_duration": 10.0
            })
            
            self.assertTrue(result.success)
            self.assertIn("video_duration", result.data)
            self.assertIn("segments", result.data)
            self.assertIn("viral_scores", result.data)
            
        finally:
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
    
    @pytest.mark.asyncio
    async def test_invalid_video_path(self):
        """Test handling of invalid video path."""
        result = await self.analyzer.process({
            "video_path": "/nonexistent/video.mp4",
            "max_clips": 5
        })
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)
    
    @pytest.mark.asyncio
    async def test_missing_required_parameters(self):
        """Test handling of missing required parameters."""
        result = await self.analyzer.process({})
        
        self.assertFalse(result.success)
        self.assertIn("video_path is required", result.error)

class TestRefactoredExporter(unittest.TestCase):
    """Test cases for RefactoredOpusClipExporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ProcessorConfig(
            max_retries=1,
            timeout_seconds=30.0,
            enable_caching=False
        )
        self.app_config = ConfigManager()
        self.exporter = RefactoredOpusClipExporter(self.config, self.app_config)
    
    def create_test_video(self) -> str:
        """Create a test video file."""
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "test_video.mp4")
        
        clip = mp.ColorClip(size=(640, 480), color=(0, 255, 0), duration=10)
        clip.write_videofile(video_path, fps=24, verbose=False, logger=None)
        
        return video_path
    
    @pytest.mark.asyncio
    async def test_clip_export(self):
        """Test clip export."""
        video_path = self.create_test_video()
        
        try:
            segments = [
                {
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "duration": 2.0,
                    "segment_id": "test_segment_1"
                },
                {
                    "start_time": 3.0,
                    "end_time": 5.0,
                    "duration": 2.0,
                    "segment_id": "test_segment_2"
                }
            ]
            
            result = await self.exporter.process({
                "video_path": video_path,
                "segments": segments,
                "output_format": "mp4",
                "quality": "high"
            })
            
            self.assertTrue(result.success)
            self.assertIn("exported_clips", result.data)
            self.assertEqual(len(result.data["exported_clips"]), 2)
            
        finally:
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
    
    @pytest.mark.asyncio
    async def test_invalid_video_path(self):
        """Test handling of invalid video path."""
        result = await self.exporter.process({
            "video_path": "/nonexistent/video.mp4",
            "segments": []
        })
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)
    
    @pytest.mark.asyncio
    async def test_empty_segments(self):
        """Test handling of empty segments."""
        video_path = self.create_test_video()
        
        try:
            result = await self.exporter.process({
                "video_path": video_path,
                "segments": [],
                "output_format": "mp4"
            })
            
            self.assertFalse(result.success)
            self.assertIn("segments are required", result.error)
            
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for PerformanceMonitor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(
            max_metrics_history=1000,
            enable_file_logging=False
        )
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection."""
        await self.monitor.start_monitoring()
        
        # Wait for some metrics to be collected
        await asyncio.sleep(2)
        
        summary = await self.monitor.get_performance_summary()
        
        self.assertIn("current_metrics", summary)
        self.assertIn("performance_stats", summary)
        self.assertIn("monitoring_active", summary)
        self.assertTrue(summary["monitoring_active"])
        
        await self.monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_request_recording(self):
        """Test request recording."""
        await self.monitor.record_request("/api/test", 1.5, True)
        await self.monitor.record_request("/api/test", 0.8, True)
        await self.monitor.record_request("/api/test", 2.0, False)
        
        summary = await self.monitor.get_performance_summary()
        
        self.assertEqual(summary["performance_stats"]["total_requests"], 3)
        self.assertEqual(summary["performance_stats"]["successful_requests"], 2)
        self.assertEqual(summary["performance_stats"]["failed_requests"], 1)
        self.assertGreater(summary["performance_stats"]["average_response_time"], 0)
    
    @pytest.mark.asyncio
    async def test_alert_system(self):
        """Test alert system."""
        # Set low thresholds for testing
        self.monitor.alert_thresholds = {
            "cpu_usage": {"warning": 0.1, "error": 0.2, "critical": 0.3}
        }
        
        # Record high CPU usage
        await self.monitor._record_metric("cpu_usage", 50.0, "percent")
        
        # Check for alerts
        await self.monitor._check_alerts()
        
        alerts = await self.monitor.get_alerts()
        self.assertGreater(len(alerts), 0)
    
    @pytest.mark.asyncio
    async def test_optimization_suggestions(self):
        """Test optimization suggestions."""
        # Set high metrics to trigger suggestions
        self.monitor.current_metrics["cpu_usage"] = 85.0
        self.monitor.current_metrics["memory_usage"] = 90.0
        self.monitor.performance_stats["average_response_time"] = 6.0
        self.monitor.performance_stats["total_requests"] = 100
        self.monitor.performance_stats["failed_requests"] = 10
        
        suggestions = await self.monitor.get_optimization_suggestions()
        
        self.assertGreater(len(suggestions), 0)
        self.assertIn("cpu_optimization", [s["type"] for s in suggestions])
        self.assertIn("memory_optimization", [s["type"] for s in suggestions])

class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full workflow from analysis to export."""
        # Initialize components
        config_manager = ConfigManager()
        job_manager = JobManager(max_workers=2, enable_persistence=False)
        
        processor_config = ProcessorConfig(
            max_retries=1,
            timeout_seconds=30.0,
            enable_caching=False
        )
        
        analyzer = RefactoredOpusClipAnalyzer(processor_config, config_manager)
        exporter = RefactoredOpusClipExporter(processor_config, config_manager)
        
        # Register processors
        job_manager.register_processor("video_analysis", analyzer.process)
        job_manager.register_processor("clip_export", exporter.process)
        
        try:
            # Create test video
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, "test_video.mp4")
            
            clip = mp.ColorClip(size=(640, 480), color=(255, 0, 0), duration=10)
            clip.write_videofile(video_path, fps=24, verbose=False, logger=None)
            
            # Submit analysis job
            analysis_job_id = await job_manager.submit_job(
                "video_analysis",
                {
                    "video_path": video_path,
                    "max_clips": 3,
                    "min_duration": 1.0,
                    "max_duration": 5.0
                }
            )
            
            # Wait for analysis to complete
            await asyncio.sleep(5)
            
            analysis_result = await job_manager.get_job_result(analysis_job_id)
            self.assertIsNotNone(analysis_result)
            self.assertTrue(analysis_result.success)
            
            # Extract segments for export
            segments = analysis_result.data.get("segments", [])
            if segments:
                # Submit export job
                export_job_id = await job_manager.submit_job(
                    "clip_export",
                    {
                        "video_path": video_path,
                        "segments": segments[:2],  # Export first 2 segments
                        "output_format": "mp4",
                        "quality": "high"
                    }
                )
                
                # Wait for export to complete
                await asyncio.sleep(5)
                
                export_result = await job_manager.get_job_result(export_job_id)
                self.assertIsNotNone(export_result)
                self.assertTrue(export_result.success)
                
                # Verify exported clips
                exported_clips = export_result.data.get("exported_clips", [])
                self.assertGreater(len(exported_clips), 0)
                
                # Clean up exported files
                for clip in exported_clips:
                    if os.path.exists(clip["path"]):
                        os.remove(clip["path"])
            
        finally:
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
            
            await job_manager.shutdown()
            await analyzer.shutdown()
            await exporter.shutdown()

class TestLoadTesting(unittest.TestCase):
    """Load testing."""
    
    @pytest.mark.asyncio
    async def test_concurrent_jobs(self):
        """Test concurrent job processing."""
        job_manager = JobManager(max_workers=4, enable_persistence=False)
        
        async def test_processor(data, metadata):
            await asyncio.sleep(0.1)
            return {"result": "processed", "job_id": data.get("job_id")}
        
        job_manager.register_processor("test", test_processor)
        
        try:
            # Submit multiple concurrent jobs
            job_ids = []
            for i in range(10):
                job_id = await job_manager.submit_job(
                    "test",
                    {"job_id": i, "data": f"test_data_{i}"}
                )
                job_ids.append(job_id)
            
            # Wait for all jobs to complete
            await asyncio.sleep(2)
            
            # Check results
            stats = await job_manager.get_statistics()
            self.assertEqual(stats["total_jobs"], 10)
            self.assertEqual(stats["completed_jobs"], 10)
            
        finally:
            await job_manager.shutdown()

def run_all_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestBaseProcessor,
        TestConfigManager,
        TestJobManager,
        TestRefactoredAnalyzer,
        TestRefactoredExporter,
        TestPerformanceMonitor,
        TestIntegration,
        TestLoadTesting
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)


