"""
Comprehensive Test Suite

Advanced testing framework for the Ultimate Opus Clip system with unit tests,
integration tests, performance tests, and end-to-end tests.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import asyncio
import pytest
import unittest
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
import json
import time
from pathlib import Path
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum

# Import refactored components
from ..core.base_processor import BaseProcessor, ProcessorConfig, ProcessingResult, ProcessorStatus
from ..core.config_manager import ConfigManager
from ..core.job_manager import JobManager, JobPriority, JobStatus
from ..processors.refactored_content_curation import RefactoredContentCurationEngine, ContentCurationConfig
from ..monitoring.performance_monitor import PerformanceMonitor, MetricType

class TestType(Enum):
    """Types of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    END_TO_END = "end_to_end"

@dataclass
class TestResult:
    """Test execution result."""
    test_name: str
    test_type: TestType
    success: bool
    duration: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None

class TestSuite:
    """Comprehensive test suite for Ultimate Opus Clip system."""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.temp_dir = None
        self.sample_video_path = None
        self.setup_complete = False
    
    async def setup(self):
        """Setup test environment."""
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="opus_clip_test_")
            
            # Create sample video for testing
            await self._create_sample_video()
            
            self.setup_complete = True
            print("Test suite setup completed")
            
        except Exception as e:
            print(f"Test suite setup failed: {e}")
            raise
    
    async def teardown(self):
        """Cleanup test environment."""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
            
            print("Test suite teardown completed")
            
        except Exception as e:
            print(f"Test suite teardown failed: {e}")
    
    async def _create_sample_video(self):
        """Create a sample video for testing."""
        try:
            # Create a simple test video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                os.path.join(self.temp_dir, "sample_video.mp4"),
                fourcc, 30.0, (640, 480)
            )
            
            # Generate 90 frames (3 seconds at 30fps)
            for i in range(90):
                # Create a frame with some content
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add some visual content
                cv2.rectangle(frame, (50, 50), (200, 150), (0, 255, 0), -1)
                cv2.putText(frame, f"Frame {i}", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
            self.sample_video_path = os.path.join(self.temp_dir, "sample_video.mp4")
            
        except Exception as e:
            print(f"Failed to create sample video: {e}")
            raise
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all tests in the suite."""
        if not self.setup_complete:
            await self.setup()
        
        try:
            print("Starting comprehensive test suite...")
            
            # Run unit tests
            await self.run_unit_tests()
            
            # Run integration tests
            await self.run_integration_tests()
            
            # Run performance tests
            await self.run_performance_tests()
            
            # Run end-to-end tests
            await self.run_end_to_end_tests()
            
            # Generate test report
            await self.generate_test_report()
            
            return self.test_results
            
        except Exception as e:
            print(f"Test suite execution failed: {e}")
            return self.test_results
    
    async def run_unit_tests(self):
        """Run unit tests for individual components."""
        print("Running unit tests...")
        
        # Test ConfigManager
        await self._test_config_manager()
        
        # Test ProcessorConfig
        await self._test_processor_config()
        
        # Test JobManager
        await self._test_job_manager()
        
        # Test ContentCurationConfig
        await self._test_content_curation_config()
        
        print("Unit tests completed")
    
    async def _test_config_manager(self):
        """Test ConfigManager functionality."""
        start_time = time.time()
        
        try:
            # Test configuration loading
            config = ConfigManager()
            
            # Test getting configuration values
            log_level = config.get("logging.level")
            assert log_level is not None
            
            # Test setting configuration values
            config.set("test.value", "test_data")
            assert config.get("test.value") == "test_data"
            
            # Test feature enabling
            config.set("features.test_feature.enabled", True)
            assert config.is_feature_enabled("test_feature") == True
            
            self._record_test_result("config_manager", TestType.UNIT, True, time.time() - start_time)
            
        except Exception as e:
            self._record_test_result("config_manager", TestType.UNIT, False, time.time() - start_time, str(e))
    
    async def _test_processor_config(self):
        """Test ProcessorConfig functionality."""
        start_time = time.time()
        
        try:
            config = ProcessorConfig(
                name="test_processor",
                version="1.0.0",
                enabled=True,
                max_workers=4,
                timeout=300.0,
                retry_attempts=3
            )
            
            assert config.name == "test_processor"
            assert config.version == "1.0.0"
            assert config.enabled == True
            assert config.max_workers == 4
            assert config.timeout == 300.0
            assert config.retry_attempts == 3
            
            self._record_test_result("processor_config", TestType.UNIT, True, time.time() - start_time)
            
        except Exception as e:
            self._record_test_result("processor_config", TestType.UNIT, False, time.time() - start_time, str(e))
    
    async def _test_job_manager(self):
        """Test JobManager functionality."""
        start_time = time.time()
        
        try:
            # Create job manager with test database
            test_db_path = os.path.join(self.temp_dir, "test_jobs.db")
            job_manager = JobManager(db_path=test_db_path)
            
            # Test job submission
            job_id = await job_manager.submit_job(
                job_type="test_job",
                input_data={"test": "data"},
                priority=JobPriority.NORMAL
            )
            
            assert job_id is not None
            
            # Test job status retrieval
            status = await job_manager.get_job_status(job_id)
            assert status is not None
            assert status["job_id"] == job_id
            
            # Test job cancellation
            cancelled = await job_manager.cancel_job(job_id)
            assert cancelled == True
            
            self._record_test_result("job_manager", TestType.UNIT, True, time.time() - start_time)
            
        except Exception as e:
            self._record_test_result("job_manager", TestType.UNIT, False, time.time() - start_time, str(e))
    
    async def _test_content_curation_config(self):
        """Test ContentCurationConfig functionality."""
        start_time = time.time()
        
        try:
            config = ContentCurationConfig(
                min_segment_duration=3.0,
                max_segment_duration=30.0,
                engagement_threshold=0.6,
                viral_threshold=0.7,
                max_clips=10
            )
            
            assert config.min_segment_duration == 3.0
            assert config.max_segment_duration == 30.0
            assert config.engagement_threshold == 0.6
            assert config.viral_threshold == 0.7
            assert config.max_clips == 10
            
            self._record_test_result("content_curation_config", TestType.UNIT, True, time.time() - start_time)
            
        except Exception as e:
            self._record_test_result("content_curation_config", TestType.UNIT, False, time.time() - start_time, str(e))
    
    async def run_integration_tests(self):
        """Run integration tests for component interactions."""
        print("Running integration tests...")
        
        # Test processor-job manager integration
        await self._test_processor_job_integration()
        
        # Test config-processor integration
        await self._test_config_processor_integration()
        
        print("Integration tests completed")
    
    async def _test_processor_job_integration(self):
        """Test integration between processors and job manager."""
        start_time = time.time()
        
        try:
            # Create test job manager
            test_db_path = os.path.join(self.temp_dir, "test_integration.db")
            job_manager = JobManager(db_path=test_db_path)
            
            # Create mock processor
            mock_processor = Mock()
            mock_processor.process = AsyncMock(return_value=ProcessingResult(
                success=True,
                processor_name="test_processor",
                processing_time=1.0,
                result_data={"test": "result"}
            ))
            
            # Register processor
            job_manager.register_processor("test_processor", mock_processor.process)
            
            # Submit job
            job_id = await job_manager.submit_job(
                job_type="test_processor",
                input_data={"test": "input"},
                priority=JobPriority.NORMAL
            )
            
            # Process job
            result = await job_manager.process_job(job_id, "test_processor", {"test": "input"})
            
            assert result.success == True
            assert result.result_data["test"] == "result"
            
            self._record_test_result("processor_job_integration", TestType.INTEGRATION, True, time.time() - start_time)
            
        except Exception as e:
            self._record_test_result("processor_job_integration", TestType.INTEGRATION, False, time.time() - start_time, str(e))
    
    async def _test_config_processor_integration(self):
        """Test integration between config manager and processors."""
        start_time = time.time()
        
        try:
            # Create config manager
            config = ConfigManager()
            
            # Set processor configuration
            config.set("features.test_processor.enabled", True)
            config.set("features.test_processor.max_workers", 2)
            config.set("features.test_processor.timeout", 60.0)
            
            # Create processor config
            processor_config = ProcessorConfig(
                name="test_processor",
                version="1.0.0",
                enabled=config.is_feature_enabled("test_processor"),
                max_workers=config.get("features.test_processor.max_workers", 4),
                timeout=config.get("features.test_processor.timeout", 300.0)
            )
            
            assert processor_config.enabled == True
            assert processor_config.max_workers == 2
            assert processor_config.timeout == 60.0
            
            self._record_test_result("config_processor_integration", TestType.INTEGRATION, True, time.time() - start_time)
            
        except Exception as e:
            self._record_test_result("config_processor_integration", TestType.INTEGRATION, False, time.time() - start_time, str(e))
    
    async def run_performance_tests(self):
        """Run performance tests to measure system performance."""
        print("Running performance tests...")
        
        # Test job processing performance
        await self._test_job_processing_performance()
        
        # Test memory usage
        await self._test_memory_usage()
        
        # Test concurrent processing
        await self._test_concurrent_processing()
        
        print("Performance tests completed")
    
    async def _test_job_processing_performance(self):
        """Test job processing performance."""
        start_time = time.time()
        
        try:
            # Create job manager
            test_db_path = os.path.join(self.temp_dir, "test_performance.db")
            job_manager = JobManager(db_path=test_db_path)
            
            # Create fast mock processor
            async def fast_processor(input_data):
                await asyncio.sleep(0.1)  # Simulate processing
                return ProcessingResult(
                    success=True,
                    processor_name="fast_processor",
                    processing_time=0.1,
                    result_data={"processed": True}
                )
            
            job_manager.register_processor("fast_processor", fast_processor)
            
            # Process multiple jobs and measure performance
            job_count = 10
            processing_times = []
            
            for i in range(job_count):
                job_start = time.time()
                
                job_id = await job_manager.submit_job(
                    job_type="fast_processor",
                    input_data={"job": i},
                    priority=JobPriority.NORMAL
                )
                
                result = await job_manager.process_job(job_id, "fast_processor", {"job": i})
                
                job_end = time.time()
                processing_times.append(job_end - job_start)
            
            # Calculate performance metrics
            avg_processing_time = np.mean(processing_times)
            max_processing_time = np.max(processing_times)
            min_processing_time = np.min(processing_times)
            
            # Performance assertions
            assert avg_processing_time < 1.0, f"Average processing time too high: {avg_processing_time}"
            assert max_processing_time < 2.0, f"Max processing time too high: {max_processing_time}"
            
            metrics = {
                "job_count": job_count,
                "avg_processing_time": avg_processing_time,
                "max_processing_time": max_processing_time,
                "min_processing_time": min_processing_time,
                "jobs_per_second": job_count / (time.time() - start_time)
            }
            
            self._record_test_result("job_processing_performance", TestType.PERFORMANCE, True, time.time() - start_time, metrics=metrics)
            
        except Exception as e:
            self._record_test_result("job_processing_performance", TestType.PERFORMANCE, False, time.time() - start_time, str(e))
    
    async def _test_memory_usage(self):
        """Test memory usage during processing."""
        start_time = time.time()
        
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Create job manager
            test_db_path = os.path.join(self.temp_dir, "test_memory.db")
            job_manager = JobManager(db_path=test_db_path)
            
            # Create memory-intensive processor
            async def memory_processor(input_data):
                # Allocate some memory
                data = np.random.random((1000, 1000))
                await asyncio.sleep(0.01)
                return ProcessingResult(
                    success=True,
                    processor_name="memory_processor",
                    processing_time=0.01,
                    result_data={"memory_used": data.nbytes}
                )
            
            job_manager.register_processor("memory_processor", memory_processor)
            
            # Process jobs
            for i in range(5):
                job_id = await job_manager.submit_job(
                    job_type="memory_processor",
                    input_data={"job": i},
                    priority=JobPriority.NORMAL
                )
                
                result = await job_manager.process_job(job_id, "memory_processor", {"job": i})
            
            # Force garbage collection
            gc.collect()
            
            # Get final memory usage
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory usage should be reasonable
            assert memory_increase < 100, f"Memory usage too high: {memory_increase} MB"
            
            metrics = {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase
            }
            
            self._record_test_result("memory_usage", TestType.PERFORMANCE, True, time.time() - start_time, metrics=metrics)
            
        except Exception as e:
            self._record_test_result("memory_usage", TestType.PERFORMANCE, False, time.time() - start_time, str(e))
    
    async def _test_concurrent_processing(self):
        """Test concurrent job processing."""
        start_time = time.time()
        
        try:
            # Create job manager
            test_db_path = os.path.join(self.temp_dir, "test_concurrent.db")
            job_manager = JobManager(db_path=test_db_path, max_workers=4)
            
            # Create processor
            async def concurrent_processor(input_data):
                await asyncio.sleep(0.5)  # Simulate processing
                return ProcessingResult(
                    success=True,
                    processor_name="concurrent_processor",
                    processing_time=0.5,
                    result_data={"processed": True}
                )
            
            job_manager.register_processor("concurrent_processor", concurrent_processor)
            
            # Submit multiple jobs concurrently
            job_count = 8
            tasks = []
            
            for i in range(job_count):
                job_id = await job_manager.submit_job(
                    job_type="concurrent_processor",
                    input_data={"job": i},
                    priority=JobPriority.NORMAL
                )
                
                task = asyncio.create_task(
                    job_manager.process_job(job_id, "concurrent_processor", {"job": i})
                )
                tasks.append(task)
            
            # Wait for all jobs to complete
            results = await asyncio.gather(*tasks)
            
            # All jobs should succeed
            assert all(result.success for result in results)
            
            # Should complete faster than sequential processing
            total_time = time.time() - start_time
            sequential_time = job_count * 0.5  # 8 * 0.5 = 4 seconds
            
            assert total_time < sequential_time, f"Concurrent processing not faster: {total_time} vs {sequential_time}"
            
            metrics = {
                "job_count": job_count,
                "total_time": total_time,
                "sequential_time": sequential_time,
                "speedup": sequential_time / total_time
            }
            
            self._record_test_result("concurrent_processing", TestType.PERFORMANCE, True, time.time() - start_time, metrics=metrics)
            
        except Exception as e:
            self._record_test_result("concurrent_processing", TestType.PERFORMANCE, False, time.time() - start_time, str(e))
    
    async def run_end_to_end_tests(self):
        """Run end-to-end tests for complete workflows."""
        print("Running end-to-end tests...")
        
        # Test complete video processing workflow
        await self._test_complete_video_processing()
        
        print("End-to-end tests completed")
    
    async def _test_complete_video_processing(self):
        """Test complete video processing workflow."""
        start_time = time.time()
        
        try:
            if not self.sample_video_path:
                raise Exception("Sample video not available")
            
            # Create content curation engine
            config = ContentCurationConfig(
                min_segment_duration=1.0,
                max_segment_duration=5.0,
                engagement_threshold=0.3,
                max_clips=3
            )
            
            engine = RefactoredContentCurationEngine()
            
            # Process video
            input_data = {
                "video_path": self.sample_video_path
            }
            
            result = await engine.process(str(uuid.uuid4()), input_data)
            
            # Verify result
            assert result.success == True
            assert "clips" in result.result_data
            assert "processing_stats" in result.result_data
            
            clips = result.result_data["clips"]
            assert len(clips) > 0, "No clips generated"
            
            # Verify clip structure
            for clip in clips:
                assert "start_time" in clip
                assert "end_time" in clip
                assert "duration" in clip
                assert "score" in clip
                assert clip["duration"] > 0
                assert clip["score"] >= 0
            
            metrics = {
                "clips_generated": len(clips),
                "processing_time": result.processing_time,
                "video_duration": 3.0,  # Our test video is 3 seconds
                "clips_per_second": len(clips) / 3.0
            }
            
            self._record_test_result("complete_video_processing", TestType.END_TO_END, True, time.time() - start_time, metrics=metrics)
            
        except Exception as e:
            self._record_test_result("complete_video_processing", TestType.END_TO_END, False, time.time() - start_time, str(e))
    
    def _record_test_result(self, test_name: str, test_type: TestType, success: bool, duration: float, error_message: str = None, metrics: Dict[str, Any] = None):
        """Record a test result."""
        result = TestResult(
            test_name=test_name,
            test_type=test_type,
            success=success,
            duration=duration,
            error_message=error_message,
            metrics=metrics or {}
        )
        
        self.test_results.append(result)
        
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {test_name} ({test_type.value}) - {duration:.2f}s")
        
        if error_message:
            print(f"    Error: {error_message}")
    
    async def generate_test_report(self):
        """Generate comprehensive test report."""
        try:
            total_tests = len(self.test_results)
            passed_tests = len([r for r in self.test_results if r.success])
            failed_tests = total_tests - passed_tests
            
            # Calculate test statistics by type
            test_stats = {}
            for test_type in TestType:
                type_results = [r for r in self.test_results if r.test_type == test_type]
                test_stats[test_type.value] = {
                    "total": len(type_results),
                    "passed": len([r for r in type_results if r.success]),
                    "failed": len([r for r in type_results if not r.success]),
                    "avg_duration": np.mean([r.duration for r in type_results]) if type_results else 0
                }
            
            # Generate report
            report = {
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                    "total_duration": sum([r.duration for r in self.test_results])
                },
                "test_statistics": test_stats,
                "failed_tests": [
                    {
                        "name": r.test_name,
                        "type": r.test_type.value,
                        "error": r.error_message,
                        "duration": r.duration
                    }
                    for r in self.test_results if not r.success
                ],
                "performance_metrics": self._extract_performance_metrics()
            }
            
            # Save report
            report_path = os.path.join(self.temp_dir, "test_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"\nTest Report Generated:")
            print(f"  Total Tests: {total_tests}")
            print(f"  Passed: {passed_tests}")
            print(f"  Failed: {failed_tests}")
            print(f"  Success Rate: {passed_tests/total_tests*100:.1f}%")
            print(f"  Report saved to: {report_path}")
            
        except Exception as e:
            print(f"Failed to generate test report: {e}")
    
    def _extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics from test results."""
        try:
            performance_tests = [r for r in self.test_results if r.test_type == TestType.PERFORMANCE and r.metrics]
            
            metrics = {}
            for test in performance_tests:
                metrics[test.test_name] = test.metrics
            
            return metrics
            
        except Exception as e:
            print(f"Failed to extract performance metrics: {e}")
            return {}

# Test runner function
async def run_tests():
    """Run the complete test suite."""
    test_suite = TestSuite()
    
    try:
        results = await test_suite.run_all_tests()
        return results
        
    finally:
        await test_suite.teardown()

if __name__ == "__main__":
    # Run tests
    asyncio.run(run_tests())


