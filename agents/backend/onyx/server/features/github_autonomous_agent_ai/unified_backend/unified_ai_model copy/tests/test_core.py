"""
Unit tests for Unified AI Model Core components
Tests config, performance monitor, and other core utilities.
"""

import pytest
import os
import time
from unittest.mock import patch, MagicMock


# Set environment variables before importing
os.environ["DEEPSEEK_API_KEY"] = "sk-test-key-12345"
os.environ["UNIFIED_AI_DEFAULT_MODEL"] = "deepseek-chat"
os.environ["UNIFIED_AI_PORT"] = "8050"


class TestConfig:
    """Tests for configuration module."""
    
    def test_config_loads_deepseek_key(self):
        """Test that DeepSeek API key is loaded from env."""
        from unified_ai_model.config import get_config
        
        config = get_config()
        assert config.deepseek.api_key == "sk-test-key-12345"
    
    def test_config_use_deepseek_property(self):
        """Test use_deepseek property when key is set."""
        from unified_ai_model.config import get_config
        
        config = get_config()
        assert config.use_deepseek is True
    
    def test_config_active_api_key(self):
        """Test active_api_key returns DeepSeek key when set."""
        from unified_ai_model.config import get_config
        
        config = get_config()
        assert config.active_api_key == "sk-test-key-12345"
    
    def test_config_default_model(self):
        """Test default model is set correctly."""
        from unified_ai_model.config import get_config
        
        config = get_config()
        assert config.default_model == "deepseek-chat"
    
    def test_config_server_port(self):
        """Test server port configuration."""
        from unified_ai_model.config import get_config
        
        config = get_config()
        assert config.server.port == 8050


class TestPerformanceMonitor:
    """Tests for performance monitor."""
    
    def test_monitor_initialization(self):
        """Test monitor initializes correctly."""
        from unified_ai_model.core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        assert monitor is not None
        assert monitor.start_time is not None
    
    def test_record_request(self):
        """Test recording a request."""
        from unified_ai_model.core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        initial_total = monitor.total_requests
        
        monitor.record_request(latency_ms=100.0, success=True)
        
        assert monitor.total_requests == initial_total + 1
        assert monitor.successful_requests >= 1
    
    def test_record_failed_request(self):
        """Test recording a failed request."""
        from unified_ai_model.core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        monitor.record_request(latency_ms=500.0, success=False)
        
        assert monitor.failed_requests >= 1
    
    def test_record_cache_hit(self):
        """Test recording cache hit."""
        from unified_ai_model.core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        initial_hits = monitor.cache_hits
        
        monitor.record_cache_hit()
        
        assert monitor.cache_hits == initial_hits + 1
    
    def test_record_cache_miss(self):
        """Test recording cache miss."""
        from unified_ai_model.core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        initial_misses = monitor.cache_misses
        
        monitor.record_cache_miss()
        
        assert monitor.cache_misses == initial_misses + 1
    
    def test_get_stats(self):
        """Test getting statistics."""
        from unified_ai_model.core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        monitor.record_request(latency_ms=100.0, success=True)
        monitor.record_cache_hit()
        
        stats = monitor.get_stats()
        
        assert "uptime_seconds" in stats
        assert "requests" in stats
        assert "cache" in stats
        assert "latency" in stats
        assert stats["requests"]["total"] >= 1
    
    def test_uptime_calculation(self):
        """Test uptime is calculated correctly."""
        from unified_ai_model.core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        time.sleep(0.1)  # Sleep 100ms
        
        stats = monitor.get_stats()
        
        assert stats["uptime_seconds"] >= 0.1
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        from unified_ai_model.core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Reset counters
        monitor.total_requests = 0
        monitor.successful_requests = 0
        monitor.failed_requests = 0
        
        # Record 3 successful and 1 failed
        monitor.record_request(latency_ms=100.0, success=True)
        monitor.record_request(latency_ms=100.0, success=True)
        monitor.record_request(latency_ms=100.0, success=True)
        monitor.record_request(latency_ms=100.0, success=False)
        
        stats = monitor.get_stats()
        
        assert stats["requests"]["total"] == 4
        assert stats["requests"]["error_rate"] == 25.0  # 1/4 = 25%


class TestContinuousAgent:
    """Tests for continuous agent."""
    
    def test_agent_creation(self):
        """Test creating an agent."""
        from unified_ai_model.core.continuous_agent import ContinuousAgent
        
        agent = ContinuousAgent(name="TestAgent")
        
        assert agent.name == "TestAgent"
        assert agent.agent_id is not None
    
    def test_agent_status(self):
        """Test agent status reporting."""
        from unified_ai_model.core.continuous_agent import ContinuousAgent, AgentStatus
        
        agent = ContinuousAgent(name="StatusAgent")
        
        status = agent.get_status()
        
        assert status["name"] == "StatusAgent"
        assert "status" in status
        assert "metrics" in status
    
    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test submitting a task."""
        from unified_ai_model.core.continuous_agent import ContinuousAgent
        
        agent = ContinuousAgent(name="TaskAgent")
        
        task_id = await agent.submit_task("Test task description", priority=5)
        
        assert task_id is not None
    
    @pytest.mark.asyncio
    async def test_pause_resume(self):
        """Test pause and resume functionality."""
        from unified_ai_model.core.continuous_agent import ContinuousAgent, AgentStatus
        
        agent = ContinuousAgent(name="PauseAgent")
        
        # Pause
        agent.pause()
        status = agent.get_status()
        assert status["status"] == AgentStatus.PAUSED.value
        
        # Resume
        agent.resume()
        # Status might be IDLE or RUNNING depending on implementation
        status = agent.get_status()
        assert status["status"] in [AgentStatus.IDLE.value, AgentStatus.RUNNING.value]
    
    @pytest.mark.asyncio
    async def test_stop_agent(self):
        """Test stopping an agent."""
        from unified_ai_model.core.continuous_agent import ContinuousAgent, AgentStatus
        
        agent = ContinuousAgent(name="StopAgent")
        
        await agent.stop()
        
        status = agent.get_status()
        assert status["status"] == AgentStatus.STOPPED.value
    
    def test_agent_metrics(self):
        """Test agent metrics initialization."""
        from unified_ai_model.core.continuous_agent import ContinuousAgent
        
        agent = ContinuousAgent(name="MetricsAgent")
        
        status = agent.get_status()
        metrics = status["metrics"]
        
        assert "tasks_completed" in metrics
        assert "tasks_failed" in metrics
        assert "total_processing_time" in metrics


class TestBatchProcessor:
    """Tests for batch processor."""
    
    def test_batch_processor_creation(self):
        """Test creating a batch processor."""
        from unified_ai_model.core.continuous_agent import BatchProcessor
        
        processor = BatchProcessor(batch_size=10, max_concurrent=5)
        
        assert processor.batch_size == 10
        assert processor.max_concurrent == 5
    
    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test processing a batch of items."""
        from unified_ai_model.core.continuous_agent import BatchProcessor
        
        processor = BatchProcessor(batch_size=3, max_concurrent=2)
        
        items = [1, 2, 3, 4, 5]
        results = []
        
        async def process_item(item):
            return item * 2
        
        batch_results = await processor.process_batch(
            items=items,
            processor_fn=process_item
        )
        
        assert len(batch_results) == 5


class TestPriorityTaskQueue:
    """Tests for priority task queue."""
    
    def test_queue_creation(self):
        """Test creating a priority queue."""
        from unified_ai_model.core.continuous_agent import PriorityTaskQueue
        
        queue = PriorityTaskQueue()
        
        assert queue is not None
        assert len(queue) == 0
    
    @pytest.mark.asyncio
    async def test_push_and_pop(self):
        """Test push and pop operations."""
        from unified_ai_model.core.continuous_agent import PriorityTaskQueue, AgentTask
        
        queue = PriorityTaskQueue()
        
        task = AgentTask(description="Test task", priority=5)
        await queue.push(task)
        
        assert len(queue) == 1
        
        popped = await queue.pop()
        
        assert popped is not None
        assert popped.description == "Test task"
        assert len(queue) == 0
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test that higher priority tasks come first."""
        from unified_ai_model.core.continuous_agent import PriorityTaskQueue, AgentTask
        
        queue = PriorityTaskQueue()
        
        # Add tasks with different priorities
        low_priority = AgentTask(description="Low", priority=1)
        high_priority = AgentTask(description="High", priority=10)
        medium_priority = AgentTask(description="Medium", priority=5)
        
        await queue.push(low_priority)
        await queue.push(high_priority)
        await queue.push(medium_priority)
        
        # Pop should return highest priority first
        first = await queue.pop()
        assert first.description == "High"
        
        second = await queue.pop()
        assert second.description == "Medium"
        
        third = await queue.pop()
        assert third.description == "Low"


class TestWorkerPool:
    """Tests for worker pool."""
    
    def test_worker_pool_creation(self):
        """Test creating a worker pool."""
        from unified_ai_model.core.continuous_agent import WorkerPool, PriorityTaskQueue
        
        queue = PriorityTaskQueue()
        pool = WorkerPool(num_workers=3, task_queue=queue)
        
        assert pool.num_workers == 3
    
    @pytest.mark.asyncio
    async def test_worker_pool_start_stop(self):
        """Test starting and stopping worker pool."""
        from unified_ai_model.core.continuous_agent import WorkerPool, PriorityTaskQueue
        
        queue = PriorityTaskQueue()
        pool = WorkerPool(num_workers=2, task_queue=queue)
        
        async def dummy_processor(task):
            pass
        
        await pool.start(dummy_processor)
        assert pool.is_running
        
        await pool.stop()
        assert not pool.is_running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



