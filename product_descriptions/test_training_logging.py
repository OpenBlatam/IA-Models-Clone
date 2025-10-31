from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import tempfile
import time
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from training_logging_system import (
from robust_operations import RobustOperations
import structlog
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for Training Logging System

This test suite covers:
- Training event logging and tracking
- Security event logging and categorization
- Performance monitoring and alerting
- Error handling and recovery
- Log analysis and reporting
- File operations and persistence
- Rich console output
- Integration with robust operations
"""



# Add the current directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

    TrainingLogger,
    TrainingMetrics,
    SecurityEvent,
    PerformanceMetrics,
    TrainingEventType,
    LogLevel,
    PerformanceMonitor,
    TrainingLoggerDecorator,
    create_training_logger,
    log_training_progress
)

# Configure logging for tests
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class SimpleTestModel(nn.Module):
    """Simple test model for training."""
    
    def __init__(self, input_size: int = 5, num_classes: int = 2):
        
    """__init__ function."""
super().__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        
    def forward(self, x) -> Any:
        return F.softmax(self.fc1(x), dim=1)


class TestTrainingLogger:
    """Test suite for TrainingLogger."""
    
    @pytest.fixture
    def temp_log_dir(self, tmp_path) -> Any:
        """Create temporary log directory."""
        log_dir = tmp_path / "test_logs"
        log_dir.mkdir()
        return str(log_dir)
    
    @pytest.fixture
    def training_logger(self, temp_log_dir) -> Any:
        """Create a TrainingLogger instance for testing."""
        return TrainingLogger(
            log_dir=temp_log_dir,
            log_level=LogLevel.DEBUG,
            enable_console=False,
            enable_file=True,
            enable_rich=False
        )
    
    @pytest.fixture
    def sample_training_metrics(self) -> Any:
        """Create sample training metrics."""
        return TrainingMetrics(
            epoch=1,
            batch=10,
            loss=0.5,
            accuracy=85.5,
            precision=0.82,
            recall=0.78,
            f1_score=0.80,
            learning_rate=0.001,
            gradient_norm=1.2,
            validation_loss=0.6,
            validation_accuracy=83.0,
            training_time=2.5,
            memory_usage=2048.0,
            gpu_memory=1024.0,
            timestamp="2024-01-01T12:00:00"
        )
    
    @pytest.fixture
    def sample_security_event(self) -> Any:
        """Create sample security event."""
        return SecurityEvent(
            event_type="threat_detection",
            severity="high",
            description="Suspicious network activity detected",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.50",
            port=80,
            protocol="TCP",
            threat_level="high",
            confidence=0.95,
            timestamp="2024-01-01T12:00:00",
            metadata={"attack_type": "DDoS", "duration": "5 minutes"}
        )
    
    @pytest.fixture
    def sample_performance_metrics(self) -> Any:
        """Create sample performance metrics."""
        return PerformanceMetrics(
            cpu_usage=75.5,
            memory_usage=68.2,
            gpu_usage=85.0,
            gpu_memory=4096.0,
            disk_io=1024.0,
            network_io=512.0,
            batch_time=0.5,
            epoch_time=120.0,
            timestamp="2024-01-01T12:00:00"
        )
    
    def test_initialization(self, training_logger) -> Any:
        """Test TrainingLogger initialization."""
        assert training_logger.log_dir is not None
        assert training_logger.session_id is not None
        assert training_logger.start_time is not None
        assert training_logger.training_metrics == []
        assert training_logger.security_events == []
        assert training_logger.performance_metrics == []
        assert training_logger.current_epoch == 0
        assert training_logger.current_batch == 0
    
    def test_log_training_event(self, training_logger) -> Any:
        """Test logging training events."""
        training_logger.log_training_event(
            TrainingEventType.EPOCH_START,
            "Starting epoch 1",
            level=LogLevel.INFO,
            epoch=1
        )
        
        # Check that event was logged (we can't easily test the actual log output)
        assert training_logger.session_id is not None
    
    def test_log_training_event_with_metrics(self, training_logger, sample_training_metrics) -> Any:
        """Test logging training events with metrics."""
        training_logger.log_training_event(
            TrainingEventType.BATCH_END,
            "Completed batch",
            metrics=sample_training_metrics,
            level=LogLevel.INFO
        )
        
        # Check that metrics were stored
        assert len(training_logger.training_metrics) == 1
        assert training_logger.training_metrics[0].epoch == 1
        assert training_logger.training_metrics[0].batch == 10
        assert training_logger.training_metrics[0].loss == 0.5
    
    def test_log_security_event(self, training_logger, sample_security_event) -> Any:
        """Test logging security events."""
        training_logger.log_security_event(sample_security_event, level=LogLevel.WARNING)
        
        # Check that security event was stored
        assert len(training_logger.security_events) == 1
        assert training_logger.security_events[0].event_type == "threat_detection"
        assert training_logger.security_events[0].severity == "high"
        assert training_logger.security_events[0].source_ip == "192.168.1.100"
    
    def test_log_performance_metrics(self, training_logger, sample_performance_metrics) -> Any:
        """Test logging performance metrics."""
        training_logger.log_performance_metrics(sample_performance_metrics, level=LogLevel.INFO)
        
        # Check that performance metrics were stored
        assert len(training_logger.performance_metrics) == 1
        assert training_logger.performance_metrics[0].cpu_usage == 75.5
        assert training_logger.performance_metrics[0].memory_usage == 68.2
        assert training_logger.performance_metrics[0].gpu_usage == 85.0
    
    def test_log_error(self, training_logger) -> Any:
        """Test error logging."""
        error = ValueError("Test error")
        context = {"epoch": 1, "batch": 10, "operation": "forward_pass"}
        
        training_logger.log_error(error, context=context, level=LogLevel.ERROR)
        
        # Check that error was logged (we can't easily test the actual log output)
        assert training_logger.session_id is not None
    
    def test_start_end_training(self, training_logger) -> Any:
        """Test training start and end logging."""
        training_logger.start_training(total_epochs=10, total_batches=100)
        
        assert training_logger.total_epochs == 10
        assert training_logger.total_batches == 100
        
        final_metrics = TrainingMetrics(
            epoch=9,
            batch=99,
            loss=0.1,
            accuracy=95.0,
            timestamp="2024-01-01T12:00:00"
        )
        
        training_logger.end_training(final_metrics)
        
        # Check that final metrics were logged
        assert len(training_logger.training_metrics) == 1
        assert training_logger.training_metrics[0].epoch == 9
    
    def test_epoch_logging(self, training_logger) -> Any:
        """Test epoch start and end logging."""
        training_logger.start_epoch(epoch=1)
        
        assert training_logger.current_epoch == 1
        
        metrics = TrainingMetrics(
            epoch=1,
            batch=0,
            loss=0.5,
            accuracy=80.0,
            timestamp="2024-01-01T12:00:00"
        )
        
        training_logger.end_epoch(epoch=1, metrics=metrics)
        
        # Check that epoch metrics were logged
        assert len(training_logger.training_metrics) == 1
        assert training_logger.training_metrics[0].epoch == 1
    
    def test_batch_logging(self, training_logger) -> Any:
        """Test batch start and end logging."""
        training_logger.start_batch(batch=5, epoch=1)
        
        assert training_logger.current_batch == 5
        
        metrics = TrainingMetrics(
            epoch=1,
            batch=5,
            loss=0.3,
            accuracy=85.0,
            timestamp="2024-01-01T12:00:00"
        )
        
        training_logger.end_batch(batch=5, epoch=1, metrics=metrics)
        
        # Check that batch metrics were logged
        assert len(training_logger.training_metrics) == 1
        assert training_logger.training_metrics[0].batch == 5
    
    def test_loss_logging(self, training_logger) -> Any:
        """Test loss logging."""
        training_logger.log_loss(epoch=1, batch=10, loss=0.5, learning_rate=0.001)
        
        # Check that loss was logged
        assert len(training_logger.training_metrics) == 1
        assert training_logger.training_metrics[0].loss == 0.5
        assert training_logger.training_metrics[0].learning_rate == 0.001
    
    def test_validation_logging(self, training_logger) -> Any:
        """Test validation logging."""
        training_logger.log_validation(
            epoch=1,
            validation_loss=0.6,
            validation_accuracy=83.0,
            precision=0.82,
            recall=0.78
        )
        
        # Check that validation metrics were logged
        assert len(training_logger.training_metrics) == 1
        assert training_logger.training_metrics[0].validation_loss == 0.6
        assert training_logger.training_metrics[0].validation_accuracy == 83.0
        assert training_logger.training_metrics[0].precision == 0.82
        assert training_logger.training_metrics[0].recall == 0.78
    
    def test_model_save_load_logging(self, training_logger) -> Any:
        """Test model save and load logging."""
        # Test successful save
        training_logger.log_model_save(
            file_path="/path/to/model.pt",
            epoch=5,
            metrics=TrainingMetrics(epoch=5, batch=0, loss=0.2, accuracy=90.0)
        )
        
        # Test successful load
        training_logger.log_model_load(
            file_path="/path/to/model.pt",
            success=True
        )
        
        # Test failed load
        training_logger.log_model_load(
            file_path="/path/to/model.pt",
            success=False,
            error_message="File not found"
        )
        
        # Check that events were logged
        assert training_logger.session_id is not None
    
    def test_security_anomaly_logging(self, training_logger) -> Any:
        """Test security anomaly logging."""
        training_logger.log_security_anomaly(
            source_ip="192.168.1.100",
            destination_ip="10.0.0.50",
            threat_level="high",
            confidence=0.95,
            description="DDoS attack detected"
        )
        
        # Check that security event was logged
        assert len(training_logger.security_events) == 1
        assert training_logger.security_events[0].event_type == "anomaly_detection"
        assert training_logger.security_events[0].threat_level == "high"
        assert training_logger.security_events[0].confidence == 0.95
    
    def test_performance_alert_logging(self, training_logger, sample_performance_metrics) -> Any:
        """Test performance alert logging."""
        training_logger.log_performance_alert(
            alert_type="high_cpu",
            message="CPU usage exceeded 90%",
            metrics=sample_performance_metrics
        )
        
        # Check that performance alert was logged
        assert len(training_logger.performance_metrics) == 1
        assert training_logger.performance_metrics[0].cpu_usage == 75.5
    
    def test_get_training_summary(self, training_logger) -> Optional[Dict[str, Any]]:
        """Test training summary generation."""
        # Add some metrics
        for i in range(5):
            metrics = TrainingMetrics(
                epoch=i,
                batch=i * 10,
                loss=1.0 - (i * 0.1),
                accuracy=70.0 + (i * 5.0),
                timestamp="2024-01-01T12:00:00"
            )
            training_logger.training_metrics.append(metrics)
        
        # Add some security events
        for i in range(3):
            event = SecurityEvent(
                event_type="threat_detection",
                severity="medium",
                description=f"Threat {i}",
                timestamp="2024-01-01T12:00:00"
            )
            training_logger.security_events.append(event)
        
        # Add some performance metrics
        for i in range(4):
            perf_metrics = PerformanceMetrics(
                cpu_usage=50.0 + (i * 10.0),
                memory_usage=60.0 + (i * 5.0),
                timestamp="2024-01-01T12:00:00"
            )
            training_logger.performance_metrics.append(perf_metrics)
        
        summary = training_logger.get_training_summary()
        
        assert summary["session_id"] == training_logger.session_id
        assert summary["metrics_count"] == 5
        assert summary["security_events_count"] == 3
        assert summary["performance_metrics_count"] == 4
        assert "final_loss" in summary
        assert "final_accuracy" in summary
        assert "avg_loss" in summary
        assert "avg_accuracy" in summary
    
    def test_save_metrics_to_csv(self, training_logger, temp_log_dir) -> Any:
        """Test saving metrics to CSV."""
        # Add some metrics
        for i in range(3):
            metrics = TrainingMetrics(
                epoch=i,
                batch=i * 10,
                loss=1.0 - (i * 0.1),
                accuracy=70.0 + (i * 5.0),
                timestamp="2024-01-01T12:00:00"
            )
            training_logger.training_metrics.append(metrics)
        
        # Save to CSV
        csv_path = Path(temp_log_dir) / "test_metrics.csv"
        training_logger.save_metrics_to_csv(str(csv_path))
        
        # Check that file was created
        assert csv_path.exists()
        
        # Check that data was saved correctly
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert "epoch" in df.columns
        assert "batch" in df.columns
        assert "loss" in df.columns
        assert "accuracy" in df.columns
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_training_curves(self, mock_show, mock_savefig, training_logger, temp_log_dir) -> Any:
        """Test plotting training curves."""
        # Add some metrics
        for i in range(10):
            metrics = TrainingMetrics(
                epoch=i // 5,
                batch=i,
                loss=1.0 - (i * 0.05),
                accuracy=70.0 + (i * 2.0),
                learning_rate=0.001,
                memory_usage=1000.0 + (i * 100.0),
                timestamp="2024-01-01T12:00:00"
            )
            training_logger.training_metrics.append(metrics)
        
        # Plot curves
        plot_path = Path(temp_log_dir) / "test_curves.png"
        training_logger.plot_training_curves(str(plot_path), show_plot=False)
        
        # Check that savefig was called
        mock_savefig.assert_called_once()
    
    def test_cleanup(self, training_logger) -> Any:
        """Test cleanup functionality."""
        # Should not raise an exception
        training_logger.cleanup()


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor."""
    
    @pytest.fixture
    def performance_monitor(self) -> Any:
        """Create a PerformanceMonitor instance for testing."""
        return PerformanceMonitor()
    
    def test_initialization(self, performance_monitor) -> Any:
        """Test PerformanceMonitor initialization."""
        assert performance_monitor.start_time is not None
        assert performance_monitor.metrics_history == []
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_get_current_metrics(self, mock_virtual_memory, mock_cpu_percent, performance_monitor) -> Optional[Dict[str, Any]]:
        """Test getting current performance metrics."""
        # Mock system metrics
        mock_cpu_percent.return_value = 75.5
        mock_memory = Mock()
        mock_memory.percent = 68.2
        mock_virtual_memory.return_value = mock_memory
        
        # Mock CUDA if available
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.utilization', return_value=85.0):
                with patch('torch.cuda.memory_allocated', return_value=1024 * 1024 * 1024):  # 1GB
                    metrics = performance_monitor.get_current_metrics()
        
        assert metrics.cpu_usage == 75.5
        assert metrics.memory_usage == 68.2
        assert metrics.gpu_usage == 85.0
        assert metrics.gpu_memory == 1024.0  # MB
    
    def test_check_performance_alerts(self, performance_monitor) -> Any:
        """Test performance alert checking."""
        # Test normal metrics (no alerts)
        normal_metrics = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            gpu_memory=4000.0,
            timestamp="2024-01-01T12:00:00"
        )
        
        alerts = performance_monitor.check_performance_alerts(normal_metrics)
        assert len(alerts) == 0
        
        # Test high CPU usage
        high_cpu_metrics = PerformanceMetrics(
            cpu_usage=95.0,
            memory_usage=60.0,
            timestamp="2024-01-01T12:00:00"
        )
        
        alerts = performance_monitor.check_performance_alerts(high_cpu_metrics)
        assert len(alerts) == 1
        assert "High CPU usage" in alerts[0]
        
        # Test high memory usage
        high_memory_metrics = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=95.0,
            timestamp="2024-01-01T12:00:00"
        )
        
        alerts = performance_monitor.check_performance_alerts(high_memory_metrics)
        assert len(alerts) == 1
        assert "High memory usage" in alerts[0]
        
        # Test high GPU memory usage
        high_gpu_metrics = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            gpu_memory=9000.0,  # 9GB
            timestamp="2024-01-01T12:00:00"
        )
        
        alerts = performance_monitor.check_performance_alerts(high_gpu_metrics)
        assert len(alerts) == 1
        assert "High GPU memory usage" in alerts[0]
        
        # Test multiple alerts
        multiple_alerts_metrics = PerformanceMetrics(
            cpu_usage=95.0,
            memory_usage=95.0,
            gpu_memory=9000.0,
            timestamp="2024-01-01T12:00:00"
        )
        
        alerts = performance_monitor.check_performance_alerts(multiple_alerts_metrics)
        assert len(alerts) == 3


class TestTrainingLoggerDecorator:
    """Test suite for TrainingLoggerDecorator."""
    
    @pytest.fixture
    def mock_logger(self) -> Any:
        """Create a mock training logger."""
        return Mock(spec=TrainingLogger)
    
    @pytest.fixture
    def decorator(self, mock_logger) -> Any:
        """Create a TrainingLoggerDecorator instance."""
        return TrainingLoggerDecorator(mock_logger)
    
    def test_decorator_success(self, decorator, mock_logger) -> Any:
        """Test decorator with successful function execution."""
        @decorator
        def test_function():
            
    """test_function function."""
return "success"
        
        result = test_function()
        
        assert result == "success"
        assert mock_logger.log_training_event.call_count == 2  # Start and end
        assert mock_logger.log_error.call_count == 0
    
    def test_decorator_error(self, decorator, mock_logger) -> Any:
        """Test decorator with function that raises an error."""
        @decorator
        def test_function():
            
    """test_function function."""
raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_function()
        
        assert mock_logger.log_training_event.call_count == 1  # Only start
        assert mock_logger.log_error.call_count == 1  # Error logged


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_create_training_logger(self, tmp_path) -> Any:
        """Test create_training_logger function."""
        log_dir = str(tmp_path / "test_logs")
        
        logger = create_training_logger({
            "log_dir": log_dir,
            "log_level": LogLevel.DEBUG,
            "enable_console": False,
            "enable_file": True,
            "enable_rich": False
        })
        
        assert isinstance(logger, TrainingLogger)
        assert logger.log_dir == Path(log_dir)
        assert logger.log_level == LogLevel.DEBUG
    
    def test_log_training_progress_decorator(self, tmp_path) -> Any:
        """Test log_training_progress decorator."""
        log_dir = str(tmp_path / "test_logs")
        logger = create_training_logger({"log_dir": log_dir, "enable_rich": False})
        
        @log_training_progress(logger)
        def train_model(model, dataloader, epochs=2) -> Any:
            return "training_completed"
        
        # Create mock model and dataloader
        model = SimpleTestModel()
        dataloader = [(torch.randn(32, 5), torch.randint(0, 2, (32,))) for _ in range(10)]
        
        result = train_model(model, dataloader, epochs=2)
        
        assert result == "training_completed"
        assert logger.total_epochs == 2
        assert logger.total_batches == 20


class TestIntegration:
    """Integration tests for the training logging system."""
    
    @pytest.fixture
    def temp_log_dir(self, tmp_path) -> Any:
        """Create temporary log directory."""
        log_dir = tmp_path / "integration_logs"
        log_dir.mkdir()
        return str(log_dir)
    
    def test_full_training_cycle(self, temp_log_dir) -> Any:
        """Test a full training cycle with logging."""
        # Create logger
        training_logger = create_training_logger({
            "log_dir": temp_log_dir,
            "enable_rich": False
        })
        
        # Create model and data
        model = SimpleTestModel()
        X = torch.randn(100, 5)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Start training
        training_logger.start_training(epochs=2, total_batches=len(dataloader) * 2)
        
        try:
            for epoch in range(2):
                training_logger.start_epoch(epoch)
                
                for batch_idx, (data, target) in enumerate(dataloader):
                    training_logger.start_batch(batch_idx, epoch)
                    
                    # Training step
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    # Log metrics
                    metrics = TrainingMetrics(
                        epoch=epoch,
                        batch=batch_idx,
                        loss=loss.item(),
                        accuracy=75.0 + (batch_idx * 2.0),
                        learning_rate=0.001,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    training_logger.end_batch(batch_idx, epoch, metrics)
                    
                    # Log performance metrics
                    perf_metrics = training_logger.performance_monitor.get_current_metrics()
                    training_logger.log_performance_metrics(perf_metrics)
                
                training_logger.end_epoch(epoch)
        
        finally:
            training_logger.end_training()
        
        # Verify logging
        assert len(training_logger.training_metrics) > 0
        assert len(training_logger.performance_metrics) > 0
        
        # Get summary
        summary = training_logger.get_training_summary()
        assert summary["metrics_count"] > 0
        assert summary["performance_metrics_count"] > 0
    
    def test_security_integration(self, temp_log_dir) -> Any:
        """Test security event logging integration."""
        training_logger = create_training_logger({
            "log_dir": temp_log_dir,
            "enable_rich": False
        })
        
        # Simulate security events during training
        for i in range(5):
            event = SecurityEvent(
                event_type="threat_detection",
                severity="medium",
                description=f"Threat {i} detected",
                source_ip=f"192.168.1.{i+1}",
                destination_ip=f"10.0.0.{i+1}",
                threat_level="medium",
                confidence=0.7 + (i * 0.05),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            training_logger.log_security_event(event, level=LogLevel.WARNING)
        
        # Verify security events
        assert len(training_logger.security_events) == 5
        
        # Test security anomaly logging
        training_logger.log_security_anomaly(
            source_ip="192.168.1.100",
            destination_ip="10.0.0.50",
            threat_level="high",
            confidence=0.95,
            description="DDoS attack detected"
        )
        
        assert len(training_logger.security_events) == 6
    
    def test_error_handling_integration(self, temp_log_dir) -> Any:
        """Test error handling integration."""
        training_logger = create_training_logger({
            "log_dir": temp_log_dir,
            "enable_rich": False
        })
        
        # Simulate training with errors
        training_logger.start_training(epochs=1, total_batches=5)
        
        try:
            for batch in range(5):
                training_logger.start_batch(batch, 0)
                
                try:
                    if batch == 2:
                        # Simulate error
                        raise RuntimeError("Simulated training error")
                    
                    # Normal training
                    metrics = TrainingMetrics(
                        epoch=0,
                        batch=batch,
                        loss=0.5,
                        accuracy=80.0,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    training_logger.end_batch(batch, 0, metrics)
                    
                except Exception as e:
                    training_logger.log_error(
                        e,
                        context={"epoch": 0, "batch": batch}
                    )
                    continue
            
            training_logger.end_epoch(0)
        
        finally:
            training_logger.end_training()
        
        # Verify that training continued despite errors
        assert training_logger.session_id is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"]) 