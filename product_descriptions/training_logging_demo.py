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

import asyncio
import time
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import structlog
from training_logging_system import (
from robust_operations import RobustOperations
                import shutil
from typing import Any, List, Dict, Optional
import logging
"""
Training Logging System Demo

This demo showcases comprehensive logging for ML training with:
- Real training scenarios with progress tracking
- Error handling and recovery
- Security event logging
- Performance monitoring
- Rich console output
- Log analysis and visualization
"""



# Add the current directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

    TrainingLogger,
    TrainingMetrics,
    SecurityEvent,
    PerformanceMetrics,
    TrainingEventType,
    LogLevel,
    create_training_logger,
    log_training_progress
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class CybersecurityModel(nn.Module):
    """Neural network for cybersecurity threat detection."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_classes: int = 2):
        
    """__init__ function."""
super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x) -> Any:
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class AnomalyDetectionModel(nn.Module):
    """Autoencoder for anomaly detection."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 32):
        
    """__init__ function."""
super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        
    def forward(self, x) -> Any:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class TrainingLoggingDemo:
    """Demo class showcasing comprehensive training logging."""
    
    def __init__(self) -> Any:
        self.config = {
            "log_dir": "demo_training_logs",
            "log_level": LogLevel.INFO,
            "enable_console": True,
            "enable_file": True,
            "enable_rich": True,
            "max_log_files": 5,
            "log_rotation_size": 50 * 1024 * 1024  # 50MB
        }
        
        # Initialize training logger
        self.training_logger = create_training_logger(self.config)
        
        # Initialize robust operations
        self.robust_ops = RobustOperations({
            "max_errors": 1000,
            "enable_persistence": True,
            "enable_profiling": True
        })
        
        # Create demo data directory
        self.demo_dir = Path("demo_data")
        self.demo_dir.mkdir(exist_ok=True)
        
        logger.info("TrainingLoggingDemo initialized", config=self.config)
    
    def create_demo_data(self) -> Tuple[DataLoader, DataLoader]:
        """Create demo cybersecurity data."""
        logger.info("Creating demo cybersecurity data")
        
        # Generate synthetic cybersecurity data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Normal traffic features
        normal_data = np.random.normal(0, 1, (n_samples // 2, n_features))
        normal_labels = np.zeros(n_samples // 2)
        
        # Malicious traffic features (anomalies)
        malicious_data = np.random.normal(2, 1.5, (n_samples // 2, n_features))
        malicious_labels = np.ones(n_samples // 2)
        
        # Combine data
        X = np.vstack([normal_data, malicious_data])
        y = np.hstack([normal_labels, malicious_labels])
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        # Split into train and validation
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        logger.info("Demo data created", 
                   train_samples=len(train_dataset),
                   val_samples=len(val_dataset),
                   features=n_features)
        
        return train_loader, val_loader
    
    def demo_basic_training_logging(self) -> Any:
        """Demonstrate basic training logging."""
        logger.info("=== Basic Training Logging Demo ===")
        
        # Create model and data
        model = CybersecurityModel(input_size=10, hidden_size=32, num_classes=2)
        train_loader, val_loader = self.create_demo_data()
        
        # Training parameters
        epochs = 5
        learning_rate = 0.001
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Start training
        self.training_logger.start_training(epochs, len(train_loader) * epochs)
        
        try:
            for epoch in range(epochs):
                self.training_logger.start_epoch(epoch)
                
                model.train()
                epoch_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    self.training_logger.start_batch(batch_idx, epoch)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    epoch_loss += loss.item()
                    
                    # Log batch metrics
                    batch_accuracy = 100 * correct / total
                    metrics = TrainingMetrics(
                        epoch=epoch,
                        batch=batch_idx,
                        loss=loss.item(),
                        accuracy=batch_accuracy,
                        learning_rate=learning_rate,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    self.training_logger.end_batch(batch_idx, epoch, metrics)
                    
                    # Log performance metrics periodically
                    if batch_idx % 10 == 0:
                        perf_metrics = self.training_logger.performance_monitor.get_current_metrics()
                        self.training_logger.log_performance_metrics(perf_metrics)
                        
                        # Check for performance alerts
                        alerts = self.training_logger.performance_monitor.check_performance_alerts(perf_metrics)
                        for alert in alerts:
                            self.training_logger.log_performance_alert("system", alert, perf_metrics)
                
                # Validation
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                self.training_logger.log_training_event(
                    TrainingEventType.VALIDATION_START,
                    f"Starting validation for epoch {epoch}",
                    level=LogLevel.INFO
                )
                
                with torch.no_grad():
                    for data, target in val_loader:
                        output = model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(output.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                
                val_accuracy = 100 * val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)
                
                # Log validation results
                val_metrics = TrainingMetrics(
                    epoch=epoch,
                    batch=0,
                    loss=avg_val_loss,
                    accuracy=val_accuracy,
                    validation_loss=avg_val_loss,
                    validation_accuracy=val_accuracy,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                self.training_logger.log_validation(epoch, avg_val_loss, val_accuracy)
                self.training_logger.end_epoch(epoch, val_metrics)
                
                # Log epoch summary
                avg_epoch_loss = epoch_loss / len(train_loader)
                epoch_accuracy = 100 * correct / total
                
                self.training_logger.log_training_event(
                    TrainingEventType.INFO,
                    f"Epoch {epoch} Summary - Train Loss: {avg_epoch_loss:.4f}, "
                    f"Train Acc: {epoch_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.2f}%",
                    level=LogLevel.INFO
                )
        
        except Exception as e:
            self.training_logger.log_error(e, context={"epoch": epoch, "batch": batch_idx})
            raise
        
        finally:
            # End training
            final_metrics = TrainingMetrics(
                epoch=epochs - 1,
                batch=len(train_loader) - 1,
                loss=avg_epoch_loss,
                accuracy=epoch_accuracy,
                validation_loss=avg_val_loss,
                validation_accuracy=val_accuracy,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            self.training_logger.end_training(final_metrics)
    
    def demo_error_handling_logging(self) -> Any:
        """Demonstrate error handling and logging."""
        logger.info("=== Error Handling Logging Demo ===")
        
        # Create model and data
        model = CybersecurityModel(input_size=10, hidden_size=32, num_classes=2)
        train_loader, val_loader = self.create_demo_data()
        
        # Training parameters
        epochs = 3
        learning_rate = 0.001
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.training_logger.start_training(epochs, len(train_loader) * epochs)
        
        try:
            for epoch in range(epochs):
                self.training_logger.start_epoch(epoch)
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    self.training_logger.start_batch(batch_idx, epoch)
                    
                    try:
                        # Simulate potential errors
                        if batch_idx == 5 and epoch == 1:
                            # Simulate NaN loss
                            raise ValueError("Simulated NaN loss error")
                        
                        if batch_idx == 10 and epoch == 1:
                            # Simulate CUDA out of memory
                            raise RuntimeError("Simulated CUDA out of memory error")
                        
                        # Normal training
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        
                        # Log successful batch
                        metrics = TrainingMetrics(
                            epoch=epoch,
                            batch=batch_idx,
                            loss=loss.item(),
                            accuracy=50.0 + random.random() * 30,  # Simulated accuracy
                            learning_rate=learning_rate,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                        )
                        
                        self.training_logger.end_batch(batch_idx, epoch, metrics)
                        
                    except Exception as e:
                        # Log error and continue training
                        self.training_logger.log_error(
                            e,
                            context={
                                "epoch": epoch,
                                "batch": batch_idx,
                                "data_shape": data.shape,
                                "target_shape": target.shape
                            }
                        )
                        
                        # Log failed batch
                        self.training_logger.log_training_event(
                            TrainingEventType.ERROR,
                            f"Batch {batch_idx} failed in epoch {epoch}",
                            level=LogLevel.ERROR,
                            error_message=str(e)
                        )
                        
                        # Continue with next batch
                        continue
                
                self.training_logger.end_epoch(epoch)
        
        except Exception as e:
            self.training_logger.log_error(e, context={"training_phase": "main_loop"})
            raise
        
        finally:
            self.training_logger.end_training()
    
    def demo_security_event_logging(self) -> Any:
        """Demonstrate security event logging."""
        logger.info("=== Security Event Logging Demo ===")
        
        # Simulate security events during training
        security_events = [
            {
                "source_ip": "192.168.1.100",
                "destination_ip": "10.0.0.50",
                "threat_level": "high",
                "confidence": 0.95,
                "description": "Suspicious network activity detected - potential DDoS attack"
            },
            {
                "source_ip": "172.16.0.25",
                "destination_ip": "192.168.1.1",
                "threat_level": "medium",
                "confidence": 0.78,
                "description": "Unusual port scanning activity detected"
            },
            {
                "source_ip": "10.0.0.100",
                "destination_ip": "8.8.8.8",
                "threat_level": "low",
                "confidence": 0.45,
                "description": "Anomalous DNS query pattern detected"
            }
        ]
        
        # Log security events
        for i, event_data in enumerate(security_events):
            event = SecurityEvent(
                event_type="threat_detection",
                severity=event_data["threat_level"],
                description=event_data["description"],
                source_ip=event_data["source_ip"],
                destination_ip=event_data["destination_ip"],
                threat_level=event_data["threat_level"],
                confidence=event_data["confidence"],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            self.training_logger.log_security_event(event, level=LogLevel.WARNING)
            
            # Simulate training progress
            time.sleep(0.5)
        
        # Simulate anomaly detection during training
        for i in range(10):
            # Simulate detecting anomalies in training data
            if random.random() < 0.3:  # 30% chance of anomaly
                anomaly_event = SecurityEvent(
                    event_type="anomaly_detection",
                    severity="medium",
                    description=f"Anomaly detected in training batch {i}",
                    source_ip=f"192.168.1.{random.randint(1, 255)}",
                    destination_ip=f"10.0.0.{random.randint(1, 255)}",
                    threat_level="medium",
                    confidence=random.uniform(0.6, 0.9),
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                self.training_logger.log_security_event(anomaly_event, level=LogLevel.INFO)
            
            time.sleep(0.2)
    
    def demo_performance_monitoring(self) -> Any:
        """Demonstrate performance monitoring and alerting."""
        logger.info("=== Performance Monitoring Demo ===")
        
        # Simulate training with performance monitoring
        epochs = 3
        batches_per_epoch = 20
        
        self.training_logger.start_training(epochs, epochs * batches_per_epoch)
        
        for epoch in range(epochs):
            self.training_logger.start_epoch(epoch)
            
            for batch in range(batches_per_epoch):
                self.training_logger.start_batch(batch, epoch)
                
                # Simulate training work
                time.sleep(0.1)
                
                # Get current performance metrics
                perf_metrics = self.training_logger.performance_monitor.get_current_metrics()
                
                # Log performance metrics
                self.training_logger.log_performance_metrics(perf_metrics)
                
                # Check for performance alerts
                alerts = self.training_logger.performance_monitor.check_performance_alerts(perf_metrics)
                for alert in alerts:
                    self.training_logger.log_performance_alert("system", alert, perf_metrics)
                
                # Simulate high resource usage occasionally
                if random.random() < 0.1:  # 10% chance
                    high_usage_metrics = PerformanceMetrics(
                        cpu_usage=95.0,
                        memory_usage=88.0,
                        gpu_usage=92.0 if torch.cuda.is_available() else None,
                        gpu_memory=7500.0 if torch.cuda.is_available() else None,
                        batch_time=2.5,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    self.training_logger.log_performance_metrics(high_usage_metrics, level=LogLevel.WARNING)
                    self.training_logger.log_performance_alert(
                        "resource_usage",
                        "High resource usage detected",
                        high_usage_metrics
                    )
                
                # Log batch completion
                metrics = TrainingMetrics(
                    epoch=epoch,
                    batch=batch,
                    loss=random.uniform(0.1, 2.0),
                    accuracy=random.uniform(60.0, 95.0),
                    learning_rate=0.001,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                self.training_logger.end_batch(batch, epoch, metrics)
            
            self.training_logger.end_epoch(epoch)
        
        self.training_logger.end_training()
    
    def demo_model_save_load_logging(self) -> Any:
        """Demonstrate model save/load logging."""
        logger.info("=== Model Save/Load Logging Demo ===")
        
        # Create model
        model = CybersecurityModel(input_size=10, hidden_size=32, num_classes=2)
        
        # Simulate training
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few steps
        train_loader, _ = self.create_demo_data()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 5:  # Train for 5 batches
                break
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Save model with logging
        model_path = self.demo_dir / "cybersecurity_model.pt"
        
        try:
            torch.save(model.state_dict(), str(model_path))
            
            metrics = TrainingMetrics(
                epoch=0,
                batch=4,
                loss=loss.item(),
                accuracy=75.0,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            self.training_logger.log_model_save(str(model_path), 0, metrics)
            
        except Exception as e:
            self.training_logger.log_error(e, context={"operation": "model_save"})
        
        # Load model with logging
        try:
            loaded_state_dict = torch.load(str(model_path))
            new_model = CybersecurityModel(input_size=10, hidden_size=32, num_classes=2)
            new_model.load_state_dict(loaded_state_dict)
            
            self.training_logger.log_model_load(str(model_path), True)
            
        except Exception as e:
            self.training_logger.log_model_load(str(model_path), False, str(e))
    
    def demo_log_analysis(self) -> Any:
        """Demonstrate log analysis and reporting."""
        logger.info("=== Log Analysis Demo ===")
        
        # Get training summary
        summary = self.training_logger.get_training_summary()
        
        logger.info("Training Summary", summary=summary)
        
        # Save metrics to CSV
        self.training_logger.save_metrics_to_csv()
        
        # Generate training curves
        self.training_logger.plot_training_curves(show_plot=False)
        
        # Analyze security events
        if self.training_logger.security_events:
            logger.info("Security Events Analysis", 
                       total_events=len(self.training_logger.security_events),
                       high_severity=len([e for e in self.training_logger.security_events if e.severity == "high"]),
                       medium_severity=len([e for e in self.training_logger.security_events if e.severity == "medium"]),
                       low_severity=len([e for e in self.training_logger.security_events if e.severity == "low"]))
        
        # Analyze performance metrics
        if self.training_logger.performance_metrics:
            cpu_usage = [m.cpu_usage for m in self.training_logger.performance_metrics if m.cpu_usage is not None]
            memory_usage = [m.memory_usage for m in self.training_logger.performance_metrics if m.memory_usage is not None]
            
            logger.info("Performance Analysis",
                       avg_cpu_usage=np.mean(cpu_usage) if cpu_usage else 0,
                       max_cpu_usage=max(cpu_usage) if cpu_usage else 0,
                       avg_memory_usage=np.mean(memory_usage) if memory_usage else 0,
                       max_memory_usage=max(memory_usage) if memory_usage else 0)
    
    def run_comprehensive_demo(self) -> Any:
        """Run the complete training logging demo."""
        logger.info("Starting Comprehensive Training Logging Demo")
        
        try:
            # Run all demos
            self.demo_basic_training_logging()
            time.sleep(1)
            
            self.demo_error_handling_logging()
            time.sleep(1)
            
            self.demo_security_event_logging()
            time.sleep(1)
            
            self.demo_performance_monitoring()
            time.sleep(1)
            
            self.demo_model_save_load_logging()
            time.sleep(1)
            
            self.demo_log_analysis()
            
            logger.info("Comprehensive demo completed successfully")
            
        except Exception as e:
            logger.error("Demo failed", error=str(e))
        
        finally:
            # Cleanup
            self.training_logger.cleanup()
            self.robust_ops.cleanup()
            
            # Clean up demo files
            try:
                shutil.rmtree(self.demo_dir)
                logger.info("Demo cleanup completed")
            except Exception as e:
                logger.warning("Demo cleanup failed", error=str(e))


async def main():
    """Main function to run the demo."""
    demo = TrainingLoggingDemo()
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 