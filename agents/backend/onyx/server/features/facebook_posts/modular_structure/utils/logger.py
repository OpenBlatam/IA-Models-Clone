"""
📝 Advanced Logging System
Provides comprehensive logging capabilities for ML experiments and training.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json
import os
from dataclasses import dataclass, asdict
import traceback

@dataclass
class LogConfig:
    """Configuration for logging"""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_dir: str = "logs"
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    file_output: bool = True
    json_format: bool = False
    include_timestamp: bool = True
    include_process_id: bool = True
    include_thread_id: bool = True

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config
        
    def format(self, record):
        if self.config.json_format:
            return self._format_json(record)
        else:
            return super().format(record)
    
    def _format_json(self, record):
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if self.config.include_process_id:
            log_entry["process_id"] = record.process
            
        if self.config.include_thread_id:
            log_entry["thread_id"] = record.thread
            
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry)

class Logger:
    """
    Advanced logging system with structured logging, file rotation,
    and experiment tracking integration.
    """
    
    def __init__(self, name: str, config: LogConfig, experiment_name: str = "experiment"):
        self.name = name
        self.config = config
        self.experiment_name = experiment_name
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatter
        if config.json_format:
            formatter = StructuredFormatter(config)
        else:
            formatter = logging.Formatter(
                config.format_string,
                datefmt=config.date_format
            )
        
        # Console handler
        if config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if config.file_output:
            self._setup_file_handler(formatter)
        
        # Store experiment metrics
        self.experiment_metrics: Dict[str, Any] = {}
        self.experiment_events: list = []
        
        self.info(f"Logger initialized for experiment: {experiment_name}")
    
    def _setup_file_handler(self, formatter):
        """Setup file handler with rotation"""
        try:
            # Create log directory
            log_dir = Path(self.config.log_dir) / self.experiment_name
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine log file path
            if self.config.log_file:
                log_file = log_dir / self.config.log_file
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = log_dir / f"{self.name}_{timestamp}.log"
            
            # Create file handler with rotation
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file_path = log_file
            
        except Exception as e:
            print(f"Failed to setup file handler: {e}")
    
    def _log_with_extra(self, level: str, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log with extra fields"""
        if extra_fields:
            record = self.logger.makeRecord(
                self.name, getattr(logging, level.upper()), 
                "", 0, message, (), None
            )
            record.extra_fields = extra_fields
            self.logger.handle(record)
        else:
            getattr(self.logger, level.lower())(message)
    
    def debug(self, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self._log_with_extra("DEBUG", message, extra_fields)
    
    def info(self, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self._log_with_extra("INFO", message, extra_fields)
    
    def warning(self, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self._log_with_extra("WARNING", message, extra_fields)
    
    def error(self, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log error message"""
        self._log_with_extra("ERROR", message, extra_fields)
    
    def critical(self, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log critical message"""
        self._log_with_extra("CRITICAL", message, extra_fields)
    
    def exception(self, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log exception with traceback"""
        if extra_fields is None:
            extra_fields = {}
        extra_fields['traceback'] = traceback.format_exc()
        self._log_with_extra("ERROR", message, extra_fields)
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None, epoch: Optional[int] = None):
        """Log a metric with metadata"""
        extra_fields = {
            'metric_name': metric_name,
            'metric_value': value,
            'log_type': 'metric'
        }
        if step is not None:
            extra_fields['step'] = step
        if epoch is not None:
            extra_fields['epoch'] = epoch
        
        self.info(f"Metric: {metric_name} = {value}", extra_fields)
        
        # Store in experiment metrics
        if metric_name not in self.experiment_metrics:
            self.experiment_metrics[metric_name] = []
        self.experiment_metrics[metric_name].append({
            'value': value,
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_event(self, event_name: str, event_data: Optional[Dict[str, Any]] = None):
        """Log an event"""
        extra_fields = {
            'event_name': event_name,
            'log_type': 'event'
        }
        if event_data:
            extra_fields.update(event_data)
        
        self.info(f"Event: {event_name}", extra_fields)
        
        # Store in experiment events
        self.experiment_events.append({
            'event_name': event_name,
            'event_data': event_data or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration"""
        extra_fields = {
            'config': config,
            'log_type': 'config'
        }
        self.info("Configuration loaded", extra_fields)
    
    def log_checkpoint(self, checkpoint_id: str, checkpoint_path: str, metrics: Optional[Dict[str, float]] = None):
        """Log checkpoint information"""
        extra_fields = {
            'checkpoint_id': checkpoint_id,
            'checkpoint_path': checkpoint_path,
            'log_type': 'checkpoint'
        }
        if metrics:
            extra_fields['metrics'] = metrics
        
        self.info(f"Checkpoint saved: {checkpoint_id}", extra_fields)
    
    def log_training_step(self, step: int, epoch: int, loss: float, metrics: Optional[Dict[str, float]] = None):
        """Log training step information"""
        extra_fields = {
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'log_type': 'training_step'
        }
        if metrics:
            extra_fields['metrics'] = metrics
        
        self.info(f"Training step {step} (epoch {epoch}): loss = {loss:.4f}", extra_fields)
    
    def log_validation(self, epoch: int, metrics: Dict[str, float]):
        """Log validation results"""
        extra_fields = {
            'epoch': epoch,
            'metrics': metrics,
            'log_type': 'validation'
        }
        
        metrics_str = ", ".join([f"{k} = {v:.4f}" for k, v in metrics.items()])
        self.info(f"Validation (epoch {epoch}): {metrics_str}", extra_fields)
    
    def log_experiment_start(self, experiment_config: Dict[str, Any]):
        """Log experiment start"""
        extra_fields = {
            'experiment_config': experiment_config,
            'log_type': 'experiment_start'
        }
        self.info("Experiment started", extra_fields)
    
    def log_experiment_end(self, final_metrics: Optional[Dict[str, float]] = None):
        """Log experiment end"""
        extra_fields = {
            'final_metrics': final_metrics,
            'log_type': 'experiment_end'
        }
        self.info("Experiment completed", extra_fields)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary from logs"""
        return {
            'experiment_name': self.experiment_name,
            'metrics': self.experiment_metrics,
            'events': self.experiment_events,
            'total_events': len(self.experiment_events),
            'total_metrics': sum(len(values) for values in self.experiment_metrics.values())
        }
    
    def save_experiment_log(self, output_path: Optional[str] = None):
        """Save experiment log to file"""
        if output_path is None:
            log_dir = Path(self.config.log_dir) / self.experiment_name
            output_path = log_dir / f"{self.experiment_name}_summary.json"
        
        summary = self.get_experiment_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.info(f"Experiment summary saved to: {output_path}")
    
    def close(self):
        """Close the logger"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        
        self.info("Logger closed")

# Convenience function to create a logger
def create_logger(
    name: str,
    experiment_name: str = "experiment",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    json_format: bool = False,
    console_output: bool = True,
    file_output: bool = True
) -> Logger:
    """Create a logger with the specified configuration"""
    config = LogConfig(
        log_level=log_level,
        log_file=log_file,
        log_dir=log_dir,
        json_format=json_format,
        console_output=console_output,
        file_output=file_output
    )
    
    return Logger(name, config, experiment_name)






