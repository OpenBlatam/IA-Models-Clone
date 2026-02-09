"""
BUL - Business Universal Language (Advanced Error Handling & Logging)
====================================================================

Advanced error handling and logging system for BUL.
"""

import logging
import logging.handlers
import traceback
import sys
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional, Callable
from functools import wraps
import asyncio
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

class AdvancedLogger:
    """Advanced logging system with multiple handlers and features."""
    
    def __init__(self, name: str = "BUL", log_dir: str = "logs"):
        """Initialize advanced logger."""
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self.setup_console_handler()
        self.setup_file_handler()
        self.setup_rotating_handler()
        self.setup_error_handler()
        self.setup_json_handler()
        
        # Error tracking
        self.error_counts = {}
        self.alert_thresholds = {
            "error_rate": 10,  # errors per minute
            "critical_errors": 3,  # critical errors per hour
            "memory_errors": 5  # memory errors per hour
        }
        
        # Alert system
        self.alert_config = {
            "email_enabled": False,
            "webhook_enabled": False,
            "email_recipients": [],
            "webhook_url": None
        }
    
    def setup_console_handler(self):
        """Setup console handler with colored output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Custom formatter with colors
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m', # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m'  # Magenta
            }
            RESET = '\033[0m'
            
            def format(self, record):
                log_color = self.COLORS.get(record.levelname, self.RESET)
                record.levelname = f"{log_color}{record.levelname}{self.RESET}"
                return super().format(record)
        
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def setup_file_handler(self):
        """Setup file handler for general logs."""
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def setup_rotating_handler(self):
        """Setup rotating file handler."""
        rotating_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_rotating.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        rotating_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        rotating_handler.setFormatter(formatter)
        self.logger.addHandler(rotating_handler)
    
    def setup_error_handler(self):
        """Setup error-specific handler."""
        error_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_errors.log"
        )
        error_handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n%(pathname)s:%(lineno)d\n'
        )
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def setup_json_handler(self):
        """Setup JSON handler for structured logging."""
        json_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_structured.json"
        )
        json_handler.setLevel(logging.INFO)
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                    "thread": record.thread,
                    "process": record.process
                }
                
                if record.exc_info:
                    log_entry["exception"] = traceback.format_exception(*record.exc_info)
                
                return json.dumps(log_entry)
        
        json_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(json_handler)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with additional context."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Track error counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Log with context
        log_data = {
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback.format_exc(),
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.error(f"Error occurred: {error_type} - {error_message}")
        self.logger.debug(f"Error context: {json.dumps(log_data, indent=2)}")
        
        # Check for alerts
        self.check_alert_conditions(error_type)
    
    def check_alert_conditions(self, error_type: str):
        """Check if alert conditions are met."""
        current_time = time.time()
        
        # Check error rate
        if self.error_counts.get(error_type, 0) > self.alert_thresholds["error_rate"]:
            self.send_alert(f"High error rate for {error_type}: {self.error_counts[error_type]} errors")
        
        # Check critical errors
        if error_type in ["CriticalError", "SystemError"]:
            critical_count = sum(
                count for error, count in self.error_counts.items() 
                if error in ["CriticalError", "SystemError"]
            )
            if critical_count > self.alert_thresholds["critical_errors"]:
                self.send_alert(f"Multiple critical errors detected: {critical_count}")
    
    def send_alert(self, message: str):
        """Send alert notification."""
        if self.alert_config["email_enabled"]:
            self.send_email_alert(message)
        
        if self.alert_config["webhook_enabled"]:
            self.send_webhook_alert(message)
    
    def send_email_alert(self, message: str):
        """Send email alert."""
        try:
            msg = MIMEMultipart()
            msg['From'] = "bul-system@example.com"
            msg['To'] = ", ".join(self.alert_config["email_recipients"])
            msg['Subject'] = f"BUL System Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            body = f"""
            BUL System Alert
            
            Time: {datetime.now().isoformat()}
            Message: {message}
            
            Error Counts: {json.dumps(self.error_counts, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Note: In production, configure SMTP server
            # server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.starttls()
            # server.login("email", "password")
            # server.send_message(msg)
            # server.quit()
            
            self.logger.info(f"Email alert sent: {message}")
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def send_webhook_alert(self, message: str):
        """Send webhook alert."""
        try:
            payload = {
                "text": f"BUL System Alert: {message}",
                "timestamp": datetime.now().isoformat(),
                "error_counts": self.error_counts
            }
            
            if self.alert_config["webhook_url"]:
                response = requests.post(
                    self.alert_config["webhook_url"],
                    json=payload,
                    timeout=10
                )
                response.raise_for_status()
                self.logger.info(f"Webhook alert sent: {message}")
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts,
            "alert_thresholds": self.alert_thresholds,
            "alert_config": self.alert_config,
            "log_files": list(self.log_dir.glob("*.log")),
            "total_errors": sum(self.error_counts.values())
        }

# Global logger instance
advanced_logger = AdvancedLogger()

def error_handler(logger_instance: AdvancedLogger = None):
    """Decorator for error handling."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logger_instance or advanced_logger
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # Limit args length
                    "kwargs": str(kwargs)[:200]
                }
                logger.log_error_with_context(e, context)
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logger_instance or advanced_logger
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }
                logger.log_error_with_context(e, context)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator

class ErrorRecovery:
    """Error recovery and retry system."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def retry_on_error(self, exceptions: tuple = (Exception,)):
        """Decorator for retry on error."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < self.max_retries:
                            wait_time = self.backoff_factor * (2 ** attempt)
                            advanced_logger.logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                f"Retrying in {wait_time} seconds..."
                            )
                            time.sleep(wait_time)
                        else:
                            advanced_logger.logger.error(
                                f"All {self.max_retries + 1} attempts failed for {func.__name__}"
                            )
                
                raise last_exception
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < self.max_retries:
                            wait_time = self.backoff_factor * (2 ** attempt)
                            advanced_logger.logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                f"Retrying in {wait_time} seconds..."
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            advanced_logger.logger.error(
                                f"All {self.max_retries + 1} attempts failed for {func.__name__}"
                            )
                
                raise last_exception
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper
        
        return decorator

# Global error recovery instance
error_recovery = ErrorRecovery()

# Example usage
@error_handler()
@error_recovery.retry_on_error((ConnectionError, TimeoutError))
def example_function():
    """Example function with error handling and retry."""
    advanced_logger.logger.info("Executing example function")
    # Simulate potential error
    import random
    if random.random() < 0.3:  # 30% chance of error
        raise ConnectionError("Simulated connection error")
    return "Success"

if __name__ == "__main__":
    # Test the logging system
    print("Testing Advanced Error Handling & Logging System...")
    
    # Test basic logging
    advanced_logger.logger.info("System started")
    advanced_logger.logger.warning("This is a warning")
    advanced_logger.logger.error("This is an error")
    
    # Test error handling
    try:
        example_function()
    except Exception as e:
        print(f"Caught exception: {e}")
    
    # Test error stats
    stats = advanced_logger.get_error_stats()
    print(f"Error stats: {json.dumps(stats, indent=2)}")
    
    print("Logging system test completed!")
