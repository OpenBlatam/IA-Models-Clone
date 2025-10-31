"""
ML NLP Benchmark Logger System
Real, working advanced logging for ML NLP Benchmark system
"""

import logging
import logging.handlers
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
from collections import defaultdict, deque

class MLNLPBenchmarkLogger:
    """Advanced logger for ML NLP Benchmark system"""
    
    def __init__(self, name: str = "ml_nlp_benchmark", log_file: str = "ml_nlp_benchmark.log"):
        self.name = name
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.lock = threading.Lock()
        
        # Setup handlers
        self._setup_handlers()
        
        # Log rotation
        self._setup_rotation()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # JSON handler for structured logging
        json_handler = logging.FileHandler(f"{self.log_file}.json")
        json_handler.setLevel(logging.INFO)
        json_formatter = JSONFormatter()
        json_handler.setFormatter(json_formatter)
        self.logger.addHandler(json_handler)
    
    def _setup_rotation(self):
        """Setup log rotation"""
        # Rotating file handler
        rotating_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        rotating_handler.setLevel(logging.DEBUG)
        rotating_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        rotating_handler.setFormatter(rotating_formatter)
        self.logger.addHandler(rotating_handler)
    
    def log_request(self, endpoint: str, method: str, processing_time: float, 
                   status_code: int = 200, user_id: Optional[str] = None):
        """Log API request"""
        with self.lock:
            self.request_counts[endpoint] += 1
            self.performance_stats[endpoint].append(processing_time)
        
        log_data = {
            "type": "request",
            "endpoint": endpoint,
            "method": method,
            "processing_time": processing_time,
            "status_code": status_code,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Request: {method} {endpoint} - {processing_time:.3f}s - {status_code}", 
                        extra={"log_data": log_data})
    
    def log_error(self, error_type: str, error_message: str, endpoint: Optional[str] = None,
                 user_id: Optional[str] = None, traceback: Optional[str] = None):
        """Log error"""
        with self.lock:
            self.error_counts[error_type] += 1
        
        log_data = {
            "type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "endpoint": endpoint,
            "user_id": user_id,
            "traceback": traceback,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.error(f"Error: {error_type} - {error_message}", 
                         extra={"log_data": log_data})
    
    def log_performance(self, operation: str, duration: float, details: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        with self.lock:
            self.performance_stats[operation].append(duration)
        
        log_data = {
            "type": "performance",
            "operation": operation,
            "duration": duration,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Performance: {operation} - {duration:.3f}s", 
                        extra={"log_data": log_data})
    
    def log_analysis(self, analysis_type: str, text_length: int, processing_time: float,
                    result_count: int, method: str = "default"):
        """Log analysis operation"""
        log_data = {
            "type": "analysis",
            "analysis_type": analysis_type,
            "text_length": text_length,
            "processing_time": processing_time,
            "result_count": result_count,
            "method": method,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Analysis: {analysis_type} - {text_length} chars - {processing_time:.3f}s - {result_count} results", 
                        extra={"log_data": log_data})
    
    def log_system_event(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log system event"""
        log_data = {
            "type": "system_event",
            "event_type": event_type,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"System Event: {event_type} - {message}", 
                        extra={"log_data": log_data})
    
    def log_security_event(self, event_type: str, message: str, ip_address: Optional[str] = None,
                          user_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Log security event"""
        log_data = {
            "type": "security_event",
            "event_type": event_type,
            "message": message,
            "ip_address": ip_address,
            "user_id": user_id,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.warning(f"Security Event: {event_type} - {message}", 
                           extra={"log_data": log_data})
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            stats = {}
            for operation, times in self.performance_stats.items():
                if times:
                    stats[operation] = {
                        "count": len(times),
                        "total_time": sum(times),
                        "average_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "recent_times": list(times[-10:])  # Last 10 operations
                    }
            return stats
    
    def get_request_stats(self) -> Dict[str, int]:
        """Get request statistics"""
        with self.lock:
            return dict(self.request_counts)
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        with self.lock:
            return dict(self.error_counts)
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get comprehensive log summary"""
        return {
            "performance_stats": self.get_performance_stats(),
            "request_stats": self.get_request_stats(),
            "error_stats": self.get_error_stats(),
            "log_file": self.log_file,
            "log_file_size": os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0,
            "json_log_file": f"{self.log_file}.json",
            "json_log_file_size": os.path.getsize(f"{self.log_file}.json") if os.path.exists(f"{self.log_file}.json") else 0
        }
    
    def clear_stats(self):
        """Clear all statistics"""
        with self.lock:
            self.performance_stats.clear()
            self.request_counts.clear()
            self.error_counts.clear()
        
        self.logger.info("Statistics cleared")

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra data if present
        if hasattr(record, 'log_data'):
            log_entry.update(record.log_data)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)

class MLNLPBenchmarkLogAnalyzer:
    """Log analyzer for ML NLP Benchmark system"""
    
    def __init__(self, log_file: str = "ml_nlp_benchmark.log.json"):
        self.log_file = log_file
    
    def analyze_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze logs for the last N hours"""
        if not os.path.exists(self.log_file):
            return {"error": "Log file not found"}
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        analysis = {
            "total_entries": 0,
            "error_count": 0,
            "request_count": 0,
            "performance_entries": 0,
            "analysis_entries": 0,
            "system_events": 0,
            "security_events": 0,
            "average_processing_time": 0.0,
            "most_common_endpoints": {},
            "error_types": {},
            "performance_by_operation": {}
        }
        
        processing_times = []
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        
                        # Check if entry is within time range
                        entry_time = datetime.fromisoformat(log_entry["timestamp"]).timestamp()
                        if entry_time < cutoff_time:
                            continue
                        
                        analysis["total_entries"] += 1
                        
                        # Analyze by type
                        if log_entry.get("type") == "error":
                            analysis["error_count"] += 1
                            error_type = log_entry.get("error_type", "unknown")
                            analysis["error_types"][error_type] = analysis["error_types"].get(error_type, 0) + 1
                        
                        elif log_entry.get("type") == "request":
                            analysis["request_count"] += 1
                            endpoint = log_entry.get("endpoint", "unknown")
                            analysis["most_common_endpoints"][endpoint] = analysis["most_common_endpoints"].get(endpoint, 0) + 1
                            processing_times.append(log_entry.get("processing_time", 0))
                        
                        elif log_entry.get("type") == "performance":
                            analysis["performance_entries"] += 1
                            operation = log_entry.get("operation", "unknown")
                            if operation not in analysis["performance_by_operation"]:
                                analysis["performance_by_operation"][operation] = []
                            analysis["performance_by_operation"][operation].append(log_entry.get("duration", 0))
                        
                        elif log_entry.get("type") == "analysis":
                            analysis["analysis_entries"] += 1
                        
                        elif log_entry.get("type") == "system_event":
                            analysis["system_events"] += 1
                        
                        elif log_entry.get("type") == "security_event":
                            analysis["security_events"] += 1
                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        continue
            
            # Calculate average processing time
            if processing_times:
                analysis["average_processing_time"] = sum(processing_times) / len(processing_times)
            
            # Sort most common endpoints
            analysis["most_common_endpoints"] = dict(
                sorted(analysis["most_common_endpoints"].items(), key=lambda x: x[1], reverse=True)[:10]
            )
            
            # Sort error types
            analysis["error_types"] = dict(
                sorted(analysis["error_types"].items(), key=lambda x: x[1], reverse=True)
            )
            
            # Calculate performance averages
            for operation, times in analysis["performance_by_operation"].items():
                if times:
                    analysis["performance_by_operation"][operation] = {
                        "count": len(times),
                        "average": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times)
                    }
        
        except Exception as e:
            analysis["error"] = f"Error analyzing logs: {str(e)}"
        
        return analysis

# Global logger instance
ml_nlp_benchmark_logger = MLNLPBenchmarkLogger()

def get_logger() -> MLNLPBenchmarkLogger:
    """Get the global logger instance"""
    return ml_nlp_benchmark_logger

def log_request(endpoint: str, method: str, processing_time: float, 
               status_code: int = 200, user_id: Optional[str] = None):
    """Log API request"""
    ml_nlp_benchmark_logger.log_request(endpoint, method, processing_time, status_code, user_id)

def log_error(error_type: str, error_message: str, endpoint: Optional[str] = None,
             user_id: Optional[str] = None, traceback: Optional[str] = None):
    """Log error"""
    ml_nlp_benchmark_logger.log_error(error_type, error_message, endpoint, user_id, traceback)

def log_performance(operation: str, duration: float, details: Optional[Dict[str, Any]] = None):
    """Log performance metrics"""
    ml_nlp_benchmark_logger.log_performance(operation, duration, details)

def log_analysis(analysis_type: str, text_length: int, processing_time: float,
                result_count: int, method: str = "default"):
    """Log analysis operation"""
    ml_nlp_benchmark_logger.log_analysis(analysis_type, text_length, processing_time, result_count, method)

def log_system_event(event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
    """Log system event"""
    ml_nlp_benchmark_logger.log_system_event(event_type, message, details)

def log_security_event(event_type: str, message: str, ip_address: Optional[str] = None,
                      user_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
    """Log security event"""
    ml_nlp_benchmark_logger.log_security_event(event_type, message, ip_address, user_id, details)











