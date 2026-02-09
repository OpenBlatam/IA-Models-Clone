"""
Debug Helpers for Testing
Utilities for debugging and troubleshooting tests
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager
import time
from datetime import datetime


class TestLogger:
    """Logger for test debugging"""
    
    def __init__(self, name: str = "test", level: int = logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_test_start(self, test_name: str):
        """Log test start"""
        self.logger.info(f"Starting test: {test_name}")
    
    def log_test_end(self, test_name: str, duration: float):
        """Log test end"""
        self.logger.info(f"Completed test: {test_name} in {duration:.3f}s")
    
    def log_assertion(self, assertion: str, result: bool):
        """Log assertion result"""
        status = "PASS" if result else "FAIL"
        self.logger.debug(f"Assertion {status}: {assertion}")
    
    def log_mock_call(self, mock_name: str, method: str, args: tuple, kwargs: dict):
        """Log mock call"""
        self.logger.debug(f"Mock {mock_name}.{method} called with args={args}, kwargs={kwargs}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        self.logger.error(f"Error in {context}: {str(error)}")
        self.logger.debug(traceback.format_exc())


class TestDebugger:
    """Debugger for test troubleshooting"""
    
    @staticmethod
    @contextmanager
    def debug_context(test_name: str, logger: Optional[TestLogger] = None):
        """Context manager for debugging test execution"""
        logger = logger or TestLogger()
        start_time = time.time()
        logger.log_test_start(test_name)
        
        try:
            yield logger
        except Exception as e:
            logger.log_error(e, test_name)
            raise
        finally:
            duration = time.time() - start_time
            logger.log_test_end(test_name, duration)
    
    @staticmethod
    def capture_mock_calls(mock: Any, logger: Optional[TestLogger] = None) -> List[Dict[str, Any]]:
        """Capture and log all mock calls"""
        logger = logger or TestLogger()
        calls = []
        
        if hasattr(mock, 'call_args_list'):
            for i, call in enumerate(mock.call_args_list):
                call_info = {
                    "call_number": i + 1,
                    "args": call[0] if call else (),
                    "kwargs": call[1] if len(call) > 1 else {}
                }
                calls.append(call_info)
                logger.log_mock_call(
                    str(mock),
                    "call",
                    call_info["args"],
                    call_info["kwargs"]
                )
        
        return calls
    
    @staticmethod
    def inspect_object(obj: Any, logger: Optional[TestLogger] = None) -> Dict[str, Any]:
        """Inspect object and log details"""
        logger = logger or TestLogger()
        info = {
            "type": type(obj).__name__,
            "module": type(obj).__module__,
            "attributes": dir(obj),
            "dict": getattr(obj, "__dict__", {})
        }
        logger.debug(f"Object inspection: {info}")
        return info


class TestProfiler:
    """Profiler for test performance"""
    
    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}
        self.current_measurements: Dict[str, float] = {}
    
    def start_measurement(self, name: str):
        """Start measuring execution time"""
        self.current_measurements[name] = time.time()
    
    def end_measurement(self, name: str) -> float:
        """End measurement and return duration"""
        if name not in self.current_measurements:
            raise ValueError(f"Measurement {name} not started")
        
        duration = time.time() - self.current_measurements[name]
        if name not in self.measurements:
            self.measurements[name] = []
        self.measurements[name].append(duration)
        del self.current_measurements[name]
        return duration
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for measurement"""
        if name not in self.measurements:
            return {}
        
        values = self.measurements[name]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "total": sum(values)
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all measurements"""
        return {name: self.get_statistics(name) for name in self.measurements}


class TestSnapshot:
    """Snapshot utility for test state"""
    
    @staticmethod
    def create_snapshot(obj: Any) -> Dict[str, Any]:
        """Create snapshot of object state"""
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": type(obj).__name__,
            "state": {}
        }
        
        if hasattr(obj, "__dict__"):
            snapshot["state"] = obj.__dict__.copy()
        elif isinstance(obj, dict):
            snapshot["state"] = obj.copy()
        else:
            snapshot["state"] = str(obj)
        
        return snapshot
    
    @staticmethod
    def compare_snapshots(snapshot1: Dict[str, Any], snapshot2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two snapshots"""
        differences = {
            "added": {},
            "removed": {},
            "changed": {}
        }
        
        state1 = snapshot1.get("state", {})
        state2 = snapshot2.get("state", {})
        
        # Find added and changed
        for key, value in state2.items():
            if key not in state1:
                differences["added"][key] = value
            elif state1[key] != value:
                differences["changed"][key] = {
                    "old": state1[key],
                    "new": value
                }
        
        # Find removed
        for key in state1:
            if key not in state2:
                differences["removed"][key] = state1[key]
        
        return differences


class TestReporter:
    """Reporter for test results"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def record_test(
        self,
        test_name: str,
        status: str,
        duration: float,
        error: Optional[Exception] = None
    ):
        """Record test result"""
        result = {
            "test_name": test_name,
            "status": status,
            "duration": duration,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if error:
            result["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            }
        
        self.results.append(result)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "passed")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        total_duration = sum(r["duration"] for r in self.results)
        
        return {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "success_rate": (passed / total * 100) if total > 0 else 0,
                "total_duration": total_duration
            },
            "results": self.results
        }


# Convenience exports
TestLogger = TestLogger
TestDebugger = TestDebugger
TestProfiler = TestProfiler
TestSnapshot = TestSnapshot
TestReporter = TestReporter



