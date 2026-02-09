"""
Test Metrics Tracker
Tracks test metrics over time for trend analysis
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class TestMetricsTracker:
    """Track and analyze test metrics over time"""
    
    def __init__(self, metrics_file: str = "test_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.metrics_history: List[Dict[str, Any]] = []
        self._load_history()
    
    def _load_history(self):
        """Load metrics history from file"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics_history = json.load(f)
            except Exception:
                self.metrics_history = []
        else:
            self.metrics_history = []
    
    def _save_history(self):
        """Save metrics history to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metrics: {e}")
    
    def record_test_run(self, results: Dict[str, Any]):
        """Record a test run's metrics"""
        metric_entry = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': results.get('total_tests', 0),
            'passed': results.get('total_tests', 0) - results.get('failures', 0) - results.get('errors', 0) - results.get('skipped', 0),
            'failures': results.get('failures', 0),
            'errors': results.get('errors', 0),
            'skipped': results.get('skipped', 0),
            'success_rate': results.get('success_rate', 0),
            'execution_time': results.get('execution_time', 0),
            'tests_per_second': results.get('tests_per_second', 0)
        }
        
        self.metrics_history.append(metric_entry)
        
        # Keep only last 100 runs
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        self._save_history()
    
    def get_trends(self) -> Dict[str, Any]:
        """Analyze trends in test metrics"""
        if len(self.metrics_history) < 2:
            return {"message": "Not enough data for trend analysis"}
        
        recent = self.metrics_history[-10:]  # Last 10 runs
        
        trends = {
            'success_rate_trend': self._calculate_trend([m['success_rate'] for m in recent]),
            'execution_time_trend': self._calculate_trend([m['execution_time'] for m in recent]),
            'total_tests_trend': self._calculate_trend([m['total_tests'] for m in recent]),
            'average_success_rate': sum(m['success_rate'] for m in recent) / len(recent),
            'average_execution_time': sum(m['execution_time'] for m in recent) / len(recent),
            'runs_analyzed': len(recent)
        }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate if values are increasing, decreasing, or stable"""
        if len(values) < 2:
            return "insufficient_data"
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        change_percent = ((avg_second - avg_first) / avg_first * 100) if avg_first > 0 else 0
        
        if abs(change_percent) < 1:
            return "stable"
        elif change_percent > 0:
            return f"increasing (+{change_percent:.1f}%)"
        else:
            return f"decreasing ({change_percent:.1f}%)"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        if not self.metrics_history:
            return {"message": "No metrics recorded yet"}
        
        all_runs = self.metrics_history
        latest = all_runs[-1]
        
        return {
            'total_runs': len(all_runs),
            'latest_run': latest,
            'best_success_rate': max(m['success_rate'] for m in all_runs),
            'worst_success_rate': min(m['success_rate'] for m in all_runs),
            'average_success_rate': sum(m['success_rate'] for m in all_runs) / len(all_runs),
            'fastest_run': min(m['execution_time'] for m in all_runs),
            'slowest_run': max(m['execution_time'] for m in all_runs),
            'average_execution_time': sum(m['execution_time'] for m in all_runs) / len(all_runs),
            'trends': self.get_trends()
        }







