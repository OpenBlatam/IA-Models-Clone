"""
Metrics System
Comprehensive metrics tracking and analysis
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, median, stdev
from collections import defaultdict

class MetricsSystem:
    """Comprehensive metrics system"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.metrics_file = project_root / "test_metrics.json"
    
    def collect_metrics(self, lookback_days: int = 30) -> Dict:
        """Collect comprehensive metrics"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Collect all metrics
        metrics = {
            'collection_date': datetime.now().isoformat(),
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'execution_metrics': self._collect_execution_metrics(recent),
            'quality_metrics': self._collect_quality_metrics(recent),
            'performance_metrics': self._collect_performance_metrics(recent),
            'reliability_metrics': self._collect_reliability_metrics(recent),
            'trend_metrics': self._collect_trend_metrics(recent)
        }
        
        # Save metrics
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _collect_execution_metrics(self, recent: List[Dict]) -> Dict:
        """Collect execution metrics"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        return {
            'avg_execution_time': round(mean(execution_times), 2) if execution_times else 0,
            'median_execution_time': round(median(execution_times), 2) if len(execution_times) > 1 else (round(execution_times[0], 2) if execution_times else 0),
            'total_execution_time': round(sum(execution_times), 2),
            'avg_tests_per_run': round(mean(total_tests), 0) if total_tests else 0,
            'total_tests_executed': sum(total_tests)
        }
    
    def _collect_quality_metrics(self, recent: List[Dict]) -> Dict:
        """Collect quality metrics"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        return {
            'avg_success_rate': round(mean(success_rates), 2) if success_rates else 0,
            'min_success_rate': round(min(success_rates), 2) if success_rates else 0,
            'max_success_rate': round(max(success_rates), 2) if success_rates else 0,
            'total_failures': sum(failures),
            'avg_failures_per_run': round(mean(failures), 2) if failures else 0
        }
    
    def _collect_performance_metrics(self, recent: List[Dict]) -> Dict:
        """Collect performance metrics"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        if execution_times and total_tests:
            avg_time = mean(execution_times)
            avg_tests = mean(total_tests)
            tests_per_second = avg_tests / avg_time if avg_time > 0 else 0
        else:
            tests_per_second = 0
        
        return {
            'tests_per_second': round(tests_per_second, 2),
            'fastest_run': round(min(execution_times), 2) if execution_times else 0,
            'slowest_run': round(max(execution_times), 2) if execution_times else 0,
            'performance_variance': round(stdev(execution_times), 2) if len(execution_times) > 1 else 0
        }
    
    def _collect_reliability_metrics(self, recent: List[Dict]) -> Dict:
        """Collect reliability metrics"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if len(success_rates) > 1:
            variance = stdev(success_rates)
            consistency = max(0, 100 - (variance * 10))
        else:
            variance = 0
            consistency = 100
        
        return {
            'consistency_score': round(consistency, 1),
            'variance': round(variance, 2),
            'reliability_score': round(mean(success_rates), 1) if success_rates else 0
        }
    
    def _collect_trend_metrics(self, recent: List[Dict]) -> Dict:
        """Collect trend metrics"""
        if len(recent) < 2:
            return {}
        
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_success = mean([r.get('success_rate', 0) for r in first_half])
        second_success = mean([r.get('success_rate', 0) for r in second_half])
        
        first_time = mean([r.get('execution_time', 0) for r in first_half])
        second_time = mean([r.get('execution_time', 0) for r in second_half])
        
        return {
            'success_rate_trend': round(second_success - first_success, 2),
            'execution_time_trend': round(second_time - first_time, 2),
            'trend_direction': 'improving' if second_success > first_success else 'declining' if second_success < first_success else 'stable'
        }
    
    def generate_metrics_report(self, metrics: Dict) -> str:
        """Generate metrics report"""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE METRICS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in metrics:
            lines.append(f"❌ {metrics['error']}")
            return "\n".join(lines)
        
        lines.append(f"Collection Date: {metrics['collection_date'][:19]}")
        lines.append(f"Period: {metrics['period']}")
        lines.append(f"Total Runs: {metrics['total_runs']}")
        lines.append("")
        
        def format_metrics_section(name, data):
            lines.append(f"📊 {name.upper().replace('_', ' ')}")
            lines.append("-" * 80)
            for key, value in data.items():
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
            lines.append("")
        
        format_metrics_section("Execution Metrics", metrics['execution_metrics'])
        format_metrics_section("Quality Metrics", metrics['quality_metrics'])
        format_metrics_section("Performance Metrics", metrics['performance_metrics'])
        format_metrics_section("Reliability Metrics", metrics['reliability_metrics'])
        
        if 'trend_metrics' in metrics and metrics['trend_metrics']:
            format_metrics_section("Trend Metrics", metrics['trend_metrics'])
        
        return "\n".join(lines)
    
    def _load_history(self) -> List[Dict]:
        """Load test history"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    metrics_system = MetricsSystem(project_root)
    metrics = metrics_system.collect_metrics(lookback_days=30)
    
    report = metrics_system.generate_metrics_report(metrics)
    print(report)
    
    print(f"\n📄 Metrics saved to: test_metrics.json")

if __name__ == "__main__":
    main()







