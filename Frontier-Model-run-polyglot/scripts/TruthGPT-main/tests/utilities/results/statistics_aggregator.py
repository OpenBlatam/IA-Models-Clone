"""
Statistics Aggregator
Aggregates and analyzes test statistics from multiple sources
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean, median, stdev

class StatisticsAggregator:
    """Aggregate statistics from multiple sources"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def aggregate_all_statistics(self) -> Dict:
        """Aggregate all available statistics"""
        stats = {
            'overall': self._get_overall_stats(),
            'trends': self._get_trend_stats(),
            'categories': self._get_category_stats(),
            'performance': self._get_performance_stats(),
            'reliability': self._get_reliability_stats()
        }
        
        return stats
    
    def _get_overall_stats(self) -> Dict:
        """Get overall statistics"""
        history = self._load_history()
        
        if not history:
            return {'error': 'No history data'}
        
        success_rates = [r.get('success_rate', 0) for r in history]
        execution_times = [r.get('execution_time', 0) for r in history]
        total_tests = [r.get('total_tests', 0) for r in history]
        
        return {
            'total_runs': len(history),
            'average_success_rate': mean(success_rates),
            'median_success_rate': median(success_rates),
            'std_success_rate': stdev(success_rates) if len(success_rates) > 1 else 0,
            'min_success_rate': min(success_rates),
            'max_success_rate': max(success_rates),
            'average_execution_time': mean(execution_times),
            'median_execution_time': median(execution_times),
            'total_tests_run': sum(total_tests),
            'average_tests_per_run': mean(total_tests)
        }
    
    def _get_trend_stats(self) -> Dict:
        """Get trend statistics"""
        history = self._load_history()
        
        if len(history) < 2:
            return {'error': 'Insufficient data for trends'}
        
        # Split into halves
        first_half = history[:len(history)//2]
        second_half = history[len(history)//2:]
        
        first_avg_success = mean([r.get('success_rate', 0) for r in first_half])
        second_avg_success = mean([r.get('success_rate', 0) for r in second_half])
        
        first_avg_time = mean([r.get('execution_time', 0) for r in first_half])
        second_avg_time = mean([r.get('execution_time', 0) for r in second_half])
        
        success_trend = ((second_avg_success - first_avg_success) / first_avg_success * 100) if first_avg_success > 0 else 0
        time_trend = ((second_avg_time - first_avg_time) / first_avg_time * 100) if first_avg_time > 0 else 0
        
        return {
            'success_rate_trend': success_trend,
            'execution_time_trend': time_trend,
            'trend_direction': 'improving' if success_trend > 0 else 'declining' if success_trend < 0 else 'stable'
        }
    
    def _get_category_stats(self) -> Dict:
        """Get statistics by category"""
        history = self._load_history()
        
        category_stats = defaultdict(list)
        
        for run in history:
            category = run.get('test_category', 'all')
            category_stats[category].append(run)
        
        stats_by_category = {}
        for category, runs in category_stats.items():
            success_rates = [r.get('success_rate', 0) for r in runs]
            stats_by_category[category] = {
                'runs': len(runs),
                'average_success_rate': mean(success_rates),
                'min_success_rate': min(success_rates),
                'max_success_rate': max(success_rates)
            }
        
        return stats_by_category
    
    def _get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        history = self._load_history()
        
        if not history:
            return {'error': 'No history data'}
        
        execution_times = [r.get('execution_time', 0) for r in history]
        tests_per_second = []
        
        for run in history:
            total_tests = run.get('total_tests', 0)
            exec_time = run.get('execution_time', 0)
            if exec_time > 0:
                tests_per_second.append(total_tests / exec_time)
        
        return {
            'average_execution_time': mean(execution_times),
            'fastest_run': min(execution_times),
            'slowest_run': max(execution_times),
            'average_tests_per_second': mean(tests_per_second) if tests_per_second else 0,
            'performance_variance': stdev(execution_times) if len(execution_times) > 1 else 0
        }
    
    def _get_reliability_stats(self) -> Dict:
        """Get reliability statistics"""
        history = self._load_history()
        
        if not history:
            return {'error': 'No history data'}
        
        # Calculate consistency
        success_rates = [r.get('success_rate', 0) for r in history]
        avg_success = mean(success_rates)
        
        # Count runs within 5% of average
        consistent_runs = sum(1 for rate in success_rates if abs(rate - avg_success) < 5)
        consistency_percentage = (consistent_runs / len(success_rates) * 100) if success_rates else 0
        
        # Calculate failure rate
        total_failures = sum(r.get('failed', 0) + r.get('errors', 0) for r in history)
        total_tests = sum(r.get('total_tests', 0) for r in history)
        failure_rate = (total_failures / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'consistency_percentage': consistency_percentage,
            'average_success_rate': avg_success,
            'failure_rate': failure_rate,
            'reliability_score': avg_success * (consistency_percentage / 100)
        }
    
    def _load_history(self) -> List[Dict]:
        """Load test history"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive statistics report"""
        stats = self.aggregate_all_statistics()
        
        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE TEST STATISTICS")
        lines.append("=" * 80)
        lines.append("")
        
        # Overall stats
        if 'error' not in stats['overall']:
            overall = stats['overall']
            lines.append("📊 OVERALL STATISTICS")
            lines.append("-" * 80)
            lines.append(f"Total Runs:              {overall['total_runs']}")
            lines.append(f"Average Success Rate:    {overall['average_success_rate']:.1f}%")
            lines.append(f"Median Success Rate:     {overall['median_success_rate']:.1f}%")
            lines.append(f"Success Rate Range:      {overall['min_success_rate']:.1f}% - {overall['max_success_rate']:.1f}%")
            lines.append(f"Std Dev Success Rate:    {overall['std_success_rate']:.2f}%")
            lines.append(f"Average Execution Time:  {overall['average_execution_time']:.2f}s")
            lines.append(f"Total Tests Run:         {overall['total_tests_run']}")
            lines.append(f"Average Tests per Run:   {overall['average_tests_per_run']:.1f}")
            lines.append("")
        
        # Trends
        if 'error' not in stats['trends']:
            trends = stats['trends']
            lines.append("📈 TRENDS")
            lines.append("-" * 80)
            lines.append(f"Success Rate Trend:       {trends['success_rate_trend']:+.1f}%")
            lines.append(f"Execution Time Trend:     {trends['execution_time_trend']:+.1f}%")
            lines.append(f"Trend Direction:          {trends['trend_direction'].upper()}")
            lines.append("")
        
        # Performance
        if 'error' not in stats['performance']:
            perf = stats['performance']
            lines.append("⏱️  PERFORMANCE")
            lines.append("-" * 80)
            lines.append(f"Average Execution Time:  {perf['average_execution_time']:.2f}s")
            lines.append(f"Fastest Run:              {perf['fastest_run']:.2f}s")
            lines.append(f"Slowest Run:              {perf['slowest_run']:.2f}s")
            lines.append(f"Average Tests/Second:     {perf['average_tests_per_second']:.1f}")
            lines.append(f"Performance Variance:     {perf['performance_variance']:.2f}s")
            lines.append("")
        
        # Reliability
        if 'error' not in stats['reliability']:
            rel = stats['reliability']
            lines.append("🔒 RELIABILITY")
            lines.append("-" * 80)
            lines.append(f"Consistency:              {rel['consistency_percentage']:.1f}%")
            lines.append(f"Average Success Rate:     {rel['average_success_rate']:.1f}%")
            lines.append(f"Failure Rate:             {rel['failure_rate']:.2f}%")
            lines.append(f"Reliability Score:        {rel['reliability_score']:.1f}")
            lines.append("")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    aggregator = StatisticsAggregator(project_root)
    report = aggregator.generate_comprehensive_report()
    
    print(report)
    
    # Save report
    report_file = project_root / "comprehensive_statistics_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Report saved to: {report_file}")

if __name__ == "__main__":
    main()







