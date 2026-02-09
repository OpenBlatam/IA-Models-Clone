"""
Benchmark Comparator
Compare test results against benchmarks
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from statistics import mean

class BenchmarkComparator:
    """Compare test results against benchmarks"""
    
    BENCHMARKS = {
        'industry_standard': {
            'success_rate': 95,
            'execution_time': 120,
            'tests_per_second': 2.0,
            'failure_rate': 5
        },
        'excellent': {
            'success_rate': 98,
            'execution_time': 60,
            'tests_per_second': 5.0,
            'failure_rate': 2
        },
        'good': {
            'success_rate': 95,
            'execution_time': 120,
            'tests_per_second': 2.0,
            'failure_rate': 5
        },
        'acceptable': {
            'success_rate': 90,
            'execution_time': 300,
            'tests_per_second': 1.0,
            'failure_rate': 10
        }
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def compare_benchmarks(self, benchmark_type: str = 'industry_standard', lookback_days: int = 30) -> Dict:
        """Compare test results against benchmarks"""
        if benchmark_type not in self.BENCHMARKS:
            return {'error': f'Unknown benchmark: {benchmark_type}'}
        
        history = self._load_history()
        benchmark = self.BENCHMARKS[benchmark_type]
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate current metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        avg_success = mean(success_rates) if success_rates else 0
        avg_time = mean(execution_times) if execution_times else 0
        total_tests_sum = sum(total_tests)
        total_time = sum(execution_times)
        total_failures = sum(failures)
        
        tests_per_second = total_tests_sum / total_time if total_time > 0 else 0
        failure_rate = (total_failures / total_tests_sum * 100) if total_tests_sum > 0 else 0
        
        # Compare against benchmarks
        comparisons = {
            'benchmark_type': benchmark_type,
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'comparisons': {
                'success_rate': self._compare_metric('success_rate', avg_success, benchmark['success_rate'], higher_better=True),
                'execution_time': self._compare_metric('execution_time', avg_time, benchmark['execution_time'], higher_better=False),
                'tests_per_second': self._compare_metric('tests_per_second', tests_per_second, benchmark['tests_per_second'], higher_better=True),
                'failure_rate': self._compare_metric('failure_rate', failure_rate, benchmark['failure_rate'], higher_better=False)
            },
            'overall_score': 0.0
        }
        
        # Calculate overall score
        scores = [comp['score'] for comp in comparisons['comparisons'].values()]
        comparisons['overall_score'] = round(mean(scores), 1)
        
        return comparisons
    
    def _compare_metric(self, name: str, actual: float, benchmark: float, higher_better: bool = True) -> Dict:
        """Compare a metric against benchmark"""
        if higher_better:
            ratio = (actual / benchmark * 100) if benchmark > 0 else 0
            meets_benchmark = actual >= benchmark
        else:
            ratio = (benchmark / actual * 100) if actual > 0 else 0
            meets_benchmark = actual <= benchmark
        
        # Score: 0-100 based on ratio
        score = min(100, max(0, ratio))
        
        return {
            'metric': name,
            'actual': round(actual, 2),
            'benchmark': benchmark,
            'meets_benchmark': meets_benchmark,
            'ratio': round(ratio, 1),
            'score': round(score, 1),
            'status': 'meets' if meets_benchmark else 'below'
        }
    
    def generate_comparison_report(self, comparison: Dict) -> str:
        """Generate comparison report"""
        lines = []
        lines.append("=" * 80)
        lines.append("BENCHMARK COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in comparison:
            lines.append(f"❌ {comparison['error']}")
            return "\n".join(lines)
        
        lines.append(f"Benchmark Type: {comparison['benchmark_type'].replace('_', ' ').title()}")
        lines.append(f"Period: {comparison['period']}")
        lines.append(f"Total Runs: {comparison['total_runs']}")
        lines.append("")
        
        score_emoji = "🟢" if comparison['overall_score'] >= 80 else "🟡" if comparison['overall_score'] >= 60 else "🔴"
        lines.append(f"{score_emoji} Overall Score: {comparison['overall_score']}/100")
        lines.append("")
        
        lines.append("📊 METRIC COMPARISONS")
        lines.append("-" * 80)
        
        for metric_name, comp in comparison['comparisons'].items():
            status_emoji = "✅" if comp['meets_benchmark'] else "⚠️"
            lines.append(f"\n{status_emoji} {metric_name.replace('_', ' ').title()}")
            lines.append(f"   Actual: {comp['actual']}")
            lines.append(f"   Benchmark: {comp['benchmark']}")
            lines.append(f"   Status: {comp['status'].upper()}")
            lines.append(f"   Score: {comp['score']}/100")
        
        lines.append("")
        
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
    import sys
    
    project_root = Path(__file__).parent.parent
    comparator = BenchmarkComparator(project_root)
    
    benchmark_type = sys.argv[1] if len(sys.argv) > 1 else 'industry_standard'
    comparison = comparator.compare_benchmarks(benchmark_type=benchmark_type)
    
    report = comparator.generate_comparison_report(comparison)
    print(report)
    
    # Save report
    report_file = project_root / f"benchmark_comparison_{benchmark_type}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Benchmark comparison report saved to: {report_file}")

if __name__ == "__main__":
    main()







