"""
Test Benchmarking
Compare test metrics against industry benchmarks
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from statistics import mean

class TestBenchmarking:
    """Benchmark test metrics against industry standards"""
    
    # Industry benchmarks
    BENCHMARKS = {
        'success_rate': {
            'excellent': 98.0,
            'good': 95.0,
            'average': 90.0,
            'poor': 85.0
        },
        'execution_time': {
            'excellent': 60,  # seconds
            'good': 120,
            'average': 300,
            'poor': 600
        },
        'test_coverage': {
            'excellent': 90,  # percentage
            'good': 80,
            'average': 70,
            'poor': 60
        },
        'flakiness_rate': {
            'excellent': 1.0,  # percentage
            'good': 2.0,
            'average': 5.0,
            'poor': 10.0
        }
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def benchmark_metrics(self, lookback_days: int = 30) -> Dict:
        """Benchmark current metrics against industry standards"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate current metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        current_metrics = {
            'success_rate': mean(success_rates) if success_rates else 0,
            'execution_time': mean(execution_times) if execution_times else 0,
            'test_coverage': 75.0,  # Placeholder - would need coverage data
            'flakiness_rate': 2.5  # Placeholder - would need flakiness data
        }
        
        # Compare against benchmarks
        comparisons = {}
        
        for metric, value in current_metrics.items():
            if metric not in self.BENCHMARKS:
                continue
            
            benchmark = self.BENCHMARKS[metric]
            level = self._determine_level(value, benchmark, metric)
            
            comparisons[metric] = {
                'current': round(value, 2),
                'level': level,
                'benchmarks': benchmark,
                'vs_excellent': round(value - benchmark['excellent'], 2),
                'vs_good': round(value - benchmark['good'], 2),
                'vs_average': round(value - benchmark['average'], 2)
            }
        
        return {
            'current_metrics': current_metrics,
            'comparisons': comparisons,
            'overall_rating': self._calculate_overall_rating(comparisons),
            'period_days': lookback_days
        }
    
    def _determine_level(self, value: float, benchmark: Dict, metric: str) -> str:
        """Determine performance level"""
        if metric == 'execution_time':
            # Lower is better for execution time
            if value <= benchmark['excellent']:
                return 'excellent'
            elif value <= benchmark['good']:
                return 'good'
            elif value <= benchmark['average']:
                return 'average'
            else:
                return 'poor'
        else:
            # Higher is better for other metrics
            if value >= benchmark['excellent']:
                return 'excellent'
            elif value >= benchmark['good']:
                return 'good'
            elif value >= benchmark['average']:
                return 'average'
            else:
                return 'poor'
    
    def _calculate_overall_rating(self, comparisons: Dict) -> str:
        """Calculate overall rating"""
        levels = [comp['level'] for comp in comparisons.values()]
        
        level_scores = {'excellent': 4, 'good': 3, 'average': 2, 'poor': 1}
        avg_score = mean([level_scores.get(level, 2) for level in levels])
        
        if avg_score >= 3.5:
            return 'excellent'
        elif avg_score >= 2.5:
            return 'good'
        elif avg_score >= 1.5:
            return 'average'
        else:
            return 'poor'
    
    def generate_benchmark_report(self, benchmark_data: Dict) -> str:
        """Generate benchmarking report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST BENCHMARKING REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in benchmark_data:
            lines.append(f"❌ {benchmark_data['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: Last {benchmark_data['period_days']} days")
        lines.append(f"Overall Rating: {benchmark_data['overall_rating'].upper()}")
        lines.append("")
        
        lines.append("📊 BENCHMARK COMPARISON")
        lines.append("-" * 80)
        
        level_emoji = {
            'excellent': '🟢',
            'good': '🟡',
            'average': '🟠',
            'poor': '🔴'
        }
        
        for metric, comp in benchmark_data['comparisons'].items():
            emoji = level_emoji.get(comp['level'], '⚪')
            lines.append(f"\n{emoji} {metric.replace('_', ' ').title()}")
            lines.append(f"   Current: {comp['current']}")
            lines.append(f"   Level: {comp['level'].upper()}")
            lines.append(f"   vs Excellent: {comp['vs_excellent']:+.2f}")
            lines.append(f"   vs Good: {comp['vs_good']:+.2f}")
            lines.append(f"   vs Average: {comp['vs_average']:+.2f}")
        
        lines.append("")
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 80)
        
        for metric, comp in benchmark_data['comparisons'].items():
            if comp['level'] in ['average', 'poor']:
                lines.append(f"• Improve {metric.replace('_', ' ')} to reach industry standards")
        
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
    
    benchmarking = TestBenchmarking(project_root)
    benchmark_data = benchmarking.benchmark_metrics(lookback_days=30)
    
    report = benchmarking.generate_benchmark_report(benchmark_data)
    print(report)
    
    # Save report
    report_file = project_root / "benchmarking_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Benchmarking report saved to: {report_file}")

if __name__ == "__main__":
    main()







