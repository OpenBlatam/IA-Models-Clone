"""
Performance Analyzer
Analyze test performance in detail
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, median, stdev

class PerformanceAnalyzer:
    """Analyze test performance"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_performance(self, lookback_days: int = 30) -> Dict:
        """Analyze test performance"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        # Calculate performance metrics
        avg_time = mean(execution_times) if execution_times else 0
        avg_tests = mean(total_tests) if total_tests else 0
        tests_per_second = avg_tests / avg_time if avg_time > 0 else 0
        
        # Performance categories
        if avg_time < 60:
            performance_category = 'Excellent'
        elif avg_time < 120:
            performance_category = 'Good'
        elif avg_time < 300:
            performance_category = 'Acceptable'
        else:
            performance_category = 'Needs Improvement'
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'execution_time': {
                'mean': round(avg_time, 2),
                'median': round(median(execution_times), 2) if len(execution_times) > 1 else (round(execution_times[0], 2) if execution_times else 0),
                'stdev': round(stdev(execution_times), 2) if len(execution_times) > 1 else 0,
                'min': round(min(execution_times), 2) if execution_times else 0,
                'max': round(max(execution_times), 2) if execution_times else 0
            },
            'throughput': {
                'tests_per_second': round(tests_per_second, 2),
                'average_tests': round(avg_tests, 0)
            },
            'performance_category': performance_category,
            'recommendations': self._generate_performance_recommendations(avg_time, tests_per_second)
        }
    
    def _generate_performance_recommendations(self, avg_time: float, tests_per_second: float) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if avg_time > 300:
            recommendations.append("Consider parallel test execution to reduce total time")
        
        if tests_per_second < 1:
            recommendations.append("Optimize individual test execution time")
        
        if avg_time > 600:
            recommendations.append("Review and optimize slow tests")
            recommendations.append("Consider test suite splitting")
        
        return recommendations
    
    def generate_performance_report(self, analysis: Dict) -> str:
        """Generate performance report"""
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append(f"Performance Category: {analysis['performance_category']}")
        lines.append("")
        
        lines.append("⏱️  EXECUTION TIME")
        lines.append("-" * 80)
        exec_time = analysis['execution_time']
        lines.append(f"Mean:    {exec_time['mean']:.2f}s")
        lines.append(f"Median:  {exec_time['median']:.2f}s")
        lines.append(f"Std Dev: {exec_time['stdev']:.2f}s")
        lines.append(f"Min:     {exec_time['min']:.2f}s")
        lines.append(f"Max:     {exec_time['max']:.2f}s")
        lines.append("")
        
        lines.append("📊 THROUGHPUT")
        lines.append("-" * 80)
        throughput = analysis['throughput']
        lines.append(f"Tests per Second: {throughput['tests_per_second']:.2f}")
        lines.append(f"Average Tests:    {throughput['average_tests']:.0f}")
        lines.append("")
        
        if analysis['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in analysis['recommendations']:
                lines.append(f"• {rec}")
        
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
    
    analyzer = PerformanceAnalyzer(project_root)
    analysis = analyzer.analyze_performance(lookback_days=30)
    
    report = analyzer.generate_performance_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "performance_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Performance analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







