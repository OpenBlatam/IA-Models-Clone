"""
Advanced Performance Analyzer
Advanced performance analysis with detailed metrics
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, median, stdev, quantiles

class AdvancedPerformanceAnalyzer:
    """Advanced performance analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_performance(self, lookback_days: int = 30) -> Dict:
        """Analyze performance in detail"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Extract performance metrics
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        # Calculate detailed statistics
        performance_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'execution_time_stats': self._calculate_execution_stats(execution_times),
            'throughput_stats': self._calculate_throughput_stats(total_tests, execution_times),
            'performance_distribution': self._calculate_performance_distribution(execution_times),
            'performance_trends': self._analyze_performance_trends(recent),
            'bottlenecks': self._identify_bottlenecks(execution_times, total_tests),
            'recommendations': []
        }
        
        # Generate recommendations
        performance_analysis['recommendations'] = self._generate_performance_recommendations(performance_analysis)
        
        return performance_analysis
    
    def _calculate_execution_stats(self, execution_times: List[float]) -> Dict:
        """Calculate execution time statistics"""
        if not execution_times:
            return {}
        
        return {
            'mean': round(mean(execution_times), 2),
            'median': round(median(execution_times), 2),
            'std': round(stdev(execution_times), 2) if len(execution_times) > 1 else 0,
            'min': round(min(execution_times), 2),
            'max': round(max(execution_times), 2),
            'range': round(max(execution_times) - min(execution_times), 2),
            'coefficient_of_variation': round((stdev(execution_times) / mean(execution_times) * 100) if mean(execution_times) > 0 else 0, 2)
        }
    
    def _calculate_throughput_stats(self, total_tests: List[int], execution_times: List[float]) -> Dict:
        """Calculate throughput statistics"""
        if not total_tests or not execution_times:
            return {}
        
        throughputs = []
        for tests, time in zip(total_tests, execution_times):
            if time > 0:
                throughputs.append(tests / time)
        
        if not throughputs:
            return {}
        
        return {
            'mean_tests_per_second': round(mean(throughputs), 2),
            'median_tests_per_second': round(median(throughputs), 2),
            'max_tests_per_second': round(max(throughputs), 2),
            'min_tests_per_second': round(min(throughputs), 2)
        }
    
    def _calculate_performance_distribution(self, execution_times: List[float]) -> Dict:
        """Calculate performance distribution"""
        if len(execution_times) < 4:
            return {}
        
        try:
            q = quantiles(execution_times, n=4)
            return {
                'q1': round(q[0], 2),
                'q2': round(q[1], 2),
                'q3': round(q[2], 2),
                'iqr': round(q[2] - q[0], 2)
            }
        except:
            return {}
    
    def _analyze_performance_trends(self, recent: List[Dict]) -> Dict:
        """Analyze performance trends"""
        if len(recent) < 4:
            return {}
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        # Split into halves
        first_half = execution_times[:len(execution_times)//2]
        second_half = execution_times[len(execution_times)//2:]
        
        first_avg = mean(first_half)
        second_avg = mean(second_half)
        
        trend = second_avg - first_avg
        percent_change = (trend / first_avg * 100) if first_avg > 0 else 0
        
        return {
            'trend': round(trend, 2),
            'percent_change': round(percent_change, 2),
            'direction': 'improving' if trend < 0 else 'degrading' if trend > 0 else 'stable',
            'first_half_avg': round(first_avg, 2),
            'second_half_avg': round(second_avg, 2)
        }
    
    def _identify_bottlenecks(self, execution_times: List[float], total_tests: List[int]) -> List[Dict]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if not execution_times or not total_tests:
            return bottlenecks
        
        # Find slow runs
        avg_time = mean(execution_times)
        slow_runs = [(i, t) for i, t in enumerate(execution_times) if t > avg_time * 1.5]
        
        for idx, time in slow_runs[:5]:  # Top 5
            tests = total_tests[idx] if idx < len(total_tests) else 0
            bottlenecks.append({
                'run_index': idx,
                'execution_time': round(time, 2),
                'tests_count': tests,
                'time_per_test': round(time / tests, 3) if tests > 0 else 0,
                'severity': 'high' if time > avg_time * 2 else 'medium'
            })
        
        return bottlenecks
    
    def _generate_performance_recommendations(self, analysis: Dict) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        stats = analysis['execution_time_stats']
        if stats.get('mean', 0) > 300:
            recommendations.append(f"Average execution time is {stats['mean']}s - optimize to <120s")
        
        if stats.get('coefficient_of_variation', 0) > 30:
            recommendations.append("High execution time variance - improve consistency")
        
        trends = analysis.get('performance_trends', {})
        if trends.get('direction') == 'degrading':
            recommendations.append("Performance is degrading - investigate recent changes")
        
        if analysis['bottlenecks']:
            recommendations.append(f"Found {len(analysis['bottlenecks'])} performance bottlenecks - optimize slow runs")
        
        if not recommendations:
            recommendations.append("Performance is optimal - maintain current practices")
        
        return recommendations
    
    def generate_performance_report(self, analysis: Dict) -> str:
        """Generate performance report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED PERFORMANCE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        lines.append("⏱️ EXECUTION TIME STATISTICS")
        lines.append("-" * 80)
        stats = analysis['execution_time_stats']
        lines.append(f"Mean: {stats['mean']}s")
        lines.append(f"Median: {stats['median']}s")
        lines.append(f"Std Dev: {stats['std']}s")
        lines.append(f"Min: {stats['min']}s")
        lines.append(f"Max: {stats['max']}s")
        lines.append(f"Range: {stats['range']}s")
        lines.append(f"Coefficient of Variation: {stats['coefficient_of_variation']}%")
        lines.append("")
        
        if analysis['throughput_stats']:
            lines.append("⚡ THROUGHPUT STATISTICS")
            lines.append("-" * 80)
            throughput = analysis['throughput_stats']
            lines.append(f"Mean Tests/Second: {throughput['mean_tests_per_second']}")
            lines.append(f"Median Tests/Second: {throughput['median_tests_per_second']}")
            lines.append(f"Max Tests/Second: {throughput['max_tests_per_second']}")
            lines.append(f"Min Tests/Second: {throughput['min_tests_per_second']}")
            lines.append("")
        
        if analysis['performance_distribution']:
            lines.append("📊 PERFORMANCE DISTRIBUTION")
            lines.append("-" * 80)
            dist = analysis['performance_distribution']
            lines.append(f"Q1: {dist['q1']}s")
            lines.append(f"Q2 (Median): {dist['q2']}s")
            lines.append(f"Q3: {dist['q3']}s")
            lines.append(f"IQR: {dist['iqr']}s")
            lines.append("")
        
        if analysis.get('performance_trends'):
            trends = analysis['performance_trends']
            trend_emoji = {'improving': '📈', 'degrading': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(trends['direction'], '➡️')
            lines.append(f"{emoji} PERFORMANCE TRENDS")
            lines.append("-" * 80)
            lines.append(f"Direction: {trends['direction'].title()}")
            lines.append(f"Trend: {trends['trend']:+.2f}s")
            lines.append(f"Percent Change: {trends['percent_change']:+.2f}%")
            lines.append(f"First Half Avg: {trends['first_half_avg']}s")
            lines.append(f"Second Half Avg: {trends['second_half_avg']}s")
            lines.append("")
        
        if analysis['bottlenecks']:
            lines.append("🔴 PERFORMANCE BOTTLENECKS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for bottleneck in analysis['bottlenecks']:
                emoji = severity_emoji.get(bottleneck['severity'], '⚪')
                lines.append(f"{emoji} Run #{bottleneck['run_index']}")
                lines.append(f"   Execution Time: {bottleneck['execution_time']}s")
                lines.append(f"   Tests: {bottleneck['tests_count']}")
                lines.append(f"   Time per Test: {bottleneck['time_per_test']}s")
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
    from statistics import quantiles
    
    project_root = Path(__file__).parent.parent
    
    analyzer = AdvancedPerformanceAnalyzer(project_root)
    analysis = analyzer.analyze_performance(lookback_days=30)
    
    report = analyzer.generate_performance_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_performance_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced performance analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







