"""
Enhanced Performance Analyzer
Enhanced performance analysis with comprehensive metrics
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, median, stdev, quantiles

class EnhancedPerformanceAnalyzer:
    """Enhanced performance analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_performance(self, lookback_days: int = 30) -> Dict:
        """Analyze performance comprehensively"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Extract performance metrics
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        # Comprehensive performance analysis
        performance_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'execution_metrics': self._calculate_execution_metrics(execution_times),
            'throughput_metrics': self._calculate_throughput_metrics(total_tests, execution_times),
            'performance_distribution': self._analyze_performance_distribution(execution_times),
            'performance_trends': self._analyze_performance_trends(recent),
            'performance_bottlenecks': self._identify_bottlenecks(execution_times, total_tests, success_rates),
            'performance_quality': self._assess_performance_quality(recent),
            'recommendations': []
        }
        
        # Calculate overall performance score
        performance_analysis['overall_performance_score'] = self._calculate_overall_performance_score(performance_analysis)
        
        # Generate recommendations
        performance_analysis['recommendations'] = self._generate_performance_recommendations(performance_analysis)
        
        return performance_analysis
    
    def _calculate_execution_metrics(self, execution_times: List[float]) -> Dict:
        """Calculate execution metrics"""
        if not execution_times:
            return {}
        
        return {
            'mean': round(mean(execution_times), 2),
            'median': round(median(execution_times), 2),
            'std_deviation': round(stdev(execution_times), 2) if len(execution_times) > 1 else 0,
            'min': round(min(execution_times), 2),
            'max': round(max(execution_times), 2),
            'range': round(max(execution_times) - min(execution_times), 2),
            'coefficient_of_variation': round((stdev(execution_times) / mean(execution_times) * 100) if mean(execution_times) > 0 else 0, 2),
            'total_time': round(sum(execution_times), 2)
        }
    
    def _calculate_throughput_metrics(self, total_tests: List[int], execution_times: List[float]) -> Dict:
        """Calculate throughput metrics"""
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
            'min_tests_per_second': round(min(throughputs), 2),
            'throughput_std': round(stdev(throughputs), 2) if len(throughputs) > 1 else 0,
            'total_tests': sum(total_tests),
            'total_time': sum(execution_times),
            'overall_throughput': round(sum(total_tests) / sum(execution_times) if sum(execution_times) > 0 else 0, 2)
        }
    
    def _analyze_performance_distribution(self, execution_times: List[float]) -> Dict:
        """Analyze performance distribution"""
        if len(execution_times) < 4:
            return {}
        
        try:
            q = quantiles(execution_times, n=4)
            return {
                'q1': round(q[0], 2),
                'q2_median': round(q[1], 2),
                'q3': round(q[2], 2),
                'iqr': round(q[2] - q[0], 2),
                'lower_fence': round(q[0] - 1.5 * (q[2] - q[0]), 2),
                'upper_fence': round(q[2] + 1.5 * (q[2] - q[0]), 2),
                'outliers': len([t for t in execution_times if t < q[0] - 1.5 * (q[2] - q[0]) or t > q[2] + 1.5 * (q[2] - q[0])])
            }
        except:
            return {}
    
    def _analyze_performance_trends(self, recent: List[Dict]) -> Dict:
        """Analyze performance trends"""
        if len(recent) < 4:
            return {}
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        # Split into thirds
        third_size = len(execution_times) // 3
        first_third = execution_times[:third_size] if third_size > 0 else execution_times[:1]
        middle_third = execution_times[third_size:2*third_size] if third_size > 0 else execution_times[1:2]
        last_third = execution_times[2*third_size:] if third_size > 0 else execution_times[2:]
        
        first_avg = mean(first_third) if first_third else 0
        middle_avg = mean(middle_third) if middle_third else 0
        last_avg = mean(last_third) if last_third else 0
        
        trend = last_avg - first_avg
        percent_change = (trend / first_avg * 100) if first_avg > 0 else 0
        
        return {
            'trend': round(trend, 2),
            'percent_change': round(percent_change, 2),
            'direction': 'improving' if trend < -5 else 'degrading' if trend > 5 else 'stable',
            'first_third_avg': round(first_avg, 2),
            'middle_third_avg': round(middle_avg, 2),
            'last_third_avg': round(last_avg, 2),
            'acceleration': round((last_avg - middle_avg) - (middle_avg - first_avg), 2)
        }
    
    def _identify_bottlenecks(self, execution_times: List[float], total_tests: List[int], success_rates: List[float]) -> List[Dict]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if not execution_times or not total_tests:
            return bottlenecks
        
        avg_time = mean(execution_times)
        avg_tests = mean(total_tests)
        avg_success = mean(success_rates) if success_rates else 100
        
        # Find slow runs with low success
        for i, (time, tests, success) in enumerate(zip(execution_times, total_tests, success_rates if success_rates else [100]*len(execution_times))):
            if time > avg_time * 1.5 and (tests < avg_tests * 0.8 or success < avg_success * 0.9):
                bottlenecks.append({
                    'run_index': i,
                    'execution_time': round(time, 2),
                    'tests_count': tests,
                    'success_rate': round(success, 2),
                    'time_per_test': round(time / tests, 3) if tests > 0 else 0,
                    'severity': 'high' if time > avg_time * 2 else 'medium',
                    'type': 'slow_with_low_success' if success < avg_success * 0.9 else 'slow_with_few_tests'
                })
        
        return sorted(bottlenecks, key=lambda x: x['execution_time'], reverse=True)[:10]  # Top 10
    
    def _assess_performance_quality(self, recent: List[Dict]) -> Dict:
        """Assess performance quality"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if not execution_times:
            return {}
        
        avg_time = mean(execution_times)
        time_std = stdev(execution_times) if len(execution_times) > 1 else 0
        
        # Performance score (lower time = higher score, but penalize variance)
        time_score = max(0, 100 - (avg_time / 5))  # Normalize
        consistency_score = max(0, 100 - (time_std * 2))
        performance_score = (time_score * 0.7 + consistency_score * 0.3)
        
        # Success-weighted performance
        if success_rates:
            avg_success = mean(success_rates)
            weighted_score = performance_score * (avg_success / 100)
        else:
            weighted_score = performance_score
        
        return {
            'performance_score': round(performance_score, 1),
            'consistency_score': round(consistency_score, 1),
            'weighted_performance_score': round(weighted_score, 1),
            'avg_execution_time': round(avg_time, 2),
            'execution_time_std': round(time_std, 2),
            'quality_level': 'excellent' if performance_score >= 90 else 'good' if performance_score >= 80 else 'acceptable' if performance_score >= 70 else 'poor'
        }
    
    def _calculate_overall_performance_score(self, analysis: Dict) -> float:
        """Calculate overall performance score"""
        quality = analysis.get('performance_quality', {})
        throughput = analysis.get('throughput_metrics', {})
        
        scores = []
        
        if quality.get('weighted_performance_score'):
            scores.append(quality['weighted_performance_score'])
        
        if throughput.get('mean_tests_per_second', 0) > 0:
            # Normalize throughput score
            throughput_score = min(100, throughput['mean_tests_per_second'] * 10)
            scores.append(throughput_score)
        
        if not scores:
            return 0.0
        
        return round(mean(scores), 1)
    
    def _generate_performance_recommendations(self, analysis: Dict) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        exec_metrics = analysis['execution_metrics']
        if exec_metrics.get('mean', 0) > 300:
            recommendations.append(f"Reduce average execution time from {exec_metrics['mean']:.0f}s to <120s")
        
        if exec_metrics.get('coefficient_of_variation', 0) > 30:
            recommendations.append(f"High execution time variance ({exec_metrics['coefficient_of_variation']:.1f}%) - improve consistency")
        
        throughput = analysis.get('throughput_metrics', {})
        if throughput.get('mean_tests_per_second', 0) < 5:
            recommendations.append(f"Increase throughput from {throughput['mean_tests_per_second']:.1f} to 10+ tests/second")
        
        trends = analysis.get('performance_trends', {})
        if trends.get('direction') == 'degrading':
            recommendations.append(f"Performance is degrading ({trends['trend']:+.2f}s) - investigate recent changes")
        
        if analysis['performance_bottlenecks']:
            high_severity = sum(1 for b in analysis['performance_bottlenecks'] if b['severity'] == 'high')
            if high_severity > 0:
                recommendations.append(f"🚨 {high_severity} high-severity bottleneck(s) - optimize slow runs")
        
        quality = analysis.get('performance_quality', {})
        if quality.get('quality_level') in ['acceptable', 'poor']:
            recommendations.append(f"Improve performance quality from {quality['quality_level']} to good+")
        
        if not recommendations:
            recommendations.append("✅ Performance is optimal - maintain current practices")
        
        return recommendations
    
    def generate_performance_report(self, analysis: Dict) -> str:
        """Generate performance report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED PERFORMANCE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        score_emoji = "🟢" if analysis['overall_performance_score'] >= 80 else "🟡" if analysis['overall_performance_score'] >= 60 else "🔴"
        lines.append(f"{score_emoji} Overall Performance Score: {analysis['overall_performance_score']}/100")
        lines.append("")
        
        lines.append("⏱️ EXECUTION METRICS")
        lines.append("-" * 80)
        exec_metrics = analysis['execution_metrics']
        lines.append(f"Mean: {exec_metrics['mean']}s")
        lines.append(f"Median: {exec_metrics['median']}s")
        lines.append(f"Std Deviation: {exec_metrics['std_deviation']}s")
        lines.append(f"Range: {exec_metrics['min']}s - {exec_metrics['max']}s")
        lines.append(f"Total Range: {exec_metrics['range']}s")
        lines.append(f"Coefficient of Variation: {exec_metrics['coefficient_of_variation']}%")
        lines.append(f"Total Time: {exec_metrics['total_time']}s")
        lines.append("")
        
        if analysis['throughput_metrics']:
            lines.append("⚡ THROUGHPUT METRICS")
            lines.append("-" * 80)
            throughput = analysis['throughput_metrics']
            lines.append(f"Mean Tests/Second: {throughput['mean_tests_per_second']}")
            lines.append(f"Median Tests/Second: {throughput['median_tests_per_second']}")
            lines.append(f"Max Tests/Second: {throughput['max_tests_per_second']}")
            lines.append(f"Min Tests/Second: {throughput['min_tests_per_second']}")
            lines.append(f"Overall Throughput: {throughput['overall_throughput']} tests/second")
            lines.append(f"Total Tests: {throughput['total_tests']:,}")
            lines.append("")
        
        if analysis['performance_distribution']:
            lines.append("📊 PERFORMANCE DISTRIBUTION")
            lines.append("-" * 80)
            dist = analysis['performance_distribution']
            lines.append(f"Q1: {dist['q1']}s")
            lines.append(f"Q2 (Median): {dist['q2_median']}s")
            lines.append(f"Q3: {dist['q3']}s")
            lines.append(f"IQR: {dist['iqr']}s")
            lines.append(f"Outliers: {dist['outliers']}")
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
            lines.append(f"Acceleration: {trends['acceleration']:+.2f}s")
            lines.append("")
        
        if analysis['performance_bottlenecks']:
            lines.append("🔴 PERFORMANCE BOTTLENECKS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡'}
            for bottleneck in analysis['performance_bottlenecks'][:5]:  # Top 5
                emoji = severity_emoji.get(bottleneck['severity'], '⚪')
                lines.append(f"{emoji} Run #{bottleneck['run_index']} - {bottleneck['type'].replace('_', ' ').title()}")
                lines.append(f"   Execution Time: {bottleneck['execution_time']}s")
                lines.append(f"   Tests: {bottleneck['tests_count']}")
                lines.append(f"   Success Rate: {bottleneck['success_rate']}%")
                lines.append(f"   Time per Test: {bottleneck['time_per_test']}s")
            lines.append("")
        
        if analysis.get('performance_quality'):
            quality = analysis['performance_quality']
            lines.append("✅ PERFORMANCE QUALITY")
            lines.append("-" * 80)
            lines.append(f"Performance Score: {quality['performance_score']}/100")
            lines.append(f"Consistency Score: {quality['consistency_score']}/100")
            lines.append(f"Weighted Performance Score: {quality['weighted_performance_score']}/100")
            lines.append(f"Quality Level: {quality['quality_level'].upper()}")
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
    
    analyzer = EnhancedPerformanceAnalyzer(project_root)
    analysis = analyzer.analyze_performance(lookback_days=30)
    
    report = analyzer.generate_performance_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "enhanced_performance_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Enhanced performance analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






