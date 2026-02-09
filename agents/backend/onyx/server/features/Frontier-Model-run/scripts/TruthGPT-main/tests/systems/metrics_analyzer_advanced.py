"""
Advanced Metrics Analyzer
Advanced metrics analysis with comprehensive insights
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev, median

class AdvancedMetricsAnalyzer:
    """Advanced metrics analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_metrics(self, lookback_days: int = 30) -> Dict:
        """Analyze metrics comprehensively"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Comprehensive metrics analysis
        metrics_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'key_metrics': self._calculate_key_metrics(recent),
            'metrics_trends': self._analyze_metrics_trends(recent),
            'metrics_correlations': self._analyze_metrics_correlations(recent),
            'metrics_distribution': self._analyze_metrics_distribution(recent),
            'metrics_health': self._assess_metrics_health(recent),
            'anomalies': self._detect_anomalies(recent),
            'recommendations': []
        }
        
        # Generate recommendations
        metrics_analysis['recommendations'] = self._generate_metrics_recommendations(metrics_analysis)
        
        return metrics_analysis
    
    def _calculate_key_metrics(self, recent: List[Dict]) -> Dict:
        """Calculate key metrics"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        return {
            'success_rate': {
                'mean': round(mean(success_rates), 2) if success_rates else 0,
                'median': round(median(success_rates), 2) if success_rates else 0,
                'std': round(stdev(success_rates), 2) if len(success_rates) > 1 else 0,
                'min': round(min(success_rates), 2) if success_rates else 0,
                'max': round(max(success_rates), 2) if success_rates else 0
            },
            'execution_time': {
                'mean': round(mean(execution_times), 2) if execution_times else 0,
                'median': round(median(execution_times), 2) if execution_times else 0,
                'std': round(stdev(execution_times), 2) if len(execution_times) > 1 else 0,
                'min': round(min(execution_times), 2) if execution_times else 0,
                'max': round(max(execution_times), 2) if execution_times else 0
            },
            'total_tests': {
                'total': sum(total_tests),
                'mean': round(mean(total_tests), 2) if total_tests else 0,
                'median': round(median(total_tests), 2) if total_tests else 0,
                'std': round(stdev(total_tests), 2) if len(total_tests) > 1 else 0
            },
            'failures': {
                'total': sum(failures),
                'mean': round(mean(failures), 2) if failures else 0,
                'median': round(median(failures), 2) if failures else 0,
                'std': round(stdev(failures), 2) if len(failures) > 1 else 0,
                'max': max(failures) if failures else 0
            }
        }
    
    def _analyze_metrics_trends(self, recent: List[Dict]) -> Dict:
        """Analyze metrics trends"""
        if len(recent) < 4:
            return {}
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        # Split into halves
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]
        
        sr_first = mean([r.get('success_rate', 0) for r in first_half])
        sr_second = mean([r.get('success_rate', 0) for r in second_half])
        sr_trend = sr_second - sr_first
        
        et_first = mean([r.get('execution_time', 0) for r in first_half])
        et_second = mean([r.get('execution_time', 0) for r in second_half])
        et_trend = et_second - et_first
        
        f_first = mean([r.get('failures', 0) + r.get('errors', 0) for r in first_half])
        f_second = mean([r.get('failures', 0) + r.get('errors', 0) for r in second_half])
        f_trend = f_second - f_first
        
        return {
            'success_rate_trend': {
                'change': round(sr_trend, 2),
                'direction': 'improving' if sr_trend > 0 else 'declining' if sr_trend < 0 else 'stable',
                'percent_change': round((sr_trend / sr_first * 100) if sr_first > 0 else 0, 2)
            },
            'execution_time_trend': {
                'change': round(et_trend, 2),
                'direction': 'improving' if et_trend < 0 else 'degrading' if et_trend > 0 else 'stable',
                'percent_change': round((et_trend / et_first * 100) if et_first > 0 else 0, 2)
            },
            'failures_trend': {
                'change': round(f_trend, 2),
                'direction': 'decreasing' if f_trend < 0 else 'increasing' if f_trend > 0 else 'stable',
                'percent_change': round((f_trend / f_first * 100) if f_first > 0 else 0, 2)
            }
        }
    
    def _analyze_metrics_correlations(self, recent: List[Dict]) -> Dict:
        """Analyze metrics correlations"""
        if len(recent) < 3:
            return {}
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        # Simple correlation indicators
        correlations = {}
        
        # Success rate vs Execution time
        sr_avg = mean(success_rates)
        et_avg = mean(execution_times)
        high_sr_low_et = sum(1 for i in range(len(recent)) 
            if success_rates[i] > sr_avg and execution_times[i] < et_avg)
        correlations['success_vs_execution_time'] = {
            'positive_correlation': high_sr_low_et > len(recent) * 0.3,
            'description': 'High success with low execution time'
        }
        
        # Success rate vs Failures
        f_avg = mean(failures)
        high_sr_low_f = sum(1 for i in range(len(recent))
            if success_rates[i] > sr_avg and failures[i] < f_avg)
        correlations['success_vs_failures'] = {
            'negative_correlation': high_sr_low_f > len(recent) * 0.3,
            'description': 'High success with low failures'
        }
        
        return correlations
    
    def _analyze_metrics_distribution(self, recent: List[Dict]) -> Dict:
        """Analyze metrics distribution"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        if not success_rates or not execution_times:
            return {}
        
        # Categorize success rates
        excellent_sr = sum(1 for sr in success_rates if sr >= 95)
        good_sr = sum(1 for sr in success_rates if 90 <= sr < 95)
        acceptable_sr = sum(1 for sr in success_rates if 80 <= sr < 90)
        poor_sr = sum(1 for sr in success_rates if sr < 80)
        
        # Categorize execution times
        fast_et = sum(1 for et in execution_times if et < 60)
        moderate_et = sum(1 for et in execution_times if 60 <= et < 180)
        slow_et = sum(1 for et in execution_times if et >= 180)
        
        total = len(recent)
        
        return {
            'success_rate_distribution': {
                'excellent': {'count': excellent_sr, 'percentage': round((excellent_sr / total * 100) if total > 0 else 0, 1)},
                'good': {'count': good_sr, 'percentage': round((good_sr / total * 100) if total > 0 else 0, 1)},
                'acceptable': {'count': acceptable_sr, 'percentage': round((acceptable_sr / total * 100) if total > 0 else 0, 1)},
                'poor': {'count': poor_sr, 'percentage': round((poor_sr / total * 100) if total > 0 else 0, 1)}
            },
            'execution_time_distribution': {
                'fast': {'count': fast_et, 'percentage': round((fast_et / total * 100) if total > 0 else 0, 1)},
                'moderate': {'count': moderate_et, 'percentage': round((moderate_et / total * 100) if total > 0 else 0, 1)},
                'slow': {'count': slow_et, 'percentage': round((slow_et / total * 100) if total > 0 else 0, 1)}
            }
        }
    
    def _assess_metrics_health(self, recent: List[Dict]) -> Dict:
        """Assess metrics health"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        if not success_rates:
            return {}
        
        avg_success = mean(success_rates)
        avg_time = mean(execution_times) if execution_times else 0
        avg_failures = mean(failures) if failures else 0
        
        # Health score components
        success_score = min(100, avg_success)
        time_score = max(0, 100 - (avg_time / 5))
        failure_score = max(0, 100 - (avg_failures * 10))
        
        # Weighted health score
        health_score = (success_score * 0.5 + time_score * 0.3 + failure_score * 0.2)
        
        return {
            'health_score': round(health_score, 1),
            'success_score': round(success_score, 1),
            'time_score': round(time_score, 1),
            'failure_score': round(failure_score, 1),
            'health_status': 'healthy' if health_score >= 80 else 'warning' if health_score >= 60 else 'critical'
        }
    
    def _detect_anomalies(self, recent: List[Dict]) -> List[Dict]:
        """Detect anomalies"""
        anomalies = []
        
        if len(recent) < 3:
            return anomalies
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        sr_mean = mean(success_rates)
        sr_std = stdev(success_rates) if len(success_rates) > 1 else 0
        et_mean = mean(execution_times)
        et_std = stdev(execution_times) if len(execution_times) > 1 else 0
        
        for i, r in enumerate(recent):
            sr = r.get('success_rate', 0)
            et = r.get('execution_time', 0)
            
            # Detect anomalies (2 standard deviations)
            if sr_std > 0 and abs(sr - sr_mean) > 2 * sr_std:
                anomalies.append({
                    'run_index': i,
                    'timestamp': r.get('timestamp', ''),
                    'type': 'success_rate_anomaly',
                    'value': round(sr, 2),
                    'expected_range': f"{sr_mean - 2*sr_std:.1f} - {sr_mean + 2*sr_std:.1f}",
                    'severity': 'high' if abs(sr - sr_mean) > 3 * sr_std else 'medium'
                })
            
            if et_std > 0 and abs(et - et_mean) > 2 * et_std:
                anomalies.append({
                    'run_index': i,
                    'timestamp': r.get('timestamp', ''),
                    'type': 'execution_time_anomaly',
                    'value': round(et, 2),
                    'expected_range': f"{et_mean - 2*et_std:.1f} - {et_mean + 2*et_std:.1f}",
                    'severity': 'high' if abs(et - et_mean) > 3 * et_std else 'medium'
                })
        
        return anomalies[:10]  # Top 10
    
    def _generate_metrics_recommendations(self, analysis: Dict) -> List[str]:
        """Generate metrics recommendations"""
        recommendations = []
        
        key_metrics = analysis['key_metrics']
        sr_metrics = key_metrics.get('success_rate', {})
        if sr_metrics.get('mean', 0) < 90:
            recommendations.append(f"Increase success rate from {sr_metrics['mean']:.1f}% to 95%+")
        
        et_metrics = key_metrics.get('execution_time', {})
        if et_metrics.get('mean', 0) > 300:
            recommendations.append(f"Reduce execution time from {et_metrics['mean']:.0f}s to <120s")
        
        health = analysis.get('metrics_health', {})
        if health.get('health_status') != 'healthy':
            recommendations.append(f"Improve metrics health from {health['health_status']} to healthy (current score: {health['health_score']:.1f})")
        
        if analysis['anomalies']:
            high_severity = sum(1 for a in analysis['anomalies'] if a['severity'] == 'high')
            if high_severity > 0:
                recommendations.append(f"🚨 {high_severity} high-severity anomaly/anomalies detected - investigate")
        
        trends = analysis.get('metrics_trends', {})
        if trends.get('success_rate_trend', {}).get('direction') == 'declining':
            recommendations.append("Success rate is declining - investigate root causes")
        
        if not recommendations:
            recommendations.append("✅ Metrics are healthy - maintain current practices")
        
        return recommendations
    
    def generate_metrics_report(self, analysis: Dict) -> str:
        """Generate metrics report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED METRICS ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        lines.append("📊 KEY METRICS")
        lines.append("-" * 80)
        key_metrics = analysis['key_metrics']
        
        lines.append("Success Rate:")
        sr = key_metrics['success_rate']
        lines.append(f"  Mean: {sr['mean']}%")
        lines.append(f"  Median: {sr['median']}%")
        lines.append(f"  Range: {sr['min']}% - {sr['max']}%")
        lines.append("")
        
        lines.append("Execution Time:")
        et = key_metrics['execution_time']
        lines.append(f"  Mean: {et['mean']}s")
        lines.append(f"  Median: {et['median']}s")
        lines.append(f"  Range: {et['min']}s - {et['max']}s")
        lines.append("")
        
        lines.append("Total Tests:")
        tt = key_metrics['total_tests']
        lines.append(f"  Total: {tt['total']:,}")
        lines.append(f"  Mean: {tt['mean']:.1f}")
        lines.append("")
        
        lines.append("Failures:")
        f = key_metrics['failures']
        lines.append(f"  Total: {f['total']}")
        lines.append(f"  Mean: {f['mean']:.1f}")
        lines.append(f"  Max: {f['max']}")
        lines.append("")
        
        if analysis.get('metrics_trends'):
            trends = analysis['metrics_trends']
            lines.append("📈 METRICS TRENDS")
            lines.append("-" * 80)
            trend_emoji = {'improving': '📈', 'declining': '📉', 'degrading': '📉', 'increasing': '📈', 'decreasing': '📉', 'stable': '➡️'}
            
            sr_trend = trends['success_rate_trend']
            emoji = trend_emoji.get(sr_trend['direction'], '➡️')
            lines.append(f"{emoji} Success Rate: {sr_trend['direction'].title()} ({sr_trend['change']:+.2f}%, {sr_trend['percent_change']:+.2f}%)")
            
            et_trend = trends['execution_time_trend']
            emoji = trend_emoji.get(et_trend['direction'], '➡️')
            lines.append(f"{emoji} Execution Time: {et_trend['direction'].title()} ({et_trend['change']:+.2f}s, {et_trend['percent_change']:+.2f}%)")
            
            f_trend = trends['failures_trend']
            emoji = trend_emoji.get(f_trend['direction'], '➡️')
            lines.append(f"{emoji} Failures: {f_trend['direction'].title()} ({f_trend['change']:+.2f}, {f_trend['percent_change']:+.2f}%)")
            lines.append("")
        
        if analysis.get('metrics_health'):
            health = analysis['metrics_health']
            status_emoji = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴'}
            emoji = status_emoji.get(health['health_status'], '⚪')
            lines.append(f"{emoji} METRICS HEALTH")
            lines.append("-" * 80)
            lines.append(f"Health Status: {health['health_status'].upper()}")
            lines.append(f"Health Score: {health['health_score']}/100")
            lines.append(f"Success Score: {health['success_score']}/100")
            lines.append(f"Time Score: {health['time_score']}/100")
            lines.append(f"Failure Score: {health['failure_score']}/100")
            lines.append("")
        
        if analysis['anomalies']:
            lines.append("🚨 ANOMALIES")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡'}
            for anomaly in analysis['anomalies'][:5]:  # Top 5
                emoji = severity_emoji.get(anomaly['severity'], '⚪')
                lines.append(f"{emoji} Run #{anomaly['run_index']} - {anomaly['type'].replace('_', ' ').title()}")
                lines.append(f"   Value: {anomaly['value']}")
                lines.append(f"   Expected Range: {anomaly['expected_range']}")
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
    
    analyzer = AdvancedMetricsAnalyzer(project_root)
    analysis = analyzer.analyze_metrics(lookback_days=30)
    
    report = analyzer.generate_metrics_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_metrics_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced metrics analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






