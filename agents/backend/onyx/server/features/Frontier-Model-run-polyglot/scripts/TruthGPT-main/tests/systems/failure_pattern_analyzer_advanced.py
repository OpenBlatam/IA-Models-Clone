"""
Advanced Failure Pattern Analyzer
Advanced analysis of failure patterns
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from collections import Counter
from statistics import mean, stdev

class AdvancedFailurePatternAnalyzer:
    """Advanced failure pattern analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_failure_patterns(self, lookback_days: int = 30) -> Dict:
        """Analyze failure patterns"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Extract failure data
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        # Analyze patterns
        pattern_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'failure_statistics': self._calculate_failure_statistics(failures, success_rates),
            'failure_distribution': self._analyze_failure_distribution(failures),
            'failure_clusters': self._identify_failure_clusters(recent),
            'failure_timing': self._analyze_failure_timing(recent),
            'failure_correlations': self._analyze_failure_correlations(recent),
            'patterns': self._identify_patterns(failures, recent),
            'recommendations': []
        }
        
        # Generate recommendations
        pattern_analysis['recommendations'] = self._generate_pattern_recommendations(pattern_analysis)
        
        return pattern_analysis
    
    def _calculate_failure_statistics(self, failures: List[int], success_rates: List[float]) -> Dict:
        """Calculate failure statistics"""
        if not failures:
            return {}
        
        total_failures = sum(failures)
        avg_failures = mean(failures)
        max_failures = max(failures)
        min_failures = min(failures)
        failure_std = stdev(failures) if len(failures) > 1 else 0
        
        avg_success = mean(success_rates) if success_rates else 0
        
        return {
            'total_failures': total_failures,
            'avg_failures_per_run': round(avg_failures, 2),
            'max_failures': max_failures,
            'min_failures': min_failures,
            'std_deviation': round(failure_std, 2),
            'coefficient_of_variation': round((failure_std / avg_failures * 100) if avg_failures > 0 else 0, 2),
            'avg_success_rate': round(avg_success, 1)
        }
    
    def _analyze_failure_distribution(self, failures: List[int]) -> Dict:
        """Analyze failure distribution"""
        if not failures:
            return {}
        
        failure_counts = Counter(failures)
        
        # Categorize
        zero_failures = failure_counts.get(0, 0)
        low_failures = sum(count for fail, count in failure_counts.items() if 1 <= fail <= 5)
        medium_failures = sum(count for fail, count in failure_counts.items() if 6 <= fail <= 10)
        high_failures = sum(count for fail, count in failure_counts.items() if fail > 10)
        
        total = len(failures)
        
        return {
            'zero_failures': {
                'count': zero_failures,
                'percentage': round((zero_failures / total * 100) if total > 0 else 0, 1)
            },
            'low_failures': {
                'count': low_failures,
                'percentage': round((low_failures / total * 100) if total > 0 else 0, 1)
            },
            'medium_failures': {
                'count': medium_failures,
                'percentage': round((medium_failures / total * 100) if total > 0 else 0, 1)
            },
            'high_failures': {
                'count': high_failures,
                'percentage': round((high_failures / total * 100) if total > 0 else 0, 1)
            },
            'distribution': dict(failure_counts)
        }
    
    def _identify_failure_clusters(self, recent: List[Dict]) -> List[Dict]:
        """Identify failure clusters"""
        clusters = []
        
        if len(recent) < 3:
            return clusters
        
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        avg_failures = mean(failures)
        
        # Find consecutive high-failure runs
        cluster_start = None
        cluster_failures = []
        
        for i, fail_count in enumerate(failures):
            if fail_count > avg_failures * 1.5:
                if cluster_start is None:
                    cluster_start = i
                cluster_failures.append(fail_count)
            else:
                if cluster_start is not None and len(cluster_failures) >= 2:
                    clusters.append({
                        'start_index': cluster_start,
                        'end_index': i - 1,
                        'duration': len(cluster_failures),
                        'avg_failures': round(mean(cluster_failures), 2),
                        'max_failures': max(cluster_failures),
                        'severity': 'high' if mean(cluster_failures) > avg_failures * 2 else 'medium'
                    })
                cluster_start = None
                cluster_failures = []
        
        # Check if cluster extends to end
        if cluster_start is not None and len(cluster_failures) >= 2:
            clusters.append({
                'start_index': cluster_start,
                'end_index': len(failures) - 1,
                'duration': len(cluster_failures),
                'avg_failures': round(mean(cluster_failures), 2),
                'max_failures': max(cluster_failures),
                'severity': 'high' if mean(cluster_failures) > avg_failures * 2 else 'medium'
            })
        
        return clusters
    
    def _analyze_failure_timing(self, recent: List[Dict]) -> Dict:
        """Analyze failure timing patterns"""
        if not recent:
            return {}
        
        # Group by day of week (if timestamps available)
        failures_by_day = {}
        failures_by_hour = {}
        
        for r in recent:
            timestamp = r.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00').split('.')[0])
                    day = dt.strftime('%A')
                    hour = dt.hour
                    
                    failures = r.get('failures', 0) + r.get('errors', 0)
                    
                    if day not in failures_by_day:
                        failures_by_day[day] = []
                    failures_by_day[day].append(failures)
                    
                    if hour not in failures_by_hour:
                        failures_by_hour[hour] = []
                    failures_by_hour[hour].append(failures)
                except:
                    pass
        
        return {
            'by_day': {day: round(mean(fails), 2) for day, fails in failures_by_day.items()} if failures_by_day else {},
            'by_hour': {hour: round(mean(fails), 2) for hour, fails in failures_by_hour.items()} if failures_by_hour else {}
        }
    
    def _analyze_failure_correlations(self, recent: List[Dict]) -> Dict:
        """Analyze failure correlations"""
        if len(recent) < 3:
            return {}
        
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        # Calculate correlations
        correlations = {}
        
        if len(failures) == len(execution_times):
            # Simple correlation: failures vs execution time
            if len(failures) > 1:
                avg_failures = mean(failures)
                avg_time = mean(execution_times)
                
                high_fail_runs = sum(1 for f in failures if f > avg_failures)
                high_time_runs = sum(1 for t in execution_times if t > avg_time)
                
                correlations['failures_vs_execution_time'] = {
                    'high_failure_high_time': sum(1 for i in range(len(failures)) 
                        if failures[i] > avg_failures and execution_times[i] > avg_time),
                    'description': 'Runs with both high failures and long execution times'
                }
        
        return correlations
    
    def _identify_patterns(self, failures: List[int], recent: List[Dict]) -> List[Dict]:
        """Identify specific patterns"""
        patterns = []
        
        if len(failures) < 3:
            return patterns
        
        # Increasing pattern
        if len(failures) >= 3:
            recent_3 = failures[-3:]
            if recent_3[0] < recent_3[1] < recent_3[2]:
                patterns.append({
                    'pattern': 'increasing_failures',
                    'description': 'Failures increasing over last 3 runs',
                    'severity': 'high',
                    'trend': 'worsening'
                })
        
        # Cyclic pattern (check for repeating pattern)
        if len(failures) >= 6:
            # Simple check: alternating high/low
            recent_6 = failures[-6:]
            if all(recent_6[i] > recent_6[i+1] for i in range(0, 5, 2)):
                patterns.append({
                    'pattern': 'cyclic_high_low',
                    'description': 'Cyclic pattern of high and low failures',
                    'severity': 'medium',
                    'trend': 'unstable'
                })
        
        # Spike pattern
        if len(failures) >= 3:
            recent_3 = failures[-3:]
            avg_prev = mean(failures[:-3]) if len(failures) > 3 else mean(failures)
            if recent_3[-1] > avg_prev * 2:
                patterns.append({
                    'pattern': 'failure_spike',
                    'description': f'Sudden spike: {recent_3[-1]} failures (avg: {avg_prev:.1f})',
                    'severity': 'high',
                    'trend': 'critical'
                })
        
        return patterns
    
    def _generate_pattern_recommendations(self, analysis: Dict) -> List[str]:
        """Generate pattern-based recommendations"""
        recommendations = []
        
        stats = analysis['failure_statistics']
        if stats.get('avg_failures_per_run', 0) > 5:
            recommendations.append(f"Reduce average failures from {stats['avg_failures_per_run']:.1f} to <3 per run")
        
        if analysis['failure_clusters']:
            critical_clusters = sum(1 for c in analysis['failure_clusters'] if c['severity'] == 'high')
            if critical_clusters > 0:
                recommendations.append(f"🚨 {critical_clusters} high-severity failure cluster(s) detected - investigate root causes")
        
        patterns = analysis.get('patterns', [])
        for pattern in patterns:
            if pattern['severity'] == 'high':
                recommendations.append(f"⚠️ {pattern['description']} - immediate attention required")
        
        dist = analysis['failure_distribution']
        if dist.get('high_failures', {}).get('percentage', 0) > 20:
            recommendations.append(f"High failure rate in {dist['high_failures']['percentage']:.1f}% of runs - improve test stability")
        
        if not recommendations:
            recommendations.append("✅ Failure patterns are within acceptable limits - maintain current practices")
        
        return recommendations
    
    def generate_pattern_report(self, analysis: Dict) -> str:
        """Generate pattern report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED FAILURE PATTERN ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        lines.append("📊 FAILURE STATISTICS")
        lines.append("-" * 80)
        stats = analysis['failure_statistics']
        lines.append(f"Total Failures: {stats['total_failures']}")
        lines.append(f"Average per Run: {stats['avg_failures_per_run']}")
        lines.append(f"Range: {stats['min_failures']} - {stats['max_failures']}")
        lines.append(f"Std Deviation: {stats['std_deviation']}")
        lines.append(f"Coefficient of Variation: {stats['coefficient_of_variation']}%")
        lines.append(f"Average Success Rate: {stats['avg_success_rate']}%")
        lines.append("")
        
        lines.append("📈 FAILURE DISTRIBUTION")
        lines.append("-" * 80)
        dist = analysis['failure_distribution']
        lines.append(f"Zero Failures: {dist['zero_failures']['count']} ({dist['zero_failures']['percentage']}%)")
        lines.append(f"Low (1-5): {dist['low_failures']['count']} ({dist['low_failures']['percentage']}%)")
        lines.append(f"Medium (6-10): {dist['medium_failures']['count']} ({dist['medium_failures']['percentage']}%)")
        lines.append(f"High (>10): {dist['high_failures']['count']} ({dist['high_failures']['percentage']}%)")
        lines.append("")
        
        if analysis['failure_clusters']:
            lines.append("🔴 FAILURE CLUSTERS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡'}
            for cluster in analysis['failure_clusters']:
                emoji = severity_emoji.get(cluster['severity'], '⚪')
                lines.append(f"{emoji} Cluster: Runs {cluster['start_index']}-{cluster['end_index']}")
                lines.append(f"   Duration: {cluster['duration']} runs")
                lines.append(f"   Avg Failures: {cluster['avg_failures']}")
                lines.append(f"   Max Failures: {cluster['max_failures']}")
            lines.append("")
        
        if analysis.get('patterns'):
            lines.append("🔍 IDENTIFIED PATTERNS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for pattern in analysis['patterns']:
                emoji = severity_emoji.get(pattern['severity'], '⚪')
                lines.append(f"{emoji} {pattern['pattern'].replace('_', ' ').title()}")
                lines.append(f"   {pattern['description']}")
                lines.append(f"   Trend: {pattern['trend']}")
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
    
    analyzer = AdvancedFailurePatternAnalyzer(project_root)
    analysis = analyzer.analyze_failure_patterns(lookback_days=30)
    
    report = analyzer.generate_pattern_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_failure_pattern_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced failure pattern analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






