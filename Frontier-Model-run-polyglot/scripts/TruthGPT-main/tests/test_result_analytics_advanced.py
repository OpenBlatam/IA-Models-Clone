"""
Advanced Test Result Analytics
Deep analytics and insights from test results
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import math


class AdvancedTestResultAnalytics:
    """Advanced analytics for test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def analyze_test_health(
        self,
        results: List[Dict],
        days: int = 30
    ) -> Dict:
        """Analyze overall test health"""
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_results = [
            r for r in results
            if datetime.fromisoformat(r.get('timestamp', '2000-01-01')) > cutoff
        ]
        
        if not recent_results:
            return {'error': 'No recent results'}
        
        # Calculate metrics
        total_runs = len(recent_results)
        total_tests = sum(r.get('summary', {}).get('total_tests', 0) for r in recent_results)
        
        success_rates = [
            r.get('summary', {}).get('success_rate', 0)
            for r in recent_results
        ]
        
        avg_success_rate = statistics.mean(success_rates) if success_rates else 0
        
        # Calculate stability
        if len(success_rates) > 1:
            stability = 1 - (statistics.stdev(success_rates) / 100) if statistics.mean(success_rates) > 0 else 0
        else:
            stability = 1.0
        
        # Health score (0-100)
        health_score = (avg_success_rate * 0.7 + stability * 100 * 0.3)
        
        return {
            'period_days': days,
            'total_runs': total_runs,
            'total_tests': total_tests,
            'average_success_rate': round(avg_success_rate, 2),
            'stability': round(stability, 3),
            'health_score': round(health_score, 2),
            'health_level': self._get_health_level(health_score)
        }
    
    def _get_health_level(self, score: float) -> str:
        """Get health level from score"""
        if score >= 90:
            return 'excellent'
        elif score >= 75:
            return 'good'
        elif score >= 60:
            return 'fair'
        elif score >= 40:
            return 'poor'
        else:
            return 'critical'
    
    def analyze_failure_patterns(
        self,
        results: List[Dict]
    ) -> Dict:
        """Analyze failure patterns"""
        failure_patterns = defaultdict(lambda: {
            'count': 0,
            'tests': set(),
            'error_types': defaultdict(int)
        })
        
        for result in results:
            test_details = result.get('test_details', {})
            
            for test_name, test_data in test_details.items():
                if test_data.get('status') in ('failed', 'error'):
                    error_msg = test_data.get('error_message', '')
                    
                    # Extract error pattern
                    pattern = self._extract_error_pattern(error_msg)
                    
                    failure_patterns[pattern]['count'] += 1
                    failure_patterns[pattern]['tests'].add(test_name)
                    
                    # Classify error type
                    error_type = self._classify_error(error_msg)
                    failure_patterns[pattern]['error_types'][error_type] += 1
        
        # Convert to list
        patterns_list = []
        for pattern, data in failure_patterns.items():
            patterns_list.append({
                'pattern': pattern,
                'occurrences': data['count'],
                'affected_tests': len(data['tests']),
                'tests': list(data['tests'])[:10],  # Top 10
                'error_types': dict(data['error_types'])
            })
        
        return {
            'total_patterns': len(patterns_list),
            'patterns': sorted(patterns_list, key=lambda x: x['occurrences'], reverse=True)
        }
    
    def _extract_error_pattern(self, error_msg: str) -> str:
        """Extract pattern from error message"""
        if not error_msg:
            return 'unknown'
        
        # Extract key parts (simplified)
        error_lower = error_msg.lower()
        
        if 'assertion' in error_lower:
            return 'assertion_error'
        elif 'timeout' in error_lower:
            return 'timeout_error'
        elif 'connection' in error_lower or 'network' in error_lower:
            return 'network_error'
        elif 'permission' in error_lower or 'access' in error_lower:
            return 'permission_error'
        elif 'not found' in error_lower:
            return 'not_found_error'
        else:
            # Use first line or first 50 chars
            first_line = error_msg.split('\n')[0][:50]
            return f"other: {first_line}"
    
    def _classify_error(self, error_msg: str) -> str:
        """Classify error type"""
        if not error_msg:
            return 'unknown'
        
        error_lower = error_msg.lower()
        
        if 'assertion' in error_lower:
            return 'assertion'
        elif 'exception' in error_lower:
            return 'exception'
        elif 'timeout' in error_lower:
            return 'timeout'
        elif 'error' in error_lower:
            return 'error'
        else:
            return 'other'
    
    def analyze_performance_trends(
        self,
        results: List[Dict],
        test_name: str = None
    ) -> Dict:
        """Analyze performance trends"""
        # Group by test or overall
        if test_name:
            durations = []
            timestamps = []
            
            for result in results:
                test_details = result.get('test_details', {})
                if test_name in test_details:
                    test_data = test_details[test_name]
                    duration = test_data.get('duration', 0)
                    if duration > 0:
                        durations.append(duration)
                        timestamps.append(result.get('timestamp', ''))
        else:
            # Overall execution time
            durations = [
                r.get('summary', {}).get('execution_time', 0)
                for r in results
                if r.get('summary', {}).get('execution_time', 0) > 0
            ]
            timestamps = [
                r.get('timestamp', '')
                for r in results
                if r.get('summary', {}).get('execution_time', 0) > 0
            ]
        
        if len(durations) < 2:
            return {'error': 'Insufficient data'}
        
        # Calculate trend
        first_half = durations[:len(durations)//2]
        second_half = durations[len(durations)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        trend_percent = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        return {
            'test_name': test_name or 'overall',
            'data_points': len(durations),
            'current': durations[-1],
            'average': statistics.mean(durations),
            'median': statistics.median(durations),
            'min': min(durations),
            'max': max(durations),
            'stdev': statistics.stdev(durations) if len(durations) > 1 else 0,
            'trend_percent': round(trend_percent, 2),
            'trend_direction': 'improving' if trend_percent < 0 else ('degrading' if trend_percent > 0 else 'stable')
        }
    
    def generate_analytics_report(
        self,
        results: List[Dict],
        output_file: Path = None
    ) -> str:
        """Generate comprehensive analytics report"""
        lines = []
        lines.append("📊 ADVANCED TEST RESULT ANALYTICS REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Health analysis
        health = self.analyze_test_health(results)
        lines.append("🏥 TEST HEALTH ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Health Score: {health.get('health_score', 0)}/100 ({health.get('health_level', 'unknown')})")
        lines.append(f"Average Success Rate: {health.get('average_success_rate', 0):.1f}%")
        lines.append(f"Stability: {health.get('stability', 0):.3f}")
        lines.append("")
        
        # Failure patterns
        patterns = self.analyze_failure_patterns(results)
        lines.append("🔍 FAILURE PATTERNS")
        lines.append("-" * 80)
        lines.append(f"Total Patterns: {patterns['total_patterns']}")
        for pattern in patterns['patterns'][:10]:
            lines.append(f"  {pattern['pattern']}: {pattern['occurrences']} occurrences, {pattern['affected_tests']} tests")
        lines.append("")
        
        # Performance trends
        perf = self.analyze_performance_trends(results)
        if 'error' not in perf:
            lines.append("⚡ PERFORMANCE TRENDS")
            lines.append("-" * 80)
            lines.append(f"Current: {perf.get('current', 0):.2f}s")
            lines.append(f"Average: {perf.get('average', 0):.2f}s")
            lines.append(f"Trend: {perf.get('trend_direction', 'unknown')} ({perf.get('trend_percent', 0):+.1f}%)")
            lines.append("")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Report saved to {output_file}")
        
        return report


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Test Result Analytics')
    parser.add_argument('--results', type=str, help='Results file or directory')
    parser.add_argument('--health', action='store_true', help='Analyze test health')
    parser.add_argument('--patterns', action='store_true', help='Analyze failure patterns')
    parser.add_argument('--performance', type=str, help='Analyze performance for test')
    parser.add_argument('--report', type=str, help='Generate full analytics report')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    analytics = AdvancedTestResultAnalytics(project_root)
    
    # Load results
    results = []
    if args.results:
        result_path = Path(args.results)
        if result_path.is_file():
            with open(result_path, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
        elif result_path.is_dir():
            for result_file in result_path.glob("*.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        results.append(json.load(f))
                except Exception:
                    pass
    
    if args.report:
        print("📊 Generating analytics report...")
        analytics.generate_analytics_report(results, Path(args.report) if args.report else None)
    elif args.health:
        print("🏥 Analyzing test health...")
        health = analytics.analyze_test_health(results)
        print(f"  Health Score: {health.get('health_score', 0)}/100")
        print(f"  Level: {health.get('health_level', 'unknown')}")
    elif args.patterns:
        print("🔍 Analyzing failure patterns...")
        patterns = analytics.analyze_failure_patterns(results)
        print(f"  Found {patterns['total_patterns']} patterns")
        for pattern in patterns['patterns'][:5]:
            print(f"    {pattern['pattern']}: {pattern['occurrences']} occurrences")
    elif args.performance:
        print(f"⚡ Analyzing performance for: {args.performance}")
        perf = analytics.analyze_performance_trends(results, args.performance)
        if 'error' not in perf:
            print(f"  Trend: {perf.get('trend_direction', 'unknown')} ({perf.get('trend_percent', 0):+.1f}%)")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

