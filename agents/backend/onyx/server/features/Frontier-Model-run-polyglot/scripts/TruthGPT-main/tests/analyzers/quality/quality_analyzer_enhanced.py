"""
Enhanced Quality Analyzer
Enhanced quality analysis with comprehensive metrics
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev

class EnhancedQualityAnalyzer:
    """Enhanced quality analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_quality(self, lookback_days: int = 30) -> Dict:
        """Analyze quality comprehensively"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Extract quality metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        # Comprehensive quality analysis
        quality_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'overall_quality_score': 0.0,
            'success_rate_analysis': self._analyze_success_rate(success_rates),
            'failure_analysis': self._analyze_failures(failures, total_tests),
            'quality_trends': self._analyze_quality_trends(recent),
            'quality_dimensions': self._analyze_quality_dimensions(recent),
            'quality_issues': self._identify_quality_issues(success_rates, failures, total_tests),
            'recommendations': []
        }
        
        # Calculate overall quality score
        quality_analysis['overall_quality_score'] = self._calculate_overall_quality_score(quality_analysis)
        
        # Generate recommendations
        quality_analysis['recommendations'] = self._generate_quality_recommendations(quality_analysis)
        
        return quality_analysis
    
    def _analyze_success_rate(self, success_rates: List[float]) -> Dict:
        """Analyze success rate"""
        if not success_rates:
            return {}
        
        return {
            'mean': round(mean(success_rates), 2),
            'std': round(stdev(success_rates), 2) if len(success_rates) > 1 else 0,
            'min': round(min(success_rates), 2),
            'max': round(max(success_rates), 2),
            'consistency': round(100 - (stdev(success_rates) * 2) if len(success_rates) > 1 else 100, 1),
            'meets_target': mean(success_rates) >= 95
        }
    
    def _analyze_failures(self, failures: List[int], total_tests: List[int]) -> Dict:
        """Analyze failure patterns"""
        if not failures or not total_tests:
            return {}
        
        total_failures = sum(failures)
        total_tests_sum = sum(total_tests)
        failure_rate = (total_failures / total_tests_sum * 100) if total_tests_sum > 0 else 0
        
        return {
            'total_failures': total_failures,
            'total_tests': total_tests_sum,
            'failure_rate': round(failure_rate, 2),
            'avg_failures_per_run': round(mean(failures), 2) if failures else 0,
            'max_failures_in_run': max(failures) if failures else 0,
            'meets_target': failure_rate < 5
        }
    
    def _analyze_quality_trends(self, recent: List[Dict]) -> Dict:
        """Analyze quality trends"""
        if len(recent) < 4:
            return {}
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        first_half = success_rates[:len(success_rates)//2]
        second_half = success_rates[len(success_rates)//2:]
        
        first_avg = mean(first_half)
        second_avg = mean(second_half)
        
        trend = second_avg - first_avg
        
        return {
            'trend': round(trend, 2),
            'direction': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable',
            'first_half_avg': round(first_avg, 2),
            'second_half_avg': round(second_avg, 2),
            'change_percent': round((trend / first_avg * 100) if first_avg > 0 else 0, 2)
        }
    
    def _analyze_quality_dimensions(self, recent: List[Dict]) -> Dict:
        """Analyze quality dimensions"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        # Reliability dimension
        reliability = mean(success_rates) if success_rates else 0
        
        # Consistency dimension
        sr_std = stdev(success_rates) if len(success_rates) > 1 else 0
        consistency = max(0, 100 - (sr_std * 2))
        
        # Efficiency dimension
        avg_time = mean(execution_times) if execution_times else 0
        efficiency = max(0, 100 - (avg_time / 5))  # Normalize
        
        return {
            'reliability': round(reliability, 1),
            'consistency': round(consistency, 1),
            'efficiency': round(efficiency, 1)
        }
    
    def _identify_quality_issues(self, success_rates: List[float], failures: List[int], total_tests: List[int]) -> List[Dict]:
        """Identify quality issues"""
        issues = []
        
        avg_success = mean(success_rates) if success_rates else 0
        if avg_success < 95:
            issues.append({
                'type': 'low_success_rate',
                'severity': 'high' if avg_success < 80 else 'medium',
                'description': f'Success rate is {avg_success:.1f}%, target is 95%+',
                'impact': 'May affect product quality'
            })
        
        total_failures = sum(failures)
        total_tests_sum = sum(total_tests)
        failure_rate = (total_failures / total_tests_sum * 100) if total_tests_sum > 0 else 0
        
        if failure_rate > 5:
            issues.append({
                'type': 'high_failure_rate',
                'severity': 'high' if failure_rate > 10 else 'medium',
                'description': f'Failure rate is {failure_rate:.1f}%, target is <5%',
                'impact': 'May delay releases'
            })
        
        return issues
    
    def _calculate_overall_quality_score(self, analysis: Dict) -> float:
        """Calculate overall quality score"""
        dimensions = analysis['quality_dimensions']
        
        # Weighted average
        score = (
            dimensions['reliability'] * 0.5 +
            dimensions['consistency'] * 0.3 +
            dimensions['efficiency'] * 0.2
        )
        
        return round(score, 1)
    
    def _generate_quality_recommendations(self, analysis: Dict) -> List[str]:
        """Generate quality recommendations"""
        recommendations = []
        
        sr_analysis = analysis['success_rate_analysis']
        if not sr_analysis.get('meets_target', False):
            recommendations.append(f"Improve success rate from {sr_analysis['mean']:.1f}% to 95%+")
        
        failure_analysis = analysis['failure_analysis']
        if not failure_analysis.get('meets_target', False):
            recommendations.append(f"Reduce failure rate from {failure_analysis['failure_rate']:.1f}% to <5%")
        
        if analysis['quality_issues']:
            recommendations.append(f"Address {len(analysis['quality_issues'])} identified quality issues")
        
        if not recommendations:
            recommendations.append("Quality metrics are meeting targets - maintain current practices")
        
        return recommendations
    
    def generate_quality_report(self, analysis: Dict) -> str:
        """Generate quality report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED QUALITY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        score_emoji = "🟢" if analysis['overall_quality_score'] >= 90 else "🟡" if analysis['overall_quality_score'] >= 75 else "🔴"
        lines.append(f"{score_emoji} Overall Quality Score: {analysis['overall_quality_score']}/100")
        lines.append("")
        
        lines.append("✅ SUCCESS RATE ANALYSIS")
        lines.append("-" * 80)
        sr = analysis['success_rate_analysis']
        status = "✅ Meets Target" if sr.get('meets_target') else "⚠️ Below Target"
        lines.append(f"{status}")
        lines.append(f"Mean: {sr['mean']}%")
        lines.append(f"Std Dev: {sr['std']}%")
        lines.append(f"Range: {sr['min']}% - {sr['max']}%")
        lines.append(f"Consistency: {sr['consistency']}%")
        lines.append("")
        
        lines.append("❌ FAILURE ANALYSIS")
        lines.append("-" * 80)
        fa = analysis['failure_analysis']
        status = "✅ Meets Target" if fa.get('meets_target') else "⚠️ Above Target"
        lines.append(f"{status}")
        lines.append(f"Total Failures: {fa['total_failures']}/{fa['total_tests']}")
        lines.append(f"Failure Rate: {fa['failure_rate']}%")
        lines.append(f"Avg Failures per Run: {fa['avg_failures_per_run']}")
        lines.append(f"Max Failures in Run: {fa['max_failures_in_run']}")
        lines.append("")
        
        lines.append("📊 QUALITY DIMENSIONS")
        lines.append("-" * 80)
        dims = analysis['quality_dimensions']
        lines.append(f"Reliability: {dims['reliability']}/100")
        lines.append(f"Consistency: {dims['consistency']}/100")
        lines.append(f"Efficiency: {dims['efficiency']}/100")
        lines.append("")
        
        if analysis.get('quality_trends'):
            trends = analysis['quality_trends']
            trend_emoji = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(trends['direction'], '➡️')
            lines.append(f"{emoji} QUALITY TRENDS")
            lines.append("-" * 80)
            lines.append(f"Direction: {trends['direction'].title()}")
            lines.append(f"Trend: {trends['trend']:+.2f}%")
            lines.append(f"Change: {trends['change_percent']:+.2f}%")
            lines.append("")
        
        if analysis['quality_issues']:
            lines.append("⚠️ QUALITY ISSUES")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for issue in analysis['quality_issues']:
                emoji = severity_emoji.get(issue['severity'], '⚪')
                lines.append(f"{emoji} [{issue['severity'].upper()}] {issue['type'].replace('_', ' ').title()}")
                lines.append(f"   {issue['description']}")
                lines.append(f"   Impact: {issue['impact']}")
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
    
    analyzer = EnhancedQualityAnalyzer(project_root)
    analysis = analyzer.analyze_quality(lookback_days=30)
    
    report = analyzer.generate_quality_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "enhanced_quality_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Enhanced quality analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







