"""
Advanced Quality Analyzer
Advanced quality analysis with comprehensive assessment
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev

class AdvancedQualityAnalyzer:
    """Advanced quality analysis"""
    
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
        
        # Comprehensive quality analysis
        quality_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'quality_dimensions': self._analyze_quality_dimensions(recent),
            'quality_trends': self._analyze_quality_trends(recent),
            'quality_distribution': self._analyze_quality_distribution(recent),
            'quality_issues': self._identify_quality_issues(recent),
            'quality_benchmarks': self._compare_quality_benchmarks(recent),
            'quality_health': self._assess_quality_health(recent),
            'recommendations': []
        }
        
        # Calculate overall quality score
        quality_analysis['overall_quality_score'] = self._calculate_overall_quality_score(quality_analysis)
        
        # Generate recommendations
        quality_analysis['recommendations'] = self._generate_quality_recommendations(quality_analysis)
        
        return quality_analysis
    
    def _analyze_quality_dimensions(self, recent: List[Dict]) -> Dict:
        """Analyze quality dimensions"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        # Reliability dimension
        reliability = mean(success_rates) if success_rates else 0
        
        # Consistency dimension
        sr_std = stdev(success_rates) if len(success_rates) > 1 else 0
        consistency = max(0, 100 - (sr_std * 2))
        
        # Efficiency dimension
        avg_time = mean(execution_times) if execution_times else 0
        efficiency = max(0, 100 - (avg_time / 5))
        
        # Stability dimension
        failure_rate = (sum(failures) / sum(total_tests) * 100) if sum(total_tests) > 0 else 0
        stability = max(0, 100 - failure_rate * 2)
        
        # Coverage dimension (using success rate as proxy)
        coverage = mean(success_rates) if success_rates else 0
        
        return {
            'reliability': round(reliability, 1),
            'consistency': round(consistency, 1),
            'efficiency': round(efficiency, 1),
            'stability': round(stability, 1),
            'coverage': round(coverage, 1)
        }
    
    def _analyze_quality_trends(self, recent: List[Dict]) -> Dict:
        """Analyze quality trends"""
        if len(recent) < 4:
            return {}
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        # Split into thirds
        third_size = len(recent) // 3
        first_third = recent[:third_size] if third_size > 0 else recent[:1]
        last_third = recent[2*third_size:] if third_size > 0 else recent[2:]
        
        sr_first = mean([r.get('success_rate', 0) for r in first_third])
        sr_last = mean([r.get('success_rate', 0) for r in last_third])
        sr_trend = sr_last - sr_first
        
        et_first = mean([r.get('execution_time', 0) for r in first_third])
        et_last = mean([r.get('execution_time', 0) for r in last_third])
        et_trend = et_last - et_first
        
        f_first = mean([r.get('failures', 0) + r.get('errors', 0) for r in first_third])
        f_last = mean([r.get('failures', 0) + r.get('errors', 0) for r in last_third])
        f_trend = f_last - f_first
        
        return {
            'reliability_trend': {
                'change': round(sr_trend, 2),
                'direction': 'improving' if sr_trend > 0 else 'declining' if sr_trend < 0 else 'stable',
                'percent_change': round((sr_trend / sr_first * 100) if sr_first > 0 else 0, 2)
            },
            'efficiency_trend': {
                'change': round(et_trend, 2),
                'direction': 'improving' if et_trend < 0 else 'degrading' if et_trend > 0 else 'stable',
                'percent_change': round((et_trend / et_first * 100) if et_first > 0 else 0, 2)
            },
            'stability_trend': {
                'change': round(f_trend, 2),
                'direction': 'improving' if f_trend < 0 else 'declining' if f_trend > 0 else 'stable',
                'percent_change': round((f_trend / f_first * 100) if f_first > 0 else 0, 2)
            }
        }
    
    def _analyze_quality_distribution(self, recent: List[Dict]) -> Dict:
        """Analyze quality distribution"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if not success_rates:
            return {}
        
        # Categorize by quality level
        excellent = sum(1 for sr in success_rates if sr >= 95)
        good = sum(1 for sr in success_rates if 90 <= sr < 95)
        acceptable = sum(1 for sr in success_rates if 80 <= sr < 90)
        poor = sum(1 for sr in success_rates if sr < 80)
        
        total = len(success_rates)
        
        return {
            'excellent': {
                'count': excellent,
                'percentage': round((excellent / total * 100) if total > 0 else 0, 1)
            },
            'good': {
                'count': good,
                'percentage': round((good / total * 100) if total > 0 else 0, 1)
            },
            'acceptable': {
                'count': acceptable,
                'percentage': round((acceptable / total * 100) if total > 0 else 0, 1)
            },
            'poor': {
                'count': poor,
                'percentage': round((poor / total * 100) if total > 0 else 0, 1)
            }
        }
    
    def _identify_quality_issues(self, recent: List[Dict]) -> List[Dict]:
        """Identify quality issues"""
        issues = []
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        if not success_rates:
            return issues
        
        avg_success = mean(success_rates)
        avg_time = mean(execution_times) if execution_times else 0
        total_failures = sum(failures)
        total_tests_sum = sum(total_tests)
        failure_rate = (total_failures / total_tests_sum * 100) if total_tests_sum > 0 else 0
        
        # Low success rate
        if avg_success < 95:
            issues.append({
                'type': 'low_success_rate',
                'severity': 'high' if avg_success < 80 else 'medium',
                'current_value': round(avg_success, 1),
                'target_value': 95,
                'gap': round(95 - avg_success, 1),
                'description': f'Success rate is {avg_success:.1f}%, target is 95%+',
                'impact': 'May affect product quality'
            })
        
        # High failure rate
        if failure_rate > 5:
            issues.append({
                'type': 'high_failure_rate',
                'severity': 'high' if failure_rate > 10 else 'medium',
                'current_value': round(failure_rate, 1),
                'target_value': 5,
                'gap': round(failure_rate - 5, 1),
                'description': f'Failure rate is {failure_rate:.1f}%, target is <5%',
                'impact': 'May delay releases'
            })
        
        # Slow execution
        if avg_time > 300:
            issues.append({
                'type': 'slow_execution',
                'severity': 'medium',
                'current_value': round(avg_time, 0),
                'target_value': 120,
                'gap': round(avg_time - 120, 0),
                'description': f'Average execution time is {avg_time:.0f}s, target is <120s',
                'impact': 'May slow down development cycle'
            })
        
        return issues
    
    def _compare_quality_benchmarks(self, recent: List[Dict]) -> Dict:
        """Compare quality against benchmarks"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        if not success_rates:
            return {}
        
        avg_success = mean(success_rates)
        avg_time = mean(execution_times) if execution_times else 0
        
        # Industry benchmarks
        industry_success_benchmark = 95
        industry_time_benchmark = 120
        
        return {
            'success_rate': {
                'current': round(avg_success, 1),
                'benchmark': industry_success_benchmark,
                'meets_benchmark': avg_success >= industry_success_benchmark,
                'gap': round(industry_success_benchmark - avg_success, 1)
            },
            'execution_time': {
                'current': round(avg_time, 2),
                'benchmark': industry_time_benchmark,
                'meets_benchmark': avg_time <= industry_time_benchmark,
                'gap': round(avg_time - industry_time_benchmark, 2)
            }
        }
    
    def _assess_quality_health(self, recent: List[Dict]) -> Dict:
        """Assess quality health"""
        dimensions = self._analyze_quality_dimensions(recent)
        
        # Calculate health score
        health_score = (
            dimensions['reliability'] * 0.3 +
            dimensions['consistency'] * 0.25 +
            dimensions['efficiency'] * 0.2 +
            dimensions['stability'] * 0.15 +
            dimensions['coverage'] * 0.1
        )
        
        return {
            'health_score': round(health_score, 1),
            'health_status': 'excellent' if health_score >= 90 else 'good' if health_score >= 80 else 'acceptable' if health_score >= 70 else 'poor',
            'dimensions': dimensions
        }
    
    def _calculate_overall_quality_score(self, analysis: Dict) -> float:
        """Calculate overall quality score"""
        health = analysis.get('quality_health', {})
        return health.get('health_score', 0.0)
    
    def _generate_quality_recommendations(self, analysis: Dict) -> List[str]:
        """Generate quality recommendations"""
        recommendations = []
        
        dimensions = analysis['quality_dimensions']
        if dimensions['reliability'] < 90:
            recommendations.append(f"Increase reliability from {dimensions['reliability']:.1f}% to 95%+")
        
        if dimensions['consistency'] < 90:
            recommendations.append(f"Improve consistency from {dimensions['consistency']:.1f}% to 95%+")
        
        if dimensions['efficiency'] < 70:
            recommendations.append(f"Improve efficiency from {dimensions['efficiency']:.1f}% to 80%+")
        
        if analysis['quality_issues']:
            high_severity = sum(1 for i in analysis['quality_issues'] if i['severity'] == 'high')
            if high_severity > 0:
                recommendations.append(f"🚨 {high_severity} high-severity quality issue(s) - prioritize fixing")
        
        benchmarks = analysis.get('quality_benchmarks', {})
        sr_bench = benchmarks.get('success_rate', {})
        if not sr_bench.get('meets_benchmark', True):
            recommendations.append(f"Success rate below industry benchmark ({sr_bench.get('gap', 0):.1f}% gap)")
        
        health = analysis.get('quality_health', {})
        if health.get('health_status') not in ['excellent', 'good']:
            recommendations.append(f"Improve quality health from {health.get('health_status', 'unknown')} to good+")
        
        if not recommendations:
            recommendations.append("✅ Quality metrics are meeting targets - maintain current practices")
        
        return recommendations
    
    def generate_quality_report(self, analysis: Dict) -> str:
        """Generate quality report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED QUALITY ANALYSIS REPORT")
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
        
        lines.append("📊 QUALITY DIMENSIONS")
        lines.append("-" * 80)
        dims = analysis['quality_dimensions']
        lines.append(f"Reliability: {dims['reliability']}/100")
        lines.append(f"Consistency: {dims['consistency']}/100")
        lines.append(f"Efficiency: {dims['efficiency']}/100")
        lines.append(f"Stability: {dims['stability']}/100")
        lines.append(f"Coverage: {dims['coverage']}/100")
        lines.append("")
        
        if analysis.get('quality_trends'):
            trends = analysis['quality_trends']
            lines.append("📈 QUALITY TRENDS")
            lines.append("-" * 80)
            trend_emoji = {'improving': '📈', 'declining': '📉', 'degrading': '📉', 'stable': '➡️'}
            
            rel_trend = trends['reliability_trend']
            emoji = trend_emoji.get(rel_trend['direction'], '➡️')
            lines.append(f"{emoji} Reliability: {rel_trend['direction'].title()} ({rel_trend['change']:+.2f}%, {rel_trend['percent_change']:+.2f}%)")
            
            eff_trend = trends['efficiency_trend']
            emoji = trend_emoji.get(eff_trend['direction'], '➡️')
            lines.append(f"{emoji} Efficiency: {eff_trend['direction'].title()} ({eff_trend['change']:+.2f}s, {eff_trend['percent_change']:+.2f}%)")
            
            stab_trend = trends['stability_trend']
            emoji = trend_emoji.get(stab_trend['direction'], '➡️')
            lines.append(f"{emoji} Stability: {stab_trend['direction'].title()} ({stab_trend['change']:+.2f}, {stab_trend['percent_change']:+.2f}%)")
            lines.append("")
        
        lines.append("📊 QUALITY DISTRIBUTION")
        lines.append("-" * 80)
        dist = analysis['quality_distribution']
        lines.append(f"Excellent (≥95%): {dist['excellent']['count']} ({dist['excellent']['percentage']}%)")
        lines.append(f"Good (90-94%): {dist['good']['count']} ({dist['good']['percentage']}%)")
        lines.append(f"Acceptable (80-89%): {dist['acceptable']['count']} ({dist['acceptable']['percentage']}%)")
        lines.append(f"Poor (<80%): {dist['poor']['count']} ({dist['poor']['percentage']}%)")
        lines.append("")
        
        if analysis['quality_issues']:
            lines.append("⚠️ QUALITY ISSUES")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for issue in analysis['quality_issues']:
                emoji = severity_emoji.get(issue['severity'], '⚪')
                lines.append(f"{emoji} [{issue['severity'].upper()}] {issue['type'].replace('_', ' ').title()}")
                lines.append(f"   Current: {issue['current_value']}")
                lines.append(f"   Target: {issue['target_value']}")
                lines.append(f"   Gap: {issue['gap']}")
                lines.append(f"   {issue['description']}")
                lines.append(f"   Impact: {issue['impact']}")
            lines.append("")
        
        if analysis.get('quality_benchmarks'):
            benchmarks = analysis['quality_benchmarks']
            lines.append("📊 QUALITY BENCHMARKS")
            lines.append("-" * 80)
            sr_bench = benchmarks['success_rate']
            status = "✅" if sr_bench['meets_benchmark'] else "⚠️"
            lines.append(f"{status} Success Rate: {sr_bench['current']}% (Benchmark: {sr_bench['benchmark']}%)")
            if not sr_bench['meets_benchmark']:
                lines.append(f"   Gap: {sr_bench['gap']}%")
            
            et_bench = benchmarks['execution_time']
            status = "✅" if et_bench['meets_benchmark'] else "⚠️"
            lines.append(f"{status} Execution Time: {et_bench['current']}s (Benchmark: {et_bench['benchmark']}s)")
            if not et_bench['meets_benchmark']:
                lines.append(f"   Gap: {et_bench['gap']}s")
            lines.append("")
        
        if analysis.get('quality_health'):
            health = analysis['quality_health']
            status_emoji = {'excellent': '🟢', 'good': '🟡', 'acceptable': '🟠', 'poor': '🔴'}
            emoji = status_emoji.get(health['health_status'], '⚪')
            lines.append(f"{emoji} QUALITY HEALTH")
            lines.append("-" * 80)
            lines.append(f"Health Status: {health['health_status'].upper()}")
            lines.append(f"Health Score: {health['health_score']}/100")
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
    
    analyzer = AdvancedQualityAnalyzer(project_root)
    analysis = analyzer.analyze_quality(lookback_days=30)
    
    report = analyzer.generate_quality_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_quality_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced quality analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






