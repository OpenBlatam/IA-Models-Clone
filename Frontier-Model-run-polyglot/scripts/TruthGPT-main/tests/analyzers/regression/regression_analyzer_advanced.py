"""
Advanced Regression Analyzer
Advanced regression analysis with detailed insights
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from statistics import mean, stdev

class AdvancedRegressionAnalyzer:
    """Advanced regression analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_regressions(self, lookback_days: int = 30, threshold: float = 0.05) -> Dict:
        """Analyze test regressions"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = sorted(
            [r for r in history if r.get('timestamp', '') >= cutoff_date],
            key=lambda x: x.get('timestamp', '')
        )
        
        if len(recent) < 2:
            return {'error': 'Insufficient data for regression analysis'}
        
        # Analyze regressions
        regression_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'threshold': threshold,
            'regressions': self._detect_regressions(recent, threshold),
            'regression_trends': self._analyze_regression_trends(recent),
            'regression_patterns': self._identify_regression_patterns(recent),
            'impact_analysis': self._analyze_regression_impact(recent),
            'recommendations': []
        }
        
        # Generate recommendations
        regression_analysis['recommendations'] = self._generate_regression_recommendations(regression_analysis)
        
        return regression_analysis
    
    def _detect_regressions(self, recent: List[Dict], threshold: float) -> List[Dict]:
        """Detect regressions"""
        regressions = []
        
        if len(recent) < 2:
            return regressions
        
        baseline = recent[0]
        baseline_success = baseline.get('success_rate', 0)
        baseline_failures = baseline.get('failures', 0) + baseline.get('errors', 0)
        
        for i, current in enumerate(recent[1:], 1):
            current_success = current.get('success_rate', 0)
            current_failures = current.get('failures', 0) + current.get('errors', 0)
            
            # Check for success rate regression
            success_decline = baseline_success - current_success
            if success_decline > threshold * 100:
                regressions.append({
                    'run_index': i,
                    'timestamp': current.get('timestamp', ''),
                    'type': 'success_rate_decline',
                    'baseline_value': round(baseline_success, 2),
                    'current_value': round(current_success, 2),
                    'decline': round(success_decline, 2),
                    'severity': self._calculate_severity(success_decline),
                    'impact': 'High' if success_decline > 10 else 'Medium' if success_decline > 5 else 'Low'
                })
            
            # Check for failure increase
            failure_increase = current_failures - baseline_failures
            if failure_increase > 0 and current_failures > baseline_failures * 1.5:
                regressions.append({
                    'run_index': i,
                    'timestamp': current.get('timestamp', ''),
                    'type': 'failure_increase',
                    'baseline_value': baseline_failures,
                    'current_value': current_failures,
                    'increase': failure_increase,
                    'severity': self._calculate_severity(failure_increase),
                    'impact': 'High' if failure_increase > 10 else 'Medium' if failure_increase > 5 else 'Low'
                })
            
            # Update baseline if significant improvement
            if current_success > baseline_success + threshold * 100:
                baseline = current
                baseline_success = current_success
                baseline_failures = current_failures
        
        return regressions
    
    def _calculate_severity(self, change: float) -> str:
        """Calculate regression severity"""
        abs_change = abs(change)
        if abs_change > 15:
            return 'critical'
        elif abs_change > 10:
            return 'high'
        elif abs_change > 5:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_regression_trends(self, recent: List[Dict]) -> Dict:
        """Analyze regression trends"""
        if len(recent) < 4:
            return {}
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        # Calculate trend
        first_half = success_rates[:len(success_rates)//2]
        second_half = success_rates[len(success_rates)//2:]
        
        sr_trend = mean(second_half) - mean(first_half)
        
        f_first = failures[:len(failures)//2]
        f_second = failures[len(failures)//2:]
        f_trend = mean(f_second) - mean(f_first)
        
        return {
            'success_rate_trend': {
                'direction': 'improving' if sr_trend > 0 else 'regressing' if sr_trend < 0 else 'stable',
                'change': round(sr_trend, 2)
            },
            'failure_trend': {
                'direction': 'increasing' if f_trend > 0 else 'decreasing' if f_trend < 0 else 'stable',
                'change': round(f_trend, 2)
            },
            'overall_trend': 'regressing' if sr_trend < -5 or f_trend > 5 else 'improving' if sr_trend > 5 or f_trend < -5 else 'stable'
        }
    
    def _identify_regression_patterns(self, recent: List[Dict]) -> List[Dict]:
        """Identify regression patterns"""
        patterns = []
        
        if len(recent) < 5:
            return patterns
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        # Check for gradual decline
        if len(success_rates) >= 5:
            recent_5 = success_rates[-5:]
            if all(recent_5[i] > recent_5[i+1] for i in range(len(recent_5)-1)):
                patterns.append({
                    'pattern': 'gradual_decline',
                    'description': 'Success rate declining over last 5 runs',
                    'severity': 'high',
                    'recommendation': 'Investigate recent changes causing gradual degradation'
                })
        
        # Check for sudden drop
        if len(success_rates) >= 3:
            recent_3 = success_rates[-3:]
            if recent_3[0] - recent_3[-1] > 20:
                patterns.append({
                    'pattern': 'sudden_drop',
                    'description': 'Sudden significant drop in success rate',
                    'severity': 'critical',
                    'recommendation': 'Immediate investigation required - check recent commits'
                })
        
        # Check for volatility
        if len(success_rates) > 1:
            sr_std = stdev(success_rates)
            if sr_std > 10:
                patterns.append({
                    'pattern': 'high_volatility',
                    'description': f'High volatility in success rate (std: {sr_std:.1f}%)',
                    'severity': 'medium',
                    'recommendation': 'Improve test stability and consistency'
                })
        
        return patterns
    
    def _analyze_regression_impact(self, recent: List[Dict]) -> Dict:
        """Analyze regression impact"""
        if not recent:
            return {}
        
        total_runs = len(recent)
        regressed_runs = sum(1 for r in recent if r.get('success_rate', 100) < 90)
        
        avg_success = mean([r.get('success_rate', 0) for r in recent])
        avg_failures = mean([r.get('failures', 0) + r.get('errors', 0) for r in recent])
        
        return {
            'total_runs': total_runs,
            'regressed_runs': regressed_runs,
            'regression_rate': round((regressed_runs / total_runs * 100) if total_runs > 0 else 0, 1),
            'avg_success_rate': round(avg_success, 1),
            'avg_failures': round(avg_failures, 1),
            'impact_level': 'high' if regressed_runs > total_runs * 0.3 else 'medium' if regressed_runs > total_runs * 0.1 else 'low'
        }
    
    def _generate_regression_recommendations(self, analysis: Dict) -> List[str]:
        """Generate regression recommendations"""
        recommendations = []
        
        if analysis['regressions']:
            critical = sum(1 for r in analysis['regressions'] if r['severity'] == 'critical')
            if critical > 0:
                recommendations.append(f"🚨 {critical} critical regression(s) detected - immediate action required")
            
            high = sum(1 for r in analysis['regressions'] if r['severity'] == 'high')
            if high > 0:
                recommendations.append(f"⚠️ {high} high-severity regression(s) - investigate within 24 hours")
        
        patterns = analysis.get('regression_patterns', [])
        for pattern in patterns:
            if pattern['severity'] == 'critical':
                recommendations.append(f"🚨 {pattern['description']} - {pattern['recommendation']}")
        
        trends = analysis.get('regression_trends', {})
        if trends.get('overall_trend') == 'regressing':
            recommendations.append("📉 Overall trend is regressing - review recent changes and test stability")
        
        impact = analysis.get('impact_analysis', {})
        if impact.get('regression_rate', 0) > 20:
            recommendations.append(f"High regression rate ({impact['regression_rate']}%) - improve test reliability")
        
        if not recommendations:
            recommendations.append("✅ No significant regressions detected - maintain current practices")
        
        return recommendations
    
    def generate_regression_report(self, analysis: Dict) -> str:
        """Generate regression report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED REGRESSION ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append(f"Regression Threshold: {analysis['threshold']*100}%")
        lines.append("")
        
        if analysis['regressions']:
            lines.append("🔴 DETECTED REGRESSIONS")
            lines.append("-" * 80)
            severity_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
            for reg in analysis['regressions']:
                emoji = severity_emoji.get(reg['severity'], '⚪')
                lines.append(f"{emoji} Run #{reg['run_index']} - {reg['type'].replace('_', ' ').title()}")
                lines.append(f"   Timestamp: {reg['timestamp'][:19]}")
                lines.append(f"   Baseline: {reg['baseline_value']}")
                lines.append(f"   Current: {reg['current_value']}")
                lines.append(f"   Change: {reg.get('decline', reg.get('increase', 0)):+.2f}")
                lines.append(f"   Severity: {reg['severity'].upper()}")
                lines.append(f"   Impact: {reg['impact']}")
            lines.append("")
        else:
            lines.append("✅ No regressions detected")
            lines.append("")
        
        if analysis.get('regression_trends'):
            trends = analysis['regression_trends']
            lines.append("📊 REGRESSION TRENDS")
            lines.append("-" * 80)
            lines.append(f"Overall Trend: {trends['overall_trend'].upper()}")
            lines.append(f"Success Rate: {trends['success_rate_trend']['direction']} ({trends['success_rate_trend']['change']:+.2f}%)")
            lines.append(f"Failures: {trends['failure_trend']['direction']} ({trends['failure_trend']['change']:+.2f})")
            lines.append("")
        
        if analysis.get('regression_patterns'):
            lines.append("🔍 REGRESSION PATTERNS")
            lines.append("-" * 80)
            severity_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
            for pattern in analysis['regression_patterns']:
                emoji = severity_emoji.get(pattern['severity'], '⚪')
                lines.append(f"{emoji} {pattern['pattern'].replace('_', ' ').title()}")
                lines.append(f"   {pattern['description']}")
                lines.append(f"   {pattern['recommendation']}")
            lines.append("")
        
        if analysis.get('impact_analysis'):
            impact = analysis['impact_analysis']
            lines.append("💥 IMPACT ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"Total Runs: {impact['total_runs']}")
            lines.append(f"Regressed Runs: {impact['regressed_runs']}")
            lines.append(f"Regression Rate: {impact['regression_rate']}%")
            lines.append(f"Average Success Rate: {impact['avg_success_rate']}%")
            lines.append(f"Average Failures: {impact['avg_failures']:.1f}")
            lines.append(f"Impact Level: {impact['impact_level'].upper()}")
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
    
    analyzer = AdvancedRegressionAnalyzer(project_root)
    analysis = analyzer.analyze_regressions(lookback_days=30, threshold=0.05)
    
    report = analyzer.generate_regression_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_regression_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced regression analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






