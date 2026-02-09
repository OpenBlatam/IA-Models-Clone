"""
Advanced Executive Report
Advanced executive-level reporting
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class AdvancedExecutiveReport:
    """Advanced executive reporting"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def generate_executive_report(self, lookback_days: int = 30) -> Dict:
        """Generate executive report"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate executive metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        total_tests_sum = sum(total_tests)
        total_time = sum(execution_times)
        avg_success = mean(success_rates) if success_rates else 0
        total_failures = sum(failures)
        
        # Executive summary
        report = {
            'period': f'Last {lookback_days} days',
            'generated_at': datetime.now().isoformat(),
            'executive_summary': {
                'overall_health': self._calculate_overall_health(avg_success, total_failures, total_tests_sum),
                'key_metrics': {
                    'total_tests_executed': total_tests_sum,
                    'success_rate': round(avg_success, 1),
                    'failure_rate': round((total_failures / total_tests_sum * 100) if total_tests_sum > 0 else 0, 1),
                    'avg_execution_time': round(mean(execution_times), 2) if execution_times else 0,
                    'total_execution_time': round(total_time, 2)
                },
                'trends': self._calculate_executive_trends(recent),
                'risks': self._identify_executive_risks(avg_success, total_failures, total_tests_sum),
                'opportunities': self._identify_executive_opportunities(avg_success, total_time, total_tests_sum)
            },
            'recommendations': self._generate_executive_recommendations(avg_success, total_failures, total_tests_sum, total_time)
        }
        
        return report
    
    def _calculate_overall_health(self, success_rate: float, failures: int, total_tests: int) -> Dict:
        """Calculate overall health score"""
        failure_rate = (failures / total_tests * 100) if total_tests > 0 else 0
        
        # Health score: 0-100
        health_score = (success_rate * 0.7) + ((100 - failure_rate) * 0.3)
        
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 75:
            status = 'good'
        elif health_score >= 60:
            status = 'fair'
        else:
            status = 'needs_attention'
        
        return {
            'score': round(health_score, 1),
            'status': status,
            'success_rate': round(success_rate, 1),
            'failure_rate': round(failure_rate, 1)
        }
    
    def _calculate_executive_trends(self, recent: List[Dict]) -> Dict:
        """Calculate executive trends"""
        if len(recent) < 4:
            return {}
        
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_sr = mean([r.get('success_rate', 0) for r in first_half])
        second_sr = mean([r.get('success_rate', 0) for r in second_half])
        
        first_tests = sum([r.get('total_tests', 0) for r in first_half])
        second_tests = sum([r.get('total_tests', 0) for r in second_half])
        
        return {
            'success_rate_trend': round(second_sr - first_sr, 1),
            'test_volume_trend': round(second_tests - first_tests, 0),
            'direction': 'improving' if second_sr > first_sr else 'declining' if second_sr < first_sr else 'stable'
        }
    
    def _identify_executive_risks(self, success_rate: float, failures: int, total_tests: int) -> List[Dict]:
        """Identify executive risks"""
        risks = []
        
        if success_rate < 90:
            risks.append({
                'type': 'quality',
                'severity': 'high' if success_rate < 80 else 'medium',
                'description': f'Success rate below target: {success_rate:.1f}%',
                'impact': 'May affect product quality and release confidence'
            })
        
        failure_rate = (failures / total_tests * 100) if total_tests > 0 else 0
        if failure_rate > 10:
            risks.append({
                'type': 'reliability',
                'severity': 'high',
                'description': f'High failure rate: {failure_rate:.1f}%',
                'impact': 'May delay releases and reduce confidence'
            })
        
        return risks
    
    def _identify_executive_opportunities(self, success_rate: float, total_time: float, total_tests: int) -> List[Dict]:
        """Identify executive opportunities"""
        opportunities = []
        
        if success_rate >= 95 and total_tests > 1000:
            opportunities.append({
                'type': 'expansion',
                'description': 'High success rate with large test suite - consider expanding coverage',
                'benefit': 'Increased confidence and faster releases'
            })
        
        if total_time < 100:
            opportunities.append({
                'type': 'efficiency',
                'description': 'Fast execution time - opportunity to add more tests',
                'benefit': 'Better coverage without significant time increase'
            })
        
        return opportunities
    
    def _generate_executive_recommendations(self, success_rate: float, failures: int, total_tests: int, total_time: float) -> List[str]:
        """Generate executive recommendations"""
        recommendations = []
        
        if success_rate < 95:
            recommendations.append("Focus on improving test success rate to 95%+ for production readiness")
        
        failure_rate = (failures / total_tests * 100) if total_tests > 0 else 0
        if failure_rate > 5:
            recommendations.append("Reduce failure rate to <5% to improve release confidence")
        
        if total_time > 300:
            recommendations.append("Optimize test execution time to enable faster feedback cycles")
        
        if not recommendations:
            recommendations.append("Test suite is performing well - maintain current standards")
        
        return recommendations
    
    def generate_report_text(self, report: Dict) -> str:
        """Generate report text"""
        lines = []
        lines.append("=" * 80)
        lines.append("EXECUTIVE TEST REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in report:
            lines.append(f"❌ {report['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {report['period']}")
        lines.append(f"Generated: {report['generated_at'][:19]}")
        lines.append("")
        
        summary = report['executive_summary']
        health = summary['overall_health']
        status_emoji = {'excellent': '🟢', 'good': '🟡', 'fair': '🟠', 'needs_attention': '🔴'}
        emoji = status_emoji.get(health['status'], '⚪')
        
        lines.append(f"{emoji} OVERALL HEALTH: {health['status'].upper().replace('_', ' ')}")
        lines.append(f"Health Score: {health['score']}/100")
        lines.append(f"Success Rate: {health['success_rate']}%")
        lines.append(f"Failure Rate: {health['failure_rate']}%")
        lines.append("")
        
        lines.append("📊 KEY METRICS")
        lines.append("-" * 80)
        metrics = summary['key_metrics']
        lines.append(f"Total Tests Executed: {metrics['total_tests_executed']:,}")
        lines.append(f"Success Rate: {metrics['success_rate']}%")
        lines.append(f"Failure Rate: {metrics['failure_rate']}%")
        lines.append(f"Average Execution Time: {metrics['avg_execution_time']}s")
        lines.append(f"Total Execution Time: {metrics['total_execution_time']}s")
        lines.append("")
        
        if 'trends' in summary and summary['trends']:
            trends = summary['trends']
            trend_emoji = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(trends.get('direction', 'stable'), '➡️')
            lines.append(f"{emoji} TRENDS")
            lines.append("-" * 80)
            lines.append(f"Success Rate Trend: {trends.get('success_rate_trend', 0):+.1f}%")
            lines.append(f"Test Volume Trend: {trends.get('test_volume_trend', 0):+.0f} tests")
            lines.append(f"Direction: {trends.get('direction', 'stable').title()}")
            lines.append("")
        
        if summary.get('risks'):
            lines.append("⚠️ RISKS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for risk in summary['risks']:
                emoji = severity_emoji.get(risk['severity'], '⚪')
                lines.append(f"{emoji} [{risk['severity'].upper()}] {risk['type'].title()}")
                lines.append(f"   {risk['description']}")
                lines.append(f"   Impact: {risk['impact']}")
            lines.append("")
        
        if summary.get('opportunities'):
            lines.append("🚀 OPPORTUNITIES")
            lines.append("-" * 80)
            for opp in summary['opportunities']:
                lines.append(f"• {opp['type'].title()}: {opp['description']}")
                lines.append(f"  Benefit: {opp['benefit']}")
            lines.append("")
        
        if report['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in report['recommendations']:
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
    
    reporter = AdvancedExecutiveReport(project_root)
    report = reporter.generate_executive_report(lookback_days=30)
    
    report_text = reporter.generate_report_text(report)
    print(report_text)
    
    # Save report
    report_file = project_root / "executive_report_advanced.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n📄 Executive report saved to: {report_file}")

if __name__ == "__main__":
    main()







