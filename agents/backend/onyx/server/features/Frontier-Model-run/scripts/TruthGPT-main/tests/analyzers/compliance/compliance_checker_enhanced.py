"""
Enhanced Compliance Checker
Enhanced compliance checking with multiple standards
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class EnhancedComplianceChecker:
    """Enhanced compliance checking"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def check_compliance(self, lookback_days: int = 30, standards: List[str] = None) -> Dict:
        """Check compliance with standards"""
        if standards is None:
            standards = ['ISO_25010', 'IEEE_829', 'CUSTOM']
        
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Comprehensive compliance checking
        compliance_check = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'standards_checked': standards,
            'iso_25010_compliance': self._check_iso_25010(recent) if 'ISO_25010' in standards else {},
            'ieee_829_compliance': self._check_ieee_829(recent) if 'IEEE_829' in standards else {},
            'custom_compliance': self._check_custom(recent) if 'CUSTOM' in standards else {},
            'overall_compliance': self._calculate_overall_compliance(recent),
            'compliance_trends': self._analyze_compliance_trends(recent),
            'non_compliance_issues': self._identify_non_compliance(recent),
            'recommendations': []
        }
        
        # Generate recommendations
        compliance_check['recommendations'] = self._generate_compliance_recommendations(compliance_check)
        
        return compliance_check
    
    def _check_iso_25010(self, recent: List[Dict]) -> Dict:
        """Check ISO 25010 compliance"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        if not success_rates:
            return {}
        
        # ISO 25010 quality characteristics
        avg_success = mean(success_rates)
        avg_time = mean(execution_times) if execution_times else 0
        total_failures = sum(failures)
        total_tests_sum = sum(total_tests)
        failure_rate = (total_failures / total_tests_sum * 100) if total_tests_sum > 0 else 0
        
        # Compliance criteria
        functional_suitability = avg_success  # Success rate
        performance_efficiency = max(0, 100 - (avg_time / 5))  # Normalized
        reliability = max(0, 100 - failure_rate * 2)  # Lower failure rate = higher reliability
        
        compliance_score = (functional_suitability * 0.4 + performance_efficiency * 0.3 + reliability * 0.3)
        
        return {
            'functional_suitability': round(functional_suitability, 1),
            'performance_efficiency': round(performance_efficiency, 1),
            'reliability': round(reliability, 1),
            'compliance_score': round(compliance_score, 1),
            'compliant': compliance_score >= 80
        }
    
    def _check_ieee_829(self, recent: List[Dict]) -> Dict:
        """Check IEEE 829 compliance"""
        # IEEE 829 test documentation requirements
        total_runs = len(recent)
        documented_runs = sum(1 for r in recent if r.get('timestamp') and r.get('total_tests', 0) > 0)
        
        # Compliance metrics
        documentation_coverage = (documented_runs / total_runs * 100) if total_runs > 0 else 0
        
        # Test execution requirements
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates) if success_rates else 0
        
        # Traceability (simulated)
        traceability_score = min(100, documentation_coverage * 1.2)
        
        compliance_score = (documentation_coverage * 0.5 + avg_success * 0.3 + traceability_score * 0.2)
        
        return {
            'documentation_coverage': round(documentation_coverage, 1),
            'test_execution_quality': round(avg_success, 1),
            'traceability_score': round(traceability_score, 1),
            'compliance_score': round(compliance_score, 1),
            'compliant': compliance_score >= 75
        }
    
    def _check_custom(self, recent: List[Dict]) -> Dict:
        """Check custom compliance"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        if not success_rates:
            return {}
        
        # Custom criteria: success rate >= 90%, execution time <= 300s
        avg_success = mean(success_rates)
        avg_time = mean(execution_times) if execution_times else 0
        
        success_compliant = avg_success >= 90
        time_compliant = avg_time <= 300
        
        compliance_score = (100 if success_compliant else avg_success) * 0.6 + (100 if time_compliant else max(0, 100 - (avg_time - 300) / 3)) * 0.4
        
        return {
            'success_rate_compliant': success_compliant,
            'execution_time_compliant': time_compliant,
            'current_success_rate': round(avg_success, 1),
            'current_execution_time': round(avg_time, 2),
            'compliance_score': round(compliance_score, 1),
            'compliant': success_compliant and time_compliant
        }
    
    def _calculate_overall_compliance(self, recent: List[Dict]) -> Dict:
        """Calculate overall compliance"""
        iso = self._check_iso_25010(recent)
        ieee = self._check_ieee_829(recent)
        custom = self._check_custom(recent)
        
        scores = []
        if iso.get('compliance_score'):
            scores.append(iso['compliance_score'])
        if ieee.get('compliance_score'):
            scores.append(ieee['compliance_score'])
        if custom.get('compliance_score'):
            scores.append(custom['compliance_score'])
        
        overall_score = mean(scores) if scores else 0
        
        return {
            'overall_score': round(overall_score, 1),
            'iso_25010_score': iso.get('compliance_score', 0),
            'ieee_829_score': ieee.get('compliance_score', 0),
            'custom_score': custom.get('compliance_score', 0),
            'compliant': overall_score >= 80
        }
    
    def _analyze_compliance_trends(self, recent: List[Dict]) -> Dict:
        """Analyze compliance trends"""
        if len(recent) < 4:
            return {}
        
        # Split into halves
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]
        
        iso_first = self._check_iso_25010(first_half)
        iso_second = self._check_iso_25010(second_half)
        
        iso_trend = iso_second.get('compliance_score', 0) - iso_first.get('compliance_score', 0)
        
        return {
            'iso_25010_trend': {
                'change': round(iso_trend, 2),
                'direction': 'improving' if iso_trend > 0 else 'declining' if iso_trend < 0 else 'stable'
            },
            'overall_trend': 'improving' if iso_trend > 2 else 'declining' if iso_trend < -2 else 'stable'
        }
    
    def _identify_non_compliance(self, recent: List[Dict]) -> List[Dict]:
        """Identify non-compliance issues"""
        issues = []
        
        iso = self._check_iso_25010(recent)
        if not iso.get('compliant', True):
            issues.append({
                'standard': 'ISO_25010',
                'issue': 'Compliance score below threshold',
                'current_score': iso.get('compliance_score', 0),
                'threshold': 80,
                'severity': 'high' if iso.get('compliance_score', 0) < 70 else 'medium'
            })
        
        ieee = self._check_ieee_829(recent)
        if not ieee.get('compliant', True):
            issues.append({
                'standard': 'IEEE_829',
                'issue': 'Compliance score below threshold',
                'current_score': ieee.get('compliance_score', 0),
                'threshold': 75,
                'severity': 'high' if ieee.get('compliance_score', 0) < 65 else 'medium'
            })
        
        custom = self._check_custom(recent)
        if not custom.get('compliant', True):
            issues.append({
                'standard': 'CUSTOM',
                'issue': 'Custom criteria not met',
                'current_score': custom.get('compliance_score', 0),
                'threshold': 80,
                'severity': 'high' if custom.get('compliance_score', 0) < 70 else 'medium'
            })
        
        return issues
    
    def _generate_compliance_recommendations(self, check: Dict) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        overall = check.get('overall_compliance', {})
        if not overall.get('compliant', True):
            recommendations.append(f"🚨 Overall compliance score is {overall['overall_score']:.1f} - improve to 80+")
        
        iso = check.get('iso_25010_compliance', {})
        if not iso.get('compliant', True):
            recommendations.append(f"Improve ISO 25010 compliance from {iso.get('compliance_score', 0):.1f} to 80+")
        
        ieee = check.get('ieee_829_compliance', {})
        if not ieee.get('compliant', True):
            recommendations.append(f"Improve IEEE 829 compliance from {ieee.get('compliance_score', 0):.1f} to 75+")
        
        custom = check.get('custom_compliance', {})
        if not custom.get('compliant', True):
            recommendations.append(f"Meet custom compliance criteria - current score: {custom.get('compliance_score', 0):.1f}")
        
        if check['non_compliance_issues']:
            high_severity = sum(1 for i in check['non_compliance_issues'] if i['severity'] == 'high')
            if high_severity > 0:
                recommendations.append(f"🚨 {high_severity} high-severity non-compliance issue(s) - immediate action required")
        
        trends = check.get('compliance_trends', {})
        if trends.get('overall_trend') == 'declining':
            recommendations.append("Compliance trend is declining - take corrective action")
        
        if not recommendations:
            recommendations.append("✅ All compliance standards are met - maintain current practices")
        
        return recommendations
    
    def generate_compliance_report(self, check: Dict) -> str:
        """Generate compliance report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED COMPLIANCE CHECK REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in check:
            lines.append(f"❌ {check['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {check['period']}")
        lines.append(f"Total Runs: {check['total_runs']}")
        lines.append(f"Standards Checked: {', '.join(check['standards_checked'])}")
        lines.append("")
        
        overall = check['overall_compliance']
        status_emoji = "✅" if overall.get('compliant') else "⚠️"
        lines.append(f"{status_emoji} Overall Compliance: {'COMPLIANT' if overall.get('compliant') else 'NON-COMPLIANT'}")
        lines.append(f"Overall Score: {overall['overall_score']}/100")
        lines.append("")
        
        if check.get('iso_25010_compliance'):
            lines.append("📋 ISO 25010 COMPLIANCE")
            lines.append("-" * 80)
            iso = check['iso_25010_compliance']
            status = "✅" if iso.get('compliant') else "⚠️"
            lines.append(f"{status} Status: {'COMPLIANT' if iso.get('compliant') else 'NON-COMPLIANT'}")
            lines.append(f"Functional Suitability: {iso['functional_suitability']}/100")
            lines.append(f"Performance Efficiency: {iso['performance_efficiency']}/100")
            lines.append(f"Reliability: {iso['reliability']}/100")
            lines.append(f"Compliance Score: {iso['compliance_score']}/100")
            lines.append("")
        
        if check.get('ieee_829_compliance'):
            lines.append("📋 IEEE 829 COMPLIANCE")
            lines.append("-" * 80)
            ieee = check['ieee_829_compliance']
            status = "✅" if ieee.get('compliant') else "⚠️"
            lines.append(f"{status} Status: {'COMPLIANT' if ieee.get('compliant') else 'NON-COMPLIANT'}")
            lines.append(f"Documentation Coverage: {ieee['documentation_coverage']}%")
            lines.append(f"Test Execution Quality: {ieee['test_execution_quality']}%")
            lines.append(f"Traceability Score: {ieee['traceability_score']}/100")
            lines.append(f"Compliance Score: {ieee['compliance_score']}/100")
            lines.append("")
        
        if check.get('custom_compliance'):
            lines.append("📋 CUSTOM COMPLIANCE")
            lines.append("-" * 80)
            custom = check['custom_compliance']
            status = "✅" if custom.get('compliant') else "⚠️"
            lines.append(f"{status} Status: {'COMPLIANT' if custom.get('compliant') else 'NON-COMPLIANT'}")
            lines.append(f"Success Rate Compliant: {'Yes' if custom.get('success_rate_compliant') else 'No'} ({custom.get('current_success_rate', 0)}%)")
            lines.append(f"Execution Time Compliant: {'Yes' if custom.get('execution_time_compliant') else 'No'} ({custom.get('current_execution_time', 0)}s)")
            lines.append(f"Compliance Score: {custom['compliance_score']}/100")
            lines.append("")
        
        if check.get('compliance_trends'):
            trends = check['compliance_trends']
            lines.append("📊 COMPLIANCE TRENDS")
            lines.append("-" * 80)
            trend_emoji = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(trends['overall_trend'], '➡️')
            lines.append(f"{emoji} Overall Trend: {trends['overall_trend'].upper()}")
            lines.append("")
        
        if check['non_compliance_issues']:
            lines.append("⚠️ NON-COMPLIANCE ISSUES")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡'}
            for issue in check['non_compliance_issues']:
                emoji = severity_emoji.get(issue['severity'], '⚪')
                lines.append(f"{emoji} [{issue['severity'].upper()}] {issue['standard']}")
                lines.append(f"   {issue['issue']}")
                lines.append(f"   Current Score: {issue['current_score']:.1f}")
                lines.append(f"   Threshold: {issue['threshold']}")
            lines.append("")
        
        if check['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in check['recommendations']:
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
    
    checker = EnhancedComplianceChecker(project_root)
    compliance = checker.check_compliance(lookback_days=30)
    
    report = checker.generate_compliance_report(compliance)
    print(report)
    
    # Save report
    report_file = project_root / "enhanced_compliance_check_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Enhanced compliance check report saved to: {report_file}")

if __name__ == "__main__":
    main()






