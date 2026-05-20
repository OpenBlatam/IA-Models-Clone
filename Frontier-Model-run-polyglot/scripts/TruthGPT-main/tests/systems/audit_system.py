"""
Audit System
Comprehensive audit of test suite
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict

class AuditSystem:
    """Comprehensive audit of test suite"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def perform_audit(self, lookback_days: int = 30) -> Dict:
        """Perform comprehensive audit"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        audit_results = {
            'audit_date': datetime.now().isoformat(),
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'compliance': self._check_compliance(recent),
            'best_practices': self._check_best_practices(recent),
            'issues': self._identify_issues(recent),
            'recommendations': self._generate_audit_recommendations(recent),
            'audit_score': 0
        }
        
        # Calculate audit score
        audit_results['audit_score'] = self._calculate_audit_score(audit_results)
        
        return audit_results
    
    def _check_compliance(self, recent: List[Dict]) -> Dict:
        """Check compliance with standards"""
        compliance = {
            'success_rate_threshold': {'required': 90, 'met': False},
            'execution_time_threshold': {'required': 300, 'met': False},
            'test_coverage_threshold': {'required': 80, 'met': False}
        }
        
        if recent:
            success_rates = [r.get('success_rate', 0) for r in recent]
            avg_success = sum(success_rates) / len(success_rates) if success_rates else 0
            compliance['success_rate_threshold']['met'] = avg_success >= 90
            
            execution_times = [r.get('execution_time', 0) for r in recent]
            avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
            compliance['execution_time_threshold']['met'] = avg_time <= 300
        
        return compliance
    
    def _check_best_practices(self, recent: List[Dict]) -> Dict:
        """Check best practices"""
        practices = {
            'regular_execution': len(recent) >= 10,
            'consistent_results': False,
            'proper_documentation': True,  # Assume true if history exists
            'error_handling': True
        }
        
        if len(recent) > 1:
            success_rates = [r.get('success_rate', 0) for r in recent]
            variance = max(success_rates) - min(success_rates)
            practices['consistent_results'] = variance < 10
        
        return practices
    
    def _identify_issues(self, recent: List[Dict]) -> List[Dict]:
        """Identify issues"""
        issues = []
        
        if not recent:
            issues.append({
                'severity': 'high',
                'type': 'insufficient_data',
                'description': 'Insufficient test execution history',
                'impact': 'Cannot perform comprehensive audit'
            })
            return issues
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = sum(success_rates) / len(success_rates) if success_rates else 0
        
        if avg_success < 90:
            issues.append({
                'severity': 'high',
                'type': 'low_success_rate',
                'description': f'Average success rate is {avg_success:.1f}%, below threshold of 90%',
                'impact': 'Quality concerns'
            })
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        if avg_time > 300:
            issues.append({
                'severity': 'medium',
                'type': 'slow_execution',
                'description': f'Average execution time is {avg_time:.1f}s, above threshold of 300s',
                'impact': 'Performance concerns'
            })
        
        return issues
    
    def _generate_audit_recommendations(self, recent: List[Dict]) -> List[str]:
        """Generate audit recommendations"""
        recommendations = []
        
        if len(recent) < 10:
            recommendations.append("Increase test execution frequency for better tracking")
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        if success_rates:
            avg_success = sum(success_rates) / len(success_rates)
            if avg_success < 95:
                recommendations.append("Improve test success rate to 95%+")
        
        return recommendations
    
    def _calculate_audit_score(self, audit_results: Dict) -> float:
        """Calculate audit score"""
        score = 100.0
        
        # Deduct for compliance issues
        compliance = audit_results['compliance']
        for check in compliance.values():
            if not check.get('met', False):
                score -= 10
        
        # Deduct for best practice violations
        practices = audit_results['best_practices']
        for practice, met in practices.items():
            if not met:
                score -= 5
        
        # Deduct for issues
        issues = audit_results['issues']
        for issue in issues:
            severity = issue.get('severity', 'medium')
            if severity == 'high':
                score -= 15
            elif severity == 'medium':
                score -= 10
            else:
                score -= 5
        
        return max(0, round(score, 1))
    
    def generate_audit_report(self, audit: Dict) -> str:
        """Generate audit report"""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE TEST SUITE AUDIT")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Audit Date: {audit['audit_date'][:19]}")
        lines.append(f"Period: {audit['period']}")
        lines.append(f"Total Runs: {audit['total_runs']}")
        lines.append(f"Audit Score: {audit['audit_score']}/100")
        lines.append("")
        
        score_emoji = "🟢" if audit['audit_score'] >= 80 else "🟡" if audit['audit_score'] >= 60 else "🔴"
        lines.append(f"{score_emoji} Overall Status: {'PASS' if audit['audit_score'] >= 80 else 'NEEDS IMPROVEMENT'}")
        lines.append("")
        
        lines.append("✅ COMPLIANCE")
        lines.append("-" * 80)
        for check_name, check_data in audit['compliance'].items():
            status = "✅" if check_data['met'] else "❌"
            lines.append(f"{status} {check_name.replace('_', ' ').title()}: {check_data['required']}")
        lines.append("")
        
        lines.append("📋 BEST PRACTICES")
        lines.append("-" * 80)
        for practice, met in audit['best_practices'].items():
            status = "✅" if met else "❌"
            lines.append(f"{status} {practice.replace('_', ' ').title()}")
        lines.append("")
        
        if audit['issues']:
            lines.append("⚠️  ISSUES")
            lines.append("-" * 80)
            for issue in audit['issues']:
                severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(issue['severity'], '⚪')
                lines.append(f"{severity_emoji} [{issue['severity'].upper()}] {issue['type']}")
                lines.append(f"   {issue['description']}")
                lines.append(f"   Impact: {issue['impact']}")
                lines.append("")
        
        if audit['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in audit['recommendations']:
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
    
    auditor = AuditSystem(project_root)
    audit = auditor.perform_audit(lookback_days=30)
    
    report = auditor.generate_audit_report(audit)
    print(report)
    
    # Save report
    report_file = project_root / "audit_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Audit report saved to: {report_file}")

if __name__ == "__main__":
    main()







