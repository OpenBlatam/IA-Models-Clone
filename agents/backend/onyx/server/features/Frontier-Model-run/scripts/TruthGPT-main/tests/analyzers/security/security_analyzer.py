"""
Security Analyzer
Analyze test security aspects
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
import re

class SecurityAnalyzer:
    """Analyze test security"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.tests_dir = project_root / "tests"
    
    def analyze_security(self) -> Dict:
        """Analyze security aspects"""
        security_issues = []
        security_score = 100
        
        # Check for hardcoded secrets
        hardcoded_secrets = self._check_hardcoded_secrets()
        if hardcoded_secrets:
            security_issues.extend(hardcoded_secrets)
            security_score -= len(hardcoded_secrets) * 10
        
        # Check for insecure test data
        insecure_data = self._check_insecure_data()
        if insecure_data:
            security_issues.extend(insecure_data)
            security_score -= len(insecure_data) * 5
        
        # Check for proper error handling
        error_handling = self._check_error_handling()
        if not error_handling['proper']:
            security_issues.append({
                'type': 'error_handling',
                'severity': 'medium',
                'description': 'Insufficient error handling in tests'
            })
            security_score -= 10
        
        return {
            'security_score': max(0, security_score),
            'total_issues': len(security_issues),
            'issues': security_issues,
            'recommendations': self._generate_security_recommendations(security_issues)
        }
    
    def _check_hardcoded_secrets(self) -> List[Dict]:
        """Check for hardcoded secrets"""
        issues = []
        secret_patterns = [
            (r'password\s*=\s*["\']\w+["\']', 'hardcoded_password'),
            (r'api_key\s*=\s*["\']\w+["\']', 'hardcoded_api_key'),
            (r'secret\s*=\s*["\']\w+["\']', 'hardcoded_secret'),
            (r'token\s*=\s*["\']\w+["\']', 'hardcoded_token')
        ]
        
        if self.tests_dir.exists():
            for test_file in self.tests_dir.glob("test_*.py"):
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern, issue_type in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            issues.append({
                                'type': issue_type,
                                'severity': 'high',
                                'file': str(test_file.name),
                                'description': f'Potential hardcoded secret found in {test_file.name}'
                            })
                except Exception:
                    continue
        
        return issues
    
    def _check_insecure_data(self) -> List[Dict]:
        """Check for insecure test data"""
        issues = []
        
        # This is a simplified check
        # In a real implementation, would check for SQL injection patterns, XSS, etc.
        
        return issues
    
    def _check_error_handling(self) -> Dict:
        """Check error handling"""
        # Simplified check
        return {'proper': True}
    
    def _generate_security_recommendations(self, issues: List[Dict]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if any(issue['type'].startswith('hardcoded') for issue in issues):
            recommendations.append("Remove hardcoded secrets - use environment variables")
        
        if len(issues) > 0:
            recommendations.append("Review and fix all security issues")
            recommendations.append("Implement security best practices")
        
        return recommendations
    
    def generate_security_report(self, analysis: Dict) -> str:
        """Generate security report"""
        lines = []
        lines.append("=" * 80)
        lines.append("SECURITY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        score_emoji = "🟢" if analysis['security_score'] >= 80 else "🟡" if analysis['security_score'] >= 60 else "🔴"
        lines.append(f"{score_emoji} Security Score: {analysis['security_score']}/100")
        lines.append(f"Total Issues: {analysis['total_issues']}")
        lines.append("")
        
        if analysis['issues']:
            lines.append("⚠️  SECURITY ISSUES")
            lines.append("-" * 80)
            for issue in analysis['issues']:
                severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(issue['severity'], '⚪')
                lines.append(f"{severity_emoji} [{issue['severity'].upper()}] {issue['type']}")
                lines.append(f"   {issue['description']}")
                if 'file' in issue:
                    lines.append(f"   File: {issue['file']}")
                lines.append("")
        
        if analysis['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in analysis['recommendations']:
                lines.append(f"• {rec}")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    analyzer = SecurityAnalyzer(project_root)
    analysis = analyzer.analyze_security()
    
    report = analyzer.generate_security_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "security_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Security analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







