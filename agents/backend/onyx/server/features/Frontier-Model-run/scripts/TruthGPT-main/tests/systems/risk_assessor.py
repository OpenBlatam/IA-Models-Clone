"""
Risk Assessor
Assess risks in test suite
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class RiskAssessor:
    """Assess risks in test suite"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def assess_risks(self, lookback_days: int = 30) -> Dict:
        """Assess risks in test suite"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        risks = []
        
        # Risk 1: Low success rate
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates) if success_rates else 0
        
        if avg_success < 90:
            risks.append({
                'category': 'Quality',
                'severity': 'high' if avg_success < 80 else 'medium',
                'title': 'Low Test Success Rate',
                'description': f'Average success rate is {avg_success:.1f}%',
                'impact': 'High risk of bugs reaching production',
                'probability': 'high' if avg_success < 80 else 'medium',
                'mitigation': 'Fix failing tests, improve test stability'
            })
        
        # Risk 2: High execution time
        execution_times = [r.get('execution_time', 0) for r in recent]
        avg_time = mean(execution_times) if execution_times else 0
        
        if avg_time > 300:
            risks.append({
                'category': 'Performance',
                'severity': 'high' if avg_time > 600 else 'medium',
                'title': 'Slow Test Execution',
                'description': f'Average execution time is {avg_time:.1f}s',
                'impact': 'Delays in development cycle, reduced test frequency',
                'probability': 'high',
                'mitigation': 'Optimize slow tests, enable parallel execution'
            })
        
        # Risk 3: High variance (unstable)
        if len(success_rates) > 1:
            variance = max(success_rates) - min(success_rates)
            if variance > 15:
                risks.append({
                    'category': 'Stability',
                    'severity': 'high' if variance > 25 else 'medium',
                    'title': 'Unstable Test Results',
                    'description': f'High variance ({variance:.1f}%) in success rates',
                    'impact': 'Unreliable test results, difficult to trust',
                    'probability': 'high',
                    'mitigation': 'Fix flaky tests, improve test isolation'
                })
        
        # Risk 4: Declining trend
        if len(recent) >= 10:
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]
            
            first_avg = mean([r.get('success_rate', 0) for r in first_half])
            second_avg = mean([r.get('success_rate', 0) for r in second_half])
            decline = first_avg - second_avg
            
            if decline > 5:
                risks.append({
                    'category': 'Trend',
                    'severity': 'high' if decline > 10 else 'medium',
                    'title': 'Declining Test Quality',
                    'description': f'Success rate declined by {decline:.1f}%',
                    'impact': 'Quality degradation over time',
                    'probability': 'medium',
                    'mitigation': 'Investigate recent changes, fix regressions'
                })
        
        # Risk 5: Low test coverage
        total_tests = [r.get('total_tests', 0) for r in recent]
        avg_tests = mean(total_tests) if total_tests else 0
        
        if avg_tests < 100:
            risks.append({
                'category': 'Coverage',
                'severity': 'medium',
                'title': 'Low Test Coverage',
                'description': f'Average {avg_tests:.0f} tests per run',
                'impact': 'Potential gaps in test coverage',
                'probability': 'medium',
                'mitigation': 'Add more tests, improve coverage metrics'
            })
        
        # Calculate overall risk score
        severity_scores = {'high': 3, 'medium': 2, 'low': 1}
        total_risk_score = sum(severity_scores.get(r['severity'], 1) for r in risks)
        max_possible_score = len(risks) * 3 if risks else 1
        risk_percentage = (total_risk_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        if risk_percentage >= 70:
            overall_risk = 'High'
        elif risk_percentage >= 40:
            overall_risk = 'Medium'
        else:
            overall_risk = 'Low'
        
        return {
            'risks': risks,
            'total_risks': len(risks),
            'high_severity': len([r for r in risks if r['severity'] == 'high']),
            'medium_severity': len([r for r in risks if r['severity'] == 'medium']),
            'low_severity': len([r for r in risks if r['severity'] == 'low']),
            'overall_risk': overall_risk,
            'risk_score': round(risk_percentage, 1),
            'period_days': lookback_days
        }
    
    def generate_risk_report(self, assessment: Dict) -> str:
        """Generate risk assessment report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST SUITE RISK ASSESSMENT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in assessment:
            lines.append(f"❌ {assessment['error']}")
            return "\n".join(lines)
        
        risk_emoji = {
            'High': '🔴',
            'Medium': '🟡',
            'Low': '🟢'
        }.get(assessment['overall_risk'], '⚪')
        
        lines.append(f"{risk_emoji} Overall Risk: {assessment['overall_risk']}")
        lines.append(f"   Risk Score: {assessment['risk_score']}/100")
        lines.append(f"   Period: Last {assessment['period_days']} days")
        lines.append(f"   Total Risks Identified: {assessment['total_risks']}")
        lines.append("")
        
        if not assessment['risks']:
            lines.append("✅ No significant risks identified!")
            return "\n".join(lines)
        
        # Group by severity
        by_severity = {}
        for risk in assessment['risks']:
            severity = risk['severity']
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(risk)
        
        severity_order = ['high', 'medium', 'low']
        severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        
        for severity in severity_order:
            if severity not in by_severity:
                continue
            
            lines.append(f"{severity_emoji[severity]} {severity.upper()} SEVERITY RISKS")
            lines.append("-" * 80)
            
            for risk in by_severity[severity]:
                lines.append(f"\n{risk['title']}")
                lines.append(f"Category: {risk['category']}")
                lines.append(f"Description: {risk['description']}")
                lines.append(f"Impact: {risk['impact']}")
                lines.append(f"Probability: {risk['probability']}")
                lines.append(f"Mitigation: {risk['mitigation']}")
                lines.append("")
        
        lines.append("💡 RISK MITIGATION STRATEGY")
        lines.append("-" * 80)
        lines.append("1. Address high-severity risks immediately")
        lines.append("2. Create action plan for medium-severity risks")
        lines.append("3. Monitor low-severity risks")
        lines.append("4. Review and update risk assessment regularly")
        
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
    
    assessor = RiskAssessor(project_root)
    assessment = assessor.assess_risks(lookback_days=30)
    
    report = assessor.generate_risk_report(assessment)
    print(report)
    
    # Save report
    report_file = project_root / "risk_assessment_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Risk assessment report saved to: {report_file}")

if __name__ == "__main__":
    main()







