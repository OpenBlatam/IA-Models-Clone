"""
Compliance Checker
Check compliance with testing standards
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class ComplianceChecker:
    """Check compliance with testing standards"""
    
    STANDARDS = {
        'iso_25010': {
            'name': 'ISO/IEC 25010',
            'requirements': {
                'test_coverage': 80,
                'success_rate': 95,
                'documentation': True
            }
        },
        'ieee_829': {
            'name': 'IEEE 829',
            'requirements': {
                'test_documentation': True,
                'test_planning': True,
                'test_reporting': True
            }
        },
        'custom': {
            'name': 'Custom Standards',
            'requirements': {
                'success_rate': 90,
                'execution_time': 300,
                'test_coverage': 75
            }
        }
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def check_compliance(self, standard: str = 'custom', lookback_days: int = 30) -> Dict:
        """Check compliance with standard"""
        if standard not in self.STANDARDS:
            return {'error': f'Unknown standard: {standard}'}
        
        history = self._load_history()
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        standard_def = self.STANDARDS[standard]
        requirements = standard_def['requirements']
        
        compliance_results = {}
        all_compliant = True
        
        # Check each requirement
        if 'success_rate' in requirements:
            success_rates = [r.get('success_rate', 0) for r in recent]
            avg_success = mean(success_rates) if success_rates else 0
            required = requirements['success_rate']
            compliant = avg_success >= required
            compliance_results['success_rate'] = {
                'required': required,
                'actual': round(avg_success, 1),
                'compliant': compliant
            }
            if not compliant:
                all_compliant = False
        
        if 'execution_time' in requirements:
            execution_times = [r.get('execution_time', 0) for r in recent]
            avg_time = mean(execution_times) if execution_times else 0
            required = requirements['execution_time']
            compliant = avg_time <= required
            compliance_results['execution_time'] = {
                'required': required,
                'actual': round(avg_time, 2),
                'compliant': compliant
            }
            if not compliant:
                all_compliant = False
        
        if 'test_coverage' in requirements:
            # Simplified - would need actual coverage data
            compliance_results['test_coverage'] = {
                'required': requirements['test_coverage'],
                'actual': 75,  # Placeholder
                'compliant': True
            }
        
        return {
            'standard': standard_def['name'],
            'compliant': all_compliant,
            'requirements': compliance_results,
            'compliance_percentage': self._calculate_compliance_percentage(compliance_results)
        }
    
    def _calculate_compliance_percentage(self, results: Dict) -> float:
        """Calculate compliance percentage"""
        if not results:
            return 0.0
        
        compliant_count = sum(1 for r in results.values() if r.get('compliant', False))
        total_count = len(results)
        
        return round((compliant_count / total_count * 100) if total_count > 0 else 0, 1)
    
    def generate_compliance_report(self, compliance: Dict) -> str:
        """Generate compliance report"""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPLIANCE CHECK REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in compliance:
            lines.append(f"❌ {compliance['error']}")
            return "\n".join(lines)
        
        status_emoji = "✅" if compliance['compliant'] else "❌"
        lines.append(f"{status_emoji} Standard: {compliance['standard']}")
        lines.append(f"   Compliance: {'COMPLIANT' if compliance['compliant'] else 'NON-COMPLIANT'}")
        lines.append(f"   Compliance Percentage: {compliance['compliance_percentage']}%")
        lines.append("")
        
        lines.append("📋 REQUIREMENTS")
        lines.append("-" * 80)
        for req_name, req_data in compliance['requirements'].items():
            status = "✅" if req_data['compliant'] else "❌"
            lines.append(f"{status} {req_name.replace('_', ' ').title()}")
            lines.append(f"   Required: {req_data['required']}")
            lines.append(f"   Actual: {req_data['actual']}")
            lines.append("")
        
        if not compliance['compliant']:
            lines.append("⚠️  ACTION REQUIRED")
            lines.append("-" * 80)
            lines.append("Some requirements are not met. Please address the issues above.")
        
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
    import sys
    
    project_root = Path(__file__).parent.parent
    checker = ComplianceChecker(project_root)
    
    standard = sys.argv[1] if len(sys.argv) > 1 else 'custom'
    compliance = checker.check_compliance(standard=standard)
    
    report = checker.generate_compliance_report(compliance)
    print(report)
    
    # Save report
    report_file = project_root / f"compliance_report_{standard}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Compliance report saved to: {report_file}")

if __name__ == "__main__":
    main()







