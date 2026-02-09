"""
ROI Calculator
Calculate Return on Investment for testing
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
from statistics import mean

class ROICalculator:
    """Calculate ROI for testing"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def calculate_roi(
        self,
        bug_cost_avg: float = 100.0,  # Average cost to fix a bug in production
        test_cost_per_run: float = 10.0,  # Cost per test run
        lookback_days: int = 30
    ) -> Dict:
        """Calculate ROI for testing"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        total_runs = len(recent)
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        total_failures = sum(r.get('failures', 0) + r.get('errors', 0) for r in recent)
        
        # Calculate investment (test execution costs)
        total_investment = total_runs * test_cost_per_run
        
        # Calculate return (bugs caught by tests)
        # Assume 80% of test failures catch real bugs
        bugs_caught = total_failures * 0.8
        bugs_caught_cost = bugs_caught * bug_cost_avg
        
        # Additional benefits
        # 1. Confidence in deployments
        confidence_value = total_runs * 5.0  # $5 value per successful run
        
        # 2. Regression prevention
        # Assume tests prevent 2 bugs per month that would have been caught in production
        regression_prevention = 2 * bug_cost_avg
        
        # Total return
        total_return = bugs_caught_cost + confidence_value + regression_prevention
        
        # Calculate ROI
        roi = ((total_return - total_investment) / total_investment * 100) if total_investment > 0 else 0
        
        # Payback period (in days)
        daily_investment = total_investment / lookback_days if lookback_days > 0 else 0
        daily_return = total_return / lookback_days if lookback_days > 0 else 0
        payback_days = (total_investment / daily_return) if daily_return > 0 else 0
        
        return {
            'period_days': lookback_days,
            'total_runs': total_runs,
            'total_tests': total_tests,
            'total_failures': total_failures,
            'investment': {
                'total': round(total_investment, 2),
                'per_run': round(test_cost_per_run, 2)
            },
            'return': {
                'bugs_caught': round(bugs_caught, 1),
                'bugs_caught_value': round(bugs_caught_cost, 2),
                'confidence_value': round(confidence_value, 2),
                'regression_prevention': round(regression_prevention, 2),
                'total': round(total_return, 2)
            },
            'roi': round(roi, 2),
            'net_benefit': round(total_return - total_investment, 2),
            'payback_days': round(payback_days, 1)
        }
    
    def generate_roi_report(self, roi_data: Dict) -> str:
        """Generate ROI report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST ROI ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in roi_data:
            lines.append(f"❌ {roi_data['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: Last {roi_data['period_days']} days")
        lines.append(f"Total Runs: {roi_data['total_runs']}")
        lines.append(f"Total Tests: {roi_data['total_tests']}")
        lines.append("")
        
        lines.append("💰 INVESTMENT")
        lines.append("-" * 80)
        lines.append(f"Total Investment:    ${roi_data['investment']['total']:,.2f}")
        lines.append(f"Investment per Run:  ${roi_data['investment']['per_run']:.2f}")
        lines.append("")
        
        lines.append("📈 RETURN")
        lines.append("-" * 80)
        returns = roi_data['return']
        lines.append(f"Bugs Caught:         {returns['bugs_caught']:.1f}")
        lines.append(f"Bugs Caught Value:   ${returns['bugs_caught_value']:,.2f}")
        lines.append(f"Confidence Value:    ${returns['confidence_value']:,.2f}")
        lines.append(f"Regression Prevention: ${returns['regression_prevention']:,.2f}")
        lines.append(f"Total Return:        ${returns['total']:,.2f}")
        lines.append("")
        
        lines.append("📊 ROI METRICS")
        lines.append("-" * 80)
        roi_emoji = "🟢" if roi_data['roi'] > 0 else "🔴"
        lines.append(f"{roi_emoji} ROI:                 {roi_data['roi']:+.1f}%")
        lines.append(f"Net Benefit:         ${roi_data['net_benefit']:,.2f}")
        lines.append(f"Payback Period:      {roi_data['payback_days']:.1f} days")
        lines.append("")
        
        lines.append("💡 INTERPRETATION")
        lines.append("-" * 80)
        if roi_data['roi'] > 100:
            lines.append("✅ Excellent ROI - Testing provides significant value")
        elif roi_data['roi'] > 0:
            lines.append("✅ Positive ROI - Testing is providing value")
        else:
            lines.append("⚠️  Negative ROI - Consider optimizing test execution")
        
        if roi_data['payback_days'] < 30:
            lines.append("✅ Quick payback - Investment recovers quickly")
        elif roi_data['payback_days'] < 90:
            lines.append("🟡 Moderate payback - Reasonable recovery time")
        else:
            lines.append("⚠️  Long payback - Consider optimization")
        
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
    
    calculator = ROICalculator(project_root)
    roi_data = calculator.calculate_roi(lookback_days=30)
    
    report = calculator.generate_roi_report(roi_data)
    print(report)
    
    # Save report
    report_file = project_root / "roi_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 ROI analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







