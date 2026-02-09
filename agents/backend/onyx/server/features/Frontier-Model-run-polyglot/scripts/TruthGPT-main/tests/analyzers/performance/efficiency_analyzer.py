"""
Efficiency Analyzer
Analyze test execution efficiency
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class EfficiencyAnalyzer:
    """Analyze test execution efficiency"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_efficiency(self, lookback_days: int = 30) -> Dict:
        """Analyze test execution efficiency"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate efficiency metrics
        total_tests = [r.get('total_tests', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        total_tests_sum = sum(total_tests)
        total_time = sum(execution_times)
        avg_success = mean(success_rates) if success_rates else 0
        
        # Calculate efficiency metrics
        tests_per_second = total_tests_sum / total_time if total_time > 0 else 0
        avg_time_per_test = total_time / total_tests_sum if total_tests_sum > 0 else 0
        
        # Efficiency score: combination of speed and success
        speed_score = min(100, tests_per_second * 10)  # Normalize
        success_score = avg_success
        efficiency_score = (speed_score * 0.4) + (success_score * 0.6)
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'efficiency_score': round(efficiency_score, 1),
            'tests_per_second': round(tests_per_second, 2),
            'avg_time_per_test': round(avg_time_per_test, 3),
            'total_tests_executed': total_tests_sum,
            'total_execution_time': round(total_time, 2),
            'avg_success_rate': round(avg_success, 2),
            'speed_score': round(speed_score, 1),
            'success_score': round(success_score, 1),
            'recommendations': self._generate_efficiency_recommendations(efficiency_score, tests_per_second, avg_success)
        }
    
    def _generate_efficiency_recommendations(self, efficiency_score: float, tests_per_second: float, success_rate: float) -> List[str]:
        """Generate efficiency recommendations"""
        recommendations = []
        
        if efficiency_score < 60:
            recommendations.append("Critical: Test efficiency is low - significant improvements needed")
            if tests_per_second < 1:
                recommendations.append("Optimize test execution speed - consider parallel execution")
            if success_rate < 90:
                recommendations.append("Improve test success rate - fix failing tests")
        elif efficiency_score < 75:
            recommendations.append("Test efficiency is moderate - room for improvement")
            if tests_per_second < 2:
                recommendations.append("Consider optimizing slow tests")
            if success_rate < 95:
                recommendations.append("Address intermittent failures")
        elif efficiency_score < 90:
            recommendations.append("Test efficiency is good - minor optimizations possible")
        else:
            recommendations.append("Excellent test efficiency - maintain current practices")
        
        return recommendations
    
    def generate_efficiency_report(self, analysis: Dict) -> str:
        """Generate efficiency report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST EFFICIENCY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        score_emoji = "🟢" if analysis['efficiency_score'] >= 90 else "🟡" if analysis['efficiency_score'] >= 75 else "🔴"
        lines.append(f"{score_emoji} Overall Efficiency Score: {analysis['efficiency_score']}/100")
        lines.append("")
        
        lines.append("⚡ EFFICIENCY METRICS")
        lines.append("-" * 80)
        lines.append(f"Tests per Second: {analysis['tests_per_second']}")
        lines.append(f"Average Time per Test: {analysis['avg_time_per_test']}s")
        lines.append(f"Total Tests Executed: {analysis['total_tests_executed']:,}")
        lines.append(f"Total Execution Time: {analysis['total_execution_time']}s")
        lines.append("")
        
        lines.append("📊 SCORE BREAKDOWN")
        lines.append("-" * 80)
        lines.append(f"Speed Score: {analysis['speed_score']}/100")
        lines.append(f"Success Score: {analysis['success_score']}/100")
        lines.append(f"Average Success Rate: {analysis['avg_success_rate']}%")
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
    
    analyzer = EfficiencyAnalyzer(project_root)
    analysis = analyzer.analyze_efficiency(lookback_days=30)
    
    report = analyzer.generate_efficiency_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "efficiency_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Efficiency analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







