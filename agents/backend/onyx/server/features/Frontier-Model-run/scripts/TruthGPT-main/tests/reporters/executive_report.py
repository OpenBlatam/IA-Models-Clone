"""
Executive Test Report Generator
Generates high-level executive reports
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta

class ExecutiveReportGenerator:
    """Generate executive-level test reports"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def generate_executive_report(self, period_days: int = 30) -> str:
        """Generate executive report for specified period"""
        cutoff_date = (datetime.now() - timedelta(days=period_days)).isoformat()
        
        # Load history
        history = self._load_history()
        recent_history = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent_history:
            return "No data available for the specified period."
        
        # Calculate metrics
        total_runs = len(recent_history)
        avg_success_rate = sum(r.get('success_rate', 0) for r in recent_history) / len(recent_history)
        total_tests_run = sum(r.get('total_tests', 0) for r in recent_history)
        avg_execution_time = sum(r.get('execution_time', 0) for r in recent_history) / len(recent_history)
        
        # Trend analysis
        if len(recent_history) >= 2:
            first_half = recent_history[:len(recent_history)//2]
            second_half = recent_history[len(recent_history)//2:]
            
            first_avg = sum(r.get('success_rate', 0) for r in first_half) / len(first_half)
            second_avg = sum(r.get('success_rate', 0) for r in second_half) / len(second_half)
            trend = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        else:
            trend = 0
        
        # Generate report
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EXECUTIVE TEST REPORT                                     ║
║                    Period: Last {period_days} Days                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 KEY METRICS                                                              ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Test Runs:              {total_runs:>6}                                           ║
║  Total Tests Executed:   {total_tests_run:>6}                                           ║
║  Average Success Rate:   {avg_success_rate:>5.1f}%                                        ║
║  Average Execution Time: {avg_execution_time:>5.1f}s                                       ║
║                                                                              ║
║  📈 TREND ANALYSIS                                                           ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Success Rate Trend:     {trend:>+5.1f}%                                        ║
║  Status:                 {'📈 IMPROVING' if trend > 0 else '📉 DECLINING' if trend < 0 else '➡️  STABLE':>20}  ║
║                                                                              ║
║  ✅ QUALITY ASSESSMENT                                                       ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Test Coverage:          {'High' if avg_success_rate > 95 else 'Medium' if avg_success_rate > 85 else 'Low':>20}  ║
║  System Stability:       {'Stable' if abs(trend) < 2 else 'Variable':>20}  ║
║  Performance:            {'Good' if avg_execution_time < 60 else 'Acceptable' if avg_execution_time < 120 else 'Needs Improvement':>20}  ║
║                                                                              ║
║  💡 RECOMMENDATIONS                                                          ║
║  ──────────────────────────────────────────────────────────────────────────  ║
"""
        
        # Add recommendations
        if avg_success_rate < 90:
            report += "║  • Focus on improving test success rate\n"
        if trend < -5:
            report += "║  • Investigate declining success rate trend\n"
        if avg_execution_time > 120:
            report += "║  • Optimize test execution time\n"
        if total_runs < 10:
            report += "║  • Increase test execution frequency\n"
        
        report += """║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        return report
    
    def _load_history(self) -> List[Dict]:
        """Load test history"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    
    def generate_summary_slide(self) -> str:
        """Generate one-page summary slide"""
        history = self._load_history()
        
        if not history:
            return "No data available."
        
        recent = history[-10:] if len(history) >= 10 else history
        
        avg_success = sum(r.get('success_rate', 0) for r in recent) / len(recent)
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        
        return f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEST QUALITY SUMMARY                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Recent Performance (Last {len(recent)} Runs)                                │
│  ────────────────────────────────────────────────────────────────────────  │
│  Average Success Rate:  {avg_success:.1f}%                                    │
│  Total Tests Run:       {total_tests}                                          │
│                                                                             │
│  Status: {'✅ EXCELLENT' if avg_success > 95 else '⚠️  GOOD' if avg_success > 85 else '❌ NEEDS ATTENTION'}                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    generator = ExecutiveReportGenerator(project_root)
    
    # Generate executive report
    report = generator.generate_executive_report(period_days=30)
    print(report)
    
    # Generate summary slide
    summary = generator.generate_summary_slide()
    print("\n" + summary)
    
    # Save reports
    report_file = project_root / "executive_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
        f.write("\n" + summary)
    
    print(f"\n📄 Executive report saved to: {report_file}")

if __name__ == "__main__":
    main()







