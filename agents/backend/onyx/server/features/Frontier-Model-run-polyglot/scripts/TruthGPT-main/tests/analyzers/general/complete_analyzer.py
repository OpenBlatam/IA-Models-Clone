"""
Complete Analyzer
Comprehensive analysis of test results
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, median, stdev
from collections import defaultdict

class CompleteAnalyzer:
    """Complete analysis of test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def complete_analysis(self, lookback_days: int = 30) -> Dict:
        """Perform complete analysis"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Extract all metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        # Comprehensive statistics
        analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'success_rate': {
                'mean': mean(success_rates) if success_rates else 0,
                'median': median(success_rates) if len(success_rates) > 1 else (success_rates[0] if success_rates else 0),
                'stdev': stdev(success_rates) if len(success_rates) > 1 else 0,
                'min': min(success_rates) if success_rates else 0,
                'max': max(success_rates) if success_rates else 0,
                'trend': self._calculate_trend(success_rates)
            },
            'execution_time': {
                'mean': mean(execution_times) if execution_times else 0,
                'median': median(execution_times) if len(execution_times) > 1 else (execution_times[0] if execution_times else 0),
                'stdev': stdev(execution_times) if len(execution_times) > 1 else 0,
                'min': min(execution_times) if execution_times else 0,
                'max': max(execution_times) if execution_times else 0
            },
            'total_tests': {
                'mean': mean(total_tests) if total_tests else 0,
                'median': median(total_tests) if len(total_tests) > 1 else (total_tests[0] if total_tests else 0),
                'stdev': stdev(total_tests) if len(total_tests) > 1 else 0,
                'min': min(total_tests) if total_tests else 0,
                'max': max(total_tests) if total_tests else 0
            },
            'failures': {
                'mean': mean(failures) if failures else 0,
                'median': median(failures) if len(failures) > 1 else (failures[0] if failures else 0),
                'stdev': stdev(failures) if len(failures) > 1 else 0,
                'min': min(failures) if failures else 0,
                'max': max(failures) if failures else 0
            },
            'health_score': self._calculate_health_score(recent),
            'recommendations': self._generate_recommendations(recent)
        }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend"""
        if len(values) < 2:
            return "insufficient_data"
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = mean(first_half)
        second_avg = mean(second_half)
        
        change = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        if abs(change) < 1:
            return "stable"
        elif change > 0:
            return f"improving (+{change:.1f}%)"
        else:
            return f"declining ({change:.1f}%)"
    
    def _calculate_health_score(self, recent: List[Dict]) -> float:
        """Calculate overall health score"""
        if not recent:
            return 0.0
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates)
        
        # Simple health score based on success rate
        health_score = min(100, max(0, avg_success))
        
        return round(health_score, 1)
    
    def _generate_recommendations(self, recent: List[Dict]) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates) if success_rates else 0
        
        if avg_success < 90:
            recommendations.append("Improve test success rate - focus on fixing failing tests")
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        avg_time = mean(execution_times) if execution_times else 0
        
        if avg_time > 300:
            recommendations.append("Optimize test execution time - consider parallel execution")
        
        if len(success_rates) > 1:
            variance = max(success_rates) - min(success_rates)
            if variance > 15:
                recommendations.append("Improve test stability - reduce variance in results")
        
        return recommendations
    
    def generate_complete_report(self, analysis: Dict) -> str:
        """Generate complete analysis report"""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPLETE TEST ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append(f"Health Score: {analysis['health_score']}/100")
        lines.append("")
        
        def format_metric(name, data):
            lines.append(f"📊 {name.upper().replace('_', ' ')}")
            lines.append("-" * 80)
            lines.append(f"Mean:    {data['mean']:.2f}")
            lines.append(f"Median:  {data['median']:.2f}")
            lines.append(f"Std Dev: {data['stdev']:.2f}")
            lines.append(f"Min:     {data['min']:.2f}")
            lines.append(f"Max:     {data['max']:.2f}")
            if 'trend' in data:
                lines.append(f"Trend:   {data['trend']}")
            lines.append("")
        
        format_metric("Success Rate", analysis['success_rate'])
        format_metric("Execution Time", analysis['execution_time'])
        format_metric("Total Tests", analysis['total_tests'])
        format_metric("Failures", analysis['failures'])
        
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
    
    analyzer = CompleteAnalyzer(project_root)
    analysis = analyzer.complete_analysis(lookback_days=30)
    
    report = analyzer.generate_complete_report(analysis)
    print(report)
    
    # Save analysis
    analysis_file = project_root / "complete_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n📄 Complete analysis saved to: {analysis_file}")

if __name__ == "__main__":
    main()







