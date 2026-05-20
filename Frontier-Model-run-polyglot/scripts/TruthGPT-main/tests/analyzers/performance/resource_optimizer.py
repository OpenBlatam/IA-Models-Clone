"""
Resource Optimizer
Optimize test resource usage
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class ResourceOptimizer:
    """Optimize test resource usage"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def optimize_resources(self, lookback_days: int = 30) -> Dict:
        """Analyze and optimize resource usage"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Analyze resource usage
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        avg_time = mean(execution_times) if execution_times else 0
        avg_tests = mean(total_tests) if total_tests else 0
        
        # Calculate resource efficiency
        time_per_test = avg_time / avg_tests if avg_tests > 0 else 0
        tests_per_minute = (avg_tests / avg_time * 60) if avg_time > 0 else 0
        
        # Identify optimization opportunities
        optimizations = []
        
        if avg_time > 300:
            optimizations.append({
                'type': 'execution_time',
                'priority': 'high',
                'current': f"{avg_time:.0f}s",
                'target': '<120s',
                'suggestion': 'Consider parallel execution or test optimization'
            })
        
        if time_per_test > 5:
            optimizations.append({
                'type': 'test_duration',
                'priority': 'medium',
                'current': f"{time_per_test:.2f}s/test",
                'target': '<2s/test',
                'suggestion': 'Optimize slow individual tests'
            })
        
        if tests_per_minute < 10:
            optimizations.append({
                'type': 'throughput',
                'priority': 'medium',
                'current': f"{tests_per_minute:.1f} tests/min",
                'target': '>30 tests/min',
                'suggestion': 'Improve test execution throughput'
            })
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'resource_metrics': {
                'avg_execution_time': round(avg_time, 2),
                'avg_tests_per_run': round(avg_tests, 0),
                'time_per_test': round(time_per_test, 3),
                'tests_per_minute': round(tests_per_minute, 1)
            },
            'optimization_opportunities': optimizations,
            'optimization_score': self._calculate_optimization_score(avg_time, time_per_test, tests_per_minute)
        }
    
    def _calculate_optimization_score(self, avg_time: float, time_per_test: float, tests_per_minute: float) -> float:
        """Calculate optimization score"""
        # Score based on multiple factors
        time_score = max(0, 100 - (avg_time / 5))  # Penalize long execution times
        test_duration_score = max(0, 100 - (time_per_test * 20))  # Penalize slow tests
        throughput_score = min(100, tests_per_minute * 3)  # Reward high throughput
        
        # Weighted average
        score = (time_score * 0.4) + (test_duration_score * 0.3) + (throughput_score * 0.3)
        return round(score, 1)
    
    def generate_optimization_report(self, analysis: Dict) -> str:
        """Generate optimization report"""
        lines = []
        lines.append("=" * 80)
        lines.append("RESOURCE OPTIMIZATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        score_emoji = "🟢" if analysis['optimization_score'] >= 80 else "🟡" if analysis['optimization_score'] >= 60 else "🔴"
        lines.append(f"{score_emoji} Optimization Score: {analysis['optimization_score']}/100")
        lines.append("")
        
        lines.append("📊 RESOURCE METRICS")
        lines.append("-" * 80)
        metrics = analysis['resource_metrics']
        lines.append(f"Average Execution Time: {metrics['avg_execution_time']}s")
        lines.append(f"Average Tests per Run: {metrics['avg_tests_per_run']}")
        lines.append(f"Time per Test: {metrics['time_per_test']}s")
        lines.append(f"Tests per Minute: {metrics['tests_per_minute']}")
        lines.append("")
        
        if analysis['optimization_opportunities']:
            lines.append("🎯 OPTIMIZATION OPPORTUNITIES")
            lines.append("-" * 80)
            priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for opp in analysis['optimization_opportunities']:
                emoji = priority_emoji.get(opp['priority'], '⚪')
                lines.append(f"\n{emoji} [{opp['priority'].upper()}] {opp['type'].replace('_', ' ').title()}")
                lines.append(f"   Current: {opp['current']}")
                lines.append(f"   Target: {opp['target']}")
                lines.append(f"   Suggestion: {opp['suggestion']}")
            lines.append("")
        else:
            lines.append("✅ No significant optimization opportunities identified")
            lines.append("")
        
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
    
    optimizer = ResourceOptimizer(project_root)
    analysis = optimizer.optimize_resources(lookback_days=30)
    
    report = optimizer.generate_optimization_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "resource_optimization_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Resource optimization report saved to: {report_file}")

if __name__ == "__main__":
    main()







