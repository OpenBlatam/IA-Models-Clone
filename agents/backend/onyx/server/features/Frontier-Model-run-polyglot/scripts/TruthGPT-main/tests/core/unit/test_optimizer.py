"""
Test Optimizer
Provides optimization suggestions for test suite
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean

class TestOptimizer:
    """Optimize test suite performance and quality"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def analyze_optimization_opportunities(self) -> Dict:
        """Analyze optimization opportunities"""
        history = self._load_history()
        
        if len(history) < 5:
            return {'error': 'Insufficient data'}
        
        recent = history[-20:] if len(history) >= 20 else history
        
        opportunities = []
        
        # Analyze execution time
        execution_times = [r.get('execution_time', 0) for r in recent]
        avg_time = mean(execution_times)
        
        if avg_time > 300:
            opportunities.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'Reduce Execution Time',
                'description': f'Average execution time is {avg_time:.1f}s, consider parallel execution',
                'impact': 'high',
                'effort': 'medium',
                'suggestions': [
                    'Enable parallel test execution',
                    'Optimize slow tests',
                    'Remove redundant tests',
                    'Use test fixtures more efficiently'
                ]
            })
        
        # Analyze success rate
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates)
        
        if avg_success < 90:
            opportunities.append({
                'type': 'quality',
                'priority': 'high',
                'title': 'Improve Success Rate',
                'description': f'Success rate is {avg_success:.1f}%, focus on fixing failures',
                'impact': 'high',
                'effort': 'high',
                'suggestions': [
                    'Fix failing tests',
                    'Investigate flaky tests',
                    'Improve test stability',
                    'Review test assertions'
                ]
            })
        
        # Analyze consistency
        if len(success_rates) > 1:
            variance = max(success_rates) - min(success_rates)
            if variance > 10:
                opportunities.append({
                    'type': 'stability',
                    'priority': 'medium',
                    'title': 'Improve Consistency',
                    'description': f'High variance ({variance:.1f}%) in success rates',
                    'impact': 'medium',
                    'effort': 'medium',
                    'suggestions': [
                        'Fix flaky tests',
                        'Improve test isolation',
                        'Reduce test dependencies',
                        'Use deterministic test data'
                    ]
                })
        
        # Analyze test count
        total_tests = [r.get('total_tests', 0) for r in recent]
        avg_tests = mean(total_tests)
        
        if avg_tests < 100:
            opportunities.append({
                'type': 'coverage',
                'priority': 'medium',
                'title': 'Increase Test Coverage',
                'description': f'Average {avg_tests:.0f} tests per run, consider adding more',
                'impact': 'medium',
                'effort': 'high',
                'suggestions': [
                    'Add unit tests for uncovered code',
                    'Add integration tests',
                    'Add edge case tests',
                    'Improve test coverage metrics'
                ]
            })
        
        return {
            'opportunities': opportunities,
            'total_opportunities': len(opportunities),
            'high_priority': len([o for o in opportunities if o['priority'] == 'high']),
            'medium_priority': len([o for o in opportunities if o['priority'] == 'medium']),
            'low_priority': len([o for o in opportunities if o['priority'] == 'low'])
        }
    
    def generate_optimization_report(self, analysis: Dict) -> str:
        """Generate optimization report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST OPTIMIZATION OPPORTUNITIES")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Total Opportunities: {analysis['total_opportunities']}")
        lines.append(f"High Priority: {analysis['high_priority']}")
        lines.append(f"Medium Priority: {analysis['medium_priority']}")
        lines.append(f"Low Priority: {analysis['low_priority']}")
        lines.append("")
        
        # Group by priority
        by_priority = defaultdict(list)
        for opp in analysis['opportunities']:
            by_priority[opp['priority']].append(opp)
        
        priority_order = ['high', 'medium', 'low']
        priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        
        for priority in priority_order:
            if priority not in by_priority:
                continue
            
            lines.append(f"{priority_emoji[priority]} {priority.upper()} PRIORITY")
            lines.append("-" * 80)
            
            for opp in by_priority[priority]:
                lines.append(f"\n{opp['title']}")
                lines.append(f"Type: {opp['type']}")
                lines.append(f"Description: {opp['description']}")
                lines.append(f"Impact: {opp['impact']} | Effort: {opp['effort']}")
                lines.append("Suggestions:")
                for suggestion in opp['suggestions']:
                    lines.append(f"  • {suggestion}")
                lines.append("")
        
        lines.append("💡 OPTIMIZATION STRATEGY")
        lines.append("-" * 80)
        lines.append("1. Start with high-impact, low-effort optimizations")
        lines.append("2. Address high-priority issues first")
        lines.append("3. Monitor improvements after each optimization")
        lines.append("4. Iterate based on results")
        
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
    
    optimizer = TestOptimizer(project_root)
    analysis = optimizer.analyze_optimization_opportunities()
    
    report = optimizer.generate_optimization_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "optimization_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Optimization report saved to: {report_file}")

if __name__ == "__main__":
    main()







