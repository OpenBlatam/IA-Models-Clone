"""
Test Recommendation Engine
Provides recommendations based on test results analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime

class RecommendationEngine:
    """Generate recommendations based on test analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def generate_recommendations(self) -> List[Dict]:
        """Generate recommendations based on analysis"""
        history = self._load_history()
        
        if len(history) < 5:
            return [{
                'type': 'info',
                'priority': 'low',
                'message': 'Need more test runs for meaningful recommendations'
            }]
        
        recommendations = []
        
        # Analyze success rate
        success_rates = [r.get('success_rate', 0) for r in history[-10:]]
        avg_success = sum(success_rates) / len(success_rates)
        
        if avg_success < 90:
            recommendations.append({
                'type': 'success_rate',
                'priority': 'high',
                'message': f'Success rate is {avg_success:.1f}% - focus on fixing failing tests',
                'action': 'Review and fix failing tests, improve test stability'
            })
        elif avg_success < 95:
            recommendations.append({
                'type': 'success_rate',
                'priority': 'medium',
                'message': f'Success rate is {avg_success:.1f}% - room for improvement',
                'action': 'Investigate intermittent failures'
            })
        
        # Analyze execution time
        execution_times = [r.get('execution_time', 0) for r in history[-10:]]
        avg_time = sum(execution_times) / len(execution_times)
        
        if avg_time > 300:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'message': f'Average execution time is {avg_time:.1f}s - tests are slow',
                'action': 'Optimize slow tests, consider parallel execution'
            })
        elif avg_time > 120:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium',
                'message': f'Average execution time is {avg_time:.1f}s',
                'action': 'Profile tests to identify bottlenecks'
            })
        
        # Analyze consistency
        if len(success_rates) > 1:
            variance = max(success_rates) - min(success_rates)
            if variance > 10:
                recommendations.append({
                    'type': 'consistency',
                    'priority': 'medium',
                    'message': f'High variance in success rates ({variance:.1f}%)',
                    'action': 'Investigate flaky tests, improve test isolation'
                })
        
        # Analyze trend
        if len(history) >= 10:
            first_half = history[:len(history)//2]
            second_half = history[len(history)//2:]
            
            first_avg = sum(r.get('success_rate', 0) for r in first_half) / len(first_half)
            second_avg = sum(r.get('success_rate', 0) for r in second_half) / len(second_half)
            trend = second_avg - first_avg
            
            if trend < -5:
                recommendations.append({
                    'type': 'trend',
                    'priority': 'high',
                    'message': f'Success rate declining by {abs(trend):.1f}%',
                    'action': 'Investigate recent changes causing test failures'
                })
            elif trend > 5:
                recommendations.append({
                    'type': 'trend',
                    'priority': 'low',
                    'message': f'Success rate improving by {trend:.1f}%',
                    'action': 'Continue current practices'
                })
        
        # Coverage recommendations
        recommendations.append({
            'type': 'coverage',
            'priority': 'medium',
            'message': 'Run coverage analysis to identify untested code',
            'action': 'python -m tests.test_coverage'
        })
        
        # Flakiness recommendations
        recommendations.append({
            'type': 'quality',
            'priority': 'medium',
            'message': 'Check for flaky tests regularly',
            'action': 'python -m tests.test_flakiness_detector'
        })
        
        return recommendations
    
    def _load_history(self) -> List[Dict]:
        """Load test history"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    
    def generate_recommendations_report(self) -> str:
        """Generate recommendations report"""
        recommendations = self.generate_recommendations()
        
        lines = []
        lines.append("=" * 80)
        lines.append("TEST RECOMMENDATIONS")
        lines.append("=" * 80)
        lines.append("")
        
        # Group by priority
        by_priority = defaultdict(list)
        for rec in recommendations:
            by_priority[rec['priority']].append(rec)
        
        priority_order = ['high', 'medium', 'low', 'info']
        priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢', 'info': 'ℹ️'}
        
        for priority in priority_order:
            if priority not in by_priority:
                continue
            
            lines.append(f"{priority_emoji[priority]} {priority.upper()} PRIORITY")
            lines.append("-" * 80)
            
            for rec in by_priority[priority]:
                lines.append(f"Type: {rec['type']}")
                lines.append(f"Message: {rec['message']}")
                if 'action' in rec:
                    lines.append(f"Action: {rec['action']}")
                lines.append("")
        
        lines.append("💡 GENERAL RECOMMENDATIONS")
        lines.append("-" * 80)
        lines.append("1. Run tests regularly to catch issues early")
        lines.append("2. Monitor trends to identify regressions")
        lines.append("3. Keep test execution time reasonable")
        lines.append("4. Maintain high test coverage")
        lines.append("5. Fix flaky tests promptly")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    engine = RecommendationEngine(project_root)
    report = engine.generate_recommendations_report()
    
    print(report)
    
    # Save report
    report_file = project_root / "recommendations_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Recommendations report saved to: {report_file}")

if __name__ == "__main__":
    main()







