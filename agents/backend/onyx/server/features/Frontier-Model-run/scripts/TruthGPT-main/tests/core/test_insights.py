"""
Test Insights
Generate actionable insights from test data
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean
from collections import defaultdict

class TestInsights:
    """Generate actionable insights"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def generate_insights(self, lookback_days: int = 30) -> Dict:
        """Generate actionable insights"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        insights = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'key_insights': [],
            'actionable_items': [],
            'trends': [],
            'opportunities': []
        }
        
        # Analyze success rate
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates) if success_rates else 0
        
        if avg_success >= 98:
            insights['key_insights'].append({
                'type': 'excellence',
                'title': 'Exceptional Test Quality',
                'description': f'Success rate of {avg_success:.1f}% indicates excellent test quality',
                'action': 'Maintain current standards'
            })
        elif avg_success < 90:
            insights['key_insights'].append({
                'type': 'improvement',
                'title': 'Test Quality Needs Attention',
                'description': f'Success rate of {avg_success:.1f}% is below optimal',
                'action': 'Focus on fixing failing tests'
            })
            insights['actionable_items'].append({
                'priority': 'high',
                'action': 'Investigate and fix failing tests',
                'expected_impact': 'Improve success rate to 95%+'
            })
        
        # Analyze execution time
        execution_times = [r.get('execution_time', 0) for r in recent]
        avg_time = mean(execution_times) if execution_times else 0
        
        if avg_time > 300:
            insights['actionable_items'].append({
                'priority': 'medium',
                'action': 'Optimize test execution time',
                'expected_impact': f'Reduce from {avg_time:.0f}s to <120s'
            })
        
        # Analyze trends
        if len(recent) >= 5:
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]
            
            first_success = mean([r.get('success_rate', 0) for r in first_half])
            second_success = mean([r.get('success_rate', 0) for r in second_half])
            
            if second_success > first_success + 2:
                insights['trends'].append({
                    'type': 'positive',
                    'description': f'Success rate improving: {first_success:.1f}% → {second_success:.1f}%',
                    'confidence': 'high'
                })
            elif second_success < first_success - 2:
                insights['trends'].append({
                    'type': 'negative',
                    'description': f'Success rate declining: {first_success:.1f}% → {second_success:.1f}%',
                    'confidence': 'high'
                })
        
        # Identify opportunities
        if avg_success >= 95 and avg_time < 120:
            insights['opportunities'].append({
                'type': 'optimization',
                'description': 'Test suite is performing well - consider adding more tests',
                'benefit': 'Increase coverage and confidence'
            })
        
        return insights
    
    def generate_insights_report(self, insights: Dict) -> str:
        """Generate insights report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST INSIGHTS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in insights:
            lines.append(f"❌ {insights['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {insights['period']}")
        lines.append(f"Total Runs: {insights['total_runs']}")
        lines.append("")
        
        if insights['key_insights']:
            lines.append("💡 KEY INSIGHTS")
            lines.append("-" * 80)
            for insight in insights['key_insights']:
                emoji = "✅" if insight['type'] == 'excellence' else "⚠️"
                lines.append(f"\n{emoji} {insight['title']}")
                lines.append(f"   {insight['description']}")
                lines.append(f"   Action: {insight['action']}")
            lines.append("")
        
        if insights['actionable_items']:
            lines.append("🎯 ACTIONABLE ITEMS")
            lines.append("-" * 80)
            priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for item in insights['actionable_items']:
                emoji = priority_emoji.get(item['priority'], '⚪')
                lines.append(f"\n{emoji} [{item['priority'].upper()}] {item['action']}")
                lines.append(f"   Expected Impact: {item['expected_impact']}")
            lines.append("")
        
        if insights['trends']:
            lines.append("📈 TRENDS")
            lines.append("-" * 80)
            for trend in insights['trends']:
                emoji = "📈" if trend['type'] == 'positive' else "📉"
                lines.append(f"{emoji} {trend['description']}")
            lines.append("")
        
        if insights['opportunities']:
            lines.append("🚀 OPPORTUNITIES")
            lines.append("-" * 80)
            for opp in insights['opportunities']:
                lines.append(f"• {opp['description']}")
                lines.append(f"  Benefit: {opp['benefit']}")
        
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
    
    insights = TestInsights(project_root)
    insights_data = insights.generate_insights(lookback_days=30)
    
    report = insights.generate_insights_report(insights_data)
    print(report)
    
    # Save report
    report_file = project_root / "insights_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Insights report saved to: {report_file}")

if __name__ == "__main__":
    main()







