"""
Recommender System
Intelligent recommendations for test improvements
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class RecommenderSystem:
    """Intelligent recommendations for test improvements"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def generate_recommendations(self, lookback_days: int = 30) -> Dict:
        """Generate intelligent recommendations"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        recommendations = []
        priority_scores = {}
        
        # Analyze and generate recommendations
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates) if success_rates else 0
        
        if avg_success < 95:
            rec = {
                'category': 'Quality',
                'title': 'Improve Test Success Rate',
                'description': f'Current success rate is {avg_success:.1f}%, target is 95%+',
                'priority': 'high',
                'impact': 'high',
                'effort': 'medium',
                'actions': [
                    'Fix failing tests',
                    'Investigate root causes',
                    'Improve test stability'
                ]
            }
            recommendations.append(rec)
            priority_scores[rec['title']] = 90
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        avg_time = mean(execution_times) if execution_times else 0
        
        if avg_time > 300:
            rec = {
                'category': 'Performance',
                'title': 'Optimize Test Execution Time',
                'description': f'Average execution time is {avg_time:.1f}s, consider optimization',
                'priority': 'medium',
                'impact': 'high',
                'effort': 'medium',
                'actions': [
                    'Enable parallel execution',
                    'Optimize slow tests',
                    'Remove redundant tests'
                ]
            }
            recommendations.append(rec)
            priority_scores[rec['title']] = 75
        
        if len(success_rates) > 1:
            variance = max(success_rates) - min(success_rates)
            if variance > 10:
                rec = {
                    'category': 'Stability',
                    'title': 'Reduce Test Result Variance',
                    'description': f'High variance ({variance:.1f}%) in success rates',
                    'priority': 'high',
                    'impact': 'medium',
                    'effort': 'high',
                    'actions': [
                        'Fix flaky tests',
                        'Improve test isolation',
                        'Use deterministic test data'
                    ]
                }
                recommendations.append(rec)
                priority_scores[rec['title']] = 85
        
        # Sort by priority score
        recommendations.sort(key=lambda x: priority_scores.get(x['title'], 50), reverse=True)
        
        return {
            'period': f'Last {lookback_days} days',
            'total_recommendations': len(recommendations),
            'recommendations': recommendations,
            'priority_breakdown': {
                'high': len([r for r in recommendations if r['priority'] == 'high']),
                'medium': len([r for r in recommendations if r['priority'] == 'medium']),
                'low': len([r for r in recommendations if r['priority'] == 'low'])
            }
        }
    
    def generate_recommendations_report(self, recommendations: Dict) -> str:
        """Generate recommendations report"""
        lines = []
        lines.append("=" * 80)
        lines.append("INTELLIGENT RECOMMENDATIONS")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in recommendations:
            lines.append(f"❌ {recommendations['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {recommendations['period']}")
        lines.append(f"Total Recommendations: {recommendations['total_recommendations']}")
        lines.append("")
        
        priority_breakdown = recommendations['priority_breakdown']
        lines.append("📊 PRIORITY BREAKDOWN")
        lines.append("-" * 80)
        lines.append(f"High:   {priority_breakdown['high']}")
        lines.append(f"Medium: {priority_breakdown['medium']}")
        lines.append(f"Low:    {priority_breakdown['low']}")
        lines.append("")
        
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 80)
        
        priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        
        for i, rec in enumerate(recommendations['recommendations'], 1):
            emoji = priority_emoji.get(rec['priority'], '⚪')
            lines.append(f"\n{i}. {emoji} {rec['title']} ({rec['priority'].upper()})")
            lines.append(f"   Category: {rec['category']}")
            lines.append(f"   Description: {rec['description']}")
            lines.append(f"   Impact: {rec['impact']} | Effort: {rec['effort']}")
            lines.append("   Actions:")
            for action in rec['actions']:
                lines.append(f"     • {action}")
        
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
    
    recommender = RecommenderSystem(project_root)
    recommendations = recommender.generate_recommendations(lookback_days=30)
    
    report = recommender.generate_recommendations_report(recommendations)
    print(report)
    
    # Save report
    report_file = project_root / "recommendations_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Recommendations report saved to: {report_file}")

if __name__ == "__main__":
    main()







