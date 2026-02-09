"""
Advanced Recommendation Engine
Advanced recommendation system with ML-like scoring
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class AdvancedRecommendationEngine:
    """Advanced recommendation engine"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def generate_recommendations(self, lookback_days: int = 30) -> Dict:
        """Generate advanced recommendations"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Analyze current state
        current_state = self._analyze_current_state(recent)
        
        # Generate recommendations
        recommendations = []
        
        # Success rate recommendations
        if current_state['avg_success_rate'] < 95:
            recommendations.append({
                'category': 'success_rate',
                'priority': 'high',
                'title': 'Improve Test Success Rate',
                'description': f"Current success rate is {current_state['avg_success_rate']:.1f}%, target is 95%+",
                'impact_score': self._calculate_impact_score('success_rate', current_state),
                'effort': 'medium',
                'actions': [
                    'Identify and fix failing tests',
                    'Review test flakiness',
                    'Improve test stability'
                ]
            })
        
        # Execution time recommendations
        if current_state['avg_execution_time'] > 300:
            recommendations.append({
                'category': 'execution_time',
                'priority': 'medium',
                'title': 'Optimize Test Execution Time',
                'description': f"Average execution time is {current_state['avg_execution_time']:.0f}s, target is <120s",
                'impact_score': self._calculate_impact_score('execution_time', current_state),
                'effort': 'high',
                'actions': [
                    'Enable parallel test execution',
                    'Optimize slow tests',
                    'Consider test sharding'
                ]
            })
        
        # Failure rate recommendations
        if current_state['failure_rate'] > 5:
            recommendations.append({
                'category': 'failure_rate',
                'priority': 'high',
                'title': 'Reduce Test Failure Rate',
                'description': f"Failure rate is {current_state['failure_rate']:.1f}%, target is <5%",
                'impact_score': self._calculate_impact_score('failure_rate', current_state),
                'effort': 'medium',
                'actions': [
                    'Fix broken tests',
                    'Address flaky tests',
                    'Improve test reliability'
                ]
            })
        
        # Stability recommendations
        if current_state['stability_score'] < 80:
            recommendations.append({
                'category': 'stability',
                'priority': 'medium',
                'title': 'Improve Test Stability',
                'description': f"Stability score is {current_state['stability_score']:.1f}/100, target is 80+",
                'impact_score': self._calculate_impact_score('stability', current_state),
                'effort': 'high',
                'actions': [
                    'Identify sources of variance',
                    'Improve test isolation',
                    'Ensure consistent test environment'
                ]
            })
        
        # Sort by priority and impact
        recommendations.sort(key=lambda x: (
            {'high': 0, 'medium': 1, 'low': 2}[x['priority']],
            -x['impact_score']
        ))
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'current_state': current_state,
            'total_recommendations': len(recommendations),
            'recommendations': recommendations,
            'top_recommendations': recommendations[:5]
        }
    
    def _analyze_current_state(self, recent: List[Dict]) -> Dict:
        """Analyze current state"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        # Calculate stability
        sr_std = stdev(success_rates) if len(success_rates) > 1 else 0
        stability_score = max(0, 100 - (sr_std * 10))
        
        return {
            'avg_success_rate': round(mean(success_rates), 2) if success_rates else 0,
            'avg_execution_time': round(mean(execution_times), 2) if execution_times else 0,
            'failure_rate': round((sum(failures) / sum(total_tests) * 100) if sum(total_tests) > 0 else 0, 2),
            'stability_score': round(stability_score, 1)
        }
    
    def _calculate_impact_score(self, category: str, state: Dict) -> float:
        """Calculate impact score for a recommendation"""
        if category == 'success_rate':
            gap = 95 - state['avg_success_rate']
            return min(100, max(0, gap * 2))
        elif category == 'execution_time':
            gap = state['avg_execution_time'] - 120
            return min(100, max(0, gap / 2))
        elif category == 'failure_rate':
            gap = state['failure_rate'] - 5
            return min(100, max(0, gap * 5))
        elif category == 'stability':
            gap = 80 - state['stability_score']
            return min(100, max(0, gap))
        return 50
    
    def generate_recommendations_report(self, recommendations: Dict) -> str:
        """Generate recommendations report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED RECOMMENDATIONS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in recommendations:
            lines.append(f"❌ {recommendations['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {recommendations['period']}")
        lines.append(f"Total Runs: {recommendations['total_runs']}")
        lines.append(f"Total Recommendations: {recommendations['total_recommendations']}")
        lines.append("")
        
        lines.append("📊 CURRENT STATE")
        lines.append("-" * 80)
        state = recommendations['current_state']
        lines.append(f"Average Success Rate: {state['avg_success_rate']}%")
        lines.append(f"Average Execution Time: {state['avg_execution_time']}s")
        lines.append(f"Failure Rate: {state['failure_rate']}%")
        lines.append(f"Stability Score: {state['stability_score']}/100")
        lines.append("")
        
        if recommendations['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            effort_emoji = {'low': '⚡', 'medium': '🔧', 'high': '🛠️'}
            
            for i, rec in enumerate(recommendations['recommendations'], 1):
                p_emoji = priority_emoji.get(rec['priority'], '⚪')
                e_emoji = effort_emoji.get(rec['effort'], '⚪')
                lines.append(f"\n{i}. {p_emoji} [{rec['priority'].upper()}] {rec['title']}")
                lines.append(f"   {rec['description']}")
                lines.append(f"   Impact Score: {rec['impact_score']}/100")
                lines.append(f"   Effort: {e_emoji} {rec['effort'].title()}")
                lines.append(f"   Actions:")
                for action in rec['actions']:
                    lines.append(f"     • {action}")
            lines.append("")
        
        if recommendations['top_recommendations']:
            lines.append("⭐ TOP 5 RECOMMENDATIONS")
            lines.append("-" * 80)
            for i, rec in enumerate(recommendations['top_recommendations'], 1):
                lines.append(f"{i}. {rec['title']} (Impact: {rec['impact_score']}/100)")
        
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
    from statistics import stdev
    
    project_root = Path(__file__).parent.parent
    
    engine = AdvancedRecommendationEngine(project_root)
    recommendations = engine.generate_recommendations(lookback_days=30)
    
    report = engine.generate_recommendations_report(recommendations)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_recommendations_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced recommendations report saved to: {report_file}")

if __name__ == "__main__":
    main()







