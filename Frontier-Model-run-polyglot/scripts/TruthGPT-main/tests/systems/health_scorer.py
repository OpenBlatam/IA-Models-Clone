"""
Test Health Scorer
Calculates health scores for test suites
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from statistics import mean

class TestHealthScorer:
    """Calculate health scores for test suites"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def calculate_health_score(self, lookback_days: int = 30) -> Dict:
        """Calculate overall health score"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate component scores
        success_score = self._calculate_success_score(recent)
        stability_score = self._calculate_stability_score(recent)
        performance_score = self._calculate_performance_score(recent)
        coverage_score = self._calculate_coverage_score(recent)
        consistency_score = self._calculate_consistency_score(recent)
        
        # Weighted overall score
        weights = {
            'success': 0.30,
            'stability': 0.25,
            'performance': 0.20,
            'coverage': 0.15,
            'consistency': 0.10
        }
        
        overall_score = (
            success_score * weights['success'] +
            stability_score * weights['stability'] +
            performance_score * weights['performance'] +
            coverage_score * weights['coverage'] +
            consistency_score * weights['consistency']
        )
        
        # Determine health level
        if overall_score >= 90:
            health_level = "Excellent"
            health_emoji = "🟢"
        elif overall_score >= 75:
            health_level = "Good"
            health_emoji = "🟡"
        elif overall_score >= 60:
            health_level = "Fair"
            health_emoji = "🟠"
        else:
            health_level = "Poor"
            health_emoji = "🔴"
        
        return {
            'overall_score': round(overall_score, 1),
            'health_level': health_level,
            'health_emoji': health_emoji,
            'components': {
                'success': round(success_score, 1),
                'stability': round(stability_score, 1),
                'performance': round(performance_score, 1),
                'coverage': round(coverage_score, 1),
                'consistency': round(consistency_score, 1)
            },
            'runs_analyzed': len(recent),
            'period_days': lookback_days
        }
    
    def _calculate_success_score(self, recent: List[Dict]) -> float:
        """Calculate success rate score"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates) if success_rates else 0
        return min(100, avg_success)  # Cap at 100
    
    def _calculate_stability_score(self, recent: List[Dict]) -> float:
        """Calculate stability score (low variance)"""
        if len(recent) < 2:
            return 50.0
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        variance = max(success_rates) - min(success_rates)
        
        # Lower variance = higher score
        stability = max(0, 100 - (variance * 2))
        return stability
    
    def _calculate_performance_score(self, recent: List[Dict]) -> float:
        """Calculate performance score (execution time)"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        avg_time = mean(execution_times) if execution_times else 0
        
        # Score based on execution time (lower is better)
        if avg_time <= 60:
            return 100
        elif avg_time <= 120:
            return 80
        elif avg_time <= 300:
            return 60
        elif avg_time <= 600:
            return 40
        else:
            return 20
    
    def _calculate_coverage_score(self, recent: List[Dict]) -> float:
        """Calculate coverage score"""
        # Simplified: based on total tests
        total_tests = [r.get('total_tests', 0) for r in recent]
        avg_tests = mean(total_tests) if total_tests else 0
        
        # More tests = better coverage (capped)
        if avg_tests >= 200:
            return 100
        elif avg_tests >= 150:
            return 80
        elif avg_tests >= 100:
            return 60
        elif avg_tests >= 50:
            return 40
        else:
            return 20
    
    def _calculate_consistency_score(self, recent: List[Dict]) -> float:
        """Calculate consistency score"""
        if len(recent) < 5:
            return 50.0
        
        # Check if runs are consistent
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg = mean(success_rates)
        
        # Count how many are within 5% of average
        within_range = sum(1 for sr in success_rates if abs(sr - avg) <= 5)
        consistency = (within_range / len(success_rates)) * 100
        
        return consistency
    
    def generate_health_report(self, score_data: Dict) -> str:
        """Generate health score report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST SUITE HEALTH SCORE")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in score_data:
            lines.append(f"❌ {score_data['error']}")
            return "\n".join(lines)
        
        lines.append(f"{score_data['health_emoji']} Overall Health: {score_data['health_level']}")
        lines.append(f"   Score: {score_data['overall_score']}/100")
        lines.append(f"   Period: Last {score_data['period_days']} days")
        lines.append(f"   Runs Analyzed: {score_data['runs_analyzed']}")
        lines.append("")
        
        lines.append("📊 COMPONENT SCORES")
        lines.append("-" * 80)
        components = score_data['components']
        
        lines.append(f"Success Rate:      {components['success']}/100")
        lines.append(f"Stability:          {components['stability']}/100")
        lines.append(f"Performance:       {components['performance']}/100")
        lines.append(f"Coverage:          {components['coverage']}/100")
        lines.append(f"Consistency:       {components['consistency']}/100")
        lines.append("")
        
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 80)
        
        if components['success'] < 80:
            lines.append("• Improve test success rate")
        if components['stability'] < 80:
            lines.append("• Reduce test result variance")
        if components['performance'] < 80:
            lines.append("• Optimize test execution time")
        if components['coverage'] < 80:
            lines.append("• Increase test coverage")
        if components['consistency'] < 80:
            lines.append("• Improve test consistency")
        
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
    
    scorer = TestHealthScorer(project_root)
    score_data = scorer.calculate_health_score(lookback_days=30)
    
    report = scorer.generate_health_report(score_data)
    print(report)
    
    # Save report
    report_file = project_root / "health_score_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Health score report saved to: {report_file}")

if __name__ == "__main__":
    main()







