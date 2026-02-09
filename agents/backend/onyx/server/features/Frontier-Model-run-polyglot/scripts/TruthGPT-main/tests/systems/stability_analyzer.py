"""
Stability Analyzer
Analyze test stability over time
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev, variance

class StabilityAnalyzer:
    """Analyze test stability"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_stability(self, lookback_days: int = 30) -> Dict:
        """Analyze test stability"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate stability metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        stability_score = self._calculate_stability_score(success_rates, execution_times)
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'stability_score': stability_score,
            'success_rate_stability': self._calculate_metric_stability(success_rates),
            'execution_time_stability': self._calculate_metric_stability(execution_times),
            'stability_trend': self._analyze_stability_trend(recent),
            'recommendations': self._generate_stability_recommendations(stability_score)
        }
    
    def _calculate_stability_score(self, success_rates: List[float], execution_times: List[float]) -> float:
        """Calculate overall stability score"""
        if not success_rates or not execution_times:
            return 0.0
        
        # Success rate stability (0-50 points)
        sr_mean = mean(success_rates)
        sr_std = stdev(success_rates) if len(success_rates) > 1 else 0
        sr_stability = max(0, 50 - (sr_std * 10))
        
        # Execution time stability (0-50 points)
        et_mean = mean(execution_times)
        et_std = stdev(execution_times) if len(execution_times) > 1 else 0
        et_cv = (et_std / et_mean * 100) if et_mean > 0 else 100
        et_stability = max(0, 50 - (et_cv * 0.5))
        
        return round((sr_stability + et_stability), 1)
    
    def _calculate_metric_stability(self, values: List[float]) -> Dict:
        """Calculate stability for a metric"""
        if not values:
            return {'stable': False, 'variance': 0, 'coefficient_of_variation': 0}
        
        mean_val = mean(values)
        std_val = stdev(values) if len(values) > 1 else 0
        cv = (std_val / mean_val * 100) if mean_val > 0 else 0
        
        # Consider stable if CV < 10%
        stable = cv < 10
        
        return {
            'stable': stable,
            'mean': round(mean_val, 2),
            'std': round(std_val, 2),
            'variance': round(variance(values) if len(values) > 1 else 0, 2),
            'coefficient_of_variation': round(cv, 2)
        }
    
    def _analyze_stability_trend(self, recent: List[Dict]) -> Dict:
        """Analyze stability trend"""
        if len(recent) < 4:
            return {}
        
        # Split into two halves
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_sr = [r.get('success_rate', 0) for r in first_half]
        second_sr = [r.get('success_rate', 0) for r in second_half]
        
        first_stability = 100 - (stdev(first_sr) * 10) if len(first_sr) > 1 else 100
        second_stability = 100 - (stdev(second_sr) * 10) if len(second_sr) > 1 else 100
        
        trend = second_stability - first_stability
        
        return {
            'trend': round(trend, 1),
            'direction': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable',
            'first_half_stability': round(first_stability, 1),
            'second_half_stability': round(second_stability, 1)
        }
    
    def _generate_stability_recommendations(self, stability_score: float) -> List[str]:
        """Generate stability recommendations"""
        recommendations = []
        
        if stability_score < 50:
            recommendations.append("Critical: Test suite is highly unstable - investigate root causes")
            recommendations.append("Review and fix flaky tests")
            recommendations.append("Ensure consistent test environment")
        elif stability_score < 70:
            recommendations.append("Test suite shows moderate instability")
            recommendations.append("Identify and address sources of variance")
            recommendations.append("Consider test isolation improvements")
        elif stability_score < 85:
            recommendations.append("Test suite is relatively stable")
            recommendations.append("Minor improvements could enhance stability further")
        else:
            recommendations.append("Excellent stability - maintain current practices")
        
        return recommendations
    
    def generate_stability_report(self, analysis: Dict) -> str:
        """Generate stability report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST STABILITY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        score_emoji = "🟢" if analysis['stability_score'] >= 85 else "🟡" if analysis['stability_score'] >= 70 else "🔴"
        lines.append(f"{score_emoji} Overall Stability Score: {analysis['stability_score']}/100")
        lines.append("")
        
        lines.append("📊 SUCCESS RATE STABILITY")
        lines.append("-" * 80)
        sr_stab = analysis['success_rate_stability']
        status = "✅ Stable" if sr_stab['stable'] else "⚠️ Unstable"
        lines.append(f"{status}")
        lines.append(f"Mean: {sr_stab['mean']}%")
        lines.append(f"Std Dev: {sr_stab['std']}")
        lines.append(f"Coefficient of Variation: {sr_stab['coefficient_of_variation']}%")
        lines.append("")
        
        lines.append("⏱️ EXECUTION TIME STABILITY")
        lines.append("-" * 80)
        et_stab = analysis['execution_time_stability']
        status = "✅ Stable" if et_stab['stable'] else "⚠️ Unstable"
        lines.append(f"{status}")
        lines.append(f"Mean: {et_stab['mean']}s")
        lines.append(f"Std Dev: {et_stab['std']}s")
        lines.append(f"Coefficient of Variation: {et_stab['coefficient_of_variation']}%")
        lines.append("")
        
        if 'stability_trend' in analysis and analysis['stability_trend']:
            trend = analysis['stability_trend']
            trend_emoji = "📈" if trend['direction'] == 'improving' else "📉" if trend['direction'] == 'declining' else "➡️"
            lines.append(f"{trend_emoji} STABILITY TREND")
            lines.append("-" * 80)
            lines.append(f"Direction: {trend['direction'].title()}")
            lines.append(f"Trend: {trend['trend']:+.1f}")
            lines.append(f"First Half: {trend['first_half_stability']:.1f}")
            lines.append(f"Second Half: {trend['second_half_stability']:.1f}")
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
    
    analyzer = StabilityAnalyzer(project_root)
    analysis = analyzer.analyze_stability(lookback_days=30)
    
    report = analyzer.generate_stability_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "stability_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Stability analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







