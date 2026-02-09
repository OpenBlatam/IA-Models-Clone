"""
Advanced Failure Predictor
Advanced failure prediction using multiple factors
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev

class AdvancedFailurePredictor:
    """Advanced failure prediction"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def predict_failures(self, lookback_days: int = 30) -> Dict:
        """Predict potential failures"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Analyze failure patterns
        failure_rates = []
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        for r in recent:
            total = r.get('total_tests', 0)
            failures = r.get('failures', 0) + r.get('errors', 0)
            failure_rate = (failures / total * 100) if total > 0 else 0
            failure_rates.append(failure_rate)
        
        # Calculate risk factors
        avg_failure_rate = mean(failure_rates) if failure_rates else 0
        failure_rate_trend = self._calculate_trend(failure_rates)
        success_rate_trend = self._calculate_trend(success_rates)
        execution_time_trend = self._calculate_trend(execution_times)
        
        # Predict failure probability
        risk_score = self._calculate_risk_score(
            avg_failure_rate,
            failure_rate_trend,
            success_rate_trend,
            execution_time_trend
        )
        
        failure_probability = min(100, max(0, risk_score))
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'current_failure_rate': round(avg_failure_rate, 2),
            'failure_rate_trend': failure_rate_trend,
            'success_rate_trend': success_rate_trend,
            'execution_time_trend': execution_time_trend,
            'risk_score': round(risk_score, 1),
            'failure_probability': round(failure_probability, 1),
            'risk_level': self._determine_risk_level(failure_probability),
            'recommendations': self._generate_recommendations(risk_score, failure_rate_trend)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict:
        """Calculate trend"""
        if len(values) < 2:
            return {'direction': 'stable', 'slope': 0}
        
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        
        if abs(slope) < 0.01:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'slope': round(slope, 4)
        }
    
    def _calculate_risk_score(self, avg_failure_rate: float, failure_trend: Dict, success_trend: Dict, execution_trend: Dict) -> float:
        """Calculate risk score"""
        # Base risk from current failure rate
        base_risk = min(100, avg_failure_rate * 10)
        
        # Trend adjustments
        if failure_trend['direction'] == 'increasing':
            base_risk += 20
        elif failure_trend['direction'] == 'decreasing':
            base_risk -= 10
        
        if success_trend['direction'] == 'decreasing':
            base_risk += 15
        
        if execution_trend['direction'] == 'increasing':
            base_risk += 5  # Longer execution might indicate issues
        
        return max(0, min(100, base_risk))
    
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level"""
        if probability >= 70:
            return 'high'
        elif probability >= 40:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, risk_score: float, failure_trend: Dict) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        if risk_score >= 70:
            recommendations.append("High failure risk detected - immediate action required")
            recommendations.append("Review recent test failures and fix root causes")
            recommendations.append("Consider increasing test coverage")
        elif risk_score >= 40:
            recommendations.append("Moderate failure risk - monitor closely")
            recommendations.append("Address any known flaky tests")
        else:
            recommendations.append("Low failure risk - maintain current practices")
        
        if failure_trend['direction'] == 'increasing':
            recommendations.append("Failure rate is trending upward - investigate causes")
        
        return recommendations
    
    def generate_prediction_report(self, prediction: Dict) -> str:
        """Generate prediction report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED FAILURE PREDICTION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in prediction:
            lines.append(f"❌ {prediction['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {prediction['period']}")
        lines.append(f"Total Runs: {prediction['total_runs']}")
        lines.append("")
        
        risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        emoji = risk_emoji.get(prediction['risk_level'], '⚪')
        lines.append(f"{emoji} Risk Level: {prediction['risk_level'].upper()}")
        lines.append(f"Failure Probability: {prediction['failure_probability']}%")
        lines.append(f"Risk Score: {prediction['risk_score']}/100")
        lines.append("")
        
        lines.append("📊 CURRENT METRICS")
        lines.append("-" * 80)
        lines.append(f"Current Failure Rate: {prediction['current_failure_rate']}%")
        lines.append("")
        
        lines.append("📈 TRENDS")
        lines.append("-" * 80)
        lines.append(f"Failure Rate Trend: {prediction['failure_rate_trend']['direction']}")
        lines.append(f"Success Rate Trend: {prediction['success_rate_trend']['direction']}")
        lines.append(f"Execution Time Trend: {prediction['execution_time_trend']['direction']}")
        lines.append("")
        
        if prediction['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in prediction['recommendations']:
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
    
    predictor = AdvancedFailurePredictor(project_root)
    prediction = predictor.predict_failures(lookback_days=30)
    
    report = predictor.generate_prediction_report(prediction)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_failure_prediction_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced failure prediction report saved to: {report_file}")

if __name__ == "__main__":
    main()







