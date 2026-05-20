"""
Test Failure Predictor
Predicts test failures based on historical patterns
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean

class TestFailurePredictor:
    """Predict test failures based on patterns"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def analyze_failure_patterns(self) -> Dict:
        """Analyze patterns in test failures"""
        history = self._load_history()
        
        if len(history) < 5:
            return {'error': 'Insufficient data for pattern analysis'}
        
        # Analyze failure patterns
        failure_rates = []
        failure_times = []
        failure_days = defaultdict(int)
        
        for run in history:
            total = run.get('total_tests', 0)
            failures = run.get('failed', 0) + run.get('errors', 0)
            
            if total > 0:
                failure_rate = (failures / total) * 100
                failure_rates.append(failure_rate)
                failure_times.append(run.get('execution_time', 0))
                
                # Day of week pattern
                try:
                    timestamp = run.get('timestamp', '')
                    if timestamp:
                        dt = datetime.fromisoformat(timestamp)
                        day_name = dt.strftime('%A')
                        failure_days[day_name] += failures
                except Exception:
                    pass
        
        # Calculate statistics
        avg_failure_rate = mean(failure_rates) if failure_rates else 0
        failure_trend = self._calculate_trend(failure_rates)
        
        return {
            'average_failure_rate': avg_failure_rate,
            'failure_trend': failure_trend,
            'failure_by_day': dict(failure_days),
            'total_runs_analyzed': len(history)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in values"""
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
            return f"increasing (+{change:.1f}%)"
        else:
            return f"decreasing ({change:.1f}%)"
    
    def predict_next_run(self) -> Dict:
        """Predict next test run outcome"""
        patterns = self.analyze_failure_patterns()
        
        if 'error' in patterns:
            return {'error': patterns['error']}
        
        # Simple prediction based on recent trend
        history = self._load_history()
        recent = history[-5:] if len(history) >= 5 else history
        
        recent_avg_success = mean([r.get('success_rate', 0) for r in recent])
        recent_avg_time = mean([r.get('execution_time', 0) for r in recent])
        
        # Predict based on trend
        if len(recent) >= 2:
            trend = recent[-1].get('success_rate', 0) - recent[0].get('success_rate', 0)
            predicted_success = recent_avg_success + (trend * 0.5)  # Damped prediction
        else:
            predicted_success = recent_avg_success
        
        # Confidence based on consistency
        success_rates = [r.get('success_rate', 0) for r in recent]
        variance = max(success_rates) - min(success_rates)
        confidence = max(0, 100 - variance)  # Lower variance = higher confidence
        
        return {
            'predicted_success_rate': max(0, min(100, predicted_success)),
            'predicted_execution_time': recent_avg_time,
            'confidence': confidence,
            'based_on_runs': len(recent),
            'pattern_analysis': patterns
        }
    
    def _load_history(self) -> List[Dict]:
        """Load test history"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    
    def generate_prediction_report(self) -> str:
        """Generate prediction report"""
        prediction = self.predict_next_run()
        patterns = self.analyze_failure_patterns()
        
        lines = []
        lines.append("=" * 80)
        lines.append("TEST FAILURE PREDICTION")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in prediction:
            lines.append(f"❌ {prediction['error']}")
            return "\n".join(lines)
        
        lines.append("🔮 PREDICTION FOR NEXT RUN")
        lines.append("-" * 80)
        lines.append(f"Predicted Success Rate:  {prediction['predicted_success_rate']:.1f}%")
        lines.append(f"Predicted Execution Time: {prediction['predicted_execution_time']:.2f}s")
        lines.append(f"Confidence:              {prediction['confidence']:.1f}%")
        lines.append(f"Based on:                {prediction['based_on_runs']} recent runs")
        lines.append("")
        
        if 'error' not in patterns:
            lines.append("📊 PATTERN ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"Average Failure Rate:    {patterns['average_failure_rate']:.1f}%")
            lines.append(f"Failure Trend:           {patterns['failure_trend']}")
            lines.append(f"Runs Analyzed:          {patterns['total_runs_analyzed']}")
            lines.append("")
        
        # Risk assessment
        predicted = prediction['predicted_success_rate']
        if predicted < 90:
            risk = "HIGH"
            recommendation = "Immediate attention required"
        elif predicted < 95:
            risk = "MEDIUM"
            recommendation = "Monitor closely"
        else:
            risk = "LOW"
            recommendation = "Expected to pass"
        
        lines.append("⚠️  RISK ASSESSMENT")
        lines.append("-" * 80)
        lines.append(f"Risk Level:              {risk}")
        lines.append(f"Recommendation:          {recommendation}")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    predictor = TestFailurePredictor(project_root)
    report = predictor.generate_prediction_report()
    
    print(report)
    
    # Save report
    report_file = project_root / "prediction_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Prediction report saved to: {report_file}")

if __name__ == "__main__":
    main()







