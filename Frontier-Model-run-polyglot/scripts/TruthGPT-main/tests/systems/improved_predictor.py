"""
Improved Predictor
Improved failure prediction with advanced algorithms
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from statistics import mean
from collections import defaultdict

class ImprovedPredictor:
    """Improved failure prediction"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def predict_failures(self, lookback_days: int = 30) -> Dict:
        """Predict potential failures"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Analyze failure patterns
        failure_patterns = defaultdict(int)
        error_patterns = defaultdict(int)
        
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                test_details = data.get('test_details', {})
                
                for failure in test_details.get('failures', []):
                    test_name = str(failure.get('test', ''))
                    failure_patterns[test_name] += 1
                
                for error in test_details.get('errors', []):
                    test_name = str(error.get('test', ''))
                    error_patterns[test_name] += 1
            except Exception:
                continue
        
        # Predict likely failures
        predictions = []
        
        # Tests that failed frequently
        for test_name, count in sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
            failure_rate = (count / len(recent)) * 100 if recent else 0
            if failure_rate > 20:  # Failed in >20% of runs
                predictions.append({
                    'test': test_name,
                    'type': 'frequent_failure',
                    'failure_count': count,
                    'failure_rate': round(failure_rate, 1),
                    'confidence': min(100, round(failure_rate * 2, 1)),
                    'risk_level': 'high' if failure_rate > 50 else 'medium'
                })
        
        # Analyze trends
        if len(recent) >= 5:
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]
            
            first_failures = sum(r.get('failures', 0) + r.get('errors', 0) for r in first_half)
            second_failures = sum(r.get('failures', 0) + r.get('errors', 0) for r in second_half)
            
            if second_failures > first_failures * 1.2:  # 20% increase
                predictions.append({
                    'test': 'Overall Suite',
                    'type': 'increasing_failures',
                    'trend': 'increasing',
                    'increase_percentage': round(((second_failures - first_failures) / first_failures * 100) if first_failures > 0 else 0, 1),
                    'confidence': 75.0,
                    'risk_level': 'high'
                })
        
        return {
            'period': f'Last {lookback_days} days',
            'total_predictions': len(predictions),
            'predictions': predictions,
            'overall_risk': self._calculate_overall_risk(predictions)
        }
    
    def _calculate_overall_risk(self, predictions: List[Dict]) -> str:
        """Calculate overall risk level"""
        if not predictions:
            return 'low'
        
        high_risk = len([p for p in predictions if p.get('risk_level') == 'high'])
        
        if high_risk >= 3:
            return 'high'
        elif high_risk >= 1:
            return 'medium'
        else:
            return 'low'
    
    def generate_prediction_report(self, predictions: Dict) -> str:
        """Generate prediction report"""
        lines = []
        lines.append("=" * 80)
        lines.append("FAILURE PREDICTION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in predictions:
            lines.append(f"❌ {predictions['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {predictions['period']}")
        lines.append(f"Total Predictions: {predictions['total_predictions']}")
        lines.append(f"Overall Risk: {predictions['overall_risk'].upper()}")
        lines.append("")
        
        risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        
        lines.append("🔮 PREDICTIONS")
        lines.append("-" * 80)
        
        for i, pred in enumerate(predictions['predictions'], 1):
            emoji = risk_emoji.get(pred.get('risk_level', 'low'), '⚪')
            lines.append(f"\n{i}. {emoji} {pred['test']}")
            lines.append(f"   Type: {pred['type'].replace('_', ' ').title()}")
            lines.append(f"   Confidence: {pred['confidence']:.1f}%")
            lines.append(f"   Risk Level: {pred['risk_level'].upper()}")
            
            if 'failure_rate' in pred:
                lines.append(f"   Failure Rate: {pred['failure_rate']:.1f}%")
            if 'increase_percentage' in pred:
                lines.append(f"   Increase: {pred['increase_percentage']:.1f}%")
        
        lines.append("")
        lines.append("💡 ACTIONS")
        lines.append("-" * 80)
        lines.append("• Investigate high-risk predictions")
        lines.append("• Fix frequently failing tests")
        lines.append("• Monitor trends closely")
        lines.append("• Implement preventive measures")
        
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
    
    predictor = ImprovedPredictor(project_root)
    predictions = predictor.predict_failures(lookback_days=30)
    
    report = predictor.generate_prediction_report(predictions)
    print(report)
    
    # Save report
    report_file = project_root / "prediction_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Prediction report saved to: {report_file}")

if __name__ == "__main__":
    main()







