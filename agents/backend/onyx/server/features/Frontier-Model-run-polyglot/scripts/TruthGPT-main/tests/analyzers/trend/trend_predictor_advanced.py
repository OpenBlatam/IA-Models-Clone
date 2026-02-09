"""
Advanced Trend Predictor
Advanced trend prediction with multiple models
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from statistics import mean, stdev

class AdvancedTrendPredictor:
    """Advanced trend prediction"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def predict_trends(self, lookback_days: int = 30, forecast_days: int = 7) -> Dict:
        """Predict future trends"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = sorted(
            [r for r in history if r.get('timestamp', '') >= cutoff_date],
            key=lambda x: x.get('timestamp', '')
        )
        
        if len(recent) < 4:
            return {'error': 'Insufficient data for prediction'}
        
        # Extract time series data
        timestamps = [r.get('timestamp', '')[:10] for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        predictions = {
            'period': f'Last {lookback_days} days',
            'forecast_days': forecast_days,
            'total_data_points': len(recent),
            'predictions': {
                'success_rate': self._predict_metric(success_rates, forecast_days),
                'execution_time': self._predict_metric(execution_times, forecast_days),
                'total_tests': self._predict_metric(total_tests, forecast_days),
                'failures': self._predict_metric(failures, forecast_days)
            },
            'trend_analysis': self._analyze_trends(recent),
            'confidence': self._calculate_confidence(recent),
            'recommendations': []
        }
        
        # Generate recommendations
        predictions['recommendations'] = self._generate_predictions_recommendations(predictions)
        
        return predictions
    
    def _predict_metric(self, values: List[float], forecast_days: int) -> Dict:
        """Predict future values using linear regression"""
        if len(values) < 2:
            return {'error': 'Insufficient data'}
        
        n = len(values)
        x = list(range(n))
        
        # Calculate linear regression
        x_mean = mean(x)
        y_mean = mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        intercept = y_mean - slope * x_mean
        
        # Predict future values
        future_x = list(range(n, n + forecast_days))
        predicted_values = [slope * x_val + intercept for x_val in future_x]
        
        # Calculate confidence intervals
        residuals = [values[i] - (slope * x[i] + intercept) for i in range(n)]
        residual_std = stdev(residuals) if len(residuals) > 1 else 0
        
        # Forecast
        last_value = values[-1]
        forecast_value = predicted_values[-1]
        change = forecast_value - last_value
        percent_change = (change / last_value * 100) if last_value > 0 else 0
        
        return {
            'current_value': round(last_value, 2),
            'forecast_value': round(forecast_value, 2),
            'change': round(change, 2),
            'percent_change': round(percent_change, 2),
            'trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'slope': round(slope, 3),
            'confidence_interval': {
                'lower': round(forecast_value - 1.96 * residual_std, 2),
                'upper': round(forecast_value + 1.96 * residual_std, 2)
            },
            'predicted_values': [round(v, 2) for v in predicted_values]
        }
    
    def _analyze_trends(self, recent: List[Dict]) -> Dict:
        """Analyze current trends"""
        if len(recent) < 4:
            return {}
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        # Split into halves
        first_half = success_rates[:len(success_rates)//2]
        second_half = success_rates[len(success_rates)//2:]
        
        sr_trend = mean(second_half) - mean(first_half)
        
        et_first = execution_times[:len(execution_times)//2]
        et_second = execution_times[len(execution_times)//2:]
        et_trend = mean(et_second) - mean(et_first)
        
        return {
            'success_rate_trend': {
                'direction': 'improving' if sr_trend > 0 else 'declining' if sr_trend < 0 else 'stable',
                'change': round(sr_trend, 2)
            },
            'execution_time_trend': {
                'direction': 'improving' if et_trend < 0 else 'degrading' if et_trend > 0 else 'stable',
                'change': round(et_trend, 2)
            }
        }
    
    def _calculate_confidence(self, recent: List[Dict]) -> Dict:
        """Calculate prediction confidence"""
        if len(recent) < 4:
            return {'confidence': 'low', 'reason': 'Insufficient data'}
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        sr_std = stdev(success_rates) if len(success_rates) > 1 else 0
        
        # Lower std = higher confidence
        if sr_std < 2:
            confidence = 'high'
        elif sr_std < 5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'confidence': confidence,
            'variance': round(sr_std, 2),
            'data_points': len(recent),
            'reason': f'Based on {len(recent)} data points with std dev of {sr_std:.2f}'
        }
    
    def _generate_predictions_recommendations(self, predictions: Dict) -> List[str]:
        """Generate recommendations based on predictions"""
        recommendations = []
        
        sr_pred = predictions['predictions']['success_rate']
        if sr_pred.get('trend') == 'decreasing' and sr_pred.get('forecast_value', 100) < 90:
            recommendations.append(f"Success rate predicted to decline to {sr_pred['forecast_value']:.1f}% - take preventive action")
        
        et_pred = predictions['predictions']['execution_time']
        if et_pred.get('trend') == 'increasing' and et_pred.get('forecast_value', 0) > 300:
            recommendations.append(f"Execution time predicted to increase to {et_pred['forecast_value']:.0f}s - optimize performance")
        
        failures_pred = predictions['predictions']['failures']
        if failures_pred.get('trend') == 'increasing' and failures_pred.get('forecast_value', 0) > 10:
            recommendations.append(f"Failures predicted to increase to {failures_pred['forecast_value']:.0f} - investigate root causes")
        
        confidence = predictions['confidence']
        if confidence['confidence'] == 'low':
            recommendations.append("Low prediction confidence - collect more data for better accuracy")
        
        if not recommendations:
            recommendations.append("Predictions look stable - maintain current practices")
        
        return recommendations
    
    def generate_prediction_report(self, predictions: Dict) -> str:
        """Generate prediction report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED TREND PREDICTION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in predictions:
            lines.append(f"❌ {predictions['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {predictions['period']}")
        lines.append(f"Forecast: Next {predictions['forecast_days']} days")
        lines.append(f"Data Points: {predictions['total_data_points']}")
        lines.append("")
        
        confidence = predictions['confidence']
        conf_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
        emoji = conf_emoji.get(confidence['confidence'], '⚪')
        lines.append(f"{emoji} Confidence: {confidence['confidence'].upper()}")
        lines.append(f"   {confidence['reason']}")
        lines.append("")
        
        lines.append("📈 PREDICTIONS")
        lines.append("-" * 80)
        
        for metric_name, pred in predictions['predictions'].items():
            if 'error' in pred:
                continue
            
            trend_emoji = {'increasing': '📈', 'decreasing': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(pred['trend'], '➡️')
            
            lines.append(f"{emoji} {metric_name.replace('_', ' ').title()}")
            lines.append(f"   Current: {pred['current_value']}")
            lines.append(f"   Forecast: {pred['forecast_value']}")
            lines.append(f"   Change: {pred['change']:+.2f} ({pred['percent_change']:+.2f}%)")
            lines.append(f"   Trend: {pred['trend'].title()}")
            lines.append(f"   Confidence Interval: [{pred['confidence_interval']['lower']}, {pred['confidence_interval']['upper']}]")
            lines.append("")
        
        if predictions.get('trend_analysis'):
            lines.append("📊 CURRENT TRENDS")
            lines.append("-" * 80)
            trends = predictions['trend_analysis']
            
            sr_trend = trends.get('success_rate_trend', {})
            trend_emoji = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(sr_trend.get('direction', 'stable'), '➡️')
            lines.append(f"{emoji} Success Rate: {sr_trend.get('direction', 'stable').title()} ({sr_trend.get('change', 0):+.2f}%)")
            
            et_trend = trends.get('execution_time_trend', {})
            trend_emoji = {'improving': '📈', 'degrading': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(et_trend.get('direction', 'stable'), '➡️')
            lines.append(f"{emoji} Execution Time: {et_trend.get('direction', 'stable').title()} ({et_trend.get('change', 0):+.2f}s)")
            lines.append("")
        
        if predictions['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in predictions['recommendations']:
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
    
    predictor = AdvancedTrendPredictor(project_root)
    predictions = predictor.predict_trends(lookback_days=30, forecast_days=7)
    
    report = predictor.generate_prediction_report(predictions)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_trend_prediction_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced trend prediction report saved to: {report_file}")

if __name__ == "__main__":
    main()






