"""
Advanced Prediction System
Advanced prediction system with multiple models
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev

class AdvancedPredictionSystem:
    """Advanced prediction system"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def predict(self, lookback_days: int = 30, forecast_days: int = 7) -> Dict:
        """Make predictions"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = sorted(
            [r for r in history if r.get('timestamp', '') >= cutoff_date],
            key=lambda x: x.get('timestamp', '')
        )
        
        if len(recent) < 3:
            return {'error': 'Insufficient data for prediction'}
        
        # Comprehensive predictions
        predictions = {
            'period': f'Last {lookback_days} days',
            'forecast_days': forecast_days,
            'total_data_points': len(recent),
            'success_rate_prediction': self._predict_success_rate(recent, forecast_days),
            'execution_time_prediction': self._predict_execution_time(recent, forecast_days),
            'failure_prediction': self._predict_failures(recent, forecast_days),
            'risk_assessment': self._assess_risks(recent),
            'prediction_confidence': self._calculate_confidence(recent),
            'recommendations': []
        }
        
        # Generate recommendations
        predictions['recommendations'] = self._generate_prediction_recommendations(predictions)
        
        return predictions
    
    def _predict_success_rate(self, recent: List[Dict], forecast_days: int) -> Dict:
        """Predict success rate"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if len(success_rates) < 3:
            return {}
        
        # Linear regression
        n = len(success_rates)
        x = list(range(n))
        x_mean = mean(x)
        y_mean = mean(success_rates)
        
        numerator = sum((x[i] - x_mean) * (success_rates[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = (numerator / denominator) if denominator > 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Forecast
        forecast_x = n - 1 + forecast_days
        forecast_value = slope * forecast_x + intercept
        
        # Confidence interval
        residuals = [success_rates[i] - (slope * x[i] + intercept) for i in range(n)]
        residual_std = stdev(residuals) if len(residuals) > 1 else 0
        
        return {
            'current_value': round(success_rates[-1], 2),
            'forecast_value': round(forecast_value, 2),
            'change': round(forecast_value - success_rates[-1], 2),
            'percent_change': round(((forecast_value - success_rates[-1]) / success_rates[-1] * 100) if success_rates[-1] > 0 else 0, 2),
            'confidence_interval': {
                'lower': round(max(0, forecast_value - 1.96 * residual_std), 2),
                'upper': round(min(100, forecast_value + 1.96 * residual_std), 2)
            },
            'trend': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
            'confidence': 'high' if abs(slope) > 0.5 else 'medium' if abs(slope) > 0.2 else 'low'
        }
    
    def _predict_execution_time(self, recent: List[Dict], forecast_days: int) -> Dict:
        """Predict execution time"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        if len(execution_times) < 3:
            return {}
        
        # Linear regression
        n = len(execution_times)
        x = list(range(n))
        x_mean = mean(x)
        y_mean = mean(execution_times)
        
        numerator = sum((x[i] - x_mean) * (execution_times[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = (numerator / denominator) if denominator > 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Forecast
        forecast_x = n - 1 + forecast_days
        forecast_value = slope * forecast_x + intercept
        
        # Confidence interval
        residuals = [execution_times[i] - (slope * x[i] + intercept) for i in range(n)]
        residual_std = stdev(residuals) if len(residuals) > 1 else 0
        
        return {
            'current_value': round(execution_times[-1], 2),
            'forecast_value': round(forecast_value, 2),
            'change': round(forecast_value - execution_times[-1], 2),
            'percent_change': round(((forecast_value - execution_times[-1]) / execution_times[-1] * 100) if execution_times[-1] > 0 else 0, 2),
            'confidence_interval': {
                'lower': round(max(0, forecast_value - 1.96 * residual_std), 2),
                'upper': round(forecast_value + 1.96 * residual_std, 2)
            },
            'trend': 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable',
            'confidence': 'high' if abs(slope) > 5 else 'medium' if abs(slope) > 2 else 'low'
        }
    
    def _predict_failures(self, recent: List[Dict], forecast_days: int) -> Dict:
        """Predict failures"""
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        if len(failures) < 3:
            return {}
        
        # Linear regression
        n = len(failures)
        x = list(range(n))
        x_mean = mean(x)
        y_mean = mean(failures)
        
        numerator = sum((x[i] - x_mean) * (failures[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = (numerator / denominator) if denominator > 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Forecast
        forecast_x = n - 1 + forecast_days
        forecast_value = slope * forecast_x + intercept
        
        return {
            'current_value': failures[-1],
            'forecast_value': round(max(0, forecast_value), 1),
            'change': round(forecast_value - failures[-1], 1),
            'trend': 'decreasing' if slope < 0 else 'increasing' if slope > 0 else 'stable',
            'confidence': 'high' if abs(slope) > 1 else 'medium' if abs(slope) > 0.5 else 'low'
        }
    
    def _assess_risks(self, recent: List[Dict]) -> Dict:
        """Assess risks"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        if not success_rates:
            return {}
        
        # Risk factors
        low_success_risk = 1 if mean(success_rates) < 90 else 0
        high_time_risk = 1 if mean(execution_times) > 300 else 0
        high_failure_risk = 1 if mean(failures) > 10 else 0
        
        total_risks = low_success_risk + high_time_risk + high_failure_risk
        risk_level = 'high' if total_risks >= 2 else 'medium' if total_risks == 1 else 'low'
        
        return {
            'low_success_risk': bool(low_success_risk),
            'high_time_risk': bool(high_time_risk),
            'high_failure_risk': bool(high_failure_risk),
            'total_risks': total_risks,
            'risk_level': risk_level
        }
    
    def _calculate_confidence(self, recent: List[Dict]) -> Dict:
        """Calculate prediction confidence"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if len(success_rates) < 3:
            return {'confidence': 'low', 'reason': 'Insufficient data'}
        
        # Confidence based on variance and data points
        sr_std = stdev(success_rates) if len(success_rates) > 1 else 0
        data_points = len(recent)
        
        if sr_std < 2 and data_points >= 10:
            confidence = 'high'
        elif sr_std < 5 and data_points >= 5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'confidence': confidence,
            'variance': round(sr_std, 2),
            'data_points': data_points,
            'reason': f'Based on {data_points} data points with std dev of {sr_std:.2f}'
        }
    
    def _generate_prediction_recommendations(self, predictions: Dict) -> List[str]:
        """Generate prediction recommendations"""
        recommendations = []
        
        sr_pred = predictions.get('success_rate_prediction', {})
        if sr_pred.get('trend') == 'declining' and sr_pred.get('forecast_value', 100) < 90:
            recommendations.append(f"🚨 Success rate predicted to decline to {sr_pred['forecast_value']:.1f}% - take preventive action")
        
        et_pred = predictions.get('execution_time_prediction', {})
        if et_pred.get('trend') == 'degrading' and et_pred.get('forecast_value', 0) > 300:
            recommendations.append(f"⚠️ Execution time predicted to increase to {et_pred['forecast_value']:.0f}s - optimize performance")
        
        f_pred = predictions.get('failure_prediction', {})
        if f_pred.get('trend') == 'increasing' and f_pred.get('forecast_value', 0) > 10:
            recommendations.append(f"⚠️ Failures predicted to increase to {f_pred['forecast_value']:.0f} - investigate root causes")
        
        risks = predictions.get('risk_assessment', {})
        if risks.get('risk_level') == 'high':
            recommendations.append(f"🚨 High risk level detected ({risks['total_risks']} risk factors) - immediate attention required")
        
        confidence = predictions.get('prediction_confidence', {})
        if confidence.get('confidence') == 'low':
            recommendations.append("Low prediction confidence - collect more data for better accuracy")
        
        if not recommendations:
            recommendations.append("✅ Predictions look stable - maintain current practices")
        
        return recommendations
    
    def generate_prediction_report(self, predictions: Dict) -> str:
        """Generate prediction report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED PREDICTION SYSTEM REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in predictions:
            lines.append(f"❌ {predictions['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {predictions['period']}")
        lines.append(f"Forecast: Next {predictions['forecast_days']} days")
        lines.append(f"Data Points: {predictions['total_data_points']}")
        lines.append("")
        
        confidence = predictions['prediction_confidence']
        conf_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
        emoji = conf_emoji.get(confidence['confidence'], '⚪')
        lines.append(f"{emoji} Prediction Confidence: {confidence['confidence'].upper()}")
        lines.append(f"   {confidence['reason']}")
        lines.append("")
        
        lines.append("📈 PREDICTIONS")
        lines.append("-" * 80)
        
        if predictions.get('success_rate_prediction'):
            sr = predictions['success_rate_prediction']
            trend_emoji = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(sr['trend'], '➡️')
            lines.append(f"{emoji} Success Rate")
            lines.append(f"   Current: {sr['current_value']}%")
            lines.append(f"   Forecast: {sr['forecast_value']}%")
            lines.append(f"   Change: {sr['change']:+.2f}% ({sr['percent_change']:+.2f}%)")
            lines.append(f"   Confidence Interval: [{sr['confidence_interval']['lower']}, {sr['confidence_interval']['upper']}]")
            lines.append(f"   Confidence: {sr['confidence'].upper()}")
            lines.append("")
        
        if predictions.get('execution_time_prediction'):
            et = predictions['execution_time_prediction']
            trend_emoji = {'improving': '📈', 'degrading': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(et['trend'], '➡️')
            lines.append(f"{emoji} Execution Time")
            lines.append(f"   Current: {et['current_value']}s")
            lines.append(f"   Forecast: {et['forecast_value']}s")
            lines.append(f"   Change: {et['change']:+.2f}s ({et['percent_change']:+.2f}%)")
            lines.append(f"   Confidence Interval: [{et['confidence_interval']['lower']}, {et['confidence_interval']['upper']}]")
            lines.append(f"   Confidence: {et['confidence'].upper()}")
            lines.append("")
        
        if predictions.get('failure_prediction'):
            f = predictions['failure_prediction']
            trend_emoji = {'decreasing': '📈', 'increasing': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(f['trend'], '➡️')
            lines.append(f"{emoji} Failures")
            lines.append(f"   Current: {f['current_value']}")
            lines.append(f"   Forecast: {f['forecast_value']:.1f}")
            lines.append(f"   Change: {f['change']:+.1f}")
            lines.append(f"   Confidence: {f['confidence'].upper()}")
            lines.append("")
        
        if predictions.get('risk_assessment'):
            risks = predictions['risk_assessment']
            risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            emoji = risk_emoji.get(risks['risk_level'], '⚪')
            lines.append(f"{emoji} RISK ASSESSMENT")
            lines.append("-" * 80)
            lines.append(f"Risk Level: {risks['risk_level'].upper()}")
            lines.append(f"Total Risk Factors: {risks['total_risks']}")
            if risks['low_success_risk']:
                lines.append("   ⚠️ Low success rate risk")
            if risks['high_time_risk']:
                lines.append("   ⚠️ High execution time risk")
            if risks['high_failure_risk']:
                lines.append("   ⚠️ High failure risk")
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
    
    predictor = AdvancedPredictionSystem(project_root)
    predictions = predictor.predict(lookback_days=30, forecast_days=7)
    
    report = predictor.generate_prediction_report(predictions)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_prediction_system_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced prediction system report saved to: {report_file}")

if __name__ == "__main__":
    main()






