"""
Trend Forecaster
Forecasts future test trends using simple linear regression
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from statistics import mean

class TrendForecaster:
    """Forecast future test trends"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def forecast_trend(
        self,
        metric: str = 'success_rate',
        days_ahead: int = 7,
        lookback: int = 30
    ) -> Dict:
        """Forecast trend for a metric"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback)).isoformat()
        recent = sorted(
            [r for r in history if r.get('timestamp', '') >= cutoff_date],
            key=lambda x: x.get('timestamp', '')
        )
        
        if len(recent) < 5:
            return {'error': 'Insufficient data for forecasting'}
        
        # Extract metric values
        values = [r.get(metric, 0) for r in recent]
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        intercept = (sum_y - slope * sum_x) / n if n > 0 else 0
        
        # Forecast future values
        forecast_dates = []
        forecast_values = []
        
        for i in range(1, days_ahead + 1):
            future_x = n + i - 1
            forecast_value = slope * future_x + intercept
            forecast_date = (datetime.now() + timedelta(days=i)).isoformat()[:10]
            
            forecast_dates.append(forecast_date)
            forecast_values.append(max(0, min(100, forecast_value)))  # Clamp to 0-100
        
        # Calculate trend direction
        current_value = values[-1] if values else 0
        forecast_value = forecast_values[-1] if forecast_values else 0
        
        trend_direction = "increasing" if forecast_value > current_value else "decreasing" if forecast_value < current_value else "stable"
        trend_change = forecast_value - current_value
        
        return {
            'metric': metric,
            'current_value': round(current_value, 2),
            'forecast_values': [round(v, 2) for v in forecast_values],
            'forecast_dates': forecast_dates,
            'trend_direction': trend_direction,
            'trend_change': round(trend_change, 2),
            'days_ahead': days_ahead,
            'confidence': self._calculate_confidence(values, slope)
        }
    
    def _calculate_confidence(self, values: List[float], slope: float) -> float:
        """Calculate forecast confidence"""
        if len(values) < 2:
            return 0.0
        
        # Simple confidence based on variance
        avg = mean(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        
        # Lower variance = higher confidence
        # Also consider trend strength
        trend_strength = abs(slope) * 10  # Scale slope
        
        confidence = max(0, min(100, 100 - (variance / 10) + trend_strength))
        return round(confidence, 1)
    
    def generate_forecast_report(self, forecast: Dict) -> str:
        """Generate forecast report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TREND FORECAST REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in forecast:
            lines.append(f"❌ {forecast['error']}")
            return "\n".join(lines)
        
        lines.append(f"Metric: {forecast['metric'].replace('_', ' ').title()}")
        lines.append(f"Current Value: {forecast['current_value']}")
        lines.append(f"Forecast Period: Next {forecast['days_ahead']} days")
        lines.append(f"Confidence: {forecast['confidence']}%")
        lines.append("")
        
        lines.append("📈 FORECAST")
        lines.append("-" * 80)
        
        trend_emoji = {
            'increasing': '📈',
            'decreasing': '📉',
            'stable': '➡️'
        }.get(forecast['trend_direction'], '➡️')
        
        lines.append(f"Trend: {trend_emoji} {forecast['trend_direction'].upper()}")
        lines.append(f"Expected Change: {forecast['trend_change']:+.2f}")
        lines.append("")
        
        lines.append("Daily Forecast:")
        for date, value in zip(forecast['forecast_dates'], forecast['forecast_values']):
            lines.append(f"  {date}: {value:.2f}")
        
        lines.append("")
        lines.append("💡 INTERPRETATION")
        lines.append("-" * 80)
        
        if forecast['trend_direction'] == 'increasing':
            lines.append("✅ Positive trend expected")
            lines.append("   Test quality is improving")
        elif forecast['trend_direction'] == 'decreasing':
            lines.append("⚠️  Negative trend expected")
            lines.append("   Action may be needed")
        else:
            lines.append("➡️  Stable trend expected")
            lines.append("   No significant changes")
        
        if forecast['confidence'] < 50:
            lines.append("")
            lines.append("⚠️  Low confidence forecast")
            lines.append("   More data needed for accurate prediction")
        
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
    
    forecaster = TrendForecaster(project_root)
    
    # Forecast success rate
    forecast = forecaster.forecast_trend(metric='success_rate', days_ahead=7)
    report = forecaster.generate_forecast_report(forecast)
    
    print(report)
    
    # Save report
    report_file = project_root / "forecast_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Forecast report saved to: {report_file}")

if __name__ == "__main__":
    main()







