"""
Advanced Trend Analyzer
Advanced trend analysis with forecasting
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean
import math

class AdvancedTrendAnalyzer:
    """Advanced trend analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_trends(self, lookback_days: int = 30, forecast_days: int = 7) -> Dict:
        """Analyze trends and forecast"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Extract time series data
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        # Analyze trends
        success_trend = self._calculate_trend(success_rates)
        execution_trend = self._calculate_trend(execution_times)
        
        # Forecast
        success_forecast = self._forecast(success_rates, forecast_days)
        execution_forecast = self._forecast(execution_times, forecast_days)
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'success_rate_trend': success_trend,
            'execution_time_trend': execution_trend,
            'success_rate_forecast': success_forecast,
            'execution_time_forecast': execution_forecast,
            'trend_summary': self._generate_trend_summary(success_trend, execution_trend)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict:
        """Calculate trend using linear regression"""
        if len(values) < 2:
            return {'direction': 'stable', 'slope': 0, 'strength': 0}
        
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate slope (simple linear regression)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        
        # Determine direction
        if abs(slope) < 0.01:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Calculate strength (R-squared approximation)
        mean_y = mean(y)
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        ss_res = sum((y[i] - (slope * x[i] + (sum_y - slope * sum_x) / n)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        strength = abs(r_squared) * 100
        
        return {
            'direction': direction,
            'slope': round(slope, 4),
            'strength': round(strength, 1),
            'current': round(values[-1], 2),
            'start': round(values[0], 2),
            'change': round(values[-1] - values[0], 2)
        }
    
    def _forecast(self, values: List[float], days: int) -> Dict:
        """Simple linear forecast"""
        if len(values) < 2:
            return {'forecast': values[-1] if values else 0, 'confidence': 0}
        
        # Use last few points for trend
        recent_values = values[-min(7, len(values)):]
        trend = self._calculate_trend(recent_values)
        
        # Forecast: current + (slope * days)
        forecast_value = values[-1] + (trend['slope'] * days)
        
        # Confidence based on trend strength
        confidence = min(100, max(0, trend['strength']))
        
        return {
            'forecast': round(forecast_value, 2),
            'confidence': round(confidence, 1),
            'direction': trend['direction']
        }
    
    def _generate_trend_summary(self, success_trend: Dict, execution_trend: Dict) -> Dict:
        """Generate trend summary"""
        summary = {
            'overall_status': 'stable',
            'key_findings': [],
            'recommendations': []
        }
        
        # Analyze success rate trend
        if success_trend['direction'] == 'decreasing' and success_trend['strength'] > 50:
            summary['overall_status'] = 'declining'
            summary['key_findings'].append('Success rate is declining significantly')
            summary['recommendations'].append('Investigate and fix failing tests')
        elif success_trend['direction'] == 'increasing' and success_trend['strength'] > 50:
            summary['key_findings'].append('Success rate is improving')
        
        # Analyze execution time trend
        if execution_trend['direction'] == 'increasing' and execution_trend['strength'] > 50:
            summary['key_findings'].append('Execution time is increasing')
            summary['recommendations'].append('Optimize slow tests')
        elif execution_trend['direction'] == 'decreasing' and execution_trend['strength'] > 50:
            summary['key_findings'].append('Execution time is improving')
        
        return summary
    
    def generate_trend_report(self, analysis: Dict) -> str:
        """Generate trend report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED TREND ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        lines.append("📈 SUCCESS RATE TREND")
        lines.append("-" * 80)
        sr_trend = analysis['success_rate_trend']
        direction_emoji = {'increasing': '📈', 'decreasing': '📉', 'stable': '➡️'}
        emoji = direction_emoji.get(sr_trend['direction'], '➡️')
        lines.append(f"{emoji} Direction: {sr_trend['direction'].title()}")
        lines.append(f"Slope: {sr_trend['slope']}")
        lines.append(f"Strength: {sr_trend['strength']}%")
        lines.append(f"Start: {sr_trend['start']}%")
        lines.append(f"Current: {sr_trend['current']}%")
        lines.append(f"Change: {sr_trend['change']:+.2f}%")
        lines.append("")
        
        lines.append("⏱️ EXECUTION TIME TREND")
        lines.append("-" * 80)
        et_trend = analysis['execution_time_trend']
        emoji = direction_emoji.get(et_trend['direction'], '➡️')
        lines.append(f"{emoji} Direction: {et_trend['direction'].title()}")
        lines.append(f"Slope: {et_trend['slope']}")
        lines.append(f"Strength: {et_trend['strength']}%")
        lines.append(f"Start: {et_trend['start']}s")
        lines.append(f"Current: {et_trend['current']}s")
        lines.append(f"Change: {et_trend['change']:+.2f}s")
        lines.append("")
        
        lines.append("🔮 FORECAST")
        lines.append("-" * 80)
        sr_forecast = analysis['success_rate_forecast']
        lines.append(f"Success Rate Forecast: {sr_forecast['forecast']}%")
        lines.append(f"  Confidence: {sr_forecast['confidence']}%")
        lines.append(f"  Direction: {sr_forecast['direction']}")
        lines.append("")
        
        et_forecast = analysis['execution_time_forecast']
        lines.append(f"Execution Time Forecast: {et_forecast['forecast']}s")
        lines.append(f"  Confidence: {et_forecast['confidence']}%")
        lines.append(f"  Direction: {et_forecast['direction']}")
        lines.append("")
        
        if analysis['trend_summary']['key_findings']:
            lines.append("💡 KEY FINDINGS")
            lines.append("-" * 80)
            for finding in analysis['trend_summary']['key_findings']:
                lines.append(f"• {finding}")
            lines.append("")
        
        if analysis['trend_summary']['recommendations']:
            lines.append("🎯 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in analysis['trend_summary']['recommendations']:
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
    
    analyzer = AdvancedTrendAnalyzer(project_root)
    analysis = analyzer.analyze_trends(lookback_days=30, forecast_days=7)
    
    report = analyzer.generate_trend_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_trend_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced trend analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







