"""
Enhanced Trend Analyzer
Enhanced trend analysis with comprehensive insights
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev

class EnhancedTrendAnalyzer:
    """Enhanced trend analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_trends(self, lookback_days: int = 30) -> Dict:
        """Analyze trends comprehensively"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = sorted(
            [r for r in history if r.get('timestamp', '') >= cutoff_date],
            key=lambda x: x.get('timestamp', '')
        )
        
        if len(recent) < 3:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Comprehensive trend analysis
        trend_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'success_rate_trends': self._analyze_success_rate_trends(recent),
            'execution_time_trends': self._analyze_execution_time_trends(recent),
            'failure_trends': self._analyze_failure_trends(recent),
            'test_count_trends': self._analyze_test_count_trends(recent),
            'trend_patterns': self._identify_trend_patterns(recent),
            'trend_forecast': self._forecast_trends(recent),
            'trend_health': self._assess_trend_health(recent),
            'recommendations': []
        }
        
        # Generate recommendations
        trend_analysis['recommendations'] = self._generate_trend_recommendations(trend_analysis)
        
        return trend_analysis
    
    def _analyze_success_rate_trends(self, recent: List[Dict]) -> Dict:
        """Analyze success rate trends"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if len(success_rates) < 3:
            return {}
        
        # Linear trend
        n = len(success_rates)
        x = list(range(n))
        x_mean = mean(x)
        y_mean = mean(success_rates)
        
        numerator = sum((x[i] - x_mean) * (success_rates[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = (numerator / denominator) if denominator > 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_pred = [slope * x[i] + intercept for i in range(n)]
        ss_res = sum((success_rates[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((success_rates[i] - y_mean) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'slope': round(slope, 3),
            'intercept': round(intercept, 2),
            'r_squared': round(r_squared, 3),
            'direction': 'improving' if slope > 0.1 else 'declining' if slope < -0.1 else 'stable',
            'strength': 'strong' if abs(slope) > 0.5 else 'moderate' if abs(slope) > 0.2 else 'weak',
            'current_value': round(success_rates[-1], 2),
            'first_value': round(success_rates[0], 2),
            'change': round(success_rates[-1] - success_rates[0], 2)
        }
    
    def _analyze_execution_time_trends(self, recent: List[Dict]) -> Dict:
        """Analyze execution time trends"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        if len(execution_times) < 3:
            return {}
        
        # Linear trend
        n = len(execution_times)
        x = list(range(n))
        x_mean = mean(x)
        y_mean = mean(execution_times)
        
        numerator = sum((x[i] - x_mean) * (execution_times[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = (numerator / denominator) if denominator > 0 else 0
        
        return {
            'slope': round(slope, 3),
            'direction': 'improving' if slope < -1 else 'degrading' if slope > 1 else 'stable',
            'strength': 'strong' if abs(slope) > 5 else 'moderate' if abs(slope) > 2 else 'weak',
            'current_value': round(execution_times[-1], 2),
            'first_value': round(execution_times[0], 2),
            'change': round(execution_times[-1] - execution_times[0], 2)
        }
    
    def _analyze_failure_trends(self, recent: List[Dict]) -> Dict:
        """Analyze failure trends"""
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        if len(failures) < 3:
            return {}
        
        # Linear trend
        n = len(failures)
        x = list(range(n))
        x_mean = mean(x)
        y_mean = mean(failures)
        
        numerator = sum((x[i] - x_mean) * (failures[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = (numerator / denominator) if denominator > 0 else 0
        
        return {
            'slope': round(slope, 3),
            'direction': 'decreasing' if slope < -0.1 else 'increasing' if slope > 0.1 else 'stable',
            'strength': 'strong' if abs(slope) > 1 else 'moderate' if abs(slope) > 0.5 else 'weak',
            'current_value': failures[-1],
            'first_value': failures[0],
            'change': failures[-1] - failures[0]
        }
    
    def _analyze_test_count_trends(self, recent: List[Dict]) -> Dict:
        """Analyze test count trends"""
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        if len(total_tests) < 3:
            return {}
        
        # Linear trend
        n = len(total_tests)
        x = list(range(n))
        x_mean = mean(x)
        y_mean = mean(total_tests)
        
        numerator = sum((x[i] - x_mean) * (total_tests[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = (numerator / denominator) if denominator > 0 else 0
        
        return {
            'slope': round(slope, 3),
            'direction': 'increasing' if slope > 1 else 'decreasing' if slope < -1 else 'stable',
            'strength': 'strong' if abs(slope) > 10 else 'moderate' if abs(slope) > 5 else 'weak',
            'current_value': total_tests[-1],
            'first_value': total_tests[0],
            'change': total_tests[-1] - total_tests[0]
        }
    
    def _identify_trend_patterns(self, recent: List[Dict]) -> List[Dict]:
        """Identify trend patterns"""
        patterns = []
        
        if len(recent) < 5:
            return patterns
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        # Linear increasing pattern
        if len(success_rates) >= 5:
            recent_5 = success_rates[-5:]
            if all(recent_5[i] < recent_5[i+1] for i in range(len(recent_5)-1)):
                patterns.append({
                    'pattern': 'linear_increase',
                    'metric': 'success_rate',
                    'description': 'Consistent linear increase in success rate',
                    'severity': 'positive',
                    'confidence': 'high'
                })
        
        # Linear decreasing pattern
        if len(execution_times) >= 5:
            recent_5 = execution_times[-5:]
            if all(recent_5[i] > recent_5[i+1] for i in range(len(recent_5)-1)):
                patterns.append({
                    'pattern': 'linear_decrease',
                    'metric': 'execution_time',
                    'description': 'Consistent linear decrease in execution time',
                    'severity': 'positive',
                    'confidence': 'high'
                })
        
        # Volatile pattern
        if len(success_rates) > 1:
            sr_std = stdev(success_rates)
            if sr_std > 10:
                patterns.append({
                    'pattern': 'high_volatility',
                    'metric': 'success_rate',
                    'description': f'High volatility in success rate (std: {sr_std:.1f}%)',
                    'severity': 'negative',
                    'confidence': 'medium'
                })
        
        return patterns
    
    def _forecast_trends(self, recent: List[Dict], forecast_periods: int = 5) -> Dict:
        """Forecast future trends"""
        if len(recent) < 3:
            return {}
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        # Simple linear forecast
        n = len(success_rates)
        x = list(range(n))
        x_mean = mean(x)
        y_mean = mean(success_rates)
        
        numerator = sum((x[i] - x_mean) * (success_rates[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = (numerator / denominator) if denominator > 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Forecast
        last_x = n - 1
        forecast_x = last_x + forecast_periods
        forecast_value = slope * forecast_x + intercept
        
        current_value = success_rates[-1]
        forecast_change = forecast_value - current_value
        
        return {
            'forecast_periods': forecast_periods,
            'current_value': round(current_value, 2),
            'forecast_value': round(forecast_value, 2),
            'forecast_change': round(forecast_change, 2),
            'forecast_direction': 'improving' if forecast_change > 0 else 'declining' if forecast_change < 0 else 'stable',
            'confidence': 'high' if abs(slope) > 0.5 else 'medium' if abs(slope) > 0.2 else 'low'
        }
    
    def _assess_trend_health(self, recent: List[Dict]) -> Dict:
        """Assess trend health"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        if not success_rates or not execution_times:
            return {}
        
        # Analyze trends
        sr_trend = self._analyze_success_rate_trends(recent)
        et_trend = self._analyze_execution_time_trends(recent)
        
        # Health score
        sr_score = 100 if sr_trend.get('direction') == 'improving' else 50 if sr_trend.get('direction') == 'stable' else 0
        et_score = 100 if et_trend.get('direction') == 'improving' else 50 if et_trend.get('direction') == 'stable' else 0
        
        health_score = (sr_score * 0.6 + et_score * 0.4)
        
        return {
            'health_score': round(health_score, 1),
            'success_rate_trend_health': sr_trend.get('direction', 'unknown'),
            'execution_time_trend_health': et_trend.get('direction', 'unknown'),
            'overall_health': 'healthy' if health_score >= 70 else 'warning' if health_score >= 40 else 'critical'
        }
    
    def _generate_trend_recommendations(self, analysis: Dict) -> List[str]:
        """Generate trend recommendations"""
        recommendations = []
        
        sr_trends = analysis.get('success_rate_trends', {})
        if sr_trends.get('direction') == 'declining':
            recommendations.append(f"Success rate is declining ({sr_trends.get('change', 0):+.2f}%) - investigate root causes")
        
        et_trends = analysis.get('execution_time_trends', {})
        if et_trends.get('direction') == 'degrading':
            recommendations.append(f"Execution time is degrading ({et_trends.get('change', 0):+.2f}s) - optimize performance")
        
        f_trends = analysis.get('failure_trends', {})
        if f_trends.get('direction') == 'increasing':
            recommendations.append(f"Failures are increasing ({f_trends.get('change', 0):+.0f}) - address failing tests")
        
        patterns = analysis.get('trend_patterns', [])
        for pattern in patterns:
            if pattern['severity'] == 'negative':
                recommendations.append(f"⚠️ {pattern['description']} - take corrective action")
        
        forecast = analysis.get('trend_forecast', {})
        if forecast.get('forecast_direction') == 'declining':
            recommendations.append(f"Forecast indicates declining trend - take preventive measures")
        
        health = analysis.get('trend_health', {})
        if health.get('overall_health') != 'healthy':
            recommendations.append(f"Trend health is {health.get('overall_health')} - improve trends")
        
        if not recommendations:
            recommendations.append("✅ Trends are positive - maintain current practices")
        
        return recommendations
    
    def generate_trend_report(self, analysis: Dict) -> str:
        """Generate trend report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED TREND ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        if analysis.get('success_rate_trends'):
            lines.append("📈 SUCCESS RATE TRENDS")
            lines.append("-" * 80)
            sr = analysis['success_rate_trends']
            trend_emoji = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(sr['direction'], '➡️')
            lines.append(f"{emoji} Direction: {sr['direction'].title()}")
            lines.append(f"Slope: {sr['slope']:+.3f}")
            lines.append(f"R-squared: {sr['r_squared']:.3f}")
            lines.append(f"Strength: {sr['strength'].title()}")
            lines.append(f"Change: {sr['change']:+.2f}% ({sr['first_value']}% → {sr['current_value']}%)")
            lines.append("")
        
        if analysis.get('execution_time_trends'):
            lines.append("⏱️ EXECUTION TIME TRENDS")
            lines.append("-" * 80)
            et = analysis['execution_time_trends']
            trend_emoji = {'improving': '📈', 'degrading': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(et['direction'], '➡️')
            lines.append(f"{emoji} Direction: {et['direction'].title()}")
            lines.append(f"Slope: {et['slope']:+.3f}")
            lines.append(f"Strength: {et['strength'].title()}")
            lines.append(f"Change: {et['change']:+.2f}s ({et['first_value']}s → {et['current_value']}s)")
            lines.append("")
        
        if analysis.get('failure_trends'):
            lines.append("❌ FAILURE TRENDS")
            lines.append("-" * 80)
            f = analysis['failure_trends']
            trend_emoji = {'decreasing': '📈', 'increasing': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(f['direction'], '➡️')
            lines.append(f"{emoji} Direction: {f['direction'].title()}")
            lines.append(f"Slope: {f['slope']:+.3f}")
            lines.append(f"Strength: {f['strength'].title()}")
            lines.append(f"Change: {f['change']:+.0f} ({f['first_value']} → {f['current_value']})")
            lines.append("")
        
        if analysis.get('trend_patterns'):
            lines.append("🔍 TREND PATTERNS")
            lines.append("-" * 80)
            for pattern in analysis['trend_patterns']:
                severity_emoji = {'positive': '✅', 'negative': '⚠️'}
                emoji = severity_emoji.get(pattern['severity'], '⚪')
                lines.append(f"{emoji} {pattern['pattern'].replace('_', ' ').title()}")
                lines.append(f"   Metric: {pattern['metric']}")
                lines.append(f"   {pattern['description']}")
                lines.append(f"   Confidence: {pattern['confidence']}")
            lines.append("")
        
        if analysis.get('trend_forecast'):
            forecast = analysis['trend_forecast']
            lines.append("🔮 TREND FORECAST")
            lines.append("-" * 80)
            lines.append(f"Forecast Periods: {forecast['forecast_periods']}")
            lines.append(f"Current Value: {forecast['current_value']}%")
            lines.append(f"Forecast Value: {forecast['forecast_value']}%")
            lines.append(f"Forecast Change: {forecast['forecast_change']:+.2f}%")
            lines.append(f"Forecast Direction: {forecast['forecast_direction'].title()}")
            lines.append(f"Confidence: {forecast['confidence'].upper()}")
            lines.append("")
        
        if analysis.get('trend_health'):
            health = analysis['trend_health']
            status_emoji = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴'}
            emoji = status_emoji.get(health['overall_health'], '⚪')
            lines.append(f"{emoji} TREND HEALTH")
            lines.append("-" * 80)
            lines.append(f"Overall Health: {health['overall_health'].upper()}")
            lines.append(f"Health Score: {health['health_score']}/100")
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
    
    analyzer = EnhancedTrendAnalyzer(project_root)
    analysis = analyzer.analyze_trends(lookback_days=30)
    
    report = analyzer.generate_trend_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "enhanced_trend_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Enhanced trend analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






