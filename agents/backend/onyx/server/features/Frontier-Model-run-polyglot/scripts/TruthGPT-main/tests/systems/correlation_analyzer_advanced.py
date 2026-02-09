"""
Advanced Correlation Analyzer
Advanced correlation analysis between test metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from statistics import mean, stdev
import math

class AdvancedCorrelationAnalyzer:
    """Advanced correlation analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_correlations(self, lookback_days: int = 30) -> Dict:
        """Analyze correlations between metrics"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Extract metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        # Calculate correlations
        correlations = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'correlations': {
                'success_vs_execution_time': self._calculate_correlation(success_rates, execution_times),
                'success_vs_total_tests': self._calculate_correlation(success_rates, total_tests),
                'success_vs_failures': self._calculate_correlation(success_rates, failures),
                'execution_time_vs_total_tests': self._calculate_correlation(execution_times, total_tests),
                'execution_time_vs_failures': self._calculate_correlation(execution_times, failures),
                'total_tests_vs_failures': self._calculate_correlation(total_tests, failures)
            },
            'strong_correlations': [],
            'insights': []
        }
        
        # Identify strong correlations
        correlations['strong_correlations'] = self._identify_strong_correlations(correlations['correlations'])
        
        # Generate insights
        correlations['insights'] = self._generate_correlation_insights(correlations)
        
        return correlations
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> Dict:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return {'correlation': 0, 'strength': 'none', 'interpretation': 'Insufficient data'}
        
        n = len(x)
        x_mean = mean(x)
        y_mean = mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        
        x_variance = sum((x[i] - x_mean) ** 2 for i in range(n))
        y_variance = sum((y[i] - y_mean) ** 2 for i in range(n))
        
        denominator = math.sqrt(x_variance * y_variance)
        
        if denominator == 0:
            return {'correlation': 0, 'strength': 'none', 'interpretation': 'No variance'}
        
        correlation = numerator / denominator
        
        # Determine strength
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            strength = 'strong'
        elif abs_corr >= 0.4:
            strength = 'moderate'
        elif abs_corr >= 0.2:
            strength = 'weak'
        else:
            strength = 'none'
        
        # Determine direction
        direction = 'positive' if correlation > 0 else 'negative' if correlation < 0 else 'none'
        
        return {
            'correlation': round(correlation, 3),
            'strength': strength,
            'direction': direction,
            'interpretation': self._interpret_correlation(correlation, strength, direction)
        }
    
    def _interpret_correlation(self, correlation: float, strength: str, direction: str) -> str:
        """Interpret correlation"""
        if strength == 'none':
            return 'No significant relationship'
        
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            if direction == 'positive':
                return 'Strong positive relationship - metrics move together'
            else:
                return 'Strong negative relationship - metrics move inversely'
        elif abs_corr >= 0.4:
            if direction == 'positive':
                return 'Moderate positive relationship'
            else:
                return 'Moderate negative relationship'
        else:
            if direction == 'positive':
                return 'Weak positive relationship'
            else:
                return 'Weak negative relationship'
    
    def _identify_strong_correlations(self, correlations: Dict) -> List[Dict]:
        """Identify strong correlations"""
        strong = []
        
        for metric_pair, data in correlations.items():
            if data['strength'] in ['strong', 'moderate']:
                strong.append({
                    'metrics': metric_pair.replace('_', ' vs ').title(),
                    'correlation': data['correlation'],
                    'strength': data['strength'],
                    'direction': data['direction'],
                    'interpretation': data['interpretation']
                })
        
        return sorted(strong, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _generate_correlation_insights(self, correlations: Dict) -> List[str]:
        """Generate insights from correlations"""
        insights = []
        
        corr_data = correlations['correlations']
        
        # Success vs Execution Time
        sr_et = corr_data['success_vs_execution_time']
        if sr_et['strength'] == 'strong' and sr_et['direction'] == 'negative':
            insights.append("Strong negative correlation: Longer execution times correlate with lower success rates")
        
        # Success vs Failures
        sr_f = corr_data['success_vs_failures']
        if sr_f['strength'] == 'strong' and sr_f['direction'] == 'negative':
            insights.append("Strong negative correlation: More failures correlate with lower success rates (expected)")
        
        # Execution Time vs Total Tests
        et_tt = corr_data['execution_time_vs_total_tests']
        if et_tt['strength'] == 'strong' and et_tt['direction'] == 'positive':
            insights.append("Strong positive correlation: More tests correlate with longer execution times (expected)")
        
        if not insights:
            insights.append("No strong correlations detected - metrics are relatively independent")
        
        return insights
    
    def generate_correlation_report(self, correlations: Dict) -> str:
        """Generate correlation report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED CORRELATION ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in correlations:
            lines.append(f"❌ {correlations['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {correlations['period']}")
        lines.append(f"Total Runs: {correlations['total_runs']}")
        lines.append("")
        
        lines.append("📊 CORRELATION MATRIX")
        lines.append("-" * 80)
        
        corr_data = correlations['correlations']
        for metric_pair, data in corr_data.items():
            strength_emoji = {
                'strong': '🔴',
                'moderate': '🟡',
                'weak': '🟢',
                'none': '⚪'
            }
            emoji = strength_emoji.get(data['strength'], '⚪')
            
            lines.append(f"{emoji} {metric_pair.replace('_', ' vs ').title()}")
            lines.append(f"   Correlation: {data['correlation']:+.3f}")
            lines.append(f"   Strength: {data['strength'].title()}")
            lines.append(f"   Direction: {data['direction'].title()}")
            lines.append(f"   {data['interpretation']}")
            lines.append("")
        
        if correlations['strong_correlations']:
            lines.append("🔍 STRONG CORRELATIONS")
            lines.append("-" * 80)
            for corr in correlations['strong_correlations']:
                strength_emoji = {'strong': '🔴', 'moderate': '🟡'}
                emoji = strength_emoji.get(corr['strength'], '⚪')
                lines.append(f"{emoji} {corr['metrics']}")
                lines.append(f"   Correlation: {corr['correlation']:+.3f}")
                lines.append(f"   {corr['interpretation']}")
            lines.append("")
        
        if correlations['insights']:
            lines.append("💡 INSIGHTS")
            lines.append("-" * 80)
            for insight in correlations['insights']:
                lines.append(f"• {insight}")
        
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
    
    analyzer = AdvancedCorrelationAnalyzer(project_root)
    correlations = analyzer.analyze_correlations(lookback_days=30)
    
    report = analyzer.generate_correlation_report(correlations)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_correlation_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced correlation analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






