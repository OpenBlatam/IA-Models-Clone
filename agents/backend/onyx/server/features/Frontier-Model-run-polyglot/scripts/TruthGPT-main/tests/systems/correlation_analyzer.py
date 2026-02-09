"""
Correlation Analyzer
Analyzes correlations between different test metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import math

class CorrelationAnalyzer:
    """Analyze correlations between test metrics"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def analyze_correlations(self, lookback: int = 20) -> Dict:
        """Analyze correlations between metrics"""
        history = self._load_history()
        
        if len(history) < lookback:
            return {'error': f'Need at least {lookback} runs, found {len(history)}'}
        
        recent = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)[:lookback]
        
        # Extract metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failed', 0) + r.get('errors', 0) for r in recent]
        
        # Calculate correlations
        correlations = {
            'success_rate_vs_execution_time': self.calculate_correlation(success_rates, execution_times),
            'success_rate_vs_total_tests': self.calculate_correlation(success_rates, total_tests),
            'success_rate_vs_failures': self.calculate_correlation(success_rates, failures),
            'execution_time_vs_total_tests': self.calculate_correlation(execution_times, total_tests),
            'execution_time_vs_failures': self.calculate_correlation(execution_times, failures),
            'total_tests_vs_failures': self.calculate_correlation(total_tests, failures)
        }
        
        return {
            'correlations': correlations,
            'runs_analyzed': len(recent),
            'interpretation': self._interpret_correlations(correlations)
        }
    
    def _interpret_correlations(self, correlations: Dict) -> Dict:
        """Interpret correlation values"""
        interpretation = {}
        
        for metric_pair, value in correlations.items():
            abs_value = abs(value)
            
            if abs_value < 0.1:
                strength = "negligible"
            elif abs_value < 0.3:
                strength = "weak"
            elif abs_value < 0.5:
                strength = "moderate"
            elif abs_value < 0.7:
                strength = "strong"
            else:
                strength = "very strong"
            
            direction = "positive" if value > 0 else "negative"
            
            interpretation[metric_pair] = {
                'strength': strength,
                'direction': direction,
                'value': value
            }
        
        return interpretation
    
    def generate_correlation_report(self, analysis: Dict) -> str:
        """Generate correlation analysis report"""
        lines = []
        lines.append("=" * 80)
        lines.append("CORRELATION ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Runs Analyzed: {analysis['runs_analyzed']}")
        lines.append("")
        
        lines.append("📊 CORRELATIONS")
        lines.append("-" * 80)
        
        for metric_pair, corr_value in analysis['correlations'].items():
            interpretation = analysis['interpretation'][metric_pair]
            strength_emoji = {
                'negligible': '⚪',
                'weak': '🟡',
                'moderate': '🟠',
                'strong': '🔴',
                'very strong': '🔴'
            }.get(interpretation['strength'], '⚪')
            
            lines.append(f"{strength_emoji} {metric_pair.replace('_', ' ').title()}")
            lines.append(f"   Correlation: {corr_value:.3f}")
            lines.append(f"   Strength: {interpretation['strength'].upper()}")
            lines.append(f"   Direction: {interpretation['direction']}")
            lines.append("")
        
        lines.append("💡 INSIGHTS")
        lines.append("-" * 80)
        
        # Generate insights
        corr = analysis['correlations']
        interp = analysis['interpretation']
        
        if abs(corr['success_rate_vs_execution_time']) > 0.5:
            lines.append("⚠️  Strong correlation between success rate and execution time")
            lines.append("   Consider: Faster tests might be more reliable")
        
        if abs(corr['success_rate_vs_total_tests']) > 0.5:
            lines.append("⚠️  Strong correlation between success rate and total tests")
            lines.append("   Consider: Test suite size affects reliability")
        
        if abs(corr['execution_time_vs_total_tests']) > 0.7:
            lines.append("⚠️  Very strong correlation between execution time and total tests")
            lines.append("   Consider: More tests = longer execution (expected)")
        
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
    
    analyzer = CorrelationAnalyzer(project_root)
    analysis = analyzer.analyze_correlations(lookback=20)
    
    report = analyzer.generate_correlation_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "correlation_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Correlation report saved to: {report_file}")

if __name__ == "__main__":
    main()







