"""
Enhanced Coverage Analyzer
Enhanced coverage analysis with detailed insights
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev

class EnhancedCoverageAnalyzer:
    """Enhanced coverage analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_coverage(self, lookback_days: int = 30, target_coverage: float = 80.0) -> Dict:
        """Analyze test coverage"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Extract coverage data
        success_rates = [r.get('success_rate', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        # Analyze coverage
        coverage_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'target_coverage': target_coverage,
            'coverage_statistics': self._calculate_coverage_statistics(success_rates, total_tests),
            'coverage_trends': self._analyze_coverage_trends(recent),
            'coverage_gaps': self._identify_coverage_gaps(recent, target_coverage),
            'coverage_distribution': self._analyze_coverage_distribution(success_rates),
            'coverage_quality': self._assess_coverage_quality(recent),
            'recommendations': []
        }
        
        # Generate recommendations
        coverage_analysis['recommendations'] = self._generate_coverage_recommendations(coverage_analysis)
        
        return coverage_analysis
    
    def _calculate_coverage_statistics(self, success_rates: List[float], total_tests: List[int]) -> Dict:
        """Calculate coverage statistics"""
        if not success_rates:
            return {}
        
        avg_coverage = mean(success_rates)
        coverage_std = stdev(success_rates) if len(success_rates) > 1 else 0
        min_coverage = min(success_rates)
        max_coverage = max(success_rates)
        
        total_tests_sum = sum(total_tests)
        
        return {
            'avg_coverage': round(avg_coverage, 2),
            'coverage_std': round(coverage_std, 2),
            'min_coverage': round(min_coverage, 2),
            'max_coverage': round(max_coverage, 2),
            'coverage_range': round(max_coverage - min_coverage, 2),
            'consistency': round(100 - (coverage_std * 2), 1),
            'total_tests': total_tests_sum,
            'meets_target': avg_coverage >= 80
        }
    
    def _analyze_coverage_trends(self, recent: List[Dict]) -> Dict:
        """Analyze coverage trends"""
        if len(recent) < 4:
            return {}
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        # Split into halves
        first_half = success_rates[:len(success_rates)//2]
        second_half = success_rates[len(success_rates)//2:]
        
        first_avg = mean(first_half)
        second_avg = mean(second_half)
        
        trend = second_avg - first_avg
        percent_change = (trend / first_avg * 100) if first_avg > 0 else 0
        
        return {
            'trend': round(trend, 2),
            'percent_change': round(percent_change, 2),
            'direction': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable',
            'first_half_avg': round(first_avg, 2),
            'second_half_avg': round(second_avg, 2),
            'velocity': round(trend / len(second_half), 2)  # Coverage change per run
        }
    
    def _identify_coverage_gaps(self, recent: List[Dict], target_coverage: float) -> List[Dict]:
        """Identify coverage gaps"""
        gaps = []
        
        for i, r in enumerate(recent):
            success_rate = r.get('success_rate', 0)
            gap = target_coverage - success_rate
            
            if gap > 5:  # Significant gap
                gaps.append({
                    'run_index': i,
                    'timestamp': r.get('timestamp', ''),
                    'current_coverage': round(success_rate, 2),
                    'target_coverage': target_coverage,
                    'gap': round(gap, 2),
                    'severity': 'high' if gap > 20 else 'medium' if gap > 10 else 'low',
                    'total_tests': r.get('total_tests', 0)
                })
        
        return sorted(gaps, key=lambda x: x['gap'], reverse=True)[:10]  # Top 10
    
    def _analyze_coverage_distribution(self, success_rates: List[float]) -> Dict:
        """Analyze coverage distribution"""
        if not success_rates:
            return {}
        
        # Categorize coverage levels
        excellent = sum(1 for sr in success_rates if sr >= 95)
        good = sum(1 for sr in success_rates if 90 <= sr < 95)
        acceptable = sum(1 for sr in success_rates if 80 <= sr < 90)
        poor = sum(1 for sr in success_rates if sr < 80)
        
        total = len(success_rates)
        
        return {
            'excellent': {
                'count': excellent,
                'percentage': round((excellent / total * 100) if total > 0 else 0, 1)
            },
            'good': {
                'count': good,
                'percentage': round((good / total * 100) if total > 0 else 0, 1)
            },
            'acceptable': {
                'count': acceptable,
                'percentage': round((acceptable / total * 100) if total > 0 else 0, 1)
            },
            'poor': {
                'count': poor,
                'percentage': round((poor / total * 100) if total > 0 else 0, 1)
            }
        }
    
    def _assess_coverage_quality(self, recent: List[Dict]) -> Dict:
        """Assess coverage quality"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        if not success_rates or not total_tests:
            return {}
        
        avg_coverage = mean(success_rates)
        coverage_std = stdev(success_rates) if len(success_rates) > 1 else 0
        
        # Quality score based on coverage and consistency
        coverage_score = min(100, avg_coverage)
        consistency_score = max(0, 100 - (coverage_std * 2))
        quality_score = (coverage_score * 0.7 + consistency_score * 0.3)
        
        # Test density (tests per percentage point of coverage)
        avg_tests = mean(total_tests)
        test_density = avg_tests / avg_coverage if avg_coverage > 0 else 0
        
        return {
            'coverage_score': round(coverage_score, 1),
            'consistency_score': round(consistency_score, 1),
            'quality_score': round(quality_score, 1),
            'test_density': round(test_density, 2),
            'quality_level': 'excellent' if quality_score >= 90 else 'good' if quality_score >= 80 else 'acceptable' if quality_score >= 70 else 'poor'
        }
    
    def _generate_coverage_recommendations(self, analysis: Dict) -> List[str]:
        """Generate coverage recommendations"""
        recommendations = []
        
        stats = analysis['coverage_statistics']
        if not stats.get('meets_target', False):
            gap = analysis['target_coverage'] - stats['avg_coverage']
            recommendations.append(f"Increase coverage from {stats['avg_coverage']:.1f}% to {analysis['target_coverage']}% (gap: {gap:.1f}%)")
        
        if stats.get('consistency', 0) < 90:
            recommendations.append(f"Improve coverage consistency from {stats['consistency']:.1f}% to 95%+")
        
        if analysis['coverage_gaps']:
            high_gaps = sum(1 for g in analysis['coverage_gaps'] if g['severity'] == 'high')
            if high_gaps > 0:
                recommendations.append(f"🚨 {high_gaps} high-severity coverage gap(s) - prioritize coverage improvement")
        
        trends = analysis.get('coverage_trends', {})
        if trends.get('direction') == 'declining':
            recommendations.append(f"Coverage is declining ({trends['trend']:+.2f}%) - investigate root causes")
        
        quality = analysis.get('coverage_quality', {})
        if quality.get('quality_level') in ['acceptable', 'poor']:
            recommendations.append(f"Improve coverage quality from {quality.get('quality_level', 'unknown')} to good+")
        
        dist = analysis['coverage_distribution']
        if dist.get('poor', {}).get('percentage', 0) > 20:
            recommendations.append(f"High percentage of poor coverage runs ({dist['poor']['percentage']:.1f}%) - improve test coverage")
        
        if not recommendations:
            recommendations.append("✅ Coverage is meeting targets - maintain current practices")
        
        return recommendations
    
    def generate_coverage_report(self, analysis: Dict) -> str:
        """Generate coverage report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED COVERAGE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append(f"Target Coverage: {analysis['target_coverage']}%")
        lines.append("")
        
        lines.append("📊 COVERAGE STATISTICS")
        lines.append("-" * 80)
        stats = analysis['coverage_statistics']
        status = "✅ Meets Target" if stats.get('meets_target') else "⚠️ Below Target"
        lines.append(f"{status}")
        lines.append(f"Average Coverage: {stats['avg_coverage']}%")
        lines.append(f"Coverage Std Dev: {stats['coverage_std']}%")
        lines.append(f"Range: {stats['min_coverage']}% - {stats['max_coverage']}%")
        lines.append(f"Coverage Range: {stats['coverage_range']}%")
        lines.append(f"Consistency: {stats['consistency']}%")
        lines.append(f"Total Tests: {stats['total_tests']:,}")
        lines.append("")
        
        if analysis.get('coverage_trends'):
            trends = analysis['coverage_trends']
            trend_emoji = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(trends['direction'], '➡️')
            lines.append(f"{emoji} COVERAGE TRENDS")
            lines.append("-" * 80)
            lines.append(f"Direction: {trends['direction'].title()}")
            lines.append(f"Trend: {trends['trend']:+.2f}%")
            lines.append(f"Percent Change: {trends['percent_change']:+.2f}%")
            lines.append(f"Velocity: {trends['velocity']:+.2f}% per run")
            lines.append(f"First Half Avg: {trends['first_half_avg']}%")
            lines.append(f"Second Half Avg: {trends['second_half_avg']}%")
            lines.append("")
        
        lines.append("📈 COVERAGE DISTRIBUTION")
        lines.append("-" * 80)
        dist = analysis['coverage_distribution']
        lines.append(f"Excellent (≥95%): {dist['excellent']['count']} ({dist['excellent']['percentage']}%)")
        lines.append(f"Good (90-94%): {dist['good']['count']} ({dist['good']['percentage']}%)")
        lines.append(f"Acceptable (80-89%): {dist['acceptable']['count']} ({dist['acceptable']['percentage']}%)")
        lines.append(f"Poor (<80%): {dist['poor']['count']} ({dist['poor']['percentage']}%)")
        lines.append("")
        
        if analysis['coverage_gaps']:
            lines.append("🔴 COVERAGE GAPS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for gap in analysis['coverage_gaps'][:5]:  # Top 5
                emoji = severity_emoji.get(gap['severity'], '⚪')
                lines.append(f"{emoji} Run #{gap['run_index']}")
                lines.append(f"   Current: {gap['current_coverage']}%")
                lines.append(f"   Target: {gap['target_coverage']}%")
                lines.append(f"   Gap: {gap['gap']}%")
            lines.append("")
        
        if analysis.get('coverage_quality'):
            quality = analysis['coverage_quality']
            lines.append("✅ COVERAGE QUALITY")
            lines.append("-" * 80)
            lines.append(f"Coverage Score: {quality['coverage_score']}/100")
            lines.append(f"Consistency Score: {quality['consistency_score']}/100")
            lines.append(f"Quality Score: {quality['quality_score']}/100")
            lines.append(f"Test Density: {quality['test_density']} tests/%")
            lines.append(f"Quality Level: {quality['quality_level'].upper()}")
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
    
    analyzer = EnhancedCoverageAnalyzer(project_root)
    analysis = analyzer.analyze_coverage(lookback_days=30, target_coverage=80.0)
    
    report = analyzer.generate_coverage_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "enhanced_coverage_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Enhanced coverage analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






