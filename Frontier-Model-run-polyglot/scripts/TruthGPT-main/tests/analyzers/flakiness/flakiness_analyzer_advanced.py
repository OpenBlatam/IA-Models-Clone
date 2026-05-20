"""
Advanced Flakiness Analyzer
Advanced flakiness analysis with detailed insights
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from collections import Counter
from statistics import mean, stdev

class AdvancedFlakinessAnalyzer:
    """Advanced flakiness analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_flakiness(self, lookback_days: int = 30, flakiness_threshold: float = 0.3) -> Dict:
        """Analyze test flakiness"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Analyze flakiness
        flakiness_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'flakiness_threshold': flakiness_threshold,
            'overall_flakiness': self._calculate_overall_flakiness(recent),
            'flakiness_distribution': self._analyze_flakiness_distribution(recent),
            'flaky_tests': self._identify_flaky_tests(recent, flakiness_threshold),
            'flakiness_trends': self._analyze_flakiness_trends(recent),
            'flakiness_patterns': self._identify_flakiness_patterns(recent),
            'impact_analysis': self._analyze_flakiness_impact(recent),
            'recommendations': []
        }
        
        # Generate recommendations
        flakiness_analysis['recommendations'] = self._generate_flakiness_recommendations(flakiness_analysis)
        
        return flakiness_analysis
    
    def _calculate_overall_flakiness(self, recent: List[Dict]) -> Dict:
        """Calculate overall flakiness metrics"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if not success_rates:
            return {}
        
        avg_success = mean(success_rates)
        success_std = stdev(success_rates) if len(success_rates) > 1 else 0
        
        # Flakiness score: higher variance = higher flakiness
        flakiness_score = min(100, success_std * 2)
        
        # Consistency score
        consistency_score = max(0, 100 - (success_std * 2))
        
        return {
            'avg_success_rate': round(avg_success, 2),
            'success_rate_std': round(success_std, 2),
            'flakiness_score': round(flakiness_score, 1),
            'consistency_score': round(consistency_score, 1),
            'is_flaky': success_std > 5
        }
    
    def _analyze_flakiness_distribution(self, recent: List[Dict]) -> Dict:
        """Analyze flakiness distribution"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if not success_rates:
            return {}
        
        # Categorize by consistency
        consistent = sum(1 for sr in success_rates if abs(sr - mean(success_rates)) < 2)
        slightly_flaky = sum(1 for sr in success_rates if 2 <= abs(sr - mean(success_rates)) < 5)
        moderately_flaky = sum(1 for sr in success_rates if 5 <= abs(sr - mean(success_rates)) < 10)
        highly_flaky = sum(1 for sr in success_rates if abs(sr - mean(success_rates)) >= 10)
        
        total = len(success_rates)
        
        return {
            'consistent': {
                'count': consistent,
                'percentage': round((consistent / total * 100) if total > 0 else 0, 1)
            },
            'slightly_flaky': {
                'count': slightly_flaky,
                'percentage': round((slightly_flaky / total * 100) if total > 0 else 0, 1)
            },
            'moderately_flaky': {
                'count': moderately_flaky,
                'percentage': round((moderately_flaky / total * 100) if total > 0 else 0, 1)
            },
            'highly_flaky': {
                'count': highly_flaky,
                'percentage': round((highly_flaky / total * 100) if total > 0 else 0, 1)
            }
        }
    
    def _identify_flaky_tests(self, recent: List[Dict], threshold: float) -> List[Dict]:
        """Identify flaky tests"""
        flaky_tests = []
        
        # Analyze success rate variance
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if len(success_rates) < 3:
            return flaky_tests
        
        avg_success = mean(success_rates)
        success_std = stdev(success_rates) if len(success_rates) > 1 else 0
        
        # Identify runs with high variance
        for i, r in enumerate(recent):
            success_rate = r.get('success_rate', 0)
            deviation = abs(success_rate - avg_success)
            
            if deviation > threshold * 100 or success_std > 5:
                flakiness_level = 'high' if deviation > 15 else 'medium' if deviation > 10 else 'low'
                
                flaky_tests.append({
                    'run_index': i,
                    'timestamp': r.get('timestamp', ''),
                    'success_rate': round(success_rate, 2),
                    'deviation': round(deviation, 2),
                    'flakiness_level': flakiness_level,
                    'total_tests': r.get('total_tests', 0),
                    'failures': r.get('failures', 0) + r.get('errors', 0)
                })
        
        return sorted(flaky_tests, key=lambda x: x['deviation'], reverse=True)[:10]  # Top 10
    
    def _analyze_flakiness_trends(self, recent: List[Dict]) -> Dict:
        """Analyze flakiness trends"""
        if len(recent) < 4:
            return {}
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        # Split into halves
        first_half = success_rates[:len(success_rates)//2]
        second_half = success_rates[len(success_rates)//2:]
        
        first_std = stdev(first_half) if len(first_half) > 1 else 0
        second_std = stdev(second_half) if len(second_half) > 1 else 0
        
        trend = second_std - first_std
        
        return {
            'first_half_std': round(first_std, 2),
            'second_half_std': round(second_std, 2),
            'trend': round(trend, 2),
            'direction': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
            'interpretation': 'Flakiness is increasing' if trend > 1 else 'Flakiness is decreasing' if trend < -1 else 'Flakiness is stable'
        }
    
    def _identify_flakiness_patterns(self, recent: List[Dict]) -> List[Dict]:
        """Identify flakiness patterns"""
        patterns = []
        
        if len(recent) < 5:
            return patterns
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates)
        
        # Alternating pattern
        if len(success_rates) >= 4:
            recent_4 = success_rates[-4:]
            if all(abs(recent_4[i] - avg_success) > abs(recent_4[i+1] - avg_success) for i in range(0, 3, 2)):
                patterns.append({
                    'pattern': 'alternating',
                    'description': 'Alternating high and low success rates',
                    'severity': 'medium',
                    'recommendation': 'Investigate timing or resource contention issues'
                })
        
        # Gradual decline pattern
        if len(success_rates) >= 5:
            recent_5 = success_rates[-5:]
            if all(recent_5[i] > recent_5[i+1] for i in range(len(recent_5)-1)):
                patterns.append({
                    'pattern': 'gradual_decline',
                    'description': 'Gradual decline in success rate over time',
                    'severity': 'high',
                    'recommendation': 'Check for resource leaks or accumulating issues'
                })
        
        # Spike pattern
        if len(success_rates) >= 3:
            recent_3 = success_rates[-3:]
            if abs(recent_3[-1] - avg_success) > 20:
                patterns.append({
                    'pattern': 'sudden_spike',
                    'description': 'Sudden spike in flakiness',
                    'severity': 'high',
                    'recommendation': 'Investigate recent changes or environmental issues'
                })
        
        return patterns
    
    def _analyze_flakiness_impact(self, recent: List[Dict]) -> Dict:
        """Analyze flakiness impact"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if not success_rates:
            return {}
        
        avg_success = mean(success_rates)
        success_std = stdev(success_rates) if len(success_rates) > 1 else 0
        
        # Calculate impact
        unreliable_runs = sum(1 for sr in success_rates if abs(sr - avg_success) > 5)
        unreliable_percentage = (unreliable_runs / len(success_rates) * 100) if success_rates else 0
        
        # Estimate time wasted
        total_time = sum(r.get('execution_time', 0) for r in recent)
        estimated_waste = total_time * (success_std / 100)
        
        return {
            'total_runs': len(recent),
            'unreliable_runs': unreliable_runs,
            'unreliable_percentage': round(unreliable_percentage, 1),
            'avg_success_rate': round(avg_success, 1),
            'success_rate_variance': round(success_std, 2),
            'estimated_time_waste': round(estimated_waste, 2),
            'impact_level': 'high' if unreliable_percentage > 30 else 'medium' if unreliable_percentage > 15 else 'low'
        }
    
    def _generate_flakiness_recommendations(self, analysis: Dict) -> List[str]:
        """Generate flakiness recommendations"""
        recommendations = []
        
        overall = analysis['overall_flakiness']
        if overall.get('is_flaky', False):
            recommendations.append(f"🚨 Test suite is flaky (std: {overall['success_rate_std']:.1f}%) - improve test stability")
        
        dist = analysis['flakiness_distribution']
        if dist.get('highly_flaky', {}).get('percentage', 0) > 10:
            recommendations.append(f"High flakiness in {dist['highly_flaky']['percentage']:.1f}% of runs - investigate root causes")
        
        if analysis['flaky_tests']:
            high_flaky = sum(1 for t in analysis['flaky_tests'] if t['flakiness_level'] == 'high')
            if high_flaky > 0:
                recommendations.append(f"🚨 {high_flaky} highly flaky test run(s) identified - prioritize fixing")
        
        patterns = analysis.get('flakiness_patterns', [])
        for pattern in patterns:
            if pattern['severity'] == 'high':
                recommendations.append(f"⚠️ {pattern['description']} - {pattern['recommendation']}")
        
        impact = analysis.get('impact_analysis', {})
        if impact.get('impact_level') == 'high':
            recommendations.append(f"High flakiness impact ({impact['unreliable_percentage']:.1f}% unreliable runs) - reduce variance")
        
        trends = analysis.get('flakiness_trends', {})
        if trends.get('direction') == 'increasing':
            recommendations.append("Flakiness trend is increasing - take preventive action")
        
        if not recommendations:
            recommendations.append("✅ Flakiness is within acceptable limits - maintain current practices")
        
        return recommendations
    
    def generate_flakiness_report(self, analysis: Dict) -> str:
        """Generate flakiness report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED FLAKINESS ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append(f"Flakiness Threshold: {analysis['flakiness_threshold']*100}%")
        lines.append("")
        
        lines.append("📊 OVERALL FLAKINESS")
        lines.append("-" * 80)
        overall = analysis['overall_flakiness']
        status_emoji = "🔴" if overall.get('is_flaky') else "🟢"
        lines.append(f"{status_emoji} Flakiness Status: {'FLAKY' if overall.get('is_flaky') else 'STABLE'}")
        lines.append(f"Average Success Rate: {overall['avg_success_rate']}%")
        lines.append(f"Success Rate Std Dev: {overall['success_rate_std']}%")
        lines.append(f"Flakiness Score: {overall['flakiness_score']}/100")
        lines.append(f"Consistency Score: {overall['consistency_score']}/100")
        lines.append("")
        
        lines.append("📈 FLAKINESS DISTRIBUTION")
        lines.append("-" * 80)
        dist = analysis['flakiness_distribution']
        lines.append(f"Consistent: {dist['consistent']['count']} ({dist['consistent']['percentage']}%)")
        lines.append(f"Slightly Flaky: {dist['slightly_flaky']['count']} ({dist['slightly_flaky']['percentage']}%)")
        lines.append(f"Moderately Flaky: {dist['moderately_flaky']['count']} ({dist['moderately_flaky']['percentage']}%)")
        lines.append(f"Highly Flaky: {dist['highly_flaky']['count']} ({dist['highly_flaky']['percentage']}%)")
        lines.append("")
        
        if analysis['flaky_tests']:
            lines.append("🔴 FLAKY TESTS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for test in analysis['flaky_tests'][:5]:  # Top 5
                emoji = severity_emoji.get(test['flakiness_level'], '⚪')
                lines.append(f"{emoji} Run #{test['run_index']}")
                lines.append(f"   Success Rate: {test['success_rate']}%")
                lines.append(f"   Deviation: {test['deviation']}%")
                lines.append(f"   Flakiness Level: {test['flakiness_level'].upper()}")
            lines.append("")
        
        if analysis.get('flakiness_trends'):
            trends = analysis['flakiness_trends']
            lines.append("📊 FLAKINESS TRENDS")
            lines.append("-" * 80)
            lines.append(f"First Half Std: {trends['first_half_std']}%")
            lines.append(f"Second Half Std: {trends['second_half_std']}%")
            lines.append(f"Trend: {trends['trend']:+.2f}%")
            lines.append(f"Direction: {trends['direction'].title()}")
            lines.append(f"Interpretation: {trends['interpretation']}")
            lines.append("")
        
        if analysis.get('flakiness_patterns'):
            lines.append("🔍 FLAKINESS PATTERNS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for pattern in analysis['flakiness_patterns']:
                emoji = severity_emoji.get(pattern['severity'], '⚪')
                lines.append(f"{emoji} {pattern['pattern'].replace('_', ' ').title()}")
                lines.append(f"   {pattern['description']}")
                lines.append(f"   {pattern['recommendation']}")
            lines.append("")
        
        if analysis.get('impact_analysis'):
            impact = analysis['impact_analysis']
            lines.append("💥 IMPACT ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"Total Runs: {impact['total_runs']}")
            lines.append(f"Unreliable Runs: {impact['unreliable_runs']} ({impact['unreliable_percentage']}%)")
            lines.append(f"Estimated Time Waste: {impact['estimated_time_waste']:.2f}s")
            lines.append(f"Impact Level: {impact['impact_level'].upper()}")
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
    
    analyzer = AdvancedFlakinessAnalyzer(project_root)
    analysis = analyzer.analyze_flakiness(lookback_days=30, flakiness_threshold=0.3)
    
    report = analyzer.generate_flakiness_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_flakiness_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced flakiness analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






