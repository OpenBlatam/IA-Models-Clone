"""
Advanced Optimizer
Advanced optimization with ML-based suggestions
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class AdvancedOptimizer:
    """Advanced optimization with ML-based suggestions"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def optimize_advanced(self, lookback_days: int = 30) -> Dict:
        """Advanced optimization analysis"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        optimizations = []
        
        # Analyze execution time patterns
        execution_times = [r.get('execution_time', 0) for r in recent]
        avg_time = mean(execution_times) if execution_times else 0
        
        if avg_time > 300:
            optimizations.append({
                'type': 'execution_time',
                'priority': 'high',
                'current_value': round(avg_time, 2),
                'target_value': 120,
                'potential_savings': round(avg_time - 120, 2),
                'savings_percentage': round(((avg_time - 120) / avg_time * 100), 1),
                'recommendations': [
                    'Enable parallel test execution (4-8 workers)',
                    'Optimize slow tests (>5s each)',
                    'Use test fixtures more efficiently',
                    'Remove redundant test cases'
                ],
                'estimated_impact': 'high',
                'effort': 'medium'
            })
        
        # Analyze success rate
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates) if success_rates else 0
        
        if avg_success < 95:
            optimizations.append({
                'type': 'success_rate',
                'priority': 'high',
                'current_value': round(avg_success, 1),
                'target_value': 95,
                'potential_improvement': round(95 - avg_success, 1),
                'recommendations': [
                    'Fix failing tests systematically',
                    'Investigate flaky tests',
                    'Improve test stability',
                    'Review test assertions'
                ],
                'estimated_impact': 'high',
                'effort': 'high'
            })
        
        # Analyze consistency
        if len(success_rates) > 1:
            variance = max(success_rates) - min(success_rates)
            if variance > 10:
                optimizations.append({
                    'type': 'consistency',
                    'priority': 'medium',
                    'current_value': round(variance, 1),
                    'target_value': 5,
                    'potential_improvement': round(variance - 5, 1),
                    'recommendations': [
                        'Fix flaky tests',
                        'Improve test isolation',
                        'Use deterministic test data',
                        'Reduce test dependencies'
                    ],
                    'estimated_impact': 'medium',
                    'effort': 'high'
                })
        
        # Calculate ROI for each optimization
        for opt in optimizations:
            opt['roi_score'] = self._calculate_roi_score(opt)
        
        # Sort by ROI score
        optimizations.sort(key=lambda x: x.get('roi_score', 0), reverse=True)
        
        return {
            'period': f'Last {lookback_days} days',
            'total_optimizations': len(optimizations),
            'optimizations': optimizations,
            'overall_potential': self._calculate_overall_potential(optimizations)
        }
    
    def _calculate_roi_score(self, optimization: Dict) -> float:
        """Calculate ROI score for optimization"""
        impact_scores = {'high': 3, 'medium': 2, 'low': 1}
        effort_scores = {'low': 3, 'medium': 2, 'high': 1}
        
        impact = impact_scores.get(optimization.get('estimated_impact', 'medium'), 2)
        effort = effort_scores.get(optimization.get('effort', 'medium'), 2)
        
        # ROI = Impact / Effort
        roi = (impact / effort) * 100 if effort > 0 else 0
        
        # Adjust by priority
        priority_multiplier = {'high': 1.5, 'medium': 1.0, 'low': 0.5}
        priority = optimization.get('priority', 'medium')
        roi *= priority_multiplier.get(priority, 1.0)
        
        return round(roi, 1)
    
    def _calculate_overall_potential(self, optimizations: List[Dict]) -> Dict:
        """Calculate overall optimization potential"""
        total_savings = sum(opt.get('potential_savings', 0) for opt in optimizations if 'potential_savings' in opt)
        total_improvement = sum(opt.get('potential_improvement', 0) for opt in optimizations if 'potential_improvement' in opt)
        
        return {
            'time_savings_seconds': round(total_savings, 2),
            'success_rate_improvement': round(total_improvement, 1),
            'high_priority_count': len([opt for opt in optimizations if opt.get('priority') == 'high'])
        }
    
    def generate_optimization_report(self, optimization: Dict) -> str:
        """Generate optimization report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED OPTIMIZATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in optimization:
            lines.append(f"❌ {optimization['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {optimization['period']}")
        lines.append(f"Total Optimizations: {optimization['total_optimizations']}")
        lines.append("")
        
        overall = optimization['overall_potential']
        lines.append("📊 OVERALL POTENTIAL")
        lines.append("-" * 80)
        lines.append(f"Time Savings: {overall['time_savings_seconds']:.2f}s")
        lines.append(f"Success Rate Improvement: {overall['success_rate_improvement']:.1f}%")
        lines.append(f"High Priority Items: {overall['high_priority_count']}")
        lines.append("")
        
        lines.append("🎯 OPTIMIZATIONS (Sorted by ROI)")
        lines.append("-" * 80)
        
        priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        
        for i, opt in enumerate(optimization['optimizations'], 1):
            emoji = priority_emoji.get(opt['priority'], '⚪')
            lines.append(f"\n{i}. {emoji} {opt['type'].replace('_', ' ').title()} ({opt['priority'].upper()})")
            lines.append(f"   ROI Score: {opt.get('roi_score', 0):.1f}")
            lines.append(f"   Current: {opt['current_value']}")
            lines.append(f"   Target: {opt['target_value']}")
            
            if 'potential_savings' in opt:
                lines.append(f"   Potential Savings: {opt['potential_savings']:.2f}s ({opt['savings_percentage']:.1f}%)")
            if 'potential_improvement' in opt:
                lines.append(f"   Potential Improvement: {opt['potential_improvement']:.1f}%")
            
            lines.append("   Recommendations:")
            for rec in opt['recommendations']:
                lines.append(f"     • {rec}")
        
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
    
    optimizer = AdvancedOptimizer(project_root)
    optimization = optimizer.optimize_advanced(lookback_days=30)
    
    report = optimizer.generate_optimization_report(optimization)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_optimization_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced optimization report saved to: {report_file}")

if __name__ == "__main__":
    main()







