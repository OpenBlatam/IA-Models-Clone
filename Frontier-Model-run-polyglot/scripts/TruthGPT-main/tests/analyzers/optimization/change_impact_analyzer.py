"""
Change Impact Analyzer
Analyze impact of changes on test results
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from statistics import mean, stdev

class ChangeImpactAnalyzer:
    """Analyze impact of changes"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.changes_file = project_root / "test_changes.json"
    
    def analyze_impact(self, lookback_days: int = 30) -> Dict:
        """Analyze impact of changes"""
        history = self._load_history()
        changes = self._load_changes()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Identify change points
        change_points = self._identify_change_points(recent, changes)
        
        # Analyze impact for each change
        impacts = []
        for change_point in change_points:
            impact = self._analyze_change_impact(recent, change_point)
            if impact:
                impacts.append(impact)
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'change_points': len(change_points),
            'impacts': impacts,
            'overall_impact': self._calculate_overall_impact(impacts)
        }
    
    def _identify_change_points(self, recent: List[Dict], changes: Dict) -> List[Dict]:
        """Identify points where changes occurred"""
        change_points = []
        
        # Look for significant changes in metrics
        if len(recent) < 3:
            return change_points
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        for i in range(1, len(recent)):
            prev_sr = success_rates[i-1]
            curr_sr = success_rates[i]
            
            # Significant change if difference > 5%
            if abs(curr_sr - prev_sr) > 5:
                change_points.append({
                    'index': i,
                    'timestamp': recent[i].get('timestamp', ''),
                    'type': 'success_rate_change',
                    'magnitude': abs(curr_sr - prev_sr)
                })
        
        return change_points
    
    def _analyze_change_impact(self, recent: List[Dict], change_point: Dict) -> Optional[Dict]:
        """Analyze impact of a specific change"""
        idx = change_point['index']
        
        if idx < 2 or idx >= len(recent) - 2:
            return None
        
        # Get before and after periods
        before = recent[:idx]
        after = recent[idx:]
        
        # Calculate metrics
        before_metrics = self._calculate_period_metrics(before)
        after_metrics = self._calculate_period_metrics(after)
        
        # Calculate impact
        impact = {
            'change_point': change_point['timestamp'][:19],
            'type': change_point['type'],
            'before_metrics': before_metrics,
            'after_metrics': after_metrics,
            'impact': {
                'success_rate_change': after_metrics['avg_success_rate'] - before_metrics['avg_success_rate'],
                'execution_time_change': after_metrics['avg_execution_time'] - before_metrics['avg_execution_time'],
                'failure_rate_change': after_metrics['failure_rate'] - before_metrics['failure_rate']
            },
            'severity': self._determine_severity(before_metrics, after_metrics)
        }
        
        return impact
    
    def _calculate_period_metrics(self, period: List[Dict]) -> Dict:
        """Calculate metrics for a period"""
        if not period:
            return {}
        
        success_rates = [r.get('success_rate', 0) for r in period]
        execution_times = [r.get('execution_time', 0) for r in period]
        total_tests = [r.get('total_tests', 0) for r in period]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in period]
        
        return {
            'avg_success_rate': round(mean(success_rates), 2) if success_rates else 0,
            'avg_execution_time': round(mean(execution_times), 2) if execution_times else 0,
            'failure_rate': round((sum(failures) / sum(total_tests) * 100) if sum(total_tests) > 0 else 0, 2)
        }
    
    def _determine_severity(self, before: Dict, after: Dict) -> str:
        """Determine severity of impact"""
        sr_change = abs(after['avg_success_rate'] - before['avg_success_rate'])
        fr_change = abs(after['failure_rate'] - before['failure_rate'])
        
        if sr_change > 10 or fr_change > 10:
            return 'high'
        elif sr_change > 5 or fr_change > 5:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_overall_impact(self, impacts: List[Dict]) -> Dict:
        """Calculate overall impact"""
        if not impacts:
            return {'score': 0, 'status': 'no_changes'}
        
        # Calculate average impact
        sr_changes = [i['impact']['success_rate_change'] for i in impacts]
        et_changes = [i['impact']['execution_time_change'] for i in impacts]
        
        avg_sr_change = mean(sr_changes) if sr_changes else 0
        avg_et_change = mean(et_changes) if et_changes else 0
        
        # Impact score (positive = improvement, negative = degradation)
        impact_score = (avg_sr_change * 0.7) - (abs(avg_et_change) / 10 * 0.3)
        
        if impact_score > 2:
            status = 'positive'
        elif impact_score < -2:
            status = 'negative'
        else:
            status = 'neutral'
        
        return {
            'score': round(impact_score, 2),
            'status': status,
            'avg_success_rate_change': round(avg_sr_change, 2),
            'avg_execution_time_change': round(avg_et_change, 2)
        }
    
    def generate_impact_report(self, analysis: Dict) -> str:
        """Generate impact report"""
        lines = []
        lines.append("=" * 80)
        lines.append("CHANGE IMPACT ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append(f"Change Points Detected: {analysis['change_points']}")
        lines.append("")
        
        overall = analysis['overall_impact']
        status_emoji = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡', 'no_changes': '⚪'}
        emoji = status_emoji.get(overall['status'], '⚪')
        lines.append(f"{emoji} Overall Impact: {overall['status'].upper()}")
        lines.append(f"Impact Score: {overall['score']:+.2f}")
        lines.append("")
        
        if analysis['impacts']:
            lines.append("📊 INDIVIDUAL IMPACTS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            
            for i, impact in enumerate(analysis['impacts'], 1):
                emoji = severity_emoji.get(impact['severity'], '⚪')
                lines.append(f"\n{emoji} Change #{i} - {impact['change_point']}")
                lines.append(f"   Type: {impact['type']}")
                lines.append(f"   Severity: {impact['severity']}")
                lines.append(f"   Success Rate Change: {impact['impact']['success_rate_change']:+.2f}%")
                lines.append(f"   Execution Time Change: {impact['impact']['execution_time_change']:+.2f}s")
                lines.append(f"   Failure Rate Change: {impact['impact']['failure_rate_change']:+.2f}%")
            lines.append("")
        
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
    
    def _load_changes(self) -> Dict:
        """Load change information"""
        if not self.changes_file.exists():
            return {}
        
        try:
            with open(self.changes_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

def main():
    """Main function"""
    from pathlib import Path
    from typing import Optional
    
    project_root = Path(__file__).parent.parent
    
    analyzer = ChangeImpactAnalyzer(project_root)
    analysis = analyzer.analyze_impact(lookback_days=30)
    
    report = analyzer.generate_impact_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "change_impact_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Change impact analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()

