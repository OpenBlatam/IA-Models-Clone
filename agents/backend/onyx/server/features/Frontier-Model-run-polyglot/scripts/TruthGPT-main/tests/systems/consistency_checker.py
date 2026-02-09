"""
Consistency Checker
Check test result consistency
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev
from collections import defaultdict

class ConsistencyChecker:
    """Check test result consistency"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def check_consistency(self, lookback_days: int = 30) -> Dict:
        """Check test result consistency"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Check consistency across multiple dimensions
        consistency_results = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'overall_consistency': 0.0,
            'success_rate_consistency': self._check_success_rate_consistency(recent),
            'execution_time_consistency': self._check_execution_time_consistency(recent),
            'failure_pattern_consistency': self._check_failure_pattern_consistency(recent),
            'inconsistencies': []
        }
        
        # Calculate overall consistency
        consistency_scores = [
            consistency_results['success_rate_consistency']['score'],
            consistency_results['execution_time_consistency']['score']
        ]
        consistency_results['overall_consistency'] = round(mean(consistency_scores), 1)
        
        # Identify inconsistencies
        consistency_results['inconsistencies'] = self._identify_inconsistencies(consistency_results)
        
        return consistency_results
    
    def _check_success_rate_consistency(self, recent: List[Dict]) -> Dict:
        """Check success rate consistency"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if not success_rates:
            return {'score': 0, 'consistent': False, 'variance': 0}
        
        mean_sr = mean(success_rates)
        std_sr = stdev(success_rates) if len(success_rates) > 1 else 0
        
        # Consistency score: 100 - (std_dev * 2)
        score = max(0, 100 - (std_sr * 2))
        consistent = std_sr < 5  # Consider consistent if std dev < 5%
        
        return {
            'score': round(score, 1),
            'consistent': consistent,
            'mean': round(mean_sr, 2),
            'std': round(std_sr, 2),
            'min': round(min(success_rates), 2),
            'max': round(max(success_rates), 2)
        }
    
    def _check_execution_time_consistency(self, recent: List[Dict]) -> Dict:
        """Check execution time consistency"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        if not execution_times:
            return {'score': 0, 'consistent': False, 'variance': 0}
        
        mean_et = mean(execution_times)
        std_et = stdev(execution_times) if len(execution_times) > 1 else 0
        
        # Coefficient of variation
        cv = (std_et / mean_et * 100) if mean_et > 0 else 100
        
        # Consistency score: 100 - (CV * 2)
        score = max(0, 100 - (cv * 2))
        consistent = cv < 15  # Consider consistent if CV < 15%
        
        return {
            'score': round(score, 1),
            'consistent': consistent,
            'mean': round(mean_et, 2),
            'std': round(std_et, 2),
            'coefficient_of_variation': round(cv, 2),
            'min': round(min(execution_times), 2),
            'max': round(max(execution_times), 2)
        }
    
    def _check_failure_pattern_consistency(self, recent: List[Dict]) -> Dict:
        """Check failure pattern consistency"""
        failure_counts = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        if not failure_counts:
            return {'consistent': True, 'pattern': 'stable'}
        
        # Check if failures are consistent (similar counts) or erratic
        mean_failures = mean(failure_counts)
        std_failures = stdev(failure_counts) if len(failure_counts) > 1 else 0
        
        # If std dev is high relative to mean, pattern is inconsistent
        if mean_failures > 0:
            cv = (std_failures / mean_failures * 100)
            consistent = cv < 50
        else:
            consistent = True
        
        return {
            'consistent': consistent,
            'mean_failures': round(mean_failures, 2),
            'std_failures': round(std_failures, 2),
            'pattern': 'stable' if consistent else 'erratic'
        }
    
    def _identify_inconsistencies(self, results: Dict) -> List[Dict]:
        """Identify specific inconsistencies"""
        inconsistencies = []
        
        if not results['success_rate_consistency']['consistent']:
            inconsistencies.append({
                'type': 'success_rate',
                'severity': 'high',
                'description': 'Success rate shows high variance',
                'details': f"Range: {results['success_rate_consistency']['min']}% - {results['success_rate_consistency']['max']}%"
            })
        
        if not results['execution_time_consistency']['consistent']:
            inconsistencies.append({
                'type': 'execution_time',
                'severity': 'medium',
                'description': 'Execution time shows high variance',
                'details': f"Range: {results['execution_time_consistency']['min']}s - {results['execution_time_consistency']['max']}s"
            })
        
        if not results['failure_pattern_consistency']['consistent']:
            inconsistencies.append({
                'type': 'failure_pattern',
                'severity': 'high',
                'description': 'Failure pattern is erratic',
                'details': 'Inconsistent failure counts across runs'
            })
        
        return inconsistencies
    
    def generate_consistency_report(self, results: Dict) -> str:
        """Generate consistency report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST CONSISTENCY CHECK REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in results:
            lines.append(f"❌ {results['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {results['period']}")
        lines.append(f"Total Runs: {results['total_runs']}")
        lines.append("")
        
        score_emoji = "🟢" if results['overall_consistency'] >= 80 else "🟡" if results['overall_consistency'] >= 60 else "🔴"
        lines.append(f"{score_emoji} Overall Consistency: {results['overall_consistency']}/100")
        lines.append("")
        
        lines.append("📊 SUCCESS RATE CONSISTENCY")
        lines.append("-" * 80)
        sr_cons = results['success_rate_consistency']
        status = "✅ Consistent" if sr_cons['consistent'] else "⚠️ Inconsistent"
        lines.append(f"{status} (Score: {sr_cons['score']}/100)")
        lines.append(f"Mean: {sr_cons['mean']}% | Std: {sr_cons['std']}")
        lines.append(f"Range: {sr_cons['min']}% - {sr_cons['max']}%")
        lines.append("")
        
        lines.append("⏱️ EXECUTION TIME CONSISTENCY")
        lines.append("-" * 80)
        et_cons = results['execution_time_consistency']
        status = "✅ Consistent" if et_cons['consistent'] else "⚠️ Inconsistent"
        lines.append(f"{status} (Score: {et_cons['score']}/100)")
        lines.append(f"Mean: {et_cons['mean']}s | Std: {et_cons['std']}s")
        lines.append(f"Range: {et_cons['min']}s - {et_cons['max']}s")
        if 'coefficient_of_variation' in et_cons:
            lines.append(f"Coefficient of Variation: {et_cons['coefficient_of_variation']}%")
        lines.append("")
        
        if results['inconsistencies']:
            lines.append("⚠️ IDENTIFIED INCONSISTENCIES")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for inc in results['inconsistencies']:
                emoji = severity_emoji.get(inc['severity'], '⚪')
                lines.append(f"{emoji} [{inc['severity'].upper()}] {inc['type'].replace('_', ' ').title()}")
                lines.append(f"   {inc['description']}")
                lines.append(f"   {inc['details']}")
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

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    checker = ConsistencyChecker(project_root)
    results = checker.check_consistency(lookback_days=30)
    
    report = checker.generate_consistency_report(results)
    print(report)
    
    # Save report
    report_file = project_root / "consistency_check_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Consistency check report saved to: {report_file}")

if __name__ == "__main__":
    main()







