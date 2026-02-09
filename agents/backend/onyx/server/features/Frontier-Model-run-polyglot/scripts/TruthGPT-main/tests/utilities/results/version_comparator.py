"""
Version Comparator
Compare test results across versions
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from statistics import mean

class VersionComparator:
    """Compare test results across versions"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.versions_file = project_root / "test_versions.json"
    
    def compare_versions(self, version1: str, version2: str, lookback_days: int = 30) -> Dict:
        """Compare two versions"""
        history = self._load_history()
        versions = self._load_versions()
        
        # Get runs for each version
        v1_runs = self._get_version_runs(history, version1, lookback_days)
        v2_runs = self._get_version_runs(history, version2, lookback_days)
        
        if not v1_runs or not v2_runs:
            return {'error': 'Insufficient data for one or both versions'}
        
        # Calculate metrics for each version
        v1_metrics = self._calculate_metrics(v1_runs)
        v2_metrics = self._calculate_metrics(v2_runs)
        
        # Compare metrics
        comparison = {
            'version1': version1,
            'version2': version2,
            'version1_metrics': v1_metrics,
            'version2_metrics': v2_metrics,
            'comparisons': self._compare_metrics(v1_metrics, v2_metrics),
            'overall_improvement': self._calculate_overall_improvement(v1_metrics, v2_metrics)
        }
        
        return comparison
    
    def _get_version_runs(self, history: List[Dict], version: str, lookback_days: int) -> List[Dict]:
        """Get runs for a specific version"""
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        
        # Filter by version (if version info is in metadata)
        version_runs = []
        for r in history:
            if r.get('timestamp', '') >= cutoff_date:
                metadata = r.get('metadata', {})
                if metadata.get('version') == version or version == 'all':
                    version_runs.append(r)
        
        # If no version-specific runs, use all recent runs
        if not version_runs:
            version_runs = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        return version_runs
    
    def _calculate_metrics(self, runs: List[Dict]) -> Dict:
        """Calculate metrics for a set of runs"""
        success_rates = [r.get('success_rate', 0) for r in runs]
        execution_times = [r.get('execution_time', 0) for r in runs]
        total_tests = [r.get('total_tests', 0) for r in runs]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in runs]
        
        return {
            'total_runs': len(runs),
            'avg_success_rate': round(mean(success_rates), 2) if success_rates else 0,
            'avg_execution_time': round(mean(execution_times), 2) if execution_times else 0,
            'total_tests': sum(total_tests),
            'total_failures': sum(failures),
            'failure_rate': round((sum(failures) / sum(total_tests) * 100) if sum(total_tests) > 0 else 0, 2)
        }
    
    def _compare_metrics(self, v1_metrics: Dict, v2_metrics: Dict) -> Dict:
        """Compare metrics between versions"""
        comparisons = {}
        
        # Success rate comparison
        sr_diff = v2_metrics['avg_success_rate'] - v1_metrics['avg_success_rate']
        comparisons['success_rate'] = {
            'v1': v1_metrics['avg_success_rate'],
            'v2': v2_metrics['avg_success_rate'],
            'difference': round(sr_diff, 2),
            'improvement': sr_diff > 0
        }
        
        # Execution time comparison
        et_diff = v2_metrics['avg_execution_time'] - v1_metrics['avg_execution_time']
        comparisons['execution_time'] = {
            'v1': v1_metrics['avg_execution_time'],
            'v2': v2_metrics['avg_execution_time'],
            'difference': round(et_diff, 2),
            'improvement': et_diff < 0  # Lower is better
        }
        
        # Failure rate comparison
        fr_diff = v2_metrics['failure_rate'] - v1_metrics['failure_rate']
        comparisons['failure_rate'] = {
            'v1': v1_metrics['failure_rate'],
            'v2': v2_metrics['failure_rate'],
            'difference': round(fr_diff, 2),
            'improvement': fr_diff < 0  # Lower is better
        }
        
        return comparisons
    
    def _calculate_overall_improvement(self, v1_metrics: Dict, v2_metrics: Dict) -> float:
        """Calculate overall improvement score"""
        improvements = []
        
        # Success rate improvement (0-50 points)
        sr_improvement = (v2_metrics['avg_success_rate'] - v1_metrics['avg_success_rate']) / 100 * 50
        improvements.append(max(0, min(50, sr_improvement + 25)))
        
        # Execution time improvement (0-30 points)
        if v1_metrics['avg_execution_time'] > 0:
            et_improvement = (1 - (v2_metrics['avg_execution_time'] / v1_metrics['avg_execution_time'])) * 30
            improvements.append(max(0, min(30, et_improvement + 15)))
        else:
            improvements.append(15)
        
        # Failure rate improvement (0-20 points)
        fr_improvement = (v1_metrics['failure_rate'] - v2_metrics['failure_rate']) / 10 * 20
        improvements.append(max(0, min(20, fr_improvement + 10)))
        
        return round(sum(improvements), 1)
    
    def generate_comparison_report(self, comparison: Dict) -> str:
        """Generate comparison report"""
        lines = []
        lines.append("=" * 80)
        lines.append("VERSION COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in comparison:
            lines.append(f"❌ {comparison['error']}")
            return "\n".join(lines)
        
        lines.append(f"Version 1: {comparison['version1']}")
        lines.append(f"Version 2: {comparison['version2']}")
        lines.append("")
        
        score_emoji = "🟢" if comparison['overall_improvement'] >= 70 else "🟡" if comparison['overall_improvement'] >= 50 else "🔴"
        lines.append(f"{score_emoji} Overall Improvement Score: {comparison['overall_improvement']}/100")
        lines.append("")
        
        lines.append("📊 METRIC COMPARISONS")
        lines.append("-" * 80)
        
        for metric_name, comp in comparison['comparisons'].items():
            improvement_emoji = "✅" if comp['improvement'] else "⚠️"
            lines.append(f"\n{improvement_emoji} {metric_name.replace('_', ' ').title()}")
            lines.append(f"   Version 1: {comp['v1']}")
            lines.append(f"   Version 2: {comp['v2']}")
            lines.append(f"   Difference: {comp['difference']:+.2f}")
            lines.append(f"   Status: {'Improved' if comp['improvement'] else 'Degraded'}")
        
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
    
    def _load_versions(self) -> Dict:
        """Load version information"""
        if not self.versions_file.exists():
            return {}
        
        try:
            with open(self.versions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

def main():
    """Main function"""
    from pathlib import Path
    import sys
    
    project_root = Path(__file__).parent.parent
    comparator = VersionComparator(project_root)
    
    version1 = sys.argv[1] if len(sys.argv) > 1 else 'v1.0'
    version2 = sys.argv[2] if len(sys.argv) > 2 else 'v2.0'
    
    comparison = comparator.compare_versions(version1, version2)
    
    report = comparator.generate_comparison_report(comparison)
    print(report)
    
    # Save report
    report_file = project_root / f"version_comparison_{version1}_vs_{version2}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Version comparison report saved to: {report_file}")

if __name__ == "__main__":
    main()







