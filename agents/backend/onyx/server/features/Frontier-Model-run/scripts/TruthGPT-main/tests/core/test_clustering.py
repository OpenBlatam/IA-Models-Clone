"""
Test Clustering
Groups similar tests together for analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re

class TestClustering:
    """Cluster tests by various criteria"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def cluster_by_name_pattern(self) -> Dict[str, List[str]]:
        """Cluster tests by name patterns"""
        clusters = defaultdict(list)
        
        # Load all test results
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                test_details = data.get('test_details', {})
                
                # Extract test names
                all_tests = []
                for test_list in [
                    test_details.get('failures', []),
                    test_details.get('errors', []),
                    test_details.get('skipped', [])
                ]:
                    for test in test_list:
                        test_name = str(test.get('test', ''))
                        if test_name:
                            all_tests.append(test_name)
                
                # Cluster by pattern
                for test_name in all_tests:
                    # Extract pattern (e.g., test_inference_*)
                    pattern = self._extract_pattern(test_name)
                    clusters[pattern].append(test_name)
            except Exception:
                continue
        
        # Remove duplicates
        for pattern in clusters:
            clusters[pattern] = list(set(clusters[pattern]))
        
        return dict(clusters)
    
    def _extract_pattern(self, test_name: str) -> str:
        """Extract pattern from test name"""
        # Try to extract common patterns
        if 'test_' in test_name:
            parts = test_name.split('test_')
            if len(parts) > 1:
                # Get category (e.g., inference, training)
                category = parts[1].split('_')[0]
                return f"test_{category}_*"
        
        # Default: use first part
        parts = test_name.split('_')
        if len(parts) > 1:
            return f"{parts[0]}_{parts[1]}_*"
        
        return "other"
    
    def cluster_by_failure_pattern(self) -> Dict[str, List[str]]:
        """Cluster tests by failure patterns"""
        clusters = defaultdict(list)
        
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                failures = data.get('test_details', {}).get('failures', [])
                
                for failure in failures:
                    test_name = str(failure.get('test', ''))
                    traceback = failure.get('traceback', '')
                    
                    # Extract error type from traceback
                    error_type = self._extract_error_type(traceback)
                    clusters[error_type].append(test_name)
            except Exception:
                continue
        
        # Remove duplicates
        for error_type in clusters:
            clusters[error_type] = list(set(clusters[error_type]))
        
        return dict(clusters)
    
    def _extract_error_type(self, traceback: str) -> str:
        """Extract error type from traceback"""
        common_errors = [
            'AssertionError',
            'ValueError',
            'TypeError',
            'AttributeError',
            'KeyError',
            'ImportError',
            'RuntimeError',
            'TimeoutError'
        ]
        
        for error in common_errors:
            if error in traceback:
                return error
        
        return "OtherError"
    
    def cluster_by_execution_time(self, history: List[Dict]) -> Dict[str, List[str]]:
        """Cluster tests by execution time"""
        # This would require per-test timing data
        # For now, return empty
        return {}
    
    def generate_clustering_report(self) -> str:
        """Generate clustering report"""
        name_clusters = self.cluster_by_name_pattern()
        failure_clusters = self.cluster_by_failure_pattern()
        
        lines = []
        lines.append("=" * 80)
        lines.append("TEST CLUSTERING REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append("📁 CLUSTERS BY NAME PATTERN")
        lines.append("-" * 80)
        for pattern, tests in sorted(name_clusters.items(), key=lambda x: len(x[1]), reverse=True):
            lines.append(f"{pattern}: {len(tests)} tests")
            for test in tests[:5]:  # Show first 5
                lines.append(f"  • {test}")
            if len(tests) > 5:
                lines.append(f"  ... and {len(tests) - 5} more")
            lines.append("")
        
        lines.append("🔴 CLUSTERS BY FAILURE PATTERN")
        lines.append("-" * 80)
        for error_type, tests in sorted(failure_clusters.items(), key=lambda x: len(x[1]), reverse=True):
            lines.append(f"{error_type}: {len(tests)} tests")
            for test in tests[:5]:  # Show first 5
                lines.append(f"  • {test}")
            if len(tests) > 5:
                lines.append(f"  ... and {len(tests) - 5} more")
            lines.append("")
        
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 80)
        lines.append("1. Tests in same cluster may share common issues")
        lines.append("2. Fix one test in a cluster to potentially fix others")
        lines.append("3. Group tests by cluster for parallel execution")
        lines.append("4. Use clusters for targeted debugging")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    clustering = TestClustering(project_root)
    report = clustering.generate_clustering_report()
    
    print(report)
    
    # Save report
    report_file = project_root / "clustering_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Clustering report saved to: {report_file}")

if __name__ == "__main__":
    main()







