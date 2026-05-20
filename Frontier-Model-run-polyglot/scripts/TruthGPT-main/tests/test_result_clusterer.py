"""
Test Result Clusterer
Groups and clusters test results by patterns, errors, and characteristics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
import difflib
import hashlib


class TestResultClusterer:
    """Cluster test results by patterns"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def cluster_by_error_pattern(
        self,
        results: List[Dict],
        similarity_threshold: float = 0.7
    ) -> Dict[str, List[Dict]]:
        """Cluster tests by similar error messages"""
        error_tests = [
            r for r in results
            if r.get('status') in ('failed', 'error') and r.get('error_message')
        ]
        
        clusters = {}
        cluster_id = 0
        
        for test in error_tests:
            error_msg = test['error_message']
            error_hash = self._hash_error(error_msg)
            
            # Try to find similar cluster
            assigned = False
            for cid, cluster_tests in clusters.items():
                if cluster_tests:
                    # Compare with first test in cluster
                    first_error = cluster_tests[0].get('error_message', '')
                    similarity = difflib.SequenceMatcher(
                        None, error_msg, first_error
                    ).ratio()
                    
                    if similarity >= similarity_threshold:
                        clusters[cid].append(test)
                        assigned = True
                        break
            
            if not assigned:
                clusters[f"cluster_{cluster_id}"] = [test]
                cluster_id += 1
        
        return clusters
    
    def cluster_by_test_pattern(
        self,
        results: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Cluster tests by name patterns"""
        clusters = defaultdict(list)
        
        for result in results:
            test_name = result.get('test_name', '')
            
            # Extract pattern (remove numbers, extract base name)
            parts = test_name.split('_')
            base_parts = [p for p in parts if not p.isdigit() and len(p) > 2]
            
            if base_parts:
                pattern = '_'.join(base_parts[:3])  # First 3 meaningful parts
                clusters[pattern].append(result)
            else:
                clusters['other'].append(result)
        
        return dict(clusters)
    
    def cluster_by_duration(
        self,
        results: List[Dict],
        buckets: List[float] = None
    ) -> Dict[str, List[Dict]]:
        """Cluster tests by execution duration"""
        if buckets is None:
            buckets = [0, 1, 5, 10, 30, float('inf')]
        
        clusters = defaultdict(list)
        
        for result in results:
            duration = result.get('duration', 0)
            
            for i, threshold in enumerate(buckets[1:], 1):
                if duration < threshold:
                    cluster_name = f"{buckets[i-1]}-{threshold}s"
                    clusters[cluster_name].append(result)
                    break
        
        return dict(clusters)
    
    def cluster_by_failure_rate(
        self,
        results: List[Dict],
        test_history: Dict[str, List[Dict]] = None
    ) -> Dict[str, List[Dict]]:
        """Cluster tests by failure rate"""
        if not test_history:
            return {}
        
        # Calculate failure rates
        failure_rates = {}
        for test_name, history in test_history.items():
            failures = sum(1 for r in history if r.get('status') in ('failed', 'error'))
            total = len(history)
            failure_rates[test_name] = failures / total if total > 0 else 0
        
        # Cluster by rate
        clusters = {
            'high_risk': [],      # > 50%
            'medium_risk': [],    # 20-50%
            'low_risk': [],       # 5-20%
            'stable': []          # < 5%
        }
        
        for result in results:
            test_name = result.get('test_name', '')
            rate = failure_rates.get(test_name, 0)
            
            if rate > 0.5:
                clusters['high_risk'].append(result)
            elif rate > 0.2:
                clusters['medium_risk'].append(result)
            elif rate > 0.05:
                clusters['low_risk'].append(result)
            else:
                clusters['stable'].append(result)
        
        return clusters
    
    def _hash_error(self, error_msg: str) -> str:
        """Create hash of error message (normalized)"""
        # Normalize: remove line numbers, file paths, etc.
        normalized = error_msg.lower()
        # Remove common variable parts
        normalized = normalized.replace('line ', 'line X')
        normalized = normalized.replace('file ', 'file X')
        return hashlib.md5(normalized.encode()).hexdigest()[:8]
    
    def generate_cluster_report(
        self,
        clusters: Dict[str, List[Dict]],
        output_file: Path = None
    ) -> str:
        """Generate cluster analysis report"""
        lines = []
        lines.append("🔗 TEST RESULT CLUSTERING REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Clusters: {len(clusters)}")
        lines.append("")
        
        # Sort clusters by size
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for cluster_name, cluster_tests in sorted_clusters:
            lines.append(f"📦 CLUSTER: {cluster_name}")
            lines.append("-" * 80)
            lines.append(f"  Size: {len(cluster_tests)} tests")
            
            # Common characteristics
            if cluster_tests:
                statuses = [t.get('status') for t in cluster_tests]
                status_counts = defaultdict(int)
                for s in statuses:
                    status_counts[s] += 1
                
                lines.append(f"  Status Distribution:")
                for status, count in status_counts.items():
                    lines.append(f"    {status}: {count}")
                
                # Sample tests
                lines.append(f"  Sample Tests:")
                for test in cluster_tests[:5]:
                    lines.append(f"    - {test.get('test_name', 'unknown')}")
            
            lines.append("")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Report saved to {output_file}")
        
        return report


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Result Clusterer')
    parser.add_argument('--results', type=str, help='Results file to cluster')
    parser.add_argument('--by-error', action='store_true', help='Cluster by error pattern')
    parser.add_argument('--by-pattern', action='store_true', help='Cluster by test name pattern')
    parser.add_argument('--by-duration', action='store_true', help='Cluster by duration')
    parser.add_argument('--report', type=str, help='Generate cluster report')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    clusterer = TestResultClusterer(project_root)
    
    if args.results:
        with open(args.results, 'r', encoding='utf-8') as f:
            results = json.load(f).get('test_details', {})
            results_list = [v for v in results.values()] if isinstance(results, dict) else results
        
        if args.by_error:
            print("🔗 Clustering by error pattern...")
            clusters = clusterer.cluster_by_error_pattern(results_list)
            print(f"Found {len(clusters)} error clusters")
        elif args.by_pattern:
            print("🔗 Clustering by test pattern...")
            clusters = clusterer.cluster_by_test_pattern(results_list)
            print(f"Found {len(clusters)} pattern clusters")
        elif args.by_duration:
            print("🔗 Clustering by duration...")
            clusters = clusterer.cluster_by_duration(results_list)
            print(f"Found {len(clusters)} duration clusters")
        
        if args.report:
            clusterer.generate_cluster_report(clusters, Path(args.report))
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

