"""
Test Result Aggregator
Aggregates test results across multiple environments and runs
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import statistics


class ResultAggregator:
    """Aggregate test results from multiple sources"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "aggregated_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def load_results_from_file(self, result_file: Path) -> Dict:
        """Load test results from JSON file"""
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            return {}
    
    def load_results_from_directory(self, results_dir: Path) -> List[Dict]:
        """Load all result files from directory"""
        results = []
        if results_dir.exists():
            for result_file in results_dir.glob("*.json"):
                result = self.load_results_from_file(result_file)
                if result:
                    result['source_file'] = str(result_file)
                    results.append(result)
        return results
    
    def aggregate_by_test(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group results by test name"""
        by_test = defaultdict(list)
        
        for result in results:
            test_details = result.get('test_details', {})
            for test_name, test_data in test_details.items():
                by_test[test_name].append({
                    'run': result.get('run_name', 'unknown'),
                    'timestamp': result.get('timestamp', ''),
                    'environment': result.get('environment', 'unknown'),
                    'status': test_data.get('status', 'unknown'),
                    'duration': test_data.get('duration', 0),
                    'error': test_data.get('error', '')
                })
        
        return dict(by_test)
    
    def aggregate_by_environment(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group results by environment"""
        by_env = defaultdict(list)
        
        for result in results:
            env = result.get('environment', 'unknown')
            by_env[env].append(result)
        
        return dict(by_env)
    
    def calculate_test_statistics(self, test_results: List[Dict]) -> Dict:
        """Calculate statistics for a specific test"""
        if not test_results:
            return {}
        
        statuses = [r['status'] for r in test_results]
        durations = [r['duration'] for r in test_results if r.get('duration', 0) > 0]
        
        passed = statuses.count('passed')
        failed = statuses.count('failed')
        skipped = statuses.count('skipped')
        total = len(statuses)
        
        stats = {
            'total_runs': total,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'success_rate': (passed / total * 100) if total > 0 else 0,
            'failure_rate': (failed / total * 100) if total > 0 else 0
        }
        
        if durations:
            stats['duration'] = {
                'min': min(durations),
                'max': max(durations),
                'avg': statistics.mean(durations),
                'median': statistics.median(durations),
                'stdev': statistics.stdev(durations) if len(durations) > 1 else 0
            }
        
        # Find most common error
        errors = [r['error'] for r in test_results if r.get('error')]
        if errors:
            error_counts = defaultdict(int)
            for error in errors:
                error_counts[error[:100]] += 1  # Truncate long errors
            stats['most_common_error'] = max(error_counts.items(), key=lambda x: x[1])
        
        return stats
    
    def aggregate_results(
        self,
        result_sources: List[Path],
        output_file: Path = None
    ) -> Dict:
        """Aggregate results from multiple sources"""
        all_results = []
        
        # Load results from all sources
        for source in result_sources:
            if source.is_file():
                result = self.load_results_from_file(source)
                if result:
                    all_results.append(result)
            elif source.is_dir():
                results = self.load_results_from_directory(source)
                all_results.extend(results)
        
        if not all_results:
            return {'error': 'No results found'}
        
        # Aggregate by test
        by_test = self.aggregate_by_test(all_results)
        
        # Aggregate by environment
        by_env = self.aggregate_by_environment(all_results)
        
        # Calculate statistics for each test
        test_stats = {}
        for test_name, test_results in by_test.items():
            test_stats[test_name] = self.calculate_test_statistics(test_results)
        
        # Overall statistics
        overall_stats = {
            'total_runs': len(all_results),
            'environments': list(by_env.keys()),
            'total_tests': len(by_test),
            'tests_with_failures': sum(1 for stats in test_stats.values() if stats.get('failed', 0) > 0),
            'avg_success_rate': statistics.mean([s.get('success_rate', 0) for s in test_stats.values()]) if test_stats else 0
        }
        
        aggregated = {
            'timestamp': datetime.now().isoformat(),
            'sources': [str(s) for s in result_sources],
            'overall_statistics': overall_stats,
            'by_environment': {
                env: {
                    'run_count': len(results),
                    'tests': len(self.aggregate_by_test(results))
                }
                for env, results in by_env.items()
            },
            'by_test': {
                test_name: {
                    'results': test_results,
                    'statistics': test_stats.get(test_name, {})
                }
                for test_name, test_results in by_test.items()
            }
        }
        
        # Save aggregated results
        if output_file is None:
            output_file = self.results_dir / f"aggregated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"✅ Aggregated results saved to {output_file}")
        
        return aggregated
    
    def generate_comparison_report(
        self,
        aggregated_results: Dict,
        output_file: Path = None
    ) -> Dict:
        """Generate comparison report between environments"""
        by_env = aggregated_results.get('by_environment', {})
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'environments': {},
            'differences': []
        }
        
        # Compare environments
        envs = list(by_env.keys())
        for i, env1 in enumerate(envs):
            for env2 in envs[i+1:]:
                # Find differences between environments
                # This would require more detailed comparison logic
                comparison['differences'].append({
                    'env1': env1,
                    'env2': env2,
                    'note': 'Detailed comparison requires test-level data'
                })
        
        if output_file is None:
            output_file = self.results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Result Aggregator')
    parser.add_argument('sources', nargs='+', help='Result files or directories to aggregate')
    parser.add_argument('--output', type=str, help='Output file for aggregated results')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    aggregator = ResultAggregator(project_root)
    
    result_sources = [Path(s) for s in args.sources]
    output_file = Path(args.output) if args.output else None
    
    print("📊 Aggregating test results...")
    aggregated = aggregator.aggregate_results(result_sources, output_file)
    
    print(f"\n✅ Aggregation complete!")
    print(f"  Total Runs: {aggregated.get('overall_statistics', {}).get('total_runs', 0)}")
    print(f"  Environments: {', '.join(aggregated.get('overall_statistics', {}).get('environments', []))}")
    print(f"  Total Tests: {aggregated.get('overall_statistics', {}).get('total_tests', 0)}")
    print(f"  Avg Success Rate: {aggregated.get('overall_statistics', {}).get('avg_success_rate', 0):.1f}%")


if __name__ == '__main__':
    main()

