"""
Test Result Deduplicator
Remove duplicate test results and merge similar results
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict


class TestResultDeduplicator:
    """Deduplicate test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def _calculate_hash(self, result: Dict) -> str:
        """Calculate hash for a test result"""
        # Create normalized version for hashing
        normalized = {
            'test_name': result.get('test_name', ''),
            'status': result.get('status', ''),
            'error_message': result.get('error_message', '')[:100]  # Truncate for comparison
        }
        
        json_str = json.dumps(normalized, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def find_duplicates(
        self,
        results: List[Dict],
        similarity_threshold: float = 1.0
    ) -> Dict[str, List[int]]:
        """Find duplicate results"""
        # Group by hash
        hash_groups = defaultdict(list)
        
        for i, result in enumerate(results):
            result_hash = self._calculate_hash(result)
            hash_groups[result_hash].append(i)
        
        # Find groups with multiple items (duplicates)
        duplicates = {
            hash_val: indices
            for hash_val, indices in hash_groups.items()
            if len(indices) > 1
        }
        
        return duplicates
    
    def remove_duplicates(
        self,
        results: List[Dict],
        keep: str = 'latest'
    ) -> List[Dict]:
        """Remove duplicate results"""
        duplicates = self.find_duplicates(results)
        
        indices_to_remove = set()
        
        for hash_val, indices in duplicates.items():
            if keep == 'latest':
                # Keep the latest (last in list)
                indices_to_remove.update(indices[:-1])
            elif keep == 'earliest':
                # Keep the earliest (first in list)
                indices_to_remove.update(indices[1:])
            elif keep == 'best':
                # Keep the one with most information
                best_idx = max(indices, key=lambda i: len(str(results[i])))
                indices_to_remove.update(set(indices) - {best_idx})
        
        # Remove duplicates
        deduplicated = [
            result for i, result in enumerate(results)
            if i not in indices_to_remove
        ]
        
        return deduplicated
    
    def merge_similar_results(
        self,
        results: List[Dict]
    ) -> List[Dict]:
        """Merge similar results into one"""
        # Group by test name
        by_test = defaultdict(list)
        
        for result in results:
            test_name = result.get('test_name', 'unknown')
            by_test[test_name].append(result)
        
        merged = []
        
        for test_name, test_results in by_test.items():
            if len(test_results) == 1:
                merged.append(test_results[0])
            else:
                # Merge multiple results for same test
                merged_result = {
                    'test_name': test_name,
                    'status': self._merge_status([r.get('status') for r in test_results]),
                    'duration': sum(r.get('duration', 0) for r in test_results) / len(test_results),
                    'error_message': self._merge_errors([r.get('error_message', '') for r in test_results]),
                    'occurrences': len(test_results),
                    'timestamps': [r.get('timestamp', '') for r in test_results]
                }
                merged.append(merged_result)
        
        return merged
    
    def _merge_status(self, statuses: List[str]) -> str:
        """Merge multiple statuses"""
        if 'failed' in statuses or 'error' in statuses:
            return 'failed'
        elif 'skipped' in statuses:
            return 'skipped'
        else:
            return 'passed'
    
    def _merge_errors(self, errors: List[str]) -> str:
        """Merge error messages"""
        unique_errors = [e for e in errors if e]
        if not unique_errors:
            return ''
        
        if len(unique_errors) == 1:
            return unique_errors[0]
        
        # Return most common error or first one
        return unique_errors[0]
    
    def deduplicate_directory(
        self,
        directory: Path,
        output_dir: Path = None
    ) -> Dict:
        """Deduplicate all files in a directory"""
        if output_dir is None:
            output_dir = directory.parent / f"{directory.name}_deduplicated"
        
        output_dir.mkdir(exist_ok=True)
        
        all_results = []
        file_mapping = {}
        
        # Load all results
        for result_file in directory.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extract test details
                    test_details = data.get('test_details', {})
                    for test_name, test_data in test_details.items():
                        result = {
                            'test_name': test_name,
                            'status': test_data.get('status'),
                            'duration': test_data.get('duration', 0),
                            'error_message': test_data.get('error_message', ''),
                            'timestamp': data.get('timestamp', ''),
                            'source_file': result_file.name
                        }
                        all_results.append(result)
                        file_mapping[test_name] = result_file.name
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
        
        # Deduplicate
        deduplicated = self.remove_duplicates(all_results, keep='latest')
        
        # Group by source file
        by_file = defaultdict(list)
        for result in deduplicated:
            source_file = result.pop('source_file', 'unknown')
            by_file[source_file].append(result)
        
        # Save deduplicated files
        saved_count = 0
        for source_file, results in by_file.items():
            output_file = output_dir / source_file
            
            # Reconstruct file structure
            output_data = {
                'timestamp': results[0].get('timestamp', ''),
                'run_name': f"deduplicated_{source_file}",
                'summary': {
                    'total_tests': len(results),
                    'passed': sum(1 for r in results if r.get('status') == 'passed'),
                    'failed': sum(1 for r in results if r.get('status') in ('failed', 'error')),
                    'skipped': sum(1 for r in results if r.get('status') == 'skipped')
                },
                'test_details': {
                    r['test_name']: {
                        'status': r.get('status'),
                        'duration': r.get('duration', 0),
                        'error_message': r.get('error_message', '')
                    }
                    for r in results
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            
            saved_count += 1
        
        return {
            'original_count': len(all_results),
            'deduplicated_count': len(deduplicated),
            'removed': len(all_results) - len(deduplicated),
            'files_saved': saved_count
        }


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Result Deduplicator')
    parser.add_argument('--deduplicate-dir', type=str, help='Deduplicate directory')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    deduplicator = TestResultDeduplicator(project_root)
    
    if args.deduplicate_dir:
        print(f"🔍 Deduplicating directory: {args.deduplicate_dir}")
        result = deduplicator.deduplicate_directory(Path(args.deduplicate_dir))
        print(f"\n📊 Results:")
        print(f"  Original: {result['original_count']} results")
        print(f"  Deduplicated: {result['deduplicated_count']} results")
        print(f"  Removed: {result['removed']} duplicates")
        print(f"  Files saved: {result['files_saved']}")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

