"""
Intelligent Merger
Intelligently merge test results with conflict resolution
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime
from collections import defaultdict

class IntelligentMerger:
    """Intelligently merge test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def merge_intelligently(
        self,
        result_files: List[str],
        conflict_resolution: str = 'latest',
        output_file: str = "intelligently_merged.json"
    ) -> Dict:
        """Intelligently merge test results with conflict resolution"""
        results = []
        
        for result_file in result_files:
            result_path = self.results_dir / result_file
            if not result_path.exists():
                continue
            
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_filename'] = result_file
                    results.append(data)
            except Exception:
                continue
        
        if not results:
            return {'error': 'No valid result files found'}
        
        # Sort by timestamp if conflict resolution is 'latest'
        if conflict_resolution == 'latest':
            results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Merge intelligently
        merged = {
            'total_tests': 0,
            'passed': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'execution_time': 0.0,
            'test_details': {
                'failures': [],
                'errors': [],
                'skipped': []
            },
            'merged_from': [r['_filename'] for r in results],
            'conflict_resolution': conflict_resolution,
            'timestamp': datetime.now().isoformat()
        }
        
        # Track seen tests
        seen_tests = set()
        
        # Merge test details with conflict resolution
        for result in results:
            test_details = result.get('test_details', {})
            
            for category in ['failures', 'errors', 'skipped']:
                tests = test_details.get(category, [])
                for test in tests:
                    test_name = str(test.get('test', ''))
                    test_key = f"{category}:{test_name}"
                    
                    if test_key not in seen_tests:
                        seen_tests.add(test_key)
                        merged['test_details'][category].append(test)
                    elif conflict_resolution == 'latest':
                        # Replace with latest
                        index = next((i for i, t in enumerate(merged['test_details'][category]) 
                                    if str(t.get('test', '')) == test_name), None)
                        if index is not None:
                            merged['test_details'][category][index] = test
        
        # Calculate merged metrics
        merged['total_tests'] = max(r.get('total_tests', 0) for r in results)
        merged['passed'] = max(r.get('passed', 0) for r in results)
        merged['failures'] = len(merged['test_details']['failures'])
        merged['errors'] = len(merged['test_details']['errors'])
        merged['skipped'] = len(merged['test_details']['skipped'])
        merged['execution_time'] = sum(r.get('execution_time', 0) for r in results)
        
        if merged['total_tests'] > 0:
            merged['success_rate'] = ((merged['passed'] / merged['total_tests']) * 100)
        else:
            merged['success_rate'] = 0
        
        # Save merged result
        output_path = self.results_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2)
        
        return merged
    
    def generate_merge_report(self, merged: Dict) -> str:
        """Generate merge report"""
        lines = []
        lines.append("=" * 80)
        lines.append("INTELLIGENT MERGE REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in merged:
            lines.append(f"❌ {merged['error']}")
            return "\n".join(lines)
        
        lines.append(f"Files Merged: {len(merged['merged_from'])}")
        lines.append(f"Conflict Resolution: {merged['conflict_resolution']}")
        lines.append("")
        
        lines.append("📊 MERGED METRICS")
        lines.append("-" * 80)
        lines.append(f"Total Tests: {merged['total_tests']}")
        lines.append(f"Passed: {merged['passed']}")
        lines.append(f"Failures: {merged['failures']}")
        lines.append(f"Errors: {merged['errors']}")
        lines.append(f"Skipped: {merged['skipped']}")
        lines.append(f"Success Rate: {merged['success_rate']:.1f}%")
        lines.append(f"Execution Time: {merged['execution_time']:.2f}s")
        lines.append("")
        
        lines.append("📋 MERGED FROM")
        lines.append("-" * 80)
        for filename in merged['merged_from']:
            lines.append(f"  • {filename}")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python intelligent_merger.py <file1.json> <file2.json> ... [--resolution latest|first|all]")
        return
    
    project_root = Path(__file__).parent.parent
    merger = IntelligentMerger(project_root)
    
    result_files = [f for f in sys.argv[1:] if not f.startswith('--')]
    conflict_resolution = 'latest'
    
    if '--resolution' in sys.argv:
        idx = sys.argv.index('--resolution')
        if idx + 1 < len(sys.argv):
            conflict_resolution = sys.argv[idx + 1]
    
    merged = merger.merge_intelligently(result_files, conflict_resolution)
    report = merger.generate_merge_report(merged)
    
    print(report)
    
    if 'error' not in merged:
        print(f"\n✅ Intelligently merged results saved to: test_results/intelligently_merged.json")

if __name__ == "__main__":
    main()







