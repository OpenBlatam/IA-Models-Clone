"""
Result Deduplicator
Remove duplicate test results
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime

class ResultDeduplicator:
    """Remove duplicate test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def deduplicate_tests(
        self,
        result_file: str,
        output_file: Optional[str] = None
    ) -> Dict:
        """Remove duplicate tests from result file"""
        from typing import Optional
        
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'error': f'File not found: {result_file}'}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'error': f'Error reading file: {e}'}
        
        test_details = data.get('test_details', {})
        
        # Track seen tests
        seen_tests = set()
        deduplicated = {
            'failures': [],
            'errors': [],
            'skipped': []
        }
        
        duplicates_removed = 0
        
        # Deduplicate each category
        for category in ['failures', 'errors', 'skipped']:
            tests = test_details.get(category, [])
            for test in tests:
                test_name = str(test.get('test', ''))
                test_key = f"{category}:{test_name}"
                
                if test_key not in seen_tests:
                    seen_tests.add(test_key)
                    deduplicated[category].append(test)
                else:
                    duplicates_removed += 1
        
        # Create deduplicated result
        deduplicated_result = data.copy()
        deduplicated_result['test_details'] = deduplicated
        deduplicated_result['_deduplication'] = {
            'duplicates_removed': duplicates_removed,
            'original_total': sum(len(test_details.get(cat, [])) for cat in ['failures', 'errors', 'skipped']),
            'deduplicated_total': sum(len(deduplicated[cat]) for cat in ['failures', 'errors', 'skipped'])
        }
        
        # Save deduplicated result
        if output_file is None:
            output_file = f"deduplicated_{result_file}"
        
        output_path = self.results_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(deduplicated_result, f, indent=2)
        
        return deduplicated_result
    
    def find_duplicates(
        self,
        result_files: List[str]
    ) -> Dict:
        """Find duplicate tests across multiple files"""
        all_tests = {}
        duplicates = {}
        
        for result_file in result_files:
            result_path = self.results_dir / result_file
            if not result_path.exists():
                continue
            
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                test_details = data.get('test_details', {})
                
                for category in ['failures', 'errors', 'skipped']:
                    tests = test_details.get(category, [])
                    for test in tests:
                        test_name = str(test.get('test', ''))
                        test_key = f"{category}:{test_name}"
                        
                        if test_key not in all_tests:
                            all_tests[test_key] = []
                        all_tests[test_key].append(result_file)
            except Exception:
                continue
        
        # Find duplicates
        for test_key, files in all_tests.items():
            if len(files) > 1:
                duplicates[test_key] = files
        
        return {
            'total_unique_tests': len(all_tests),
            'duplicate_tests': len(duplicates),
            'duplicates': duplicates
        }
    
    def generate_deduplication_report(self, result: Dict) -> str:
        """Generate deduplication report"""
        lines = []
        lines.append("=" * 80)
        lines.append("DEDUPLICATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in result:
            lines.append(f"❌ {result['error']}")
            return "\n".join(lines)
        
        if '_deduplication' in result:
            dedup_info = result['_deduplication']
            lines.append(f"Duplicates Removed: {dedup_info['duplicates_removed']}")
            lines.append(f"Original Total: {dedup_info['original_total']}")
            lines.append(f"Deduplicated Total: {dedup_info['deduplicated_total']}")
        else:
            lines.append(f"Total Unique Tests: {result['total_unique_tests']}")
            lines.append(f"Duplicate Tests: {result['duplicate_tests']}")
            lines.append("")
            lines.append("🔍 DUPLICATES FOUND")
            lines.append("-" * 80)
            for test_key, files in list(result['duplicates'].items())[:20]:
                lines.append(f"{test_key}: {len(files)} occurrences")
                for file in files:
                    lines.append(f"  • {file}")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    from typing import Optional
    
    if len(sys.argv) < 2:
        print("Usage: python result_deduplicator.py <file.json> [output.json]")
        print("   or: python result_deduplicator.py find <file1.json> <file2.json> ...")
        return
    
    project_root = Path(__file__).parent.parent
    deduplicator = ResultDeduplicator(project_root)
    
    if sys.argv[1] == 'find':
        # Find duplicates across files
        result_files = sys.argv[2:]
        duplicates = deduplicator.find_duplicates(result_files)
        report = deduplicator.generate_deduplication_report(duplicates)
        print(report)
    else:
        # Deduplicate single file
        result_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        result = deduplicator.deduplicate_tests(result_file, output_file)
        report = deduplicator.generate_deduplication_report(result)
        print(report)
        
        if 'error' not in result:
            print(f"\n✅ Deduplicated result saved")

if __name__ == "__main__":
    main()







