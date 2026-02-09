"""
Result Splitter
Split large test result files into smaller chunks
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class ResultSplitter:
    """Split test results into smaller files"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def split_by_tests(
        self,
        result_file: str,
        tests_per_file: int = 50,
        output_prefix: str = "split_"
    ) -> List[str]:
        """Split result file by number of tests"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return []
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️  Error reading file: {e}")
            return []
        
        test_details = data.get('test_details', {})
        all_tests = []
        
        # Collect all tests
        for test_list in [
            test_details.get('failures', []),
            test_details.get('errors', []),
            test_details.get('skipped', [])
        ]:
            all_tests.extend(test_list)
        
        # Split into chunks
        split_files = []
        for i in range(0, len(all_tests), tests_per_file):
            chunk = all_tests[i:i + tests_per_file]
            
            split_data = {
                'original_file': result_file,
                'chunk_number': i // tests_per_file + 1,
                'total_chunks': (len(all_tests) + tests_per_file - 1) // tests_per_file,
                'tests_in_chunk': len(chunk),
                'timestamp': datetime.now().isoformat(),
                'test_details': {
                    'failures': [t for t in chunk if t in test_details.get('failures', [])],
                    'errors': [t for t in chunk if t in test_details.get('errors', [])],
                    'skipped': [t for t in chunk if t in test_details.get('skipped', [])]
                }
            }
            
            output_file = f"{output_prefix}{i // tests_per_file + 1:03d}.json"
            output_path = self.results_dir / output_file
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2)
            
            split_files.append(output_file)
        
        return split_files
    
    def split_by_category(
        self,
        result_file: str,
        output_prefix: str = "category_"
    ) -> List[str]:
        """Split result file by test category"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return []
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️  Error reading file: {e}")
            return []
        
        test_details = data.get('test_details', {})
        
        # Categorize tests
        categories = {
            'failures': test_details.get('failures', []),
            'errors': test_details.get('errors', []),
            'skipped': test_details.get('skipped', [])
        }
        
        split_files = []
        for category, tests in categories.items():
            if not tests:
                continue
            
            split_data = {
                'original_file': result_file,
                'category': category,
                'tests_count': len(tests),
                'timestamp': datetime.now().isoformat(),
                'test_details': {
                    category: tests
                }
            }
            
            output_file = f"{output_prefix}{category}.json"
            output_path = self.results_dir / output_file
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2)
            
            split_files.append(output_file)
        
        return split_files

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python result_splitter.py <file.json> [method] [size]")
        print("Methods: by_tests, by_category")
        return
    
    project_root = Path(__file__).parent.parent
    splitter = ResultSplitter(project_root)
    
    result_file = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'by_tests'
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    if method == 'by_tests':
        split_files = splitter.split_by_tests(result_file, size)
    elif method == 'by_category':
        split_files = splitter.split_by_category(result_file)
    else:
        print(f"Unknown method: {method}")
        return
    
    print(f"✅ Split into {len(split_files)} files:")
    for file in split_files:
        print(f"  • {file}")

if __name__ == "__main__":
    main()







