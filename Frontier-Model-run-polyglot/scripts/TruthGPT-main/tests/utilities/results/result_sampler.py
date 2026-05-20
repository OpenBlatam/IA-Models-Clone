"""
Result Sampler
Sample test results for analysis
"""

import json
import random
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class ResultSampler:
    """Sample test results for analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def sample_results(
        self,
        result_file: str,
        sample_size: int = 10,
        method: str = 'random'
    ) -> Dict:
        """Sample test results from a file"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'error': f'File not found: {result_file}'}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'error': f'Error reading file: {e}'}
        
        test_details = data.get('test_details', {})
        all_tests = []
        
        # Collect all test results
        for test_list in [
            test_details.get('failures', []),
            test_details.get('errors', []),
            test_details.get('skipped', [])
        ]:
            all_tests.extend(test_list)
        
        # Sample based on method
        if method == 'random':
            sampled = random.sample(all_tests, min(sample_size, len(all_tests)))
        elif method == 'first':
            sampled = all_tests[:sample_size]
        elif method == 'last':
            sampled = all_tests[-sample_size:]
        elif method == 'failures_first':
            failures = test_details.get('failures', [])
            errors = test_details.get('errors', [])
            sampled = (failures + errors)[:sample_size]
        else:
            sampled = all_tests[:sample_size]
        
        # Create sampled result
        sampled_result = {
            'original_file': result_file,
            'sample_size': len(sampled),
            'total_tests': len(all_tests),
            'sampling_method': method,
            'timestamp': datetime.now().isoformat(),
            'test_details': {
                'failures': [t for t in sampled if t in test_details.get('failures', [])],
                'errors': [t for t in sampled if t in test_details.get('errors', [])],
                'skipped': [t for t in sampled if t in test_details.get('skipped', [])]
            }
        }
        
        return sampled_result
    
    def save_sample(
        self,
        sampled: Dict,
        output_file: str = "sampled_results.json"
    ) -> Path:
        """Save sampled results"""
        output_path = self.results_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sampled, f, indent=2)
        
        return output_path
    
    def generate_sample_report(self, sampled: Dict) -> str:
        """Generate sample report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST RESULT SAMPLE")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in sampled:
            lines.append(f"❌ {sampled['error']}")
            return "\n".join(lines)
        
        lines.append(f"Original File: {sampled['original_file']}")
        lines.append(f"Sampling Method: {sampled['sampling_method']}")
        lines.append(f"Sample Size: {sampled['sample_size']}")
        lines.append(f"Total Tests: {sampled['total_tests']}")
        lines.append(f"Sampling Ratio: {(sampled['sample_size'] / sampled['total_tests'] * 100) if sampled['total_tests'] > 0 else 0:.1f}%")
        lines.append("")
        
        test_details = sampled['test_details']
        lines.append("📊 SAMPLE BREAKDOWN")
        lines.append("-" * 80)
        lines.append(f"Failures: {len(test_details['failures'])}")
        lines.append(f"Errors: {len(test_details['errors'])}")
        lines.append(f"Skipped: {len(test_details['skipped'])}")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python result_sampler.py <file.json> [sample_size] [method]")
        print("Methods: random, first, last, failures_first")
        return
    
    project_root = Path(__file__).parent.parent
    sampler = ResultSampler(project_root)
    
    result_file = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    method = sys.argv[3] if len(sys.argv) > 3 else 'random'
    
    sampled = sampler.sample_results(result_file, sample_size, method)
    report = sampler.generate_sample_report(sampled)
    
    print(report)
    
    if 'error' not in sampled:
        output_path = sampler.save_sample(sampled)
        print(f"\n✅ Sampled results saved to: {output_path}")

if __name__ == "__main__":
    main()







