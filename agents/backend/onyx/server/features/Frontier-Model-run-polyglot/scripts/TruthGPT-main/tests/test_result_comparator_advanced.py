"""
Advanced Test Result Comparator
Compares test results with advanced diff analysis, regression detection, and insights
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
import difflib


class AdvancedTestResultComparator:
    """Advanced comparison of test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def compare_runs(
        self,
        run1_path: Path,
        run2_path: Path,
        detailed: bool = True
    ) -> Dict:
        """Compare two test runs with detailed analysis"""
        run1 = self._load_result(run1_path)
        run2 = self._load_result(run2_path)
        
        if not run1 or not run2:
            return {'error': 'Could not load one or both results'}
        
        comparison = {
            'run1': {
                'name': run1.get('run_name', 'unknown'),
                'timestamp': run1.get('timestamp'),
                'summary': run1.get('summary', {})
            },
            'run2': {
                'name': run2.get('run_name', 'unknown'),
                'timestamp': run2.get('timestamp'),
                'summary': run2.get('summary', {})
            },
            'summary_changes': self._compare_summaries(
                run1.get('summary', {}),
                run2.get('summary', {})
            ),
            'test_changes': self._compare_tests(
                run1.get('test_details', {}),
                run2.get('test_details', {})
            ),
            'new_tests': [],
            'removed_tests': [],
            'regressions': [],
            'improvements': [],
            'unchanged': []
        }
        
        if detailed:
            comparison['detailed_analysis'] = self._detailed_analysis(
                run1.get('test_details', {}),
                run2.get('test_details', {})
            )
        
        return comparison
    
    def _load_result(self, path: Path) -> Optional[Dict]:
        """Load test result from file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def _compare_summaries(self, summary1: Dict, summary2: Dict) -> Dict:
        """Compare summary statistics"""
        return {
            'total_tests': {
                'run1': summary1.get('total_tests', 0),
                'run2': summary2.get('total_tests', 0),
                'change': summary2.get('total_tests', 0) - summary1.get('total_tests', 0)
            },
            'passed': {
                'run1': summary1.get('passed', 0),
                'run2': summary2.get('passed', 0),
                'change': summary2.get('passed', 0) - summary1.get('passed', 0)
            },
            'failed': {
                'run1': summary1.get('failed', 0),
                'run2': summary2.get('failed', 0),
                'change': summary2.get('failed', 0) - summary1.get('failed', 0)
            },
            'success_rate': {
                'run1': summary1.get('success_rate', 0),
                'run2': summary2.get('success_rate', 0),
                'change': summary2.get('success_rate', 0) - summary1.get('success_rate', 0)
            },
            'execution_time': {
                'run1': summary1.get('execution_time', 0),
                'run2': summary2.get('execution_time', 0),
                'change': summary2.get('execution_time', 0) - summary1.get('execution_time', 0),
                'change_percent': self._percent_change(
                    summary1.get('execution_time', 0),
                    summary2.get('execution_time', 0)
                )
            }
        }
    
    def _compare_tests(
        self,
        tests1: Dict,
        tests2: Dict
    ) -> Dict:
        """Compare individual test results"""
        all_tests = set(tests1.keys()) | set(tests2.keys())
        
        changes = {
            'status_changes': [],
            'duration_changes': [],
            'new_tests': [],
            'removed_tests': [],
            'regressions': [],
            'improvements': []
        }
        
        for test_name in all_tests:
            test1 = tests1.get(test_name)
            test2 = tests2.get(test_name)
            
            if not test1:
                changes['new_tests'].append({
                    'test_name': test_name,
                    'status': test2.get('status'),
                    'duration': test2.get('duration', 0)
                })
                continue
            
            if not test2:
                changes['removed_tests'].append({
                    'test_name': test_name,
                    'status': test1.get('status'),
                    'duration': test1.get('duration', 0)
                })
                continue
            
            # Status change
            status1 = test1.get('status', 'unknown')
            status2 = test2.get('status', 'unknown')
            
            if status1 != status2:
                change = {
                    'test_name': test_name,
                    'from': status1,
                    'to': status2
                }
                
                if status1 == 'passed' and status2 in ('failed', 'error'):
                    changes['regressions'].append(change)
                elif status1 in ('failed', 'error') and status2 == 'passed':
                    changes['improvements'].append(change)
                
                changes['status_changes'].append(change)
            
            # Duration change
            dur1 = test1.get('duration', 0)
            dur2 = test2.get('duration', 0)
            
            if dur1 > 0 and dur2 > 0:
                change_percent = self._percent_change(dur1, dur2)
                if abs(change_percent) > 10:  # Significant change
                    changes['duration_changes'].append({
                        'test_name': test_name,
                        'from': dur1,
                        'to': dur2,
                        'change_percent': change_percent
                    })
        
        return changes
    
    def _detailed_analysis(
        self,
        tests1: Dict,
        tests2: Dict
    ) -> Dict:
        """Perform detailed analysis of changes"""
        analysis = {
            'error_analysis': self._analyze_errors(tests1, tests2),
            'performance_analysis': self._analyze_performance(tests1, tests2),
            'stability_analysis': self._analyze_stability(tests1, tests2)
        }
        
        return analysis
    
    def _analyze_errors(self, tests1: Dict, tests2: Dict) -> Dict:
        """Analyze error patterns"""
        errors1 = [
            (name, test.get('error_message', ''))
            for name, test in tests1.items()
            if test.get('status') in ('failed', 'error')
        ]
        
        errors2 = [
            (name, test.get('error_message', ''))
            for name, test in tests2.items()
            if test.get('status') in ('failed', 'error')
        ]
        
        # Find similar errors
        similar_errors = []
        for name1, err1 in errors1:
            for name2, err2 in errors2:
                if name1 == name2 and err1 and err2:
                    similarity = difflib.SequenceMatcher(None, err1, err2).ratio()
                    if similarity > 0.7:
                        similar_errors.append({
                            'test_name': name1,
                            'similarity': similarity,
                            'error1': err1[:100],
                            'error2': err2[:100]
                        })
        
        return {
            'total_errors_run1': len(errors1),
            'total_errors_run2': len(errors2),
            'similar_errors': similar_errors
        }
    
    def _analyze_performance(self, tests1: Dict, tests2: Dict) -> Dict:
        """Analyze performance changes"""
        performance_changes = []
        
        for test_name in set(tests1.keys()) & set(tests2.keys()):
            dur1 = tests1[test_name].get('duration', 0)
            dur2 = tests2[test_name].get('duration', 0)
            
            if dur1 > 0 and dur2 > 0:
                change = dur2 - dur1
                change_percent = self._percent_change(dur1, dur2)
                
                if abs(change_percent) > 20:  # Significant change
                    performance_changes.append({
                        'test_name': test_name,
                        'change': change,
                        'change_percent': change_percent,
                        'severity': 'high' if abs(change_percent) > 50 else 'medium'
                    })
        
        return {
            'significant_changes': len(performance_changes),
            'changes': sorted(performance_changes, key=lambda x: abs(x['change_percent']), reverse=True)
        }
    
    def _analyze_stability(self, tests1: Dict, tests2: Dict) -> Dict:
        """Analyze test stability"""
        stable = 0
        unstable = 0
        
        for test_name in set(tests1.keys()) & set(tests2.keys()):
            status1 = tests1[test_name].get('status')
            status2 = tests2[test_name].get('status')
            
            if status1 == status2:
                stable += 1
            else:
                unstable += 1
        
        total = stable + unstable
        
        return {
            'stable_tests': stable,
            'unstable_tests': unstable,
            'stability_rate': (stable / total * 100) if total > 0 else 0
        }
    
    def _percent_change(self, old: float, new: float) -> float:
        """Calculate percentage change"""
        if old == 0:
            return 0
        return ((new - old) / old) * 100
    
    def generate_comparison_report(
        self,
        comparison: Dict,
        output_file: Path = None
    ) -> str:
        """Generate human-readable comparison report"""
        lines = []
        lines.append("📊 TEST RESULT COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("📋 SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Run 1: {comparison['run1']['name']} ({comparison['run1']['timestamp']})")
        lines.append(f"Run 2: {comparison['run2']['name']} ({comparison['run2']['timestamp']})")
        lines.append("")
        
        # Summary changes
        summary = comparison['summary_changes']
        lines.append("📈 SUMMARY CHANGES")
        lines.append("-" * 80)
        lines.append(f"Total Tests: {summary['total_tests']['run1']} → {summary['total_tests']['run2']} ({summary['total_tests']['change']:+.0f})")
        lines.append(f"Passed: {summary['passed']['run1']} → {summary['passed']['run2']} ({summary['passed']['change']:+.0f})")
        lines.append(f"Failed: {summary['failed']['run1']} → {summary['failed']['run2']} ({summary['failed']['change']:+.0f})")
        lines.append(f"Success Rate: {summary['success_rate']['run1']:.1f}% → {summary['success_rate']['run2']:.1f}% ({summary['success_rate']['change']:+.1f}%)")
        lines.append(f"Execution Time: {summary['execution_time']['run1']:.1f}s → {summary['execution_time']['run2']:.1f}s ({summary['execution_time']['change_percent']:+.1f}%)")
        lines.append("")
        
        # Test changes
        test_changes = comparison['test_changes']
        if test_changes['regressions']:
            lines.append("🔴 REGRESSIONS")
            lines.append("-" * 80)
            for reg in test_changes['regressions'][:10]:
                lines.append(f"  {reg['test_name']}: {reg['from']} → {reg['to']}")
            lines.append("")
        
        if test_changes['improvements']:
            lines.append("🟢 IMPROVEMENTS")
            lines.append("-" * 80)
            for imp in test_changes['improvements'][:10]:
                lines.append(f"  {imp['test_name']}: {imp['from']} → {imp['to']}")
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
    
    parser = argparse.ArgumentParser(description='Advanced Test Result Comparator')
    parser.add_argument('run1', type=str, help='First run result file')
    parser.add_argument('run2', type=str, help='Second run result file')
    parser.add_argument('--report', type=str, help='Output report file')
    parser.add_argument('--detailed', action='store_true', help='Include detailed analysis')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    comparator = AdvancedTestResultComparator(project_root)
    
    run1_path = Path(args.run1)
    run2_path = Path(args.run2)
    
    print("📊 Comparing test runs...")
    comparison = comparator.compare_runs(run1_path, run2_path, args.detailed)
    
    if args.report:
        comparator.generate_comparison_report(comparison, Path(args.report))
    else:
        print(f"\n📈 Summary Changes:")
        print(f"  Success Rate: {comparison['summary_changes']['success_rate']['change']:+.1f}%")
        print(f"  Regressions: {len(comparison['test_changes']['regressions'])}")
        print(f"  Improvements: {len(comparison['test_changes']['improvements'])}")


if __name__ == '__main__':
    main()

