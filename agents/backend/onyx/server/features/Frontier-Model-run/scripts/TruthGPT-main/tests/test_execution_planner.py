"""
Test Execution Planner
Intelligent planning of test execution order and parallelization
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
import statistics


class TestExecutionPlanner:
    """Plan optimal test execution strategy"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_execution_history.json"
        self.history = self._load_history()
        self.dependency_graph = {}  # Will be populated from dependency analyzer
    
    def _load_history(self) -> List[Dict]:
        """Load execution history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save_history(self):
        """Save execution history"""
        self.history_file.parent.mkdir(exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
    
    def record_execution(
        self,
        test_name: str,
        duration: float,
        status: str,
        dependencies: List[str] = None
    ):
        """Record test execution"""
        record = {
            'test_name': test_name,
            'duration': duration,
            'status': status,
            'dependencies': dependencies or [],
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.append(record)
        
        # Keep last 500 records
        if len(self.history) > 500:
            self.history = self.history[-500:]
        
        self._save_history()
    
    def get_test_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all tests"""
        by_test = defaultdict(lambda: {
            'durations': [],
            'failure_count': 0,
            'total_count': 0,
            'dependencies': set()
        })
        
        for record in self.history:
            test_name = record['test_name']
            by_test[test_name]['durations'].append(record['duration'])
            by_test[test_name]['total_count'] += 1
            if record['status'] in ('failed', 'error'):
                by_test[test_name]['failure_count'] += 1
            by_test[test_name]['dependencies'].update(record.get('dependencies', []))
        
        metrics = {}
        for test_name, data in by_test.items():
            durations = data['durations']
            metrics[test_name] = {
                'avg_duration': statistics.mean(durations) if durations else 0,
                'median_duration': statistics.median(durations) if durations else 0,
                'failure_rate': data['failure_count'] / data['total_count'] if data['total_count'] > 0 else 0,
                'total_runs': data['total_count'],
                'dependencies': list(data['dependencies'])
            }
        
        return metrics
    
    def plan_execution(
        self,
        test_names: List[str],
        max_parallel: int = 4,
        strategy: str = 'balanced'
    ) -> Dict:
        """Plan optimal test execution"""
        metrics = self.get_test_metrics()
        
        # Filter to available tests
        available_tests = [t for t in test_names if t in metrics]
        
        if strategy == 'fast_first':
            # Execute fastest tests first
            available_tests.sort(key=lambda t: metrics[t]['avg_duration'])
        elif strategy == 'slow_first':
            # Execute slowest tests first
            available_tests.sort(key=lambda t: metrics[t]['avg_duration'], reverse=True)
        elif strategy == 'risky_first':
            # Execute high-risk tests first
            available_tests.sort(key=lambda t: metrics[t]['failure_rate'], reverse=True)
        elif strategy == 'balanced':
            # Balanced: prioritize by risk, then by duration
            available_tests.sort(
                key=lambda t: (
                    metrics[t]['failure_rate'] * 0.6 +  # Risk weight
                    (1 - min(metrics[t]['avg_duration'] / 10, 1)) * 0.4  # Speed weight (normalized)
                ),
                reverse=True
            )
        
        # Group into parallel batches
        batches = []
        current_batch = []
        current_batch_duration = 0
        
        for test_name in available_tests:
            test_duration = metrics[test_name]['avg_duration']
            
            # Start new batch if current is full or adding would exceed average
            if len(current_batch) >= max_parallel:
                batches.append({
                    'tests': current_batch.copy(),
                    'estimated_duration': current_batch_duration,
                    'parallel': len(current_batch)
                })
                current_batch = []
                current_batch_duration = 0
            
            current_batch.append(test_name)
            current_batch_duration = max(current_batch_duration, test_duration)
        
        # Add final batch
        if current_batch:
            batches.append({
                'tests': current_batch,
                'estimated_duration': current_batch_duration,
                'parallel': len(current_batch)
            })
        
        # Calculate total estimated time
        total_time = sum(b['estimated_duration'] for b in batches)
        
        return {
            'strategy': strategy,
            'total_tests': len(available_tests),
            'total_batches': len(batches),
            'estimated_total_time': round(total_time, 2),
            'max_parallel': max_parallel,
            'batches': batches,
            'metrics_summary': {
                'avg_duration': statistics.mean([metrics[t]['avg_duration'] for t in available_tests]),
                'total_failure_rate': statistics.mean([metrics[t]['failure_rate'] for t in available_tests])
            }
        }
    
    def generate_pytest_command(
        self,
        plan: Dict,
        output_file: Path = None
    ) -> str:
        """Generate pytest command from plan"""
        lines = []
        lines.append("# Generated test execution plan")
        lines.append(f"# Strategy: {plan['strategy']}")
        lines.append(f"# Estimated time: {plan['estimated_total_time']}s")
        lines.append("")
        
        for i, batch in enumerate(plan['batches'], 1):
            lines.append(f"# Batch {i} ({batch['estimated_duration']:.1f}s, {batch['parallel']} parallel)")
            test_args = " ".join(batch['tests'])
            lines.append(f"pytest -n {batch['parallel']} {test_args}")
            lines.append("")
        
        command = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(command)
            print(f"✅ Plan saved to {output_file}")
        
        return command
    
    def optimize_for_time(
        self,
        test_names: List[str],
        target_time: float,
        max_parallel: int = 4
    ) -> Dict:
        """Optimize test selection to fit within target time"""
        metrics = self.get_test_metrics()
        
        # Sort by value (failure_rate / duration ratio)
        test_values = []
        for test_name in test_names:
            if test_name in metrics:
                m = metrics[test_name]
                value = m['failure_rate'] / max(m['avg_duration'], 0.1)  # Value per second
                test_values.append({
                    'test_name': test_name,
                    'value': value,
                    'duration': m['avg_duration'],
                    'failure_rate': m['failure_rate']
                })
        
        # Greedy selection
        selected = []
        total_time = 0
        
        # Sort by value descending
        test_values.sort(key=lambda x: x['value'], reverse=True)
        
        for test in test_values:
            if total_time + test['duration'] <= target_time:
                selected.append(test['test_name'])
                total_time += test['duration']
        
        return {
            'selected_tests': selected,
            'total_tests': len(selected),
            'estimated_time': round(total_time, 2),
            'target_time': target_time,
            'coverage': len(selected) / len(test_names) * 100 if test_names else 0
        }


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Execution Planner')
    parser.add_argument('--plan', nargs='+', help='Plan execution for tests')
    parser.add_argument('--strategy', choices=['fast_first', 'slow_first', 'risky_first', 'balanced'], 
                       default='balanced', help='Execution strategy')
    parser.add_argument('--parallel', type=int, default=4, help='Max parallel tests')
    parser.add_argument('--optimize', nargs='+', help='Optimize test selection')
    parser.add_argument('--target-time', type=float, help='Target time for optimization (seconds)')
    parser.add_argument('--output', type=str, help='Output plan file')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    planner = TestExecutionPlanner(project_root)
    
    if args.optimize and args.target_time:
        print(f"⚡ Optimizing test selection for {args.target_time}s...")
        result = planner.optimize_for_time(args.optimize, args.target_time, args.parallel)
        print(f"\n✅ Selected {result['total_tests']} tests")
        print(f"   Estimated time: {result['estimated_time']}s")
        print(f"   Coverage: {result['coverage']:.1f}%")
    elif args.plan:
        print(f"📋 Planning execution for {len(args.plan)} tests...")
        plan = planner.plan_execution(args.plan, args.parallel, args.strategy)
        
        print(f"\n📊 Execution Plan:")
        print(f"  Strategy: {plan['strategy']}")
        print(f"  Total Batches: {plan['total_batches']}")
        print(f"  Estimated Time: {plan['estimated_total_time']}s")
        
        if args.output:
            planner.generate_pytest_command(plan, Path(args.output))
        else:
            print("\n📝 Plan:")
            for i, batch in enumerate(plan['batches'][:5], 1):
                print(f"  Batch {i}: {len(batch['tests'])} tests, ~{batch['estimated_duration']:.1f}s")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

