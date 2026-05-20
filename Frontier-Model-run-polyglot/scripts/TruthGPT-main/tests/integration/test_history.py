"""
Test History Tracker
Tracks test results over time to identify trends and regressions
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

HISTORY_FILE = "test_history.json"

def load_history() -> List[Dict[str, Any]]:
    """Load test history from file"""
    if Path(HISTORY_FILE).exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_history(history: List[Dict[str, Any]]):
    """Save test history to file"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, default=str)

def add_test_result(results: Dict[str, Any]):
    """Add a new test result to history"""
    history = load_history()
    
    entry = {
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'total_tests': results.get('total_tests', 0),
        'passed': results.get('passed', 0),
        'failures': results.get('failures', 0),
        'errors': results.get('errors', 0),
        'skipped': results.get('skipped', 0),
        'success_rate': results.get('success_rate', 0.0),
        'execution_time': results.get('execution_time', 0.0),
        'tests_per_second': results.get('tests_per_second', 0.0),
        'category': results.get('category', 'all')
    }
    
    history.append(entry)
    
    # Keep only last 100 entries
    if len(history) > 100:
        history = history[-100:]
    
    save_history(history)
    return entry

def get_statistics() -> Dict[str, Any]:
    """Get statistics from test history"""
    history = load_history()
    
    if not history:
        return {
            'total_runs': 0,
            'message': 'No test history available'
        }
    
    total_runs = len(history)
    success_rates = [h['success_rate'] for h in history]
    execution_times = [h['execution_time'] for h in history]
    
    return {
        'total_runs': total_runs,
        'average_success_rate': sum(success_rates) / len(success_rates) if success_rates else 0,
        'min_success_rate': min(success_rates) if success_rates else 0,
        'max_success_rate': max(success_rates) if success_rates else 0,
        'average_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
        'min_execution_time': min(execution_times) if execution_times else 0,
        'max_execution_time': max(execution_times) if execution_times else 0,
        'latest_run': history[-1] if history else None,
        'first_run': history[0] if history else None
    }

def print_statistics():
    """Print test history statistics"""
    stats = get_statistics()
    
    if stats['total_runs'] == 0:
        print("📊 No test history available")
        return
    
    print("📊 Test History Statistics")
    print("=" * 60)
    print(f"Total Runs: {stats['total_runs']}")
    print(f"\nSuccess Rate:")
    print(f"  Average: {stats['average_success_rate']:.1f}%")
    print(f"  Min:     {stats['min_success_rate']:.1f}%")
    print(f"  Max:     {stats['max_success_rate']:.1f}%")
    print(f"\nExecution Time:")
    print(f"  Average: {stats['average_execution_time']:.2f}s")
    print(f"  Min:     {stats['min_execution_time']:.2f}s")
    print(f"  Max:     {stats['max_execution_time']:.2f}s")
    
    if stats['latest_run']:
        latest = stats['latest_run']
        print(f"\nLatest Run ({latest['date']} {latest['time']}):")
        print(f"  Tests: {latest['total_tests']} | Passed: {latest['passed']} | Failed: {latest['failures'] + latest['errors']}")
        print(f"  Success Rate: {latest['success_rate']:.1f}%")

def print_recent_runs(limit: int = 10):
    """Print recent test runs"""
    history = load_history()
    
    if not history:
        print("📊 No test history available")
        return
    
    print(f"📊 Recent Test Runs (last {min(limit, len(history))})")
    print("=" * 80)
    print(f"{'Date':<12} {'Time':<10} {'Tests':<8} {'Passed':<8} {'Failed':<8} {'Success':<10} {'Time':<10}")
    print("-" * 80)
    
    for entry in history[-limit:]:
        status = "✅" if entry['failures'] + entry['errors'] == 0 else "❌"
        print(f"{entry['date']:<12} {entry['time']:<10} {entry['total_tests']:<8} "
              f"{entry['passed']:<8} {entry['failures'] + entry['errors']:<8} "
              f"{entry['success_rate']:>6.1f}% {status} {entry['execution_time']:>6.2f}s")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_history.py add <results.json>  # Add result to history")
        print("  python test_history.py stats               # Show statistics")
        print("  python test_history.py recent [limit]      # Show recent runs")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'add':
        if len(sys.argv) < 3:
            print("❌ Error: Please provide results JSON file")
            sys.exit(1)
        
        json_file = sys.argv[2]
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            entry = add_test_result(results)
            print(f"✅ Test result added to history: {entry['date']} {entry['time']}")
        except FileNotFoundError:
            print(f"❌ Error: File not found: {json_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON file: {e}")
            sys.exit(1)
    
    elif command == 'stats':
        print_statistics()
    
    elif command == 'recent':
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print_recent_runs(limit)
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()






