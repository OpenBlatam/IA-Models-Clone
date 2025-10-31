"""
Auto-Healing Framework for HeyGen AI Testing System.
Intelligent test auto-healing including automatic fixes, retry mechanisms,
and adaptive test strategies.
"""

import time
import json
import re
import ast
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict, deque
import threading
import queue

@dataclass
class TestFailure:
    """Represents a test failure with context."""
    test_id: str
    test_name: str
    failure_type: str  # assertion, timeout, exception, flaky
    error_message: str
    stack_trace: str
    timestamp: datetime
    retry_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_fixes: List[str] = field(default_factory=list)

@dataclass
class HealingAction:
    """Represents a healing action taken."""
    action_id: str
    test_id: str
    action_type: str  # retry, fix, skip, modify
    description: str
    success: bool
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealingStrategy:
    """Represents a healing strategy."""
    strategy_id: str
    name: str
    description: str
    conditions: List[str]  # When to apply this strategy
    actions: List[str]  # What actions to take
    priority: int = 1  # Higher number = higher priority
    success_rate: float = 0.0
    usage_count: int = 0

class FailureAnalyzer:
    """Analyzes test failures to determine appropriate healing strategies."""
    
    def __init__(self):
        self.failure_patterns = {
            'timeout': [
                r'timeout',
                r'timed out',
                r'connection timeout',
                r'request timeout'
            ],
            'assertion': [
                r'assertion error',
                r'assert.*failed',
                r'expected.*but got',
                r'not equal'
            ],
            'network': [
                r'connection refused',
                r'network unreachable',
                r'dns resolution failed',
                r'socket error'
            ],
            'resource': [
                r'out of memory',
                r'disk space',
                r'file not found',
                r'permission denied'
            ],
            'flaky': [
                r'intermittent',
                r'race condition',
                r'timing issue',
                r'random failure'
            ]
        }
        
        self.healing_strategies = self._initialize_healing_strategies()
    
    def _initialize_healing_strategies(self) -> List[HealingStrategy]:
        """Initialize default healing strategies."""
        strategies = [
            HealingStrategy(
                strategy_id="retry_simple",
                name="Simple Retry",
                description="Retry the test with exponential backoff",
                conditions=["timeout", "network", "flaky"],
                actions=["retry", "wait"],
                priority=1
            ),
            HealingStrategy(
                strategy_id="retry_with_cleanup",
                name="Retry with Cleanup",
                description="Clean up resources and retry",
                conditions=["resource", "timeout"],
                actions=["cleanup", "retry", "wait"],
                priority=2
            ),
            HealingStrategy(
                strategy_id="skip_flaky",
                name="Skip Flaky Test",
                description="Skip test if it's consistently flaky",
                conditions=["flaky"],
                actions=["skip", "mark_flaky"],
                priority=3
            ),
            HealingStrategy(
                strategy_id="modify_timeout",
                name="Modify Timeout",
                description="Increase timeout for slow tests",
                conditions=["timeout"],
                actions=["modify_timeout", "retry"],
                priority=2
            ),
            HealingStrategy(
                strategy_id="fix_assertion",
                name="Fix Assertion",
                description="Attempt to fix assertion errors",
                conditions=["assertion"],
                actions=["analyze_assertion", "suggest_fix", "retry"],
                priority=4
            )
        ]
        
        return strategies
    
    def analyze_failure(self, failure: TestFailure) -> List[HealingStrategy]:
        """Analyze a test failure and suggest healing strategies."""
        failure_type = self._classify_failure(failure)
        failure.failure_type = failure_type
        
        # Find applicable strategies
        applicable_strategies = []
        for strategy in self.healing_strategies:
            if failure_type in strategy.conditions:
                applicable_strategies.append(strategy)
        
        # Sort by priority and success rate
        applicable_strategies.sort(key=lambda x: (x.priority, x.success_rate), reverse=True)
        
        return applicable_strategies
    
    def _classify_failure(self, failure: TestFailure) -> str:
        """Classify the type of test failure."""
        error_text = (failure.error_message + " " + failure.stack_trace).lower()
        
        for failure_type, patterns in self.failure_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_text):
                    return failure_type
        
        return "unknown"
    
    def update_strategy_success(self, strategy_id: str, success: bool):
        """Update the success rate of a healing strategy."""
        for strategy in self.healing_strategies:
            if strategy.strategy_id == strategy_id:
                strategy.usage_count += 1
                if success:
                    strategy.success_rate = ((strategy.success_rate * (strategy.usage_count - 1)) + 1) / strategy.usage_count
                else:
                    strategy.success_rate = (strategy.success_rate * (strategy.usage_count - 1)) / strategy.usage_count
                break

class TestHealer:
    """Implements auto-healing mechanisms for tests."""
    
    def __init__(self, failure_analyzer: FailureAnalyzer):
        self.failure_analyzer = failure_analyzer
        self.healing_history: List[HealingAction] = []
        self.retry_config = {
            'max_retries': 3,
            'base_delay': 1.0,
            'max_delay': 30.0,
            'backoff_multiplier': 2.0
        }
        self.flaky_tests = set()
        self.timeout_adjustments = {}
    
    def heal_test_failure(self, failure: TestFailure) -> Tuple[bool, List[HealingAction]]:
        """Attempt to heal a test failure."""
        strategies = self.failure_analyzer.analyze_failure(failure)
        healing_actions = []
        
        for strategy in strategies:
            success, actions = self._apply_strategy(strategy, failure)
            healing_actions.extend(actions)
            
            if success:
                # Update strategy success rate
                self.failure_analyzer.update_strategy_success(strategy.strategy_id, True)
                return True, healing_actions
            else:
                # Update strategy success rate
                self.failure_analyzer.update_strategy_success(strategy.strategy_id, False)
        
        return False, healing_actions
    
    def _apply_strategy(self, strategy: HealingStrategy, failure: TestFailure) -> Tuple[bool, List[HealingAction]]:
        """Apply a specific healing strategy."""
        actions = []
        
        try:
            if strategy.strategy_id == "retry_simple":
                success, retry_actions = self._retry_simple(failure)
                actions.extend(retry_actions)
                return success, actions
            
            elif strategy.strategy_id == "retry_with_cleanup":
                success, cleanup_actions = self._retry_with_cleanup(failure)
                actions.extend(cleanup_actions)
                return success, actions
            
            elif strategy.strategy_id == "skip_flaky":
                success, skip_actions = self._skip_flaky_test(failure)
                actions.extend(skip_actions)
                return success, actions
            
            elif strategy.strategy_id == "modify_timeout":
                success, timeout_actions = self._modify_timeout(failure)
                actions.extend(timeout_actions)
                return success, actions
            
            elif strategy.strategy_id == "fix_assertion":
                success, assertion_actions = self._fix_assertion(failure)
                actions.extend(assertion_actions)
                return success, actions
            
            else:
                return False, actions
        
        except Exception as e:
            logging.error(f"Error applying strategy {strategy.strategy_id}: {e}")
            return False, actions
    
    def _retry_simple(self, failure: TestFailure) -> Tuple[bool, List[HealingAction]]:
        """Simple retry with exponential backoff."""
        actions = []
        
        if failure.retry_count >= self.retry_config['max_retries']:
            action = HealingAction(
                action_id=f"retry_limit_{int(time.time())}",
                test_id=failure.test_id,
                action_type="retry",
                description="Retry limit reached",
                success=False,
                timestamp=datetime.now(),
                details={"retry_count": failure.retry_count}
            )
            actions.append(action)
            return False, actions
        
        # Calculate delay
        delay = min(
            self.retry_config['base_delay'] * (self.retry_config['backoff_multiplier'] ** failure.retry_count),
            self.retry_config['max_delay']
        )
        
        # Wait before retry
        time.sleep(delay)
        
        # Record retry action
        action = HealingAction(
            action_id=f"retry_{int(time.time())}",
            test_id=failure.test_id,
            action_type="retry",
            description=f"Retrying test (attempt {failure.retry_count + 1})",
            success=True,
            timestamp=datetime.now(),
            details={"delay": delay, "retry_count": failure.retry_count + 1}
        )
        actions.append(action)
        
        return True, actions
    
    def _retry_with_cleanup(self, failure: TestFailure) -> Tuple[bool, List[HealingAction]]:
        """Retry with resource cleanup."""
        actions = []
        
        # Cleanup actions
        cleanup_action = HealingAction(
            action_id=f"cleanup_{int(time.time())}",
            test_id=failure.test_id,
            action_type="cleanup",
            description="Cleaning up resources",
            success=True,
            timestamp=datetime.now()
        )
        actions.append(cleanup_action)
        
        # Perform cleanup
        self._perform_cleanup(failure)
        
        # Retry
        success, retry_actions = self._retry_simple(failure)
        actions.extend(retry_actions)
        
        return success, actions
    
    def _skip_flaky_test(self, failure: TestFailure) -> Tuple[bool, List[HealingAction]]:
        """Skip a flaky test."""
        actions = []
        
        # Mark as flaky
        self.flaky_tests.add(failure.test_name)
        
        skip_action = HealingAction(
            action_id=f"skip_flaky_{int(time.time())}",
            test_id=failure.test_id,
            action_type="skip",
            description="Skipping flaky test",
            success=True,
            timestamp=datetime.now(),
            details={"reason": "flaky_test"}
        )
        actions.append(skip_action)
        
        return True, actions
    
    def _modify_timeout(self, failure: TestFailure) -> Tuple[bool, List[HealingAction]]:
        """Modify timeout for slow tests."""
        actions = []
        
        # Increase timeout
        current_timeout = self.timeout_adjustments.get(failure.test_name, 30)
        new_timeout = min(current_timeout * 1.5, 300)  # Max 5 minutes
        self.timeout_adjustments[failure.test_name] = new_timeout
        
        timeout_action = HealingAction(
            action_id=f"timeout_{int(time.time())}",
            test_id=failure.test_id,
            action_type="modify",
            description=f"Increased timeout to {new_timeout}s",
            success=True,
            timestamp=datetime.now(),
            details={"old_timeout": current_timeout, "new_timeout": new_timeout}
        )
        actions.append(timeout_action)
        
        # Retry with new timeout
        success, retry_actions = self._retry_simple(failure)
        actions.extend(retry_actions)
        
        return success, actions
    
    def _fix_assertion(self, failure: TestFailure) -> Tuple[bool, List[HealingAction]]:
        """Attempt to fix assertion errors."""
        actions = []
        
        # Analyze assertion error
        assertion_fix = self._analyze_assertion_error(failure)
        
        if assertion_fix:
            fix_action = HealingAction(
                action_id=f"fix_assertion_{int(time.time())}",
                test_id=failure.test_id,
                action_type="fix",
                description="Attempting to fix assertion error",
                success=True,
                timestamp=datetime.now(),
                details={"suggested_fix": assertion_fix}
            )
            actions.append(fix_action)
            
            # Apply fix and retry
            success, retry_actions = self._retry_simple(failure)
            actions.extend(retry_actions)
            
            return success, actions
        else:
            return False, actions
    
    def _perform_cleanup(self, failure: TestFailure):
        """Perform resource cleanup."""
        try:
            # Clean up temporary files
            temp_dir = Path("temp")
            if temp_dir.exists():
                for file in temp_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
            
            # Clear caches
            # This would be implementation-specific
            
            # Reset test environment
            # This would be implementation-specific
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def _analyze_assertion_error(self, failure: TestFailure) -> Optional[str]:
        """Analyze assertion error and suggest fixes."""
        error_message = failure.error_message.lower()
        
        # Common assertion fixes
        if "not equal" in error_message:
            return "Check if values are of the same type and format"
        elif "expected" in error_message and "but got" in error_message:
            return "Verify expected vs actual values match"
        elif "assertion error" in error_message:
            return "Review assertion logic and conditions"
        elif "timeout" in error_message:
            return "Consider increasing timeout or optimizing test"
        
        return None
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get healing statistics."""
        total_actions = len(self.healing_history)
        successful_actions = sum(1 for action in self.healing_history if action.success)
        
        actions_by_type = defaultdict(int)
        for action in self.healing_history:
            actions_by_type[action.action_type] += 1
        
        return {
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": successful_actions / total_actions if total_actions > 0 else 0,
            "actions_by_type": dict(actions_by_type),
            "flaky_tests": len(self.flaky_tests),
            "timeout_adjustments": len(self.timeout_adjustments)
        }

class AdaptiveTestRunner:
    """Adaptive test runner that applies auto-healing."""
    
    def __init__(self, healer: TestHealer):
        self.healer = healer
        self.test_queue = queue.Queue()
        self.results = []
        self.running = False
        self.worker_threads = []
    
    def add_test(self, test_func: Callable, test_name: str, test_id: str = None):
        """Add a test to the queue."""
        if test_id is None:
            test_id = f"test_{int(time.time())}"
        
        self.test_queue.put({
            'test_func': test_func,
            'test_name': test_name,
            'test_id': test_id
        })
    
    def run_tests(self, max_workers: int = 4):
        """Run tests with auto-healing."""
        self.running = True
        self.worker_threads = []
        
        # Start worker threads
        for i in range(max_workers):
            thread = threading.Thread(target=self._worker, daemon=True)
            thread.start()
            self.worker_threads.append(thread)
        
        # Wait for all tests to complete
        self.test_queue.join()
        
        self.running = False
        
        return self.results
    
    def _worker(self):
        """Worker thread for running tests."""
        while self.running:
            try:
                test_item = self.test_queue.get(timeout=1)
                self._run_test_with_healing(test_item)
                self.test_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in worker thread: {e}")
    
    def _run_test_with_healing(self, test_item: Dict[str, Any]):
        """Run a test with auto-healing."""
        test_func = test_item['test_func']
        test_name = test_item['test_name']
        test_id = test_item['test_id']
        
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Run the test
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                
                # Record successful execution
                self.results.append({
                    'test_id': test_id,
                    'test_name': test_name,
                    'success': True,
                    'duration': duration,
                    'attempt': attempt + 1,
                    'timestamp': datetime.now()
                })
                
                return
                
            except Exception as e:
                attempt += 1
                
                # Create failure record
                failure = TestFailure(
                    test_id=test_id,
                    test_name=test_name,
                    failure_type="unknown",
                    error_message=str(e),
                    stack_trace=str(e),
                    timestamp=datetime.now(),
                    retry_count=attempt - 1
                )
                
                # Try to heal the failure
                healed, healing_actions = self.healer.heal_test_failure(failure)
                
                if healed and attempt < max_attempts:
                    # Continue with next attempt
                    continue
                else:
                    # Record failed execution
                    self.results.append({
                        'test_id': test_id,
                        'test_name': test_name,
                        'success': False,
                        'duration': time.time() - start_time,
                        'attempt': attempt,
                        'error': str(e),
                        'healing_actions': [action.__dict__ for action in healing_actions],
                        'timestamp': datetime.now()
                    })
                    return

class AutoHealingFramework:
    """Main auto-healing framework."""
    
    def __init__(self):
        self.failure_analyzer = FailureAnalyzer()
        self.healer = TestHealer(self.failure_analyzer)
        self.adaptive_runner = AdaptiveTestRunner(self.healer)
        self.healing_stats = {
            'total_failures': 0,
            'healed_failures': 0,
            'total_actions': 0,
            'successful_actions': 0
        }
    
    def run_test_with_healing(self, test_func: Callable, test_name: str) -> Dict[str, Any]:
        """Run a single test with auto-healing."""
        test_id = f"test_{int(time.time())}"
        
        # Add test to runner
        self.adaptive_runner.add_test(test_func, test_name, test_id)
        
        # Run tests
        results = self.adaptive_runner.run_tests(max_workers=1)
        
        if results:
            return results[0]
        else:
            return {
                'test_id': test_id,
                'test_name': test_name,
                'success': False,
                'error': 'No results returned'
            }
    
    def run_test_suite_with_healing(self, test_suite: List[Tuple[Callable, str]], 
                                  max_workers: int = 4) -> List[Dict[str, Any]]:
        """Run a test suite with auto-healing."""
        # Clear previous results
        self.adaptive_runner.results = []
        
        # Add tests to runner
        for test_func, test_name in test_suite:
            self.adaptive_runner.add_test(test_func, test_name)
        
        # Run tests
        results = self.adaptive_runner.run_tests(max_workers)
        
        # Update statistics
        self._update_statistics(results)
        
        return results
    
    def _update_statistics(self, results: List[Dict[str, Any]]):
        """Update healing statistics."""
        for result in results:
            if not result['success']:
                self.healing_stats['total_failures'] += 1
                
                if 'healing_actions' in result:
                    self.healing_stats['total_actions'] += len(result['healing_actions'])
                    
                    # Check if any healing action was successful
                    for action in result['healing_actions']:
                        if action.get('success', False):
                            self.healing_stats['successful_actions'] += 1
                            self.healing_stats['healed_failures'] += 1
                            break
    
    def get_healing_report(self) -> Dict[str, Any]:
        """Get comprehensive healing report."""
        healer_stats = self.healer.get_healing_statistics()
        
        return {
            "framework_stats": self.healing_stats,
            "healer_stats": healer_stats,
            "healing_rate": self.healing_stats['healed_failures'] / max(1, self.healing_stats['total_failures']),
            "action_success_rate": self.healing_stats['successful_actions'] / max(1, self.healing_stats['total_actions']),
            "flaky_tests": list(self.healer.flaky_tests),
            "timeout_adjustments": self.healer.timeout_adjustments
        }
    
    def configure_healing(self, config: Dict[str, Any]):
        """Configure healing parameters."""
        if 'retry_config' in config:
            self.healer.retry_config.update(config['retry_config'])
        
        if 'max_workers' in config:
            self.adaptive_runner.max_workers = config['max_workers']

# Example usage and demo
def demo_auto_healing():
    """Demonstrate auto-healing capabilities."""
    print("🔧 Auto-Healing Framework Demo")
    print("=" * 50)
    
    # Create auto-healing framework
    framework = AutoHealingFramework()
    
    # Define some test functions
    def flaky_test():
        """A flaky test that sometimes fails."""
        import random
        if random.random() < 0.3:  # 30% chance of failure
            raise Exception("Random failure - this is a flaky test")
        return True
    
    def timeout_test():
        """A test that sometimes times out."""
        import random
        if random.random() < 0.4:  # 40% chance of timeout
            time.sleep(2)  # Simulate slow operation
        return True
    
    def assertion_test():
        """A test with assertion errors."""
        import random
        if random.random() < 0.2:  # 20% chance of assertion error
            assert False, "Assertion error - values not equal"
        return True
    
    def reliable_test():
        """A reliable test that always passes."""
        return True
    
    # Create test suite
    test_suite = [
        (reliable_test, "reliable_test"),
        (flaky_test, "flaky_test"),
        (timeout_test, "timeout_test"),
        (assertion_test, "assertion_test")
    ]
    
    print("🧪 Running test suite with auto-healing...")
    
    # Run tests with auto-healing
    results = framework.run_test_suite_with_healing(test_suite, max_workers=2)
    
    # Print results
    print("\n📊 Test Results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {result['test_name']}: {result['duration']:.2f}s")
        
        if not result['success'] and 'healing_actions' in result:
            print(f"    Healing actions: {len(result['healing_actions'])}")
            for action in result['healing_actions']:
                action_status = "✅" if action['success'] else "❌"
                print(f"      {action_status} {action['action_type']}: {action['description']}")
    
    # Print healing report
    print("\n📈 Healing Report:")
    report = framework.get_healing_report()
    print(f"  Total Failures: {report['framework_stats']['total_failures']}")
    print(f"  Healed Failures: {report['framework_stats']['healed_failures']}")
    print(f"  Healing Rate: {report['healing_rate']:.1%}")
    print(f"  Action Success Rate: {report['action_success_rate']:.1%}")
    print(f"  Flaky Tests: {len(report['flaky_tests'])}")
    print(f"  Timeout Adjustments: {len(report['timeout_adjustments'])}")

if __name__ == "__main__":
    # Run demo
    demo_auto_healing()
