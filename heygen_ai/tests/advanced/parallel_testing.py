"""
Advanced parallel testing capabilities for HeyGen AI system.
Implements concurrent test execution with load balancing and resource management.
"""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import time
import psutil
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import queue
import logging

@dataclass
class TestTask:
    """Represents a test task for parallel execution."""
    id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 1  # 1 = highest, 5 = lowest
    timeout: float = 30.0
    retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TestResult:
    """Result of a test task execution."""
    task_id: str
    name: str
    status: str  # passed, failed, timeout, error, skipped
    duration: float
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    executed_at: datetime = field(default_factory=datetime.now)
    worker_id: Optional[str] = None

class ResourceMonitor:
    """Monitors system resources during parallel test execution."""
    
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self, interval: float):
        """Monitor system resources."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_percent)
                
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        if not self.cpu_usage or not self.memory_usage:
            return {}
        
        return {
            "cpu": {
                "current": self.cpu_usage[-1] if self.cpu_usage else 0,
                "average": sum(self.cpu_usage) / len(self.cpu_usage),
                "max": max(self.cpu_usage),
                "min": min(self.cpu_usage)
            },
            "memory": {
                "current": self.memory_usage[-1] if self.memory_usage else 0,
                "average": sum(self.memory_usage) / len(self.memory_usage),
                "max": max(self.memory_usage),
                "min": min(self.memory_usage)
            }
        }

class ParallelTestExecutor:
    """Advanced parallel test executor with load balancing and resource management."""
    
    def __init__(self, max_workers: Optional[int] = None, max_cpu_percent: float = 80.0):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.max_cpu_percent = max_cpu_percent
        self.resource_monitor = ResourceMonitor()
        self.task_queue = queue.PriorityQueue()
        self.results: List[TestResult] = []
        self.running_tasks: Dict[str, TestTask] = {}
        self.completed_tasks: set = set()
        self.failed_tasks: set = set()
        self.executor = None
        self.start_time = None
        self.end_time = None
    
    def add_task(self, task: TestTask):
        """Add a test task to the execution queue."""
        # Check dependencies
        if not self._check_dependencies(task):
            return False
        
        # Add to priority queue (lower priority number = higher priority)
        self.task_queue.put((task.priority, task.id, task))
        return True
    
    def _check_dependencies(self, task: TestTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def execute_parallel(self) -> List[TestResult]:
        """Execute all tasks in parallel with load balancing."""
        print(f"🚀 Starting parallel test execution with {self.max_workers} workers")
        print("=" * 60)
        
        self.start_time = time.time()
        self.resource_monitor.start_monitoring()
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                self.executor = executor
                
                # Submit initial tasks
                self._submit_available_tasks()
                
                # Monitor and submit new tasks as they complete
                while not self.task_queue.empty() or self.running_tasks:
                    self._check_completed_tasks()
                    self._submit_available_tasks()
                    
                    # Check resource usage
                    if self._should_throttle():
                        time.sleep(0.1)
                
                # Wait for remaining tasks
                self._wait_for_completion()
        
        finally:
            self.resource_monitor.stop_monitoring()
            self.end_time = time.time()
        
        self._print_summary()
        return self.results
    
    def _submit_available_tasks(self):
        """Submit available tasks to the executor."""
        while (len(self.running_tasks) < self.max_workers and 
               not self.task_queue.empty()):
            
            try:
                priority, task_id, task = self.task_queue.get_nowait()
                
                if self._check_dependencies(task):
                    future = self.executor.submit(self._execute_task, task)
                    self.running_tasks[task_id] = (task, future)
                else:
                    # Put back in queue if dependencies not met
                    self.task_queue.put((priority, task_id, task))
                    break
                    
            except queue.Empty:
                break
    
    def _check_completed_tasks(self):
        """Check for completed tasks and process results."""
        completed_task_ids = []
        
        for task_id, (task, future) in self.running_tasks.items():
            if future.done():
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    if result.status == "passed":
                        self.completed_tasks.add(task_id)
                    else:
                        self.failed_tasks.add(task_id)
                    
                    completed_task_ids.append(task_id)
                    
                except Exception as e:
                    error_result = TestResult(
                        task_id=task_id,
                        name=task.name,
                        status="error",
                        duration=0.0,
                        error=str(e)
                    )
                    self.results.append(error_result)
                    self.failed_tasks.add(task_id)
                    completed_task_ids.append(task_id)
        
        # Remove completed tasks
        for task_id in completed_task_ids:
            del self.running_tasks[task_id]
    
    def _execute_task(self, task: TestTask) -> TestResult:
        """Execute a single test task."""
        start_time = time.time()
        worker_id = threading.current_thread().name
        
        try:
            # Execute the task with timeout
            result = asyncio.run(
                asyncio.wait_for(
                    self._run_async_task(task),
                    timeout=task.timeout
                )
            )
            
            duration = time.time() - start_time
            
            return TestResult(
                task_id=task.id,
                name=task.name,
                status="passed",
                duration=duration,
                result=result,
                worker_id=worker_id
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return TestResult(
                task_id=task.id,
                name=task.name,
                status="timeout",
                duration=duration,
                error=f"Task timed out after {task.timeout}s",
                worker_id=worker_id
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                task_id=task.id,
                name=task.name,
                status="error",
                duration=duration,
                error=str(e),
                worker_id=worker_id
            )
    
    async def _run_async_task(self, task: TestTask):
        """Run task asynchronously."""
        if asyncio.iscoroutinefunction(task.function):
            return await task.function(*task.args, **task.kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, task.function, *task.args, **task.kwargs
            )
    
    def _should_throttle(self) -> bool:
        """Check if execution should be throttled due to resource usage."""
        stats = self.resource_monitor.get_stats()
        if not stats:
            return False
        
        return stats["cpu"]["current"] > self.max_cpu_percent
    
    def _wait_for_completion(self):
        """Wait for all running tasks to complete."""
        for task, future in self.running_tasks.values():
            try:
                result = future.result()
                self.results.append(result)
            except Exception as e:
                error_result = TestResult(
                    task_id=task.id,
                    name=task.name,
                    status="error",
                    duration=0.0,
                    error=str(e)
                )
                self.results.append(error_result)
    
    def _print_summary(self):
        """Print execution summary."""
        if not self.start_time or not self.end_time:
            return
        
        total_duration = self.end_time - self.start_time
        total_tasks = len(self.results)
        passed_tasks = sum(1 for r in self.results if r.status == "passed")
        failed_tasks = total_tasks - passed_tasks
        
        print("\n" + "=" * 60)
        print("📊 PARALLEL TEST EXECUTION SUMMARY")
        print("=" * 60)
        print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
        print(f"👥 Workers Used: {self.max_workers}")
        print(f"📋 Total Tasks: {total_tasks}")
        print(f"✅ Passed: {passed_tasks}")
        print(f"❌ Failed: {failed_tasks}")
        print(f"📈 Success Rate: {passed_tasks/total_tasks*100:.1f}%")
        
        # Resource usage
        stats = self.resource_monitor.get_stats()
        if stats:
            print(f"💻 CPU Usage: {stats['cpu']['average']:.1f}% avg, {stats['cpu']['max']:.1f}% max")
            print(f"🧠 Memory Usage: {stats['memory']['average']:.1f}% avg, {stats['memory']['max']:.1f}% max")
        
        # Worker utilization
        worker_stats = {}
        for result in self.results:
            if result.worker_id:
                worker_stats[result.worker_id] = worker_stats.get(result.worker_id, 0) + 1
        
        print(f"\n👥 Worker Utilization:")
        for worker, count in worker_stats.items():
            print(f"   {worker}: {count} tasks")
        
        print("=" * 60)

class LoadBalancedTestRunner:
    """Load-balanced test runner with intelligent task distribution."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.executor = ParallelTestExecutor(max_workers)
        self.test_suites: List[Dict[str, Any]] = []
    
    def add_test_suite(self, name: str, tests: List[Callable], priority: int = 1):
        """Add a test suite to the runner."""
        suite = {
            "name": name,
            "tests": tests,
            "priority": priority,
            "tasks": []
        }
        
        # Create tasks for each test
        for i, test_func in enumerate(tests):
            task = TestTask(
                id=f"{name}_{i}",
                name=f"{name}_{test_func.__name__}",
                function=test_func,
                priority=priority,
                timeout=30.0
            )
            suite["tasks"].append(task)
            self.executor.add_task(task)
        
        self.test_suites.append(suite)
    
    def run_all_suites(self) -> Dict[str, Any]:
        """Run all test suites in parallel."""
        print("🔄 Starting Load-Balanced Test Execution")
        print("=" * 60)
        
        results = self.executor.execute_parallel()
        
        # Organize results by suite
        suite_results = {}
        for suite in self.test_suites:
            suite_name = suite["name"]
            suite_tasks = {task.id for task in suite["tasks"]}
            
            suite_results[suite_name] = [
                result for result in results 
                if result.task_id in suite_tasks
            ]
        
        # Generate summary
        summary = self._generate_summary(suite_results)
        self._print_suite_summary(suite_results, summary)
        
        return {
            "suite_results": suite_results,
            "summary": summary,
            "all_results": results
        }
    
    def _generate_summary(self, suite_results: Dict[str, List[TestResult]]) -> Dict[str, Any]:
        """Generate execution summary."""
        total_tasks = sum(len(results) for results in suite_results.values())
        total_passed = sum(
            sum(1 for r in results if r.status == "passed")
            for results in suite_results.values()
        )
        
        suite_stats = {}
        for suite_name, results in suite_results.items():
            passed = sum(1 for r in results if r.status == "passed")
            total = len(results)
            suite_stats[suite_name] = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": (passed / total * 100) if total > 0 else 0
            }
        
        return {
            "total_tasks": total_tasks,
            "total_passed": total_passed,
            "total_failed": total_tasks - total_passed,
            "overall_success_rate": (total_passed / total_tasks * 100) if total_tasks > 0 else 0,
            "suite_stats": suite_stats
        }
    
    def _print_suite_summary(self, suite_results: Dict[str, List[TestResult]], summary: Dict[str, Any]):
        """Print suite execution summary."""
        print("\n📊 SUITE EXECUTION SUMMARY")
        print("=" * 60)
        
        for suite_name, results in suite_results.items():
            stats = summary["suite_stats"][suite_name]
            status_icon = "✅" if stats["success_rate"] == 100 else "⚠️"
            
            print(f"{status_icon} {suite_name}:")
            print(f"   Tests: {stats['passed']}/{stats['total']} passed ({stats['success_rate']:.1f}%)")
            
            # Show failed tests
            failed_tests = [r for r in results if r.status != "passed"]
            if failed_tests:
                print(f"   Failed: {', '.join(r.name for r in failed_tests)}")
        
        print(f"\n🎯 Overall: {summary['total_passed']}/{summary['total_tasks']} passed ({summary['overall_success_rate']:.1f}%)")
        print("=" * 60)

# Example usage and test functions
def sample_test_function_1():
    """Sample test function 1."""
    time.sleep(0.1)
    return "test_1_result"

def sample_test_function_2():
    """Sample test function 2."""
    time.sleep(0.2)
    return "test_2_result"

def sample_test_function_3():
    """Sample test function 3."""
    time.sleep(0.15)
    return "test_3_result"

async def sample_async_test_function():
    """Sample async test function."""
    await asyncio.sleep(0.1)
    return "async_test_result"

def demo_parallel_testing():
    """Demonstrate parallel testing capabilities."""
    print("🔄 Parallel Testing Demo")
    print("=" * 40)
    
    # Create load-balanced runner
    runner = LoadBalancedTestRunner(max_workers=4)
    
    # Add test suites
    runner.add_test_suite("Basic Tests", [
        sample_test_function_1,
        sample_test_function_2,
        sample_test_function_3
    ], priority=1)
    
    runner.add_test_suite("Async Tests", [
        sample_async_test_function,
        sample_async_test_function,
        sample_async_test_function
    ], priority=2)
    
    # Run all suites
    results = runner.run_all_suites()
    
    return results

if __name__ == "__main__":
    # Run demo
    demo_parallel_testing()
