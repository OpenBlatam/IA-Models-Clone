#!/usr/bin/env python3
"""
Performance Benchmark Suite for HeyGen AI
==========================================

Comprehensive performance benchmarking and optimization testing.
"""

import time
import asyncio
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
    name: str
    duration: float
    iterations: int
    avg_duration: float
    min_duration: float
    max_duration: float
    std_deviation: float
    throughput: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

class PerformanceBenchmark:
    """Performance benchmarking suite for HeyGen AI"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_results: Dict[str, BenchmarkResult] = {}
    
    def benchmark_function(self, func: Callable, name: str, iterations: int = 1000, 
                          *args, **kwargs) -> BenchmarkResult:
        """Benchmark a synchronous function"""
        print(f"🔬 Benchmarking {name} ({iterations} iterations)...")
        
        durations = []
        for i in range(iterations):
            start_time = time.perf_counter()
            try:
                func(*args, **kwargs)
                end_time = time.perf_counter()
                durations.append(end_time - start_time)
            except Exception as e:
                print(f"  ⚠️ Error in iteration {i}: {e}")
                continue
        
        if not durations:
            raise ValueError(f"No successful iterations for {name}")
        
        avg_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        std_deviation = statistics.stdev(durations) if len(durations) > 1 else 0
        throughput = 1.0 / avg_duration if avg_duration > 0 else 0
        
        result = BenchmarkResult(
            name=name,
            duration=sum(durations),
            iterations=len(durations),
            avg_duration=avg_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            std_deviation=std_deviation,
            throughput=throughput
        )
        
        self.results.append(result)
        return result
    
    async def benchmark_async_function(self, func: Callable, name: str, 
                                     iterations: int = 1000, *args, **kwargs) -> BenchmarkResult:
        """Benchmark an asynchronous function"""
        print(f"🔬 Benchmarking async {name} ({iterations} iterations)...")
        
        durations = []
        for i in range(iterations):
            start_time = time.perf_counter()
            try:
                await func(*args, **kwargs)
                end_time = time.perf_counter()
                durations.append(end_time - start_time)
            except Exception as e:
                print(f"  ⚠️ Error in iteration {i}: {e}")
                continue
        
        if not durations:
            raise ValueError(f"No successful iterations for {name}")
        
        avg_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        std_deviation = statistics.stdev(durations) if len(durations) > 1 else 0
        throughput = 1.0 / avg_duration if avg_duration > 0 else 0
        
        result = BenchmarkResult(
            name=name,
            duration=sum(durations),
            iterations=len(durations),
            avg_duration=avg_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            std_deviation=std_deviation,
            throughput=throughput
        )
        
        self.results.append(result)
        return result
    
    def benchmark_enterprise_features(self):
        """Benchmark enterprise features performance"""
        print("\n🏢 Benchmarking Enterprise Features...")
        
        try:
            from core.enterprise_features import EnterpriseFeatures, User, Role, Permission
            
            # Initialize enterprise features
            enterprise = EnterpriseFeatures()
            
            # Benchmark user creation
            def create_user():
                user = User(
                    username="benchmark_user",
                    email="benchmark@example.com",
                    full_name="Benchmark User",
                    role="user"
                )
                return user
            
            self.benchmark_function(create_user, "User Creation", 10000)
            
            # Benchmark role creation
            def create_role():
                role = Role(
                    name="benchmark_role",
                    description="Benchmark Role",
                    permissions=["read", "write"]
                )
                return role
            
            self.benchmark_function(create_role, "Role Creation", 10000)
            
            # Benchmark permission creation
            def create_permission():
                permission = Permission(
                    name="benchmark_permission",
                    description="Benchmark Permission",
                    resource="benchmark_resource",
                    actions=["read", "write"]
                )
                return permission
            
            self.benchmark_function(create_permission, "Permission Creation", 10000)
            
        except ImportError as e:
            print(f"  ⚠️ Enterprise features not available: {e}")
    
    def benchmark_core_structures(self):
        """Benchmark core data structures"""
        print("\n🔧 Benchmarking Core Structures...")
        
        try:
            from core.base_service import ServiceStatus, ServiceType, HealthCheckResult
            from core.dependency_manager import ServicePriority, ServiceInfo
            
            # Benchmark ServiceStatus enum
            def service_status_operations():
                status = ServiceStatus.RUNNING
                return str(status), status.value
            
            self.benchmark_function(service_status_operations, "ServiceStatus Operations", 50000)
            
            # Benchmark ServiceInfo creation
            def create_service_info():
                info = ServiceInfo(
                    name="benchmark_service",
                    version="1.0.0",
                    status=ServiceStatus.RUNNING,
                    priority=ServicePriority.HIGH,
                    dependencies=[],
                    metadata={"benchmark": True}
                )
                return info
            
            self.benchmark_function(create_service_info, "ServiceInfo Creation", 10000)
            
            # Benchmark HealthCheckResult
            def create_health_check():
                result = HealthCheckResult(
                    status=ServiceStatus.RUNNING,
                    message="Benchmark health check",
                    timestamp=datetime.now(),
                    details={"benchmark": True}
                )
                return result
            
            self.benchmark_function(create_health_check, "HealthCheckResult Creation", 10000)
            
        except ImportError as e:
            print(f"  ⚠️ Core structures not available: {e}")
    
    def benchmark_import_performance(self):
        """Benchmark import performance"""
        print("\n📦 Benchmarking Import Performance...")
        
        modules_to_test = [
            "core.base_service",
            "core.dependency_manager",
            "core.error_handler",
            "core.config_manager",
            "core.logging_service",
            "core.enterprise_features"
        ]
        
        for module_name in modules_to_test:
            def import_module():
                try:
                    __import__(module_name)
                    return True
                except ImportError:
                    return False
            
            self.benchmark_function(import_module, f"Import {module_name}", 1000)
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage patterns"""
        print("\n💾 Benchmarking Memory Usage...")
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            def memory_intensive_operation():
                # Create a large data structure
                data = [i for i in range(10000)]
                result = sum(data)
                return result
            
            # Get baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            result = self.benchmark_function(memory_intensive_operation, "Memory Intensive Operation", 100)
            
            # Get peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = peak_memory - baseline_memory
            
            result.memory_usage = memory_usage
            print(f"  📊 Memory usage: {memory_usage:.2f} MB")
            
        except ImportError:
            print("  ⚠️ psutil not available for memory benchmarking")
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report"""
        report = []
        report.append("🚀 HeyGen AI Performance Benchmark Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Benchmarks: {len(self.results)}")
        report.append("")
        
        # Sort results by throughput (descending)
        sorted_results = sorted(self.results, key=lambda x: x.throughput, reverse=True)
        
        report.append("📊 Performance Rankings (by Throughput):")
        report.append("-" * 60)
        for i, result in enumerate(sorted_results[:10], 1):
            report.append(f"{i:2d}. {result.name}")
            report.append(f"    Throughput: {result.throughput:.2f} ops/sec")
            report.append(f"    Avg Duration: {result.avg_duration*1000:.3f} ms")
            report.append(f"    Std Dev: {result.std_deviation*1000:.3f} ms")
            if result.memory_usage:
                report.append(f"    Memory: {result.memory_usage:.2f} MB")
            report.append("")
        
        # Performance summary
        if self.results:
            avg_throughput = statistics.mean([r.throughput for r in self.results])
            total_operations = sum([r.iterations for r in self.results])
            total_duration = sum([r.duration for r in self.results])
            
            report.append("📈 Performance Summary:")
            report.append(f"  Average Throughput: {avg_throughput:.2f} ops/sec")
            report.append(f"  Total Operations: {total_operations:,}")
            report.append(f"  Total Duration: {total_duration:.2f} seconds")
            report.append("")
        
        # Performance recommendations
        report.append("💡 Performance Recommendations:")
        report.append("-" * 40)
        
        # Find slowest operations
        slowest = min(self.results, key=lambda x: x.throughput)
        report.append(f"  🐌 Slowest Operation: {slowest.name} ({slowest.throughput:.2f} ops/sec)")
        
        # Find most variable operations
        most_variable = max(self.results, key=lambda x: x.std_deviation)
        report.append(f"  📊 Most Variable: {most_variable.name} (σ={most_variable.std_deviation*1000:.3f} ms)")
        
        # Find fastest operations
        fastest = max(self.results, key=lambda x: x.throughput)
        report.append(f"  ⚡ Fastest Operation: {fastest.name} ({fastest.throughput:.2f} ops/sec)")
        
        return "\n".join(report)
    
    def save_benchmark_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file"""
        results_data = []
        for result in self.results:
            results_data.append({
                "name": result.name,
                "duration": result.duration,
                "iterations": result.iterations,
                "avg_duration": result.avg_duration,
                "min_duration": result.min_duration,
                "max_duration": result.max_duration,
                "std_deviation": result.std_deviation,
                "throughput": result.throughput,
                "memory_usage": result.memory_usage,
                "cpu_usage": result.cpu_usage
            })
        
        base_dir = Path(__file__).parent
        results_file = base_dir / filename
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Benchmark results saved to: {results_file}")
    
    def run_all_benchmarks(self):
        """Run all benchmark tests"""
        print("🚀 Starting HeyGen AI Performance Benchmarks")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all benchmark categories
        self.benchmark_import_performance()
        self.benchmark_core_structures()
        self.benchmark_enterprise_features()
        self.benchmark_memory_usage()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Generate and display report
        report = self.generate_benchmark_report()
        print(f"\n{report}")
        
        # Save results
        self.save_benchmark_results()
        
        print(f"\n⏱️ Total benchmark duration: {total_duration:.2f} seconds")
        print(f"📊 Benchmarked {len(self.results)} operations")

def main():
    """Main benchmark function"""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()
    return 0

if __name__ == "__main__":
    sys.exit(main())





