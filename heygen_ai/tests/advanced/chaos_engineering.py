"""
Chaos Engineering Framework for HeyGen AI Testing System.
Advanced chaos testing including fault injection, resilience testing,
and system stability validation.
"""

import asyncio
import random
import time
import psutil
import threading
import subprocess
import signal
import os
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import requests
import socket
import tempfile
import shutil

@dataclass
class ChaosExperiment:
    """Represents a chaos engineering experiment."""
    experiment_id: str
    name: str
    description: str
    target_system: str
    chaos_type: str  # cpu, memory, network, disk, process, service
    intensity: float  # 0.0 to 1.0
    duration: int  # seconds
    recovery_time: int  # seconds
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChaosResult:
    """Result of a chaos experiment."""
    experiment_id: str
    success: bool
    system_impact: Dict[str, Any]
    recovery_time: float
    metrics_before: Dict[str, Any]
    metrics_during: Dict[str, Any]
    metrics_after: Dict[str, Any]
    observations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class SystemMonitor:
    """Monitors system metrics during chaos experiments."""
    
    def __init__(self):
        self.metrics_history = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_system,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_system(self, interval: float):
        """Monitor system metrics."""
        while self.monitoring:
            try:
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
                    'disk_usage_percent': psutil.disk_usage('/').percent,
                    'network_io': psutil.net_io_counters()._asdict(),
                    'process_count': len(psutil.pids()),
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
                }
                
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 records
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(interval)
                
            except Exception as e:
                logging.error(f"Error monitoring system: {e}")
                time.sleep(interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.metrics_history:
            return {}
        
        return self.metrics_history[-1]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.metrics_history:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics_history]
        memory_values = [m['memory_percent'] for m in self.metrics_history]
        
        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': sum(memory_values) / len(memory_values),
            'max_memory_percent': max(memory_values),
            'total_samples': len(self.metrics_history)
        }

class CPUChaos:
    """CPU chaos injection."""
    
    def __init__(self):
        self.chaos_threads = []
        self.running = False
    
    def inject_cpu_stress(self, intensity: float, duration: int):
        """Inject CPU stress."""
        self.running = True
        self.chaos_threads = []
        
        # Calculate number of threads based on intensity
        num_threads = int(psutil.cpu_count() * intensity)
        
        def cpu_stress_worker():
            end_time = time.time() + duration
            while self.running and time.time() < end_time:
                # CPU intensive operation
                sum(range(1000000))
        
        # Start stress threads
        for _ in range(num_threads):
            thread = threading.Thread(target=cpu_stress_worker, daemon=True)
            thread.start()
            self.chaos_threads.append(thread)
    
    def stop_cpu_stress(self):
        """Stop CPU stress."""
        self.running = False
        for thread in self.chaos_threads:
            thread.join(timeout=1)
        self.chaos_threads = []

class MemoryChaos:
    """Memory chaos injection."""
    
    def __init__(self):
        self.memory_blocks = []
        self.running = False
    
    def inject_memory_stress(self, intensity: float, duration: int):
        """Inject memory stress."""
        self.running = True
        
        # Calculate memory to allocate based on intensity
        total_memory = psutil.virtual_memory().total
        target_memory = int(total_memory * intensity)
        
        def memory_stress_worker():
            end_time = time.time() + duration
            while self.running and time.time() < end_time:
                try:
                    # Allocate memory blocks
                    block_size = min(1024 * 1024, target_memory // 100)  # 1MB blocks
                    memory_block = bytearray(block_size)
                    self.memory_blocks.append(memory_block)
                    
                    # Check if we've reached target
                    allocated = sum(len(block) for block in self.memory_blocks)
                    if allocated >= target_memory:
                        break
                        
                except MemoryError:
                    break
        
        # Start memory stress
        thread = threading.Thread(target=memory_stress_worker, daemon=True)
        thread.start()
    
    def stop_memory_stress(self):
        """Stop memory stress."""
        self.running = False
        self.memory_blocks.clear()

class NetworkChaos:
    """Network chaos injection."""
    
    def __init__(self):
        self.blocked_ports = set()
        self.original_connections = []
    
    def inject_network_delay(self, target_host: str, delay_ms: int, duration: int):
        """Inject network delay using iptables (Linux only)."""
        try:
            # This is a simplified version - in practice, you'd use iptables
            # or network namespaces for more sophisticated network chaos
            print(f"Simulating network delay of {delay_ms}ms to {target_host}")
            time.sleep(duration)
        except Exception as e:
            logging.error(f"Error injecting network delay: {e}")
    
    def inject_packet_loss(self, target_host: str, loss_percent: float, duration: int):
        """Inject packet loss."""
        try:
            print(f"Simulating packet loss of {loss_percent}% to {target_host}")
            time.sleep(duration)
        except Exception as e:
            logging.error(f"Error injecting packet loss: {e}")

class DiskChaos:
    """Disk chaos injection."""
    
    def __init__(self):
        self.temp_files = []
        self.running = False
    
    def inject_disk_stress(self, intensity: float, duration: int):
        """Inject disk stress."""
        self.running = True
        
        def disk_stress_worker():
            end_time = time.time() + duration
            while self.running and time.time() < end_time:
                try:
                    # Create temporary files
                    with tempfile.NamedTemporaryFile(delete=False) as f:
                        # Write random data
                        data = os.urandom(1024 * 1024)  # 1MB
                        f.write(data)
                        self.temp_files.append(f.name)
                    
                    # Randomly delete some files to simulate disk churn
                    if len(self.temp_files) > 10 and random.random() < 0.3:
                        file_to_delete = random.choice(self.temp_files)
                        try:
                            os.unlink(file_to_delete)
                            self.temp_files.remove(file_to_delete)
                        except:
                            pass
                            
                except Exception as e:
                    logging.error(f"Error in disk stress: {e}")
                    break
        
        thread = threading.Thread(target=disk_stress_worker, daemon=True)
        thread.start()
    
    def stop_disk_stress(self):
        """Stop disk stress and cleanup."""
        self.running = False
        
        # Cleanup temporary files
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except:
                pass
        self.temp_files.clear()

class ProcessChaos:
    """Process chaos injection."""
    
    def __init__(self):
        self.killed_processes = []
    
    def kill_random_processes(self, count: int = 1, exclude_important: bool = True):
        """Kill random processes (be very careful with this!)."""
        try:
            # Get all processes
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'status']):
                try:
                    proc_info = proc.info
                    if exclude_important:
                        # Exclude important system processes
                        important_names = ['systemd', 'kernel', 'init', 'sshd', 'bash']
                        if any(name in proc_info['name'].lower() for name in important_names):
                            continue
                    processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Kill random processes
            if processes:
                selected = random.sample(processes, min(count, len(processes)))
                for proc_info in selected:
                    try:
                        proc = psutil.Process(proc_info['pid'])
                        proc.kill()
                        self.killed_processes.append(proc_info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
        except Exception as e:
            logging.error(f"Error killing processes: {e}")
    
    def restart_service(self, service_name: str):
        """Restart a service."""
        try:
            # This is a simplified version - in practice, you'd use systemctl
            print(f"Restarting service: {service_name}")
            # subprocess.run(['sudo', 'systemctl', 'restart', service_name])
        except Exception as e:
            logging.error(f"Error restarting service {service_name}: {e}")

class ChaosExperimentRunner:
    """Runs chaos engineering experiments."""
    
    def __init__(self):
        self.monitor = SystemMonitor()
        self.cpu_chaos = CPUChaos()
        self.memory_chaos = MemoryChaos()
        self.network_chaos = NetworkChaos()
        self.disk_chaos = DiskChaos()
        self.process_chaos = ProcessChaos()
        self.experiments: List[ChaosExperiment] = []
        self.results: List[ChaosResult] = []
    
    def create_experiment(self, name: str, description: str, target_system: str,
                         chaos_type: str, intensity: float, duration: int) -> ChaosExperiment:
        """Create a new chaos experiment."""
        experiment = ChaosExperiment(
            experiment_id=f"exp_{int(time.time())}",
            name=name,
            description=description,
            target_system=target_system,
            chaos_type=chaos_type,
            intensity=intensity,
            duration=duration,
            recovery_time=duration // 2
        )
        
        self.experiments.append(experiment)
        return experiment
    
    def run_experiment(self, experiment: ChaosExperiment) -> ChaosResult:
        """Run a chaos experiment."""
        print(f"🧪 Running Chaos Experiment: {experiment.name}")
        print(f"   Type: {experiment.chaos_type}")
        print(f"   Intensity: {experiment.intensity}")
        print(f"   Duration: {experiment.duration}s")
        print("=" * 50)
        
        experiment.status = "running"
        experiment.start_time = datetime.now()
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Get baseline metrics
        baseline_metrics = self.monitor.get_current_metrics()
        
        try:
            # Run the chaos injection
            self._inject_chaos(experiment)
            
            # Wait for experiment duration
            time.sleep(experiment.duration)
            
            # Get metrics during chaos
            chaos_metrics = self.monitor.get_current_metrics()
            
            # Stop chaos injection
            self._stop_chaos(experiment)
            
            # Wait for recovery
            time.sleep(experiment.recovery_time)
            
            # Get recovery metrics
            recovery_metrics = self.monitor.get_current_metrics()
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Calculate results
            result = self._calculate_results(experiment, baseline_metrics, 
                                          chaos_metrics, recovery_metrics)
            
            experiment.status = "completed"
            experiment.end_time = datetime.now()
            experiment.results = result.__dict__
            
            self.results.append(result)
            
            # Print results
            self._print_experiment_results(result)
            
            return result
            
        except Exception as e:
            experiment.status = "failed"
            experiment.end_time = datetime.now()
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Stop chaos injection
            self._stop_chaos(experiment)
            
            logging.error(f"Chaos experiment failed: {e}")
            
            return ChaosResult(
                experiment_id=experiment.experiment_id,
                success=False,
                system_impact={},
                recovery_time=0.0,
                metrics_before=baseline_metrics,
                metrics_during={},
                metrics_after={},
                observations=[f"Experiment failed: {str(e)}"],
                recommendations=["Fix the underlying issue before retrying"]
            )
    
    def _inject_chaos(self, experiment: ChaosExperiment):
        """Inject chaos based on experiment type."""
        chaos_type = experiment.chaos_type.lower()
        intensity = experiment.intensity
        duration = experiment.duration
        
        if chaos_type == "cpu":
            self.cpu_chaos.inject_cpu_stress(intensity, duration)
        elif chaos_type == "memory":
            self.memory_chaos.inject_memory_stress(intensity, duration)
        elif chaos_type == "network":
            self.network_chaos.inject_network_delay("localhost", 100, duration)
        elif chaos_type == "disk":
            self.disk_chaos.inject_disk_stress(intensity, duration)
        elif chaos_type == "process":
            self.process_chaos.kill_random_processes(1, True)
        else:
            raise ValueError(f"Unknown chaos type: {chaos_type}")
    
    def _stop_chaos(self, experiment: ChaosExperiment):
        """Stop chaos injection."""
        chaos_type = experiment.chaos_type.lower()
        
        if chaos_type == "cpu":
            self.cpu_chaos.stop_cpu_stress()
        elif chaos_type == "memory":
            self.memory_chaos.stop_memory_stress()
        elif chaos_type == "disk":
            self.disk_chaos.stop_disk_stress()
    
    def _calculate_results(self, experiment: ChaosExperiment, baseline: Dict[str, Any],
                          chaos: Dict[str, Any], recovery: Dict[str, Any]) -> ChaosResult:
        """Calculate experiment results."""
        # Calculate system impact
        cpu_impact = chaos.get('cpu_percent', 0) - baseline.get('cpu_percent', 0)
        memory_impact = chaos.get('memory_percent', 0) - baseline.get('memory_percent', 0)
        
        system_impact = {
            'cpu_impact_percent': cpu_impact,
            'memory_impact_percent': memory_impact,
            'baseline_cpu': baseline.get('cpu_percent', 0),
            'chaos_cpu': chaos.get('cpu_percent', 0),
            'recovery_cpu': recovery.get('cpu_percent', 0),
            'baseline_memory': baseline.get('memory_percent', 0),
            'chaos_memory': chaos.get('memory_percent', 0),
            'recovery_memory': recovery.get('memory_percent', 0)
        }
        
        # Calculate recovery time
        recovery_time = 0.0
        if experiment.start_time and experiment.end_time:
            recovery_time = (experiment.end_time - experiment.start_time).total_seconds()
        
        # Generate observations
        observations = []
        if cpu_impact > 20:
            observations.append(f"High CPU impact: {cpu_impact:.1f}% increase")
        if memory_impact > 20:
            observations.append(f"High memory impact: {memory_impact:.1f}% increase")
        
        # Check if system recovered
        cpu_recovered = abs(recovery.get('cpu_percent', 0) - baseline.get('cpu_percent', 0)) < 10
        memory_recovered = abs(recovery.get('memory_percent', 0) - baseline.get('memory_percent', 0)) < 10
        
        if cpu_recovered and memory_recovered:
            observations.append("System recovered successfully")
        else:
            observations.append("System did not fully recover")
        
        # Generate recommendations
        recommendations = []
        if not cpu_recovered:
            recommendations.append("Improve CPU resource management")
        if not memory_recovered:
            recommendations.append("Improve memory management and cleanup")
        if cpu_impact > 50:
            recommendations.append("Consider CPU throttling or load balancing")
        if memory_impact > 50:
            recommendations.append("Implement memory limits and monitoring")
        
        success = cpu_recovered and memory_recovered
        
        return ChaosResult(
            experiment_id=experiment.experiment_id,
            success=success,
            system_impact=system_impact,
            recovery_time=recovery_time,
            metrics_before=baseline,
            metrics_during=chaos,
            metrics_after=recovery,
            observations=observations,
            recommendations=recommendations
        )
    
    def _print_experiment_results(self, result: ChaosResult):
        """Print experiment results."""
        print("\n" + "=" * 60)
        print("🧪 CHAOS EXPERIMENT RESULTS")
        print("=" * 60)
        
        print(f"✅ Success: {'Yes' if result.success else 'No'}")
        print(f"⏱️  Recovery Time: {result.recovery_time:.2f}s")
        
        print(f"\n📊 System Impact:")
        impact = result.system_impact
        print(f"   CPU Impact: {impact.get('cpu_impact_percent', 0):.1f}%")
        print(f"   Memory Impact: {impact.get('memory_impact_percent', 0):.1f}%")
        
        print(f"\n📈 Metrics:")
        print(f"   CPU: {impact.get('baseline_cpu', 0):.1f}% → {impact.get('chaos_cpu', 0):.1f}% → {impact.get('recovery_cpu', 0):.1f}%")
        print(f"   Memory: {impact.get('baseline_memory', 0):.1f}% → {impact.get('chaos_memory', 0):.1f}% → {impact.get('recovery_memory', 0):.1f}%")
        
        print(f"\n👀 Observations:")
        for obs in result.observations:
            print(f"   - {obs}")
        
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   - {rec}")
        
        print("=" * 60)
    
    def run_chaos_suite(self) -> List[ChaosResult]:
        """Run a suite of chaos experiments."""
        print("💥 Starting Chaos Engineering Suite")
        print("=" * 50)
        
        # Create experiments
        experiments = [
            self.create_experiment(
                "CPU Stress Test",
                "Test system resilience under CPU stress",
                "CPU",
                "cpu",
                0.5,  # 50% intensity
                30    # 30 seconds
            ),
            self.create_experiment(
                "Memory Stress Test",
                "Test system resilience under memory pressure",
                "Memory",
                "memory",
                0.3,  # 30% intensity
                30    # 30 seconds
            ),
            self.create_experiment(
                "Disk I/O Stress Test",
                "Test system resilience under disk I/O stress",
                "Disk",
                "disk",
                0.4,  # 40% intensity
                30    # 30 seconds
            )
        ]
        
        # Run experiments
        results = []
        for experiment in experiments:
            result = self.run_experiment(experiment)
            results.append(result)
            
            # Wait between experiments
            time.sleep(10)
        
        # Generate suite report
        self._generate_suite_report(results)
        
        return results
    
    def _generate_suite_report(self, results: List[ChaosResult]):
        """Generate chaos suite report."""
        print("\n" + "=" * 60)
        print("📊 CHAOS ENGINEERING SUITE REPORT")
        print("=" * 60)
        
        total_experiments = len(results)
        successful_experiments = sum(1 for r in results if r.success)
        success_rate = (successful_experiments / total_experiments * 100) if total_experiments > 0 else 0
        
        print(f"📈 Summary:")
        print(f"   Total Experiments: {total_experiments}")
        print(f"   Successful: {successful_experiments}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        avg_recovery_time = sum(r.recovery_time for r in results) / len(results) if results else 0
        print(f"   Average Recovery Time: {avg_recovery_time:.2f}s")
        
        print(f"\n🔍 Experiment Details:")
        for i, result in enumerate(results, 1):
            status_icon = "✅" if result.success else "❌"
            print(f"   {status_icon} Experiment {i}: {result.recovery_time:.2f}s recovery")
        
        # Overall assessment
        if success_rate >= 80:
            print(f"\n🎉 System Resilience: EXCELLENT")
        elif success_rate >= 60:
            print(f"\n👍 System Resilience: GOOD")
        elif success_rate >= 40:
            print(f"\n⚠️  System Resilience: FAIR")
        else:
            print(f"\n❌ System Resilience: POOR")
        
        print("=" * 60)

# Example usage and demo
def demo_chaos_engineering():
    """Demonstrate chaos engineering capabilities."""
    print("💥 Chaos Engineering Framework Demo")
    print("=" * 40)
    
    # Create chaos runner
    chaos_runner = ChaosExperimentRunner()
    
    # Run a single experiment
    print("\n🧪 Running Single Experiment...")
    experiment = chaos_runner.create_experiment(
        "CPU Stress Test",
        "Test system under CPU stress",
        "CPU",
        "cpu",
        0.3,  # 30% intensity
        10    # 10 seconds
    )
    
    result = chaos_runner.run_experiment(experiment)
    
    # Run chaos suite
    print("\n💥 Running Chaos Suite...")
    results = chaos_runner.run_chaos_suite()
    
    return results

if __name__ == "__main__":
    # Run demo
    demo_chaos_engineering()
