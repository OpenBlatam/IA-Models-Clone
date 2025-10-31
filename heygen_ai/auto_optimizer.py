#!/usr/bin/env python3
"""
HeyGen AI - Automatic Optimizer

This module provides automatic optimization capabilities for the HeyGen AI system,
including performance tuning, resource management, and intelligent configuration
adjustments.
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationRule:
    """Optimization rule definition."""
    name: str
    condition: str  # Python expression string
    action: str     # Action to take
    priority: int = 1
    enabled: bool = True
    cooldown: float = 300.0  # seconds
    last_executed: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    rule_name: str
    timestamp: float
    success: bool
    message: str
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    improvement: float
    execution_time: float

class AutoOptimizer:
    """Automatic optimization system for HeyGen AI."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.optimization_rules: List[OptimizationRule] = []
        self.optimization_history: List[OptimizationResult] = []
        self.optimization_active = False
        self.optimizer_thread: Optional[threading.Thread] = None
        self.optimization_interval = self.config.get('optimization_interval', 60.0)
        
        # Performance tracking
        self.performance_baseline: Dict[str, Any] = {}
        self.optimization_stats: Dict[str, Any] = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'total_improvement': 0.0,
            'last_optimization': None
        }
        
        # Initialize optimizer
        self._setup_optimizer()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default optimization configuration."""
        return {
            'optimization_interval': 60.0,  # seconds
            'max_optimizations_per_hour': 10,
            'performance_threshold': 0.05,  # 5% improvement threshold
            'auto_apply': True,
            'backup_before_optimization': True,
            'optimization_rules_file': 'optimization_rules.json',
            'monitoring': {
                'enabled': True,
                'metrics_tracking': True,
                'performance_analysis': True
            }
        }
    
    def _setup_optimizer(self):
        """Setup optimization infrastructure."""
        try:
            # Load default optimization rules
            self._load_default_rules()
            
            # Load custom rules if available
            self._load_custom_rules()
            
            # Initialize performance baseline
            self._initialize_performance_baseline()
            
            logger.info(f"Auto-optimizer initialized with {len(self.optimization_rules)} rules")
            
        except Exception as e:
            logger.error(f"Failed to setup optimizer: {e}")
    
    def _load_default_rules(self):
        """Load default optimization rules."""
        default_rules = [
            OptimizationRule(
                name="Memory Cleanup",
                condition="metrics.get('memory_percent', 0) > 80",
                action="cleanup_memory",
                priority=1,
                cooldown=300.0,
                parameters={'aggressive': False}
            ),
            OptimizationRule(
                name="CPU Throttling",
                condition="metrics.get('cpu_percent', 0) > 90",
                action="throttle_cpu",
                priority=2,
                cooldown=180.0,
                parameters={'throttle_level': 'medium'}
            ),
            OptimizationRule(
                name="Disk Cleanup",
                condition="metrics.get('disk_usage_percent', 0) > 85",
                action="cleanup_disk",
                priority=1,
                cooldown=600.0,
                parameters={'cleanup_temp': True, 'cleanup_logs': True}
            ),
            OptimizationRule(
                name="Process Optimization",
                condition="metrics.get('active_processes', 0) > 100",
                action="optimize_processes",
                priority=3,
                cooldown=120.0,
                parameters={'kill_zombie': True, 'optimize_priority': True}
            ),
            OptimizationRule(
                name="GPU Memory Optimization",
                condition="metrics.get('gpu_memory_percent', 0) > 90",
                action="optimize_gpu_memory",
                priority=2,
                cooldown=240.0,
                parameters={'clear_cache': True, 'defragment': False}
            )
        ]
        
        self.optimization_rules.extend(default_rules)
    
    def _load_custom_rules(self):
        """Load custom optimization rules from file."""
        rules_file = Path(self.config.get('optimization_rules_file', 'optimization_rules.json'))
        
        if rules_file.exists():
            try:
                with open(rules_file, 'r') as f:
                    custom_rules_data = json.load(f)
                
                for rule_data in custom_rules_data:
                    rule = OptimizationRule(**rule_data)
                    self.optimization_rules.append(rule)
                
                logger.info(f"Loaded {len(custom_rules_data)} custom optimization rules")
                
            except Exception as e:
                logger.error(f"Failed to load custom rules: {e}")
    
    def _initialize_performance_baseline(self):
        """Initialize performance baseline metrics."""
        try:
            self.performance_baseline = self._capture_current_metrics()
            logger.info("Performance baseline established")
        except Exception as e:
            logger.error(f"Failed to establish performance baseline: {e}")
    
    def _capture_current_metrics(self) -> Dict[str, Any]:
        """Capture current system metrics."""
        metrics = {}
        
        try:
            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_used_gb'] = memory.used / (1024**3)
            metrics['memory_available_gb'] = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_usage_percent'] = (disk.used / disk.total) * 100
            
            # Process metrics
            metrics['active_processes'] = len(psutil.pids())
            metrics['active_threads'] = threading.active_count()
            
            # GPU metrics (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_stats()
                    allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                    reserved = gpu_memory.get('reserved_bytes.all.current', 0)
                    
                    if reserved > 0:
                        metrics['gpu_memory_percent'] = (allocated / reserved) * 100
                    else:
                        metrics['gpu_memory_percent'] = 0.0
            except ImportError:
                pass
            
        except Exception as e:
            logger.error(f"Error capturing metrics: {e}")
        
        return metrics
    
    def start_optimization(self):
        """Start automatic optimization."""
        if self.optimization_active:
            logger.warning("Optimization is already active")
            return
        
        self.optimization_active = True
        self.optimizer_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimizer_thread.start()
        logger.info("Automatic optimization started")
    
    def stop_optimization(self):
        """Stop automatic optimization."""
        self.optimization_active = False
        if self.optimizer_thread:
            self.optimizer_thread.join(timeout=5.0)
        logger.info("Automatic optimization stopped")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.optimization_active:
            try:
                # Check if optimization is needed
                if self._should_optimize():
                    # Execute optimization
                    self._execute_optimization_cycle()
                
                # Wait for next optimization cycle
                time.sleep(self.optimization_interval)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(self.optimization_interval)
    
    def _should_optimize(self) -> bool:
        """Check if optimization should be performed."""
        # Check rate limiting
        if self.optimization_stats['total_optimizations'] > 0:
            last_opt = self.optimization_stats['last_optimization']
            if last_opt and time.time() - last_opt < 3600:  # 1 hour
                if self.optimization_stats['total_optimizations'] >= self.config.get('max_optimizations_per_hour', 10):
                    return False
        
        # Check if any rules are triggered
        current_metrics = self._capture_current_metrics()
        for rule in self.optimization_rules:
            if rule.enabled and self._is_rule_triggered(rule, current_metrics):
                return True
        
        return False
    
    def _is_rule_triggered(self, rule: OptimizationRule, metrics: Dict[str, Any]) -> bool:
        """Check if a rule is triggered by current metrics."""
        # Check cooldown
        if rule.last_executed and time.time() - rule.last_executed < rule.cooldown:
            return False
        
        # Evaluate condition
        try:
            # Create a safe evaluation context
            context = {
                'metrics': metrics,
                'time': time.time(),
                'datetime': datetime.now()
            }
            
            # Simple condition evaluation (in production, use ast.literal_eval for safety)
            condition_met = eval(rule.condition, {"__builtins__": {}}, context)
            return bool(condition_met)
            
        except Exception as e:
            logger.error(f"Error evaluating rule condition '{rule.condition}': {e}")
            return False
    
    def _execute_optimization_cycle(self):
        """Execute one optimization cycle."""
        logger.info("Starting optimization cycle")
        
        # Capture metrics before optimization
        metrics_before = self._capture_current_metrics()
        
        # Find applicable rules
        applicable_rules = self._find_applicable_rules(metrics_before)
        
        if not applicable_rules:
            logger.info("No applicable optimization rules found")
            return
        
        # Sort rules by priority
        applicable_rules.sort(key=lambda r: r.priority)
        
        # Execute optimizations
        for rule in applicable_rules:
            try:
                result = self._execute_optimization_rule(rule, metrics_before)
                if result:
                    self.optimization_history.append(result)
                    self._update_optimization_stats(result)
                    
                    # Update rule execution time
                    rule.last_executed = time.time()
                    
                    # Break if significant improvement achieved
                    if result.improvement > self.config.get('performance_threshold', 0.05):
                        logger.info(f"Significant improvement achieved: {result.improvement:.2%}")
                        break
                        
            except Exception as e:
                logger.error(f"Error executing optimization rule '{rule.name}': {e}")
    
    def _find_applicable_rules(self, metrics: Dict[str, Any]) -> List[OptimizationRule]:
        """Find rules that are currently applicable."""
        applicable_rules = []
        
        for rule in self.optimization_rules:
            if rule.enabled and self._is_rule_triggered(rule, metrics):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _execute_optimization_rule(self, rule: OptimizationRule, metrics_before: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Execute a specific optimization rule."""
        start_time = time.time()
        
        try:
            logger.info(f"Executing optimization rule: {rule.name}")
            
            # Execute the optimization action
            success, message = self._execute_action(rule.action, rule.parameters)
            
            # Capture metrics after optimization
            time.sleep(1)  # Wait for system to stabilize
            metrics_after = self._capture_current_metrics()
            
            # Calculate improvement
            improvement = self._calculate_improvement(metrics_before, metrics_after)
            
            # Create result
            result = OptimizationResult(
                rule_name=rule.name,
                timestamp=time.time(),
                success=success,
                message=message,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                improvement=improvement,
                execution_time=time.time() - start_time
            )
            
            if success:
                logger.info(f"Optimization '{rule.name}' completed successfully. Improvement: {improvement:.2%}")
            else:
                logger.warning(f"Optimization '{rule.name}' failed: {message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing optimization rule '{rule.name}': {e}")
            return None
    
    def _execute_action(self, action: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute an optimization action."""
        try:
            if action == "cleanup_memory":
                return self._cleanup_memory(parameters)
            elif action == "throttle_cpu":
                return self._throttle_cpu(parameters)
            elif action == "cleanup_disk":
                return self._cleanup_disk(parameters)
            elif action == "optimize_processes":
                return self._optimize_processes(parameters)
            elif action == "optimize_gpu_memory":
                return self._optimize_gpu_memory(parameters)
            else:
                return False, f"Unknown action: {action}"
                
        except Exception as e:
            return False, f"Action execution failed: {e}"
    
    def _cleanup_memory(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Clean up system memory."""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear Python cache if aggressive
            if parameters.get('aggressive', False):
                import sys
                for module in list(sys.modules.keys()):
                    if module.startswith('__pycache__'):
                        del sys.modules[module]
            
            return True, "Memory cleanup completed"
            
        except Exception as e:
            return False, f"Memory cleanup failed: {e}"
    
    def _throttle_cpu(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Throttle CPU usage."""
        try:
            # This is a simplified implementation
            # In production, you might use more sophisticated CPU throttling
            throttle_level = parameters.get('throttle_level', 'medium')
            
            if throttle_level == 'high':
                time.sleep(0.1)  # High throttling
            elif throttle_level == 'medium':
                time.sleep(0.05)  # Medium throttling
            else:
                time.sleep(0.01)  # Low throttling
            
            return True, f"CPU throttling applied (level: {throttle_level})"
            
        except Exception as e:
            return False, f"CPU throttling failed: {e}"
    
    def _cleanup_disk(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Clean up disk space."""
        try:
            cleanup_temp = parameters.get('cleanup_temp', True)
            cleanup_logs = parameters.get('cleanup_logs', True)
            
            cleaned_files = 0
            
            if cleanup_temp:
                # Clean temporary files
                temp_dirs = ['/tmp', '/var/tmp']
                for temp_dir in temp_dirs:
                    temp_path = Path(temp_dir)
                    if temp_path.exists():
                        for temp_file in temp_path.glob('*'):
                            if temp_file.is_file():
                                try:
                                    temp_file.unlink()
                                    cleaned_files += 1
                                except:
                                    pass
            
            if cleanup_logs:
                # Clean old log files
                log_dirs = ['/var/log', './logs']
                for log_dir in log_dirs:
                    log_path = Path(log_dir)
                    if log_path.exists():
                        for log_file in log_path.glob('*.log.*'):
                            if log_file.is_file():
                                try:
                                    log_file.unlink()
                                    cleaned_files += 1
                                except:
                                    pass
            
            return True, f"Disk cleanup completed. Cleaned {cleaned_files} files"
            
        except Exception as e:
            return False, f"Disk cleanup failed: {e}"
    
    def _optimize_processes(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Optimize system processes."""
        try:
            optimized = 0
            
            kill_zombie = parameters.get('kill_zombie', True)
            optimize_priority = parameters.get('optimize_priority', True)
            
            if kill_zombie:
                # Find and kill zombie processes
                for proc in psutil.process_iter(['pid', 'name', 'status']):
                    try:
                        if proc.info['status'] == 'zombie':
                            proc.kill()
                            optimized += 1
                    except:
                        pass
            
            if optimize_priority:
                # Optimize process priorities
                for proc in psutil.process_iter(['pid', 'name', 'nice']):
                    try:
                        if proc.info['nice'] < 0:  # High priority processes
                            proc.nice(0)  # Set to normal priority
                            optimized += 1
                    except:
                        pass
            
            return True, f"Process optimization completed. Optimized {optimized} processes"
            
        except Exception as e:
            return False, f"Process optimization failed: {e}"
    
    def _optimize_gpu_memory(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Optimize GPU memory usage."""
        try:
            clear_cache = parameters.get('clear_cache', True)
            defragment = parameters.get('defragment', False)
            
            if clear_cache:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if defragment:
                # GPU memory defragmentation (simplified)
                import torch
                if torch.cuda.is_available():
                    # This is a placeholder - actual defragmentation would be more complex
                    pass
            
            return True, "GPU memory optimization completed"
            
        except Exception as e:
            return False, f"GPU memory optimization failed: {e}"
    
    def _calculate_improvement(self, metrics_before: Dict[str, Any], metrics_after: Dict[str, Any]) -> float:
        """Calculate performance improvement percentage."""
        try:
            # Calculate improvement based on key metrics
            improvements = []
            
            # CPU improvement (lower is better)
            if 'cpu_percent' in metrics_before and 'cpu_percent' in metrics_after:
                cpu_improvement = (metrics_before['cpu_percent'] - metrics_after['cpu_percent']) / max(metrics_before['cpu_percent'], 1)
                improvements.append(max(0, cpu_improvement))
            
            # Memory improvement (lower is better)
            if 'memory_percent' in metrics_before and 'memory_percent' in metrics_after:
                mem_improvement = (metrics_before['memory_percent'] - metrics_after['memory_percent']) / max(metrics_before['memory_percent'], 1)
                improvements.append(max(0, mem_improvement))
            
            # Disk improvement (lower is better)
            if 'disk_usage_percent' in metrics_before and 'disk_usage_percent' in metrics_after:
                disk_improvement = (metrics_before['disk_usage_percent'] - metrics_after['disk_usage_percent']) / max(metrics_before['disk_usage_percent'], 1)
                improvements.append(max(0, disk_improvement))
            
            # Calculate average improvement
            if improvements:
                return sum(improvements) / len(improvements)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating improvement: {e}")
            return 0.0
    
    def _update_optimization_stats(self, result: OptimizationResult):
        """Update optimization statistics."""
        self.optimization_stats['total_optimizations'] += 1
        
        if result.success:
            self.optimization_stats['successful_optimizations'] += 1
            self.optimization_stats['total_improvement'] += result.improvement
        
        self.optimization_stats['last_optimization'] = result.timestamp
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            "active": self.optimization_active,
            "rules_count": len(self.optimization_rules),
            "enabled_rules": len([r for r in self.optimization_rules if r.enabled]),
            "stats": self.optimization_stats.copy(),
            "recent_optimizations": [
                {
                    "rule_name": r.rule_name,
                    "timestamp": datetime.fromtimestamp(r.timestamp).isoformat(),
                    "success": r.success,
                    "improvement": f"{r.improvement:.2%}"
                }
                for r in self.optimization_history[-10:]  # Last 10 optimizations
            ]
        }
    
    def add_optimization_rule(self, rule: OptimizationRule):
        """Add a new optimization rule."""
        self.optimization_rules.append(rule)
        logger.info(f"Added optimization rule: {rule.name}")
    
    def enable_optimization_rule(self, rule_name: str):
        """Enable an optimization rule."""
        for rule in self.optimization_rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"Enabled optimization rule: {rule_name}")
                return
        logger.warning(f"Optimization rule not found: {rule_name}")
    
    def disable_optimization_rule(self, rule_name: str):
        """Disable an optimization rule."""
        for rule in self.optimization_rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Disabled optimization rule: {rule_name}")
                return
        logger.warning(f"Optimization rule not found: {rule_name}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current system state."""
        recommendations = []
        current_metrics = self._capture_current_metrics()
        
        # Memory recommendations
        if current_metrics.get('memory_percent', 0) > 80:
            recommendations.append("High memory usage detected. Consider memory cleanup.")
        
        # CPU recommendations
        if current_metrics.get('cpu_percent', 0) > 90:
            recommendations.append("High CPU usage detected. Consider process optimization.")
        
        # Disk recommendations
        if current_metrics.get('disk_usage_percent', 0) > 85:
            recommendations.append("High disk usage detected. Consider disk cleanup.")
        
        # Process recommendations
        if current_metrics.get('active_processes', 0) > 100:
            recommendations.append("High process count detected. Consider process optimization.")
        
        # GPU recommendations
        if current_metrics.get('gpu_memory_percent', 0) > 90:
            recommendations.append("High GPU memory usage detected. Consider GPU optimization.")
        
        if not recommendations:
            recommendations.append("System performance is within optimal ranges.")
        
        return recommendations

def main():
    """Demo function for the auto-optimizer."""
    print("🚀 HeyGen AI Auto-Optimizer Demo")
    print("=" * 50)
    
    # Create optimizer instance
    optimizer = AutoOptimizer()
    
    try:
        # Show initial status
        print("\n📊 Initial Optimization Status:")
        status = optimizer.get_optimization_status()
        print(json.dumps(status, indent=2, default=str))
        
        # Start optimization
        print("\n🚀 Starting automatic optimization...")
        optimizer.start_optimization()
        
        # Let it run for a bit
        print("Running optimization for 60 seconds...")
        time.sleep(60)
        
        # Show final status
        print("\n📊 Final Optimization Status:")
        status = optimizer.get_optimization_status()
        print(json.dumps(status, indent=2, default=str))
        
        # Get recommendations
        print("\n💡 Optimization Recommendations:")
        recommendations = optimizer.get_optimization_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Stop optimization
        optimizer.stop_optimization()
        print("\n✅ Auto-optimizer demo completed!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
        optimizer.stop_optimization()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        optimizer.stop_optimization()

if __name__ == "__main__":
    main()
