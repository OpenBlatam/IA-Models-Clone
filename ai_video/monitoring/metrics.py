from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
from collections import defaultdict, deque
import logging
            import psutil
    import argparse
from typing import Any, List, Dict, Optional
"""
Advanced Metrics and Monitoring System for AI Video Workflow

This module provides comprehensive metrics tracking for:
- Extractor performance and success rates
- Video generator performance and quality metrics
- Workflow statistics and timing
- Real-time monitoring and alerting
- Performance analytics and reporting
"""


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be tracked."""
    EXTRACTION = "extraction"
    SUGGESTION = "suggestion"
    GENERATION = "generation"
    WORKFLOW = "workflow"
    SYSTEM = "system"


@dataclass
class ExtractorMetrics:
    """Metrics for web content extractors."""
    name: str
    total_attempts: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None
    domain_success: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    domain_failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def update(self, success: bool, duration: float, domain: str = "unknown"):
        """Update metrics with a new extraction attempt."""
        self.total_attempts += 1
        self.total_time += duration
        
        if success:
            self.successful_extractions += 1
            self.domain_success[domain] += 1
        else:
            self.failed_extractions += 1
            self.domain_failures[domain] += 1
        
        self.avg_time = self.total_time / self.total_attempts
        self.success_rate = self.successful_extractions / self.total_attempts
        self.last_used = datetime.now()


@dataclass
class GeneratorMetrics:
    """Metrics for video generators."""
    name: str
    total_attempts: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None
    quality_scores: List[float] = field(default_factory=list)
    avg_quality: float = 0.0
    
    def update(self, success: bool, duration: float, quality_score: Optional[float] = None):
        """Update metrics with a new generation attempt."""
        self.total_attempts += 1
        self.total_time += duration
        
        if success:
            self.successful_generations += 1
            if quality_score is not None:
                self.quality_scores.append(quality_score)
                self.avg_quality = statistics.mean(self.quality_scores)
        else:
            self.failed_generations += 1
        
        self.avg_time = self.total_time / self.total_attempts
        self.success_rate = self.successful_generations / self.total_attempts
        self.last_used = datetime.now()


@dataclass
class WorkflowMetrics:
    """Metrics for workflow execution."""
    total_workflows: int = 0
    completed_workflows: int = 0
    failed_workflows: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    success_rate: float = 0.0
    stage_timings: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    stage_avg_timings: Dict[str, float] = field(default_factory=dict)
    recent_workflows: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, success: bool, total_duration: float, stage_timings: Dict[str, float]):
        """Update metrics with a new workflow execution."""
        self.total_workflows += 1
        self.total_time += total_duration
        
        if success:
            self.completed_workflows += 1
        else:
            self.failed_workflows += 1
        
        self.avg_time = self.total_time / self.total_workflows
        self.success_rate = self.completed_workflows / self.total_workflows
        
        # Update stage timings
        for stage, duration in stage_timings.items():
            self.stage_timings[stage].append(duration)
            self.stage_avg_timings[stage] = statistics.mean(self.stage_timings[stage])
        
        # Add to recent workflows
        self.recent_workflows.append({
            'timestamp': datetime.now(),
            'success': success,
            'duration': total_duration,
            'stages': stage_timings
        })


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    uptime: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_workflows: int = 0
    queue_size: int = 0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Main metrics collection and management system."""
    
    def __init__(self, storage_path: str = ".metrics"):
        
    """__init__ function."""
self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize metrics
        self.extractors: Dict[str, ExtractorMetrics] = {}
        self.generators: Dict[str, GeneratorMetrics] = {}
        self.workflows = WorkflowMetrics()
        self.system = SystemMetrics()
        
        # Load existing metrics
        self._load_metrics()
        
        # Start background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_monitoring()
    
    def _load_metrics(self) -> Any:
        """Load metrics from persistent storage."""
        try:
            # Load extractor metrics
            extractor_file = self.storage_path / "extractors.json"
            if extractor_file.exists():
                with open(extractor_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                    for name, metrics_data in data.items():
                        self.extractors[name] = ExtractorMetrics(**metrics_data)
            
            # Load generator metrics
            generator_file = self.storage_path / "generators.json"
            if generator_file.exists():
                with open(generator_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                    for name, metrics_data in data.items():
                        self.generators[name] = GeneratorMetrics(**metrics_data)
            
            # Load workflow metrics
            workflow_file = self.storage_path / "workflows.json"
            if workflow_file.exists():
                with open(workflow_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                    self.workflows = WorkflowMetrics(**data)
            
            logger.info("Metrics loaded from storage")
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
    
    def _save_metrics(self) -> Any:
        """Save metrics to persistent storage."""
        try:
            # Save extractor metrics
            extractor_data = {name: asdict(metrics) for name, metrics in self.extractors.items()}
            with open(self.storage_path / "extractors.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(extractor_data, f, indent=2, default=str)
            
            # Save generator metrics
            generator_data = {name: asdict(metrics) for name, metrics in self.generators.items()}
            with open(self.storage_path / "generators.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(generator_data, f, indent=2, default=str)
            
            # Save workflow metrics
            with open(self.storage_path / "workflows.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(asdict(self.workflows), f, indent=2, default=str)
            
            logger.debug("Metrics saved to storage")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _start_monitoring(self) -> Any:
        """Start background monitoring tasks."""
        async def monitor():
            
    """monitor function."""
while True:
                try:
                    await self._update_system_metrics()
                    await asyncio.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(60)
        
        self._monitoring_task = asyncio.create_task(monitor())
    
    async def _update_system_metrics(self) -> Any:
        """Update system metrics."""
        try:
            
            self.system.uptime = time.time()
            self.system.memory_usage = psutil.virtual_memory().percent
            self.system.cpu_usage = psutil.cpu_percent()
            self.system.last_updated = datetime.now()
            
            # Calculate error rate from recent workflows
            if self.workflows.recent_workflows:
                recent_errors = sum(1 for w in self.workflows.recent_workflows 
                                  if not w['success'])
                self.system.error_rate = recent_errors / len(self.workflows.recent_workflows)
        except ImportError:
            logger.warning("psutil not available, system metrics disabled")
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def record_extraction(self, extractor_name: str, success: bool, duration: float, domain: str = "unknown"):
        """Record an extraction attempt."""
        if extractor_name not in self.extractors:
            self.extractors[extractor_name] = ExtractorMetrics(name=extractor_name)
        
        self.extractors[extractor_name].update(success, duration, domain)
        self._save_metrics()
    
    def record_generation(self, generator_name: str, success: bool, duration: float, quality_score: Optional[float] = None):
        """Record a video generation attempt."""
        if generator_name not in self.generators:
            self.generators[generator_name] = GeneratorMetrics(name=generator_name)
        
        self.generators[generator_name].update(success, duration, quality_score)
        self._save_metrics()
    
    def record_workflow(self, success: bool, total_duration: float, stage_timings: Dict[str, float]):
        """Record a workflow execution."""
        self.workflows.update(success, total_duration, stage_timings)
        self._save_metrics()
    
    def get_extractor_performance(self, extractor_name: str) -> Optional[ExtractorMetrics]:
        """Get performance metrics for a specific extractor."""
        return self.extractors.get(extractor_name)
    
    def get_generator_performance(self, generator_name: str) -> Optional[GeneratorMetrics]:
        """Get performance metrics for a specific generator."""
        return self.generators.get(generator_name)
    
    def get_best_extractor_for_domain(self, domain: str) -> Optional[str]:
        """Get the best performing extractor for a specific domain."""
        best_extractor = None
        best_success_rate = 0.0
        
        for name, metrics in self.extractors.items():
            domain_attempts = metrics.domain_success.get(domain, 0) + metrics.domain_failures.get(domain, 0)
            if domain_attempts >= 3:  # Minimum attempts for reliability
                success_rate = metrics.domain_success.get(domain, 0) / domain_attempts
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_extractor = name
        
        return best_extractor
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': asdict(self.system),
            'workflows': asdict(self.workflows),
            'extractors': {},
            'generators': {},
            'recommendations': []
        }
        
        # Add extractor metrics
        for name, metrics in self.extractors.items():
            report['extractors'][name] = asdict(metrics)
        
        # Add generator metrics
        for name, metrics in self.generators.items():
            report['generators'][name] = asdict(metrics)
        
        # Generate recommendations
        recommendations = []
        
        # Check for low-performing extractors
        for name, metrics in self.extractors.items():
            if metrics.success_rate < 0.5 and metrics.total_attempts > 10:
                recommendations.append(f"Extractor '{name}' has low success rate ({metrics.success_rate:.1%})")
        
        # Check for slow generators
        for name, metrics in self.generators.items():
            if metrics.avg_time > 60 and metrics.total_attempts > 5:
                recommendations.append(f"Generator '{name}' is slow (avg: {metrics.avg_time:.1f}s)")
        
        # Check system health
        if self.system.error_rate > 0.2:
            recommendations.append(f"High error rate detected ({self.system.error_rate:.1%})")
        
        if self.system.memory_usage > 80:
            recommendations.append(f"High memory usage ({self.system.memory_usage:.1f}%)")
        
        report['recommendations'] = recommendations
        
        return report
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """Get real-time statistics for monitoring."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'active_workflows': self.system.active_workflows,
            'queue_size': self.system.queue_size,
            'system_health': {
                'memory_usage': self.system.memory_usage,
                'cpu_usage': self.system.cpu_usage,
                'error_rate': self.system.error_rate
            },
            'recent_performance': {
                'workflows_last_hour': len([w for w in self.workflows.recent_workflows 
                                          if w['timestamp'] > datetime.now() - timedelta(hours=1)]),
                'avg_workflow_time': self.workflows.avg_time,
                'success_rate': self.workflows.success_rate
            }
        }
        
        return stats
    
    async def cleanup_old_metrics(self, days: int = 30):
        """Clean up old metrics data."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Clean old workflow records
        self.workflows.recent_workflows = deque(
            [w for w in self.workflows.recent_workflows 
             if w['timestamp'] > cutoff_date],
            maxlen=100
        )
        
        # Clean old quality scores (keep last 100)
        for generator in self.generators.values():
            if len(generator.quality_scores) > 100:
                generator.quality_scores = generator.quality_scores[-100:]
                generator.avg_quality = statistics.mean(generator.quality_scores)
        
        self._save_metrics()
        logger.info(f"Cleaned up metrics older than {days} days")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_extraction_metrics(extractor_name: str, success: bool, duration: float, domain: str = "unknown"):
    """Record extraction metrics."""
    collector = get_metrics_collector()
    collector.record_extraction(extractor_name, success, duration, domain)


def record_generation_metrics(generator_name: str, success: bool, duration: float, quality_score: Optional[float] = None):
    """Record generation metrics."""
    collector = get_metrics_collector()
    collector.record_generation(generator_name, success, duration, quality_score)


def record_workflow_metrics(success: bool, total_duration: float, stage_timings: Dict[str, float]):
    """Record workflow metrics."""
    collector = get_metrics_collector()
    collector.record_workflow(success, total_duration, stage_timings)


# CLI for metrics management
if __name__ == "__main__":
    
    async def main():
        
    """main function."""
parser = argparse.ArgumentParser(description="AI Video Metrics CLI")
        parser.add_argument('command', choices=['report', 'stats', 'cleanup'], help='Command to execute')
        parser.add_argument('--days', type=int, default=30, help='Days for cleanup')
        
        args = parser.parse_args()
        
        collector = get_metrics_collector()
        
        if args.command == 'report':
            report = collector.get_performance_report()
            print(json.dumps(report, indent=2, default=str))
        
        elif args.command == 'stats':
            stats = collector.get_realtime_stats()
            print(json.dumps(stats, indent=2, default=str))
        
        elif args.command == 'cleanup':
            await collector.cleanup_old_metrics(args.days)
            print(f"✅ Cleaned up metrics older than {args.days} days")
    
    asyncio.run(main()) 