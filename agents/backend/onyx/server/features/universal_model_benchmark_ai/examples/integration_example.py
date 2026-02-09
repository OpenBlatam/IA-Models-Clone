"""
Integration Example - Complete integration example.

Shows how to use all major features together.
"""

import asyncio
from core.results import ResultsManager
from core.experiments import ExperimentManager, ExperimentConfig
from core.model_registry import ModelRegistry, ModelMetadata, ModelStatus
from core.cost_tracking import CostTracker, ResourceType
from core.queue import TaskQueue
from core.scheduler import TaskScheduler, ScheduleType
from core.metrics import metrics_collector
from core.rate_limiter import RateLimiter, RateLimit


def benchmark_handler(payload: dict):
    """Handle benchmark task."""
    print(f"Running benchmark: {payload['benchmark']} on {payload['model']}")
    # Simulate benchmark execution
    return {"accuracy": 0.85, "throughput": 100.0}


def main():
    """Main integration example."""
    print("🚀 Universal Model Benchmark AI - Integration Example\n")
    
    # 1. Initialize managers
    print("1. Initializing managers...")
    results_manager = ResultsManager()
    experiment_manager = ExperimentManager()
    model_registry = ModelRegistry()
    cost_tracker = CostTracker()
    task_queue = TaskQueue()
    scheduler = TaskScheduler(task_queue=task_queue)
    
    # 2. Register model
    print("\n2. Registering model...")
    metadata = ModelMetadata(
        name="llama2-7b",
        version="1.0.0",
        architecture="llama",
        parameters=7_000_000_000,
    )
    model_version = model_registry.register_model(metadata, "/path/to/model")
    print(f"   ✓ Registered: {model_version.model_name} v{model_version.version}")
    
    # 3. Create experiment
    print("\n3. Creating experiment...")
    exp_config = ExperimentConfig(
        name="llama2-7b-mmlu",
        model_name="llama2-7b",
        benchmark_name="mmlu",
        hyperparameters={"temperature": 0.7},
    )
    experiment = experiment_manager.create_experiment(exp_config)
    print(f"   ✓ Created experiment: {experiment.id}")
    
    # 4. Setup task queue
    print("\n4. Setting up task queue...")
    task_queue.register_handler("benchmark", benchmark_handler)
    task_queue.start_workers(num_workers=2)
    print("   ✓ Task queue started with 2 workers")
    
    # 5. Enqueue tasks
    print("\n5. Enqueuing tasks...")
    task1 = task_queue.enqueue(
        task_type="benchmark",
        payload={"model": "llama2-7b", "benchmark": "mmlu"},
        priority=5,
    )
    task2 = task_queue.enqueue(
        task_type="benchmark",
        payload={"model": "llama2-7b", "benchmark": "hellaswag"},
        priority=3,  # Higher priority
    )
    print(f"   ✓ Enqueued {task_queue.get_queue_size()} tasks")
    
    # 6. Schedule recurring task
    print("\n6. Scheduling recurring task...")
    scheduled = scheduler.schedule_task(
        name="Daily benchmark",
        task_type="benchmark",
        payload={"model": "llama2-7b", "benchmark": "mmlu"},
        schedule_type=ScheduleType.CRON,
        schedule="0 9 * * *",  # 9 AM daily
    )
    scheduler.start()
    print(f"   ✓ Scheduled: {scheduled.name}")
    
    # 7. Track costs
    print("\n7. Tracking costs...")
    cost_tracker.set_budget(1000.0)
    cost_tracker.record_usage(
        "llama2-7b",
        "mmlu",
        ResourceType.GPU,
        amount=1,
        duration_seconds=3600,
    )
    status = cost_tracker.get_budget_status()
    print(f"   ✓ Cost: ${status['spent']:.2f} / ${status['budget']:.2f}")
    
    # 8. Record metrics
    print("\n8. Recording metrics...")
    metrics_collector.record_benchmark(
        model="llama2-7b",
        benchmark="mmlu",
        accuracy=0.85,
        throughput=100.0,
        duration=120.0,
    )
    print("   ✓ Metrics recorded")
    
    # 9. Setup rate limiting
    print("\n9. Setting up rate limiting...")
    rate_limiter = RateLimiter()
    allowed, remaining = rate_limiter.check_rate_limit(
        key="user_123",
        limit=RateLimit(requests=100, window_seconds=60),
    )
    print(f"   ✓ Rate limit check: allowed={allowed}, remaining={remaining}")
    
    # 10. Get statistics
    print("\n10. System statistics:")
    stats = results_manager.get_statistics()
    print(f"    Total results: {stats.get('total_results', 0)}")
    print(f"    Unique models: {stats.get('unique_models', 0)}")
    
    queue_stats = task_queue.get_stats()
    print(f"    Queue size: {queue_stats['queue_size']}")
    print(f"    Total tasks: {queue_stats['total_tasks']}")
    
    print("\n✅ Integration example complete!")
    
    # Cleanup
    task_queue.stop_workers()
    scheduler.stop()


if __name__ == "__main__":
    main()












