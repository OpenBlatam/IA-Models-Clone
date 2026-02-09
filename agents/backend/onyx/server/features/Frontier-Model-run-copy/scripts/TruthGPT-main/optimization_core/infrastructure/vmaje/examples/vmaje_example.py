"""
VMAJE System Example
====================

Comprehensive example demonstrating the VMAJE (Virtual Machine, Application, Job, and Environment)
management system for TruthGPT optimization framework.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Import VMAJE components
from ..core.vm_manager import VMManager, VMConfig, VMType, CloudProvider
from ..core.job_scheduler import JobScheduler, JobConfig, JobType, JobPriority
from ..core.environment_manager import EnvironmentManager, EnvironmentConfig, EnvironmentType
from ..core.resource_optimizer import ResourceOptimizer
from ..core.cost_manager import CostManager
from ..core.monitoring import VMAJEMonitor


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vmaje_example.log')
        ]
    )


async def demonstrate_vm_management():
    """Demonstrate VM management capabilities"""
    print("\n🖥️  VM Management Demo")
    print("=" * 50)
    
    # Initialize VM Manager
    vm_config = {
        'azure': {
            'enabled': True,
            'subscription_id': 'your-subscription-id'
        },
        'aws': {
            'enabled': True,
            'region': 'us-east-1'
        },
        'gcp': {
            'enabled': True,
            'project_id': 'your-project-id'
        }
    }
    
    vm_manager = VMManager(vm_config)
    
    # Create VM configurations
    cpu_vm_config = VMConfig(
        name="truthgpt-cpu-vm",
        provider=CloudProvider.AZURE,
        vm_type=VMType.CPU_OPTIMIZED,
        size="Standard_D4s_v3",
        region="eastus",
        image="Ubuntu2204",
        disk_size=100,
        tags={'purpose': 'cpu-optimization', 'team': 'ml-engineering'}
    )
    
    gpu_vm_config = VMConfig(
        name="truthgpt-gpu-vm",
        provider=CloudProvider.AZURE,
        vm_type=VMType.GPU_OPTIMIZED,
        size="Standard_NC6s_v3",
        region="eastus",
        image="Ubuntu2204",
        disk_size=200,
        gpu_count=1,
        gpu_type="Tesla V100",
        tags={'purpose': 'gpu-optimization', 'team': 'ml-engineering'}
    )
    
    print("Creating CPU-optimized VM...")
    cpu_vm = await vm_manager.create_vm(cpu_vm_config)
    print(f"✅ CPU VM created: {cpu_vm.name} (ID: {cpu_vm.id})")
    
    print("Creating GPU-optimized VM...")
    gpu_vm = await vm_manager.create_vm(gpu_vm_config)
    print(f"✅ GPU VM created: {gpu_vm.name} (ID: {gpu_vm.id})")
    
    # Start VMs
    print("\nStarting VMs...")
    await vm_manager.start_vm(cpu_vm.id)
    await vm_manager.start_vm(gpu_vm.id)
    print("✅ VMs started successfully")
    
    # Get VM metrics
    print("\nVM Metrics:")
    cpu_metrics = vm_manager.get_vm_metrics(cpu_vm.id)
    gpu_metrics = vm_manager.get_vm_metrics(gpu_vm.id)
    
    print(f"CPU VM: {cpu_metrics}")
    print(f"GPU VM: {gpu_metrics}")
    
    # Get VM summary
    summary = vm_manager.get_vm_summary()
    print(f"\nVM Summary: {summary}")
    
    return vm_manager, [cpu_vm.id, gpu_vm.id]


async def demonstrate_job_scheduling(vm_manager):
    """Demonstrate job scheduling capabilities"""
    print("\n📋 Job Scheduling Demo")
    print("=" * 50)
    
    # Initialize Job Scheduler
    job_scheduler = JobScheduler(vm_manager)
    
    # Define job callbacks
    async def training_job_callback(job):
        """Training job callback"""
        print(f"🚀 Executing training job: {job.name}")
        await asyncio.sleep(2)  # Simulate training
        return {"model_accuracy": 0.95, "training_time": 120}
    
    async def inference_job_callback(job):
        """Inference job callback"""
        print(f"🔮 Executing inference job: {job.name}")
        await asyncio.sleep(1)  # Simulate inference
        return {"predictions": 1000, "inference_time": 5}
    
    async def optimization_job_callback(job):
        """Optimization job callback"""
        print(f"⚡ Executing optimization job: {job.name}")
        await asyncio.sleep(3)  # Simulate optimization
        return {"optimization_score": 0.98, "optimization_time": 180}
    
    # Register job callbacks
    job_scheduler.register_job_callback(JobType.TRAINING, training_job_callback)
    job_scheduler.register_job_callback(JobType.INFERENCE, inference_job_callback)
    job_scheduler.register_job_callback(JobType.OPTIMIZATION, optimization_job_callback)
    
    # Create job configurations
    training_job_config = JobConfig(
        name="model_training_job",
        job_type=JobType.TRAINING,
        priority=JobPriority.HIGH,
        timeout=300,
        gpu_required=True,
        gpu_count=1,
        memory_required=8000,  # 8GB
        cpu_required=4,
        tags={'model': 'transformer', 'dataset': 'large'}
    )
    
    inference_job_config = JobConfig(
        name="batch_inference_job",
        job_type=JobType.INFERENCE,
        priority=JobPriority.NORMAL,
        timeout=120,
        gpu_required=False,
        memory_required=4000,  # 4GB
        cpu_required=2,
        tags={'batch_size': '1000', 'model': 'transformer'}
    )
    
    optimization_job_config = JobConfig(
        name="hyperparameter_optimization",
        job_type=JobType.OPTIMIZATION,
        priority=JobPriority.URGENT,
        timeout=600,
        gpu_required=True,
        gpu_count=1,
        memory_required=16000,  # 16GB
        cpu_required=8,
        tags={'optimization_type': 'hyperparameter', 'trials': '100'}
    )
    
    # Submit jobs
    print("Submitting jobs...")
    training_job_id = job_scheduler.submit_job(training_job_config)
    inference_job_id = job_scheduler.submit_job(inference_job_config)
    optimization_job_id = job_scheduler.submit_job(optimization_job_config)
    
    print(f"✅ Training job submitted: {training_job_id}")
    print(f"✅ Inference job submitted: {inference_job_id}")
    print(f"✅ Optimization job submitted: {optimization_job_id}")
    
    # Monitor job progress
    print("\nMonitoring job progress...")
    for i in range(10):
        await asyncio.sleep(2)
        
        # Get job status
        training_job = job_scheduler.get_job(training_job_id)
        inference_job = job_scheduler.get_job(inference_job_id)
        optimization_job = job_scheduler.get_job(optimization_job_id)
        
        print(f"Training job status: {training_job.status.value if training_job else 'Unknown'}")
        print(f"Inference job status: {inference_job.status.value if inference_job else 'Unknown'}")
        print(f"Optimization job status: {optimization_job.status.value if optimization_job else 'Unknown'}")
        
        # Check if all jobs completed
        if (training_job and training_job.status.value in ['completed', 'failed']) and \
           (inference_job and inference_job.status.value in ['completed', 'failed']) and \
           (optimization_job and optimization_job.status.value in ['completed', 'failed']):
            break
    
    # Get job results
    print("\nJob Results:")
    for job_id in [training_job_id, inference_job_id, optimization_job_id]:
        job = job_scheduler.get_job(job_id)
        if job:
            metrics = job_scheduler.get_job_metrics(job_id)
            print(f"Job {job.name}: {metrics}")
    
    # Get scheduler summary
    summary = job_scheduler.get_scheduler_summary()
    print(f"\nScheduler Summary: {summary}")
    
    return job_scheduler


async def demonstrate_environment_management(vm_manager):
    """Demonstrate environment management capabilities"""
    print("\n🌍 Environment Management Demo")
    print("=" * 50)
    
    # Initialize Environment Manager
    env_manager = EnvironmentManager(vm_manager)
    
    # Create environment configurations
    dev_env_config = EnvironmentConfig(
        name="truthgpt-dev",
        env_type=EnvironmentType.DEVELOPMENT,
        region="eastus",
        vm_count=1,
        vm_size="Standard_D2s_v3",
        gpu_enabled=False,
        storage_size=50,
        tags={'environment': 'development', 'team': 'ml-engineering'}
    )
    
    staging_env_config = EnvironmentConfig(
        name="truthgpt-staging",
        env_type=EnvironmentType.STAGING,
        region="eastus",
        vm_count=2,
        vm_size="Standard_D4s_v3",
        gpu_enabled=True,
        gpu_count=1,
        storage_size=100,
        tags={'environment': 'staging', 'team': 'ml-engineering'}
    )
    
    # Create environments
    print("Creating development environment...")
    dev_env = await env_manager.create_environment(dev_env_config)
    print(f"✅ Development environment created: {dev_env.config.name}")
    
    print("Creating staging environment...")
    staging_env = await env_manager.create_environment(staging_env_config)
    print(f"✅ Staging environment created: {staging_env.config.name}")
    
    # Start environments
    print("\nStarting environments...")
    await env_manager.start_environment(dev_env.id)
    await env_manager.start_environment(staging_env.id)
    print("✅ Environments started successfully")
    
    # Get environment metrics
    print("\nEnvironment Metrics:")
    dev_metrics = env_manager.get_environment_metrics(dev_env.id)
    staging_metrics = env_manager.get_environment_metrics(staging_env.id)
    
    print(f"Development Environment: {dev_metrics}")
    print(f"Staging Environment: {staging_metrics}")
    
    # Get environment summary
    summary = env_manager.get_environment_summary()
    print(f"\nEnvironment Summary: {summary}")
    
    return env_manager, [dev_env.id, staging_env.id]


async def demonstrate_resource_optimization(vm_manager, job_scheduler, env_manager):
    """Demonstrate resource optimization capabilities"""
    print("\n⚡ Resource Optimization Demo")
    print("=" * 50)
    
    # Initialize Resource Optimizer
    resource_optimizer = ResourceOptimizer(vm_manager, job_scheduler, env_manager)
    
    # Analyze resource usage
    print("Analyzing resource usage...")
    analysis = resource_optimizer.analyze_resource_usage()
    print(f"Resource Analysis: {analysis}")
    
    # Get optimization recommendations
    print("\nGetting optimization recommendations...")
    recommendations = resource_optimizer.get_optimization_recommendations()
    print(f"Optimization Recommendations: {recommendations}")
    
    # Apply optimizations
    print("\nApplying optimizations...")
    optimization_result = resource_optimizer.apply_optimizations()
    print(f"Optimization Result: {optimization_result}")
    
    return resource_optimizer


async def demonstrate_cost_management(vm_manager, job_scheduler, env_manager):
    """Demonstrate cost management capabilities"""
    print("\n💰 Cost Management Demo")
    print("=" * 50)
    
    # Initialize Cost Manager
    cost_manager = CostManager(vm_manager, job_scheduler, env_manager)
    
    # Get cost analysis
    print("Analyzing costs...")
    cost_analysis = cost_manager.analyze_costs()
    print(f"Cost Analysis: {cost_analysis}")
    
    # Get cost optimization recommendations
    print("\nGetting cost optimization recommendations...")
    cost_recommendations = cost_manager.get_cost_optimization_recommendations()
    print(f"Cost Optimization Recommendations: {cost_recommendations}")
    
    # Set cost budgets
    print("\nSetting cost budgets...")
    budget_result = cost_manager.set_cost_budgets({
        'daily_budget': 100.0,
        'monthly_budget': 3000.0,
        'vm_budget': 50.0,
        'job_budget': 10.0
    })
    print(f"Budget Result: {budget_result}")
    
    return cost_manager


async def demonstrate_monitoring(vm_manager, job_scheduler, env_manager):
    """Demonstrate monitoring capabilities"""
    print("\n📊 Monitoring Demo")
    print("=" * 50)
    
    # Initialize VMAJE Monitor
    monitor = VMAJEMonitor(vm_manager, job_scheduler, env_manager)
    
    # Get system metrics
    print("Getting system metrics...")
    system_metrics = monitor.get_system_metrics()
    print(f"System Metrics: {system_metrics}")
    
    # Get performance metrics
    print("\nGetting performance metrics...")
    performance_metrics = monitor.get_performance_metrics()
    print(f"Performance Metrics: {performance_metrics}")
    
    # Get health status
    print("\nGetting health status...")
    health_status = monitor.get_health_status()
    print(f"Health Status: {health_status}")
    
    # Get alerts
    print("\nGetting alerts...")
    alerts = monitor.get_alerts()
    print(f"Alerts: {alerts}")
    
    return monitor


async def cleanup_resources(vm_manager, job_scheduler, env_manager, vm_ids, env_ids):
    """Cleanup resources"""
    print("\n🧹 Cleanup Demo")
    print("=" * 50)
    
    # Stop environments
    print("Stopping environments...")
    for env_id in env_ids:
        await env_manager.stop_environment(env_id)
    print("✅ Environments stopped")
    
    # Stop VMs
    print("Stopping VMs...")
    for vm_id in vm_ids:
        await vm_manager.stop_vm(vm_id)
    print("✅ VMs stopped")
    
    # Cleanup
    vm_manager.cleanup()
    job_scheduler.cleanup()
    env_manager.cleanup()
    print("✅ Cleanup completed")


async def main():
    """Main demonstration function"""
    print("🚀 VMAJE System Demonstration")
    print("=" * 60)
    print("Virtual Machine, Application, Job, and Environment Management")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    try:
        # Demonstrate VM Management
        vm_manager, vm_ids = await demonstrate_vm_management()
        
        # Demonstrate Job Scheduling
        job_scheduler = await demonstrate_job_scheduling(vm_manager)
        
        # Demonstrate Environment Management
        env_manager, env_ids = await demonstrate_environment_management(vm_manager)
        
        # Demonstrate Resource Optimization
        resource_optimizer = await demonstrate_resource_optimization(vm_manager, job_scheduler, env_manager)
        
        # Demonstrate Cost Management
        cost_manager = await demonstrate_cost_management(vm_manager, job_scheduler, env_manager)
        
        # Demonstrate Monitoring
        monitor = await demonstrate_monitoring(vm_manager, job_scheduler, env_manager)
        
        # Cleanup resources
        await cleanup_resources(vm_manager, job_scheduler, env_manager, vm_ids, env_ids)
        
        print("\n🎉 VMAJE demonstration completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Multi-cloud VM management (Azure, AWS, GCP)")
        print("✅ Intelligent job scheduling and prioritization")
        print("✅ Environment management and isolation")
        print("✅ Resource optimization and cost management")
        print("✅ Comprehensive monitoring and alerting")
        print("✅ Automated scaling and lifecycle management")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


