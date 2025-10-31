"""
Automated Benchmarking System
=============================

Advanced automated benchmarking system for AI models with comprehensive
evaluation, performance tracking, and intelligent benchmarking capabilities.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BenchmarkType(str, Enum):
    """Types of benchmarks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CUSTOM = "custom"


class BenchmarkMetric(str, Enum):
    """Benchmark metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    TRAINING_TIME = "training_time"
    INFERENCE_TIME = "inference_time"
    MEMORY_USAGE = "memory_usage"
    MODEL_SIZE = "model_size"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ENERGY_CONSUMPTION = "energy_consumption"
    COST = "cost"


class BenchmarkStatus(str, Enum):
    """Benchmark status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class BenchmarkDataset:
    """Benchmark dataset configuration"""
    dataset_id: str
    name: str
    benchmark_type: BenchmarkType
    data_generator: str
    parameters: Dict[str, Any]
    size: int
    features: int
    classes: int = None
    description: str = ""
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BenchmarkTask:
    """Benchmark task"""
    task_id: str
    model_name: str
    model_class: str
    model_parameters: Dict[str, Any]
    dataset: BenchmarkDataset
    benchmark_metrics: List[BenchmarkMetric]
    timeout: int = 3600  # 1 hour
    priority: int = 1
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    error_message: str = ""
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BenchmarkResult:
    """Benchmark result"""
    result_id: str
    task_id: str
    model_name: str
    dataset_name: str
    benchmark_type: BenchmarkType
    metrics: Dict[str, float]
    performance_metrics: Dict[str, Any]
    execution_time: float
    memory_usage: float
    cpu_usage: float
    status: BenchmarkStatus
    error_message: str = ""
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BenchmarkSuite:
    """Benchmark suite"""
    suite_id: str
    name: str
    description: str
    benchmark_type: BenchmarkType
    datasets: List[BenchmarkDataset]
    models: List[Dict[str, Any]]
    metrics: List[BenchmarkMetric]
    timeout: int = 3600
    parallel_execution: bool = True
    max_workers: int = 4
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AutomatedBenchmarkingSystem:
    """Advanced automated benchmarking system for AI models"""
    
    def __init__(self, max_concurrent_benchmarks: int = 4):
        self.max_concurrent_benchmarks = max_concurrent_benchmarks
        self.benchmark_tasks: Dict[str, BenchmarkTask] = {}
        self.benchmark_results: List[BenchmarkResult] = []
        self.benchmark_suites: Dict[str, BenchmarkSuite] = {}
        self.benchmark_datasets: Dict[str, BenchmarkDataset] = {}
        
        # Execution tracking
        self.running_tasks: Dict[str, threading.Thread] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_benchmarks)
        
        # Performance monitoring
        self.system_metrics = {
            "cpu_usage": deque(maxlen=100),
            "memory_usage": deque(maxlen=100),
            "active_benchmarks": 0,
            "completed_benchmarks": 0,
            "failed_benchmarks": 0
        }
        
        # Cache for results
        self.result_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
    
    async def create_benchmark_dataset(self, 
                                     name: str,
                                     benchmark_type: BenchmarkType,
                                     data_generator: str,
                                     parameters: Dict[str, Any],
                                     description: str = "") -> BenchmarkDataset:
        """Create a benchmark dataset"""
        try:
            dataset_id = hashlib.md5(f"{name}_{benchmark_type}_{datetime.now()}".encode()).hexdigest()
            
            # Generate sample data to get dimensions
            sample_data = await self._generate_sample_data(data_generator, parameters)
            
            dataset = BenchmarkDataset(
                dataset_id=dataset_id,
                name=name,
                benchmark_type=benchmark_type,
                data_generator=data_generator,
                parameters=parameters,
                size=len(sample_data[0]),
                features=sample_data[0].shape[1] if len(sample_data) > 0 else 0,
                classes=len(np.unique(sample_data[1])) if len(sample_data) > 1 else None,
                description=description
            )
            
            self.benchmark_datasets[dataset_id] = dataset
            
            logger.info(f"Created benchmark dataset: {name}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating benchmark dataset: {str(e)}")
            raise e
    
    async def create_benchmark_suite(self, 
                                   name: str,
                                   description: str,
                                   benchmark_type: BenchmarkType,
                                   datasets: List[BenchmarkDataset],
                                   models: List[Dict[str, Any]],
                                   metrics: List[BenchmarkMetric],
                                   timeout: int = 3600,
                                   parallel_execution: bool = True) -> BenchmarkSuite:
        """Create a benchmark suite"""
        try:
            suite_id = hashlib.md5(f"{name}_{benchmark_type}_{datetime.now()}".encode()).hexdigest()
            
            suite = BenchmarkSuite(
                suite_id=suite_id,
                name=name,
                description=description,
                benchmark_type=benchmark_type,
                datasets=datasets,
                models=models,
                metrics=metrics,
                timeout=timeout,
                parallel_execution=parallel_execution,
                max_workers=self.max_concurrent_benchmarks
            )
            
            self.benchmark_suites[suite_id] = suite
            
            logger.info(f"Created benchmark suite: {name}")
            
            return suite
            
        except Exception as e:
            logger.error(f"Error creating benchmark suite: {str(e)}")
            raise e
    
    async def run_benchmark_suite(self, suite_id: str) -> List[BenchmarkResult]:
        """Run a complete benchmark suite"""
        try:
            if suite_id not in self.benchmark_suites:
                raise ValueError(f"Benchmark suite {suite_id} not found")
            
            suite = self.benchmark_suites[suite_id]
            results = []
            
            logger.info(f"Starting benchmark suite: {suite.name}")
            
            # Create benchmark tasks
            tasks = []
            for dataset in suite.datasets:
                for model_config in suite.models:
                    task = await self._create_benchmark_task(
                        model_name=model_config["name"],
                        model_class=model_config["class"],
                        model_parameters=model_config.get("parameters", {}),
                        dataset=dataset,
                        benchmark_metrics=suite.metrics,
                        timeout=suite.timeout
                    )
                    tasks.append(task)
            
            # Execute benchmarks
            if suite.parallel_execution:
                results = await self._run_parallel_benchmarks(tasks)
            else:
                results = await self._run_sequential_benchmarks(tasks)
            
            logger.info(f"Completed benchmark suite: {suite.name} with {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running benchmark suite: {str(e)}")
            return []
    
    async def run_single_benchmark(self, 
                                 model_name: str,
                                 model_class: str,
                                 model_parameters: Dict[str, Any],
                                 dataset: BenchmarkDataset,
                                 benchmark_metrics: List[BenchmarkMetric],
                                 timeout: int = 3600) -> BenchmarkResult:
        """Run a single benchmark"""
        try:
            # Create benchmark task
            task = await self._create_benchmark_task(
                model_name=model_name,
                model_class=model_class,
                model_parameters=model_parameters,
                dataset=dataset,
                benchmark_metrics=benchmark_metrics,
                timeout=timeout
            )
            
            # Execute benchmark
            result = await self._execute_benchmark(task)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running single benchmark: {str(e)}")
            raise e
    
    async def get_benchmark_results(self, 
                                  suite_id: str = None,
                                  model_name: str = None,
                                  dataset_name: str = None,
                                  benchmark_type: BenchmarkType = None) -> List[BenchmarkResult]:
        """Get benchmark results with filtering"""
        try:
            results = self.benchmark_results.copy()
            
            # Apply filters
            if suite_id:
                # Filter by suite (would need to track suite_id in results)
                pass
            
            if model_name:
                results = [r for r in results if r.model_name == model_name]
            
            if dataset_name:
                results = [r for r in results if r.dataset_name == dataset_name]
            
            if benchmark_type:
                results = [r for r in results if r.benchmark_type == benchmark_type]
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting benchmark results: {str(e)}")
            return []
    
    async def get_benchmark_analytics(self, 
                                    suite_id: str = None,
                                    time_range_days: int = 30) -> Dict[str, Any]:
        """Get comprehensive benchmark analytics"""
        try:
            # Filter results by time range
            cutoff_date = datetime.now() - timedelta(days=time_range_days)
            recent_results = [r for r in self.benchmark_results if r.created_at >= cutoff_date]
            
            if suite_id:
                # Filter by suite
                pass
            
            analytics = {
                "total_benchmarks": len(recent_results),
                "successful_benchmarks": len([r for r in recent_results if r.status == BenchmarkStatus.COMPLETED]),
                "failed_benchmarks": len([r for r in recent_results if r.status == BenchmarkStatus.FAILED]),
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "model_performance": {},
                "dataset_performance": {},
                "benchmark_type_performance": {},
                "top_performers": [],
                "performance_trends": {},
                "system_metrics": dict(self.system_metrics)
            }
            
            if recent_results:
                # Calculate success rate
                successful = len([r for r in recent_results if r.status == BenchmarkStatus.COMPLETED])
                analytics["success_rate"] = successful / len(recent_results)
                
                # Calculate average execution time
                execution_times = [r.execution_time for r in recent_results if r.execution_time > 0]
                if execution_times:
                    analytics["average_execution_time"] = np.mean(execution_times)
                
                # Analyze model performance
                analytics["model_performance"] = await self._analyze_model_performance(recent_results)
                
                # Analyze dataset performance
                analytics["dataset_performance"] = await self._analyze_dataset_performance(recent_results)
                
                # Analyze benchmark type performance
                analytics["benchmark_type_performance"] = await self._analyze_benchmark_type_performance(recent_results)
                
                # Identify top performers
                analytics["top_performers"] = await self._identify_top_performers(recent_results)
                
                # Analyze performance trends
                analytics["performance_trends"] = await self._analyze_performance_trends(recent_results)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting benchmark analytics: {str(e)}")
            return {"error": str(e)}
    
    async def compare_models(self, 
                           model_names: List[str],
                           dataset_name: str = None,
                           benchmark_type: BenchmarkType = None) -> Dict[str, Any]:
        """Compare multiple models"""
        try:
            # Get results for specified models
            results = self.benchmark_results.copy()
            
            if model_names:
                results = [r for r in results if r.model_name in model_names]
            
            if dataset_name:
                results = [r for r in results if r.dataset_name == dataset_name]
            
            if benchmark_type:
                results = [r for r in results if r.benchmark_type == benchmark_type]
            
            if not results:
                return {"error": "No results found for comparison"}
            
            # Group results by model
            model_results = defaultdict(list)
            for result in results:
                model_results[result.model_name].append(result)
            
            # Calculate comparison metrics
            comparison = {
                "models_compared": list(model_results.keys()),
                "total_benchmarks": len(results),
                "model_statistics": {},
                "performance_ranking": [],
                "best_performers": {},
                "recommendations": []
            }
            
            # Analyze each model
            for model_name, model_result_list in model_results.items():
                model_stats = await self._calculate_model_statistics(model_result_list)
                comparison["model_statistics"][model_name] = model_stats
            
            # Create performance ranking
            comparison["performance_ranking"] = await self._create_performance_ranking(model_results)
            
            # Identify best performers
            comparison["best_performers"] = await self._identify_best_performers(model_results)
            
            # Generate recommendations
            comparison["recommendations"] = await self._generate_comparison_recommendations(model_results)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _generate_sample_data(self, data_generator: str, parameters: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample data for dataset configuration"""
        try:
            if data_generator == "make_classification":
                X, y = make_classification(**parameters)
            elif data_generator == "make_regression":
                X, y = make_regression(**parameters)
            else:
                # Default classification dataset
                X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            # Return default data
            return make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    async def _create_benchmark_task(self, 
                                   model_name: str,
                                   model_class: str,
                                   model_parameters: Dict[str, Any],
                                   dataset: BenchmarkDataset,
                                   benchmark_metrics: List[BenchmarkMetric],
                                   timeout: int) -> BenchmarkTask:
        """Create a benchmark task"""
        try:
            task_id = hashlib.md5(f"{model_name}_{dataset.dataset_id}_{datetime.now()}".encode()).hexdigest()
            
            task = BenchmarkTask(
                task_id=task_id,
                model_name=model_name,
                model_class=model_class,
                model_parameters=model_parameters,
                dataset=dataset,
                benchmark_metrics=benchmark_metrics,
                timeout=timeout
            )
            
            self.benchmark_tasks[task_id] = task
            
            return task
            
        except Exception as e:
            logger.error(f"Error creating benchmark task: {str(e)}")
            raise e
    
    async def _run_parallel_benchmarks(self, tasks: List[BenchmarkTask]) -> List[BenchmarkResult]:
        """Run benchmarks in parallel"""
        try:
            results = []
            
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                future = self.executor.submit(self._execute_benchmark_sync, task)
                future_to_task[future] = task
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Benchmark task {task.task_id} failed: {str(e)}")
                    # Create failed result
                    failed_result = BenchmarkResult(
                        result_id=hashlib.md5(f"{task.task_id}_failed_{datetime.now()}".encode()).hexdigest(),
                        task_id=task.task_id,
                        model_name=task.model_name,
                        dataset_name=task.dataset.name,
                        benchmark_type=task.dataset.benchmark_type,
                        metrics={},
                        performance_metrics={},
                        execution_time=0.0,
                        memory_usage=0.0,
                        cpu_usage=0.0,
                        status=BenchmarkStatus.FAILED,
                        error_message=str(e)
                    )
                    results.append(failed_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running parallel benchmarks: {str(e)}")
            return []
    
    async def _run_sequential_benchmarks(self, tasks: List[BenchmarkTask]) -> List[BenchmarkResult]:
        """Run benchmarks sequentially"""
        try:
            results = []
            
            for task in tasks:
                try:
                    result = await self._execute_benchmark(task)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Benchmark task {task.task_id} failed: {str(e)}")
                    # Create failed result
                    failed_result = BenchmarkResult(
                        result_id=hashlib.md5(f"{task.task_id}_failed_{datetime.now()}".encode()).hexdigest(),
                        task_id=task.task_id,
                        model_name=task.model_name,
                        dataset_name=task.dataset.name,
                        benchmark_type=task.dataset.benchmark_type,
                        metrics={},
                        performance_metrics={},
                        execution_time=0.0,
                        memory_usage=0.0,
                        cpu_usage=0.0,
                        status=BenchmarkStatus.FAILED,
                        error_message=str(e)
                    )
                    results.append(failed_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running sequential benchmarks: {str(e)}")
            return []
    
    async def _execute_benchmark(self, task: BenchmarkTask) -> BenchmarkResult:
        """Execute a benchmark task"""
        try:
            # Update task status
            task.status = BenchmarkStatus.RUNNING
            task.started_at = datetime.now()
            
            # Generate data
            X, y = await self._generate_sample_data(task.dataset.data_generator, task.dataset.parameters)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create model
            model = await self._create_model(task.model_class, task.model_parameters)
            
            # Start performance monitoring
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            start_cpu = psutil.cpu_percent()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = await self._calculate_benchmark_metrics(
                y_test, y_pred, task.benchmark_metrics, task.dataset.benchmark_type
            )
            
            # End performance monitoring
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            end_cpu = psutil.cpu_percent()
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = (start_cpu + end_cpu) / 2
            
            # Create result
            result = BenchmarkResult(
                result_id=hashlib.md5(f"{task.task_id}_{datetime.now()}".encode()).hexdigest(),
                task_id=task.task_id,
                model_name=task.model_name,
                dataset_name=task.dataset.name,
                benchmark_type=task.dataset.benchmark_type,
                metrics=metrics,
                performance_metrics={
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": X.shape[1],
                    "model_parameters": task.model_parameters
                },
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                status=BenchmarkStatus.COMPLETED
            )
            
            # Update task status
            task.status = BenchmarkStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Store result
            self.benchmark_results.append(result)
            
            # Update system metrics
            self.system_metrics["completed_benchmarks"] += 1
            
            logger.info(f"Completed benchmark: {task.model_name} on {task.dataset.name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing benchmark: {str(e)}")
            
            # Update task status
            task.status = BenchmarkStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            # Create failed result
            failed_result = BenchmarkResult(
                result_id=hashlib.md5(f"{task.task_id}_failed_{datetime.now()}".encode()).hexdigest(),
                task_id=task.task_id,
                model_name=task.model_name,
                dataset_name=task.dataset.name,
                benchmark_type=task.dataset.benchmark_type,
                metrics={},
                performance_metrics={},
                execution_time=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                status=BenchmarkStatus.FAILED,
                error_message=str(e)
            )
            
            # Store result
            self.benchmark_results.append(failed_result)
            
            # Update system metrics
            self.system_metrics["failed_benchmarks"] += 1
            
            return failed_result
    
    def _execute_benchmark_sync(self, task: BenchmarkTask) -> BenchmarkResult:
        """Synchronous version of benchmark execution for thread pool"""
        try:
            # This is a simplified synchronous version
            # In practice, you'd need to handle async operations differently
            
            # Update task status
            task.status = BenchmarkStatus.RUNNING
            task.started_at = datetime.now()
            
            # Generate data
            X, y = self._generate_sample_data_sync(task.dataset.data_generator, task.dataset.parameters)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create model
            model = self._create_model_sync(task.model_class, task.model_parameters)
            
            # Start performance monitoring
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            start_cpu = psutil.cpu_percent()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_benchmark_metrics_sync(
                y_test, y_pred, task.benchmark_metrics, task.dataset.benchmark_type
            )
            
            # End performance monitoring
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            end_cpu = psutil.cpu_percent()
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = (start_cpu + end_cpu) / 2
            
            # Create result
            result = BenchmarkResult(
                result_id=hashlib.md5(f"{task.task_id}_{datetime.now()}".encode()).hexdigest(),
                task_id=task.task_id,
                model_name=task.model_name,
                dataset_name=task.dataset.name,
                benchmark_type=task.dataset.benchmark_type,
                metrics=metrics,
                performance_metrics={
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": X.shape[1],
                    "model_parameters": task.model_parameters
                },
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                status=BenchmarkStatus.COMPLETED
            )
            
            # Update task status
            task.status = BenchmarkStatus.COMPLETED
            task.completed_at = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing benchmark sync: {str(e)}")
            
            # Update task status
            task.status = BenchmarkStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            # Create failed result
            failed_result = BenchmarkResult(
                result_id=hashlib.md5(f"{task.task_id}_failed_{datetime.now()}".encode()).hexdigest(),
                task_id=task.task_id,
                model_name=task.model_name,
                dataset_name=task.dataset.name,
                benchmark_type=task.dataset.benchmark_type,
                metrics={},
                performance_metrics={},
                execution_time=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                status=BenchmarkStatus.FAILED,
                error_message=str(e)
            )
            
            return failed_result
    
    def _generate_sample_data_sync(self, data_generator: str, parameters: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronous version of data generation"""
        try:
            if data_generator == "make_classification":
                X, y = make_classification(**parameters)
            elif data_generator == "make_regression":
                X, y = make_regression(**parameters)
            else:
                # Default classification dataset
                X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error generating sample data sync: {str(e)}")
            # Return default data
            return make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    async def _create_model(self, model_class: str, parameters: Dict[str, Any]) -> Any:
        """Create model instance"""
        try:
            # Import model class dynamically
            if model_class == "RandomForestClassifier":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**parameters)
            elif model_class == "LogisticRegression":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(**parameters)
            elif model_class == "SVC":
                from sklearn.svm import SVC
                return SVC(**parameters)
            elif model_class == "RandomForestRegressor":
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(**parameters)
            elif model_class == "LinearRegression":
                from sklearn.linear_model import LinearRegression
                return LinearRegression(**parameters)
            else:
                raise ValueError(f"Unsupported model class: {model_class}")
                
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise e
    
    def _create_model_sync(self, model_class: str, parameters: Dict[str, Any]) -> Any:
        """Synchronous version of model creation"""
        try:
            # Import model class dynamically
            if model_class == "RandomForestClassifier":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**parameters)
            elif model_class == "LogisticRegression":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(**parameters)
            elif model_class == "SVC":
                from sklearn.svm import SVC
                return SVC(**parameters)
            elif model_class == "RandomForestRegressor":
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(**parameters)
            elif model_class == "LinearRegression":
                from sklearn.linear_model import LinearRegression
                return LinearRegression(**parameters)
            else:
                raise ValueError(f"Unsupported model class: {model_class}")
                
        except Exception as e:
            logger.error(f"Error creating model sync: {str(e)}")
            raise e
    
    async def _calculate_benchmark_metrics(self, 
                                         y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         metrics: List[BenchmarkMetric],
                                         benchmark_type: BenchmarkType) -> Dict[str, float]:
        """Calculate benchmark metrics"""
        try:
            results = {}
            
            for metric in metrics:
                try:
                    if metric == BenchmarkMetric.ACCURACY:
                        results[metric.value] = accuracy_score(y_true, y_pred)
                    elif metric == BenchmarkMetric.PRECISION:
                        results[metric.value] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    elif metric == BenchmarkMetric.RECALL:
                        results[metric.value] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    elif metric == BenchmarkMetric.F1_SCORE:
                        results[metric.value] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    elif metric == BenchmarkMetric.MSE:
                        results[metric.value] = mean_squared_error(y_true, y_pred)
                    elif metric == BenchmarkMetric.R2_SCORE:
                        results[metric.value] = r2_score(y_true, y_pred)
                    else:
                        results[metric.value] = 0.0
                except Exception as e:
                    logger.warning(f"Error calculating {metric.value}: {str(e)}")
                    results[metric.value] = 0.0
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics: {str(e)}")
            return {}
    
    def _calculate_benchmark_metrics_sync(self, 
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        metrics: List[BenchmarkMetric],
                                        benchmark_type: BenchmarkType) -> Dict[str, float]:
        """Synchronous version of metric calculation"""
        try:
            results = {}
            
            for metric in metrics:
                try:
                    if metric == BenchmarkMetric.ACCURACY:
                        results[metric.value] = accuracy_score(y_true, y_pred)
                    elif metric == BenchmarkMetric.PRECISION:
                        results[metric.value] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    elif metric == BenchmarkMetric.RECALL:
                        results[metric.value] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    elif metric == BenchmarkMetric.F1_SCORE:
                        results[metric.value] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    elif metric == BenchmarkMetric.MSE:
                        results[metric.value] = mean_squared_error(y_true, y_pred)
                    elif metric == BenchmarkMetric.R2_SCORE:
                        results[metric.value] = r2_score(y_true, y_pred)
                    else:
                        results[metric.value] = 0.0
                except Exception as e:
                    logger.warning(f"Error calculating {metric.value}: {str(e)}")
                    results[metric.value] = 0.0
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics sync: {str(e)}")
            return {}
    
    def _monitor_system(self):
        """Monitor system performance"""
        try:
            while True:
                # Update system metrics
                self.system_metrics["cpu_usage"].append(psutil.cpu_percent())
                self.system_metrics["memory_usage"].append(psutil.virtual_memory().percent)
                self.system_metrics["active_benchmarks"] = len([t for t in self.benchmark_tasks.values() 
                                                              if t.status == BenchmarkStatus.RUNNING])
                
                time.sleep(5)  # Update every 5 seconds
                
        except Exception as e:
            logger.error(f"Error in system monitoring: {str(e)}")
    
    async def _analyze_model_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze model performance"""
        try:
            model_performance = defaultdict(list)
            
            for result in results:
                if result.status == BenchmarkStatus.COMPLETED:
                    model_performance[result.model_name].append(result)
            
            analysis = {}
            for model_name, model_results in model_performance.items():
                if model_results:
                    # Calculate average metrics
                    avg_metrics = defaultdict(list)
                    for result in model_results:
                        for metric, value in result.metrics.items():
                            avg_metrics[metric].append(value)
                    
                    model_analysis = {}
                    for metric, values in avg_metrics.items():
                        model_analysis[metric] = {
                            "average": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                            "count": len(values)
                        }
                    
                    analysis[model_name] = model_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model performance: {str(e)}")
            return {}
    
    async def _analyze_dataset_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze dataset performance"""
        try:
            dataset_performance = defaultdict(list)
            
            for result in results:
                if result.status == BenchmarkStatus.COMPLETED:
                    dataset_performance[result.dataset_name].append(result)
            
            analysis = {}
            for dataset_name, dataset_results in dataset_performance.items():
                if dataset_results:
                    # Calculate average metrics
                    avg_metrics = defaultdict(list)
                    for result in dataset_results:
                        for metric, value in result.metrics.items():
                            avg_metrics[metric].append(value)
                    
                    dataset_analysis = {}
                    for metric, values in avg_metrics.items():
                        dataset_analysis[metric] = {
                            "average": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                            "count": len(values)
                        }
                    
                    analysis[dataset_name] = dataset_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing dataset performance: {str(e)}")
            return {}
    
    async def _analyze_benchmark_type_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze benchmark type performance"""
        try:
            type_performance = defaultdict(list)
            
            for result in results:
                if result.status == BenchmarkStatus.COMPLETED:
                    type_performance[result.benchmark_type.value].append(result)
            
            analysis = {}
            for benchmark_type, type_results in type_performance.items():
                if type_results:
                    # Calculate average metrics
                    avg_metrics = defaultdict(list)
                    for result in type_results:
                        for metric, value in result.metrics.items():
                            avg_metrics[metric].append(value)
                    
                    type_analysis = {}
                    for metric, values in avg_metrics.items():
                        type_analysis[metric] = {
                            "average": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                            "count": len(values)
                        }
                    
                    analysis[benchmark_type] = type_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing benchmark type performance: {str(e)}")
            return {}
    
    async def _identify_top_performers(self, results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        """Identify top performing models"""
        try:
            if not results:
                return []
            
            # Calculate composite scores
            model_scores = defaultdict(list)
            
            for result in results:
                if result.status == BenchmarkStatus.COMPLETED and result.metrics:
                    # Calculate composite score
                    scores = list(result.metrics.values())
                    if scores:
                        composite_score = np.mean(scores)
                        model_scores[result.model_name].append(composite_score)
            
            # Calculate average scores per model
            model_averages = {}
            for model_name, scores in model_scores.items():
                model_averages[model_name] = np.mean(scores)
            
            # Sort by average score
            sorted_models = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
            
            # Create top performers list
            top_performers = []
            for i, (model_name, avg_score) in enumerate(sorted_models[:10]):  # Top 10
                top_performers.append({
                    "rank": i + 1,
                    "model_name": model_name,
                    "average_score": float(avg_score),
                    "benchmark_count": len(model_scores[model_name])
                })
            
            return top_performers
            
        except Exception as e:
            logger.error(f"Error identifying top performers: {str(e)}")
            return []
    
    async def _analyze_performance_trends(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        try:
            if not results:
                return {}
            
            # Group results by time periods
            daily_results = defaultdict(list)
            
            for result in results:
                if result.status == BenchmarkStatus.COMPLETED:
                    date_key = result.created_at.date()
                    daily_results[date_key].append(result)
            
            # Calculate daily averages
            trends = {}
            for date, day_results in daily_results.items():
                if day_results:
                    # Calculate average metrics for the day
                    daily_metrics = defaultdict(list)
                    for result in day_results:
                        for metric, value in result.metrics.items():
                            daily_metrics[metric].append(value)
                    
                    daily_averages = {}
                    for metric, values in daily_metrics.items():
                        daily_averages[metric] = float(np.mean(values))
                    
                    trends[date.isoformat()] = daily_averages
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {str(e)}")
            return {}
    
    async def _calculate_model_statistics(self, model_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate statistics for a model"""
        try:
            if not model_results:
                return {}
            
            # Calculate execution time statistics
            execution_times = [r.execution_time for r in model_results if r.execution_time > 0]
            memory_usage = [r.memory_usage for r in model_results if r.memory_usage > 0]
            cpu_usage = [r.cpu_usage for r in model_results if r.cpu_usage > 0]
            
            # Calculate metric statistics
            all_metrics = defaultdict(list)
            for result in model_results:
                for metric, value in result.metrics.items():
                    all_metrics[metric].append(value)
            
            statistics = {
                "total_benchmarks": len(model_results),
                "successful_benchmarks": len([r for r in model_results if r.status == BenchmarkStatus.COMPLETED]),
                "failed_benchmarks": len([r for r in model_results if r.status == BenchmarkStatus.FAILED]),
                "success_rate": len([r for r in model_results if r.status == BenchmarkStatus.COMPLETED]) / len(model_results) if model_results else 0,
                "execution_time": {
                    "average": float(np.mean(execution_times)) if execution_times else 0,
                    "std": float(np.std(execution_times)) if execution_times else 0,
                    "min": float(np.min(execution_times)) if execution_times else 0,
                    "max": float(np.max(execution_times)) if execution_times else 0
                },
                "memory_usage": {
                    "average": float(np.mean(memory_usage)) if memory_usage else 0,
                    "std": float(np.std(memory_usage)) if memory_usage else 0,
                    "min": float(np.min(memory_usage)) if memory_usage else 0,
                    "max": float(np.max(memory_usage)) if memory_usage else 0
                },
                "cpu_usage": {
                    "average": float(np.mean(cpu_usage)) if cpu_usage else 0,
                    "std": float(np.std(cpu_usage)) if cpu_usage else 0,
                    "min": float(np.min(cpu_usage)) if cpu_usage else 0,
                    "max": float(np.max(cpu_usage)) if cpu_usage else 0
                },
                "metrics": {}
            }
            
            # Calculate metric statistics
            for metric, values in all_metrics.items():
                statistics["metrics"][metric] = {
                    "average": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values)
                }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating model statistics: {str(e)}")
            return {}
    
    async def _create_performance_ranking(self, model_results: Dict[str, List[BenchmarkResult]]) -> List[Dict[str, Any]]:
        """Create performance ranking of models"""
        try:
            rankings = []
            
            for model_name, results in model_results.items():
                if results:
                    # Calculate composite score
                    all_scores = []
                    for result in results:
                        if result.status == BenchmarkStatus.COMPLETED and result.metrics:
                            scores = list(result.metrics.values())
                            if scores:
                                all_scores.append(np.mean(scores))
                    
                    if all_scores:
                        avg_score = np.mean(all_scores)
                        rankings.append({
                            "model_name": model_name,
                            "average_score": float(avg_score),
                            "benchmark_count": len(results),
                            "success_rate": len([r for r in results if r.status == BenchmarkStatus.COMPLETED]) / len(results)
                        })
            
            # Sort by average score
            rankings.sort(key=lambda x: x["average_score"], reverse=True)
            
            # Add ranks
            for i, ranking in enumerate(rankings):
                ranking["rank"] = i + 1
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error creating performance ranking: {str(e)}")
            return []
    
    async def _identify_best_performers(self, model_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Identify best performers by metric"""
        try:
            best_performers = {}
            
            # Collect all metrics across all models
            all_metrics = defaultdict(list)
            for model_name, results in model_results.items():
                for result in results:
                    if result.status == BenchmarkStatus.COMPLETED:
                        for metric, value in result.metrics.items():
                            all_metrics[metric].append((model_name, value))
            
            # Find best performer for each metric
            for metric, model_values in all_metrics.items():
                if model_values:
                    # Sort by value (higher is better for most metrics)
                    model_values.sort(key=lambda x: x[1], reverse=True)
                    best_model, best_value = model_values[0]
                    
                    best_performers[metric] = {
                        "model": best_model,
                        "value": float(best_value),
                        "count": len(model_values)
                    }
            
            return best_performers
            
        except Exception as e:
            logger.error(f"Error identifying best performers: {str(e)}")
            return {}
    
    async def _generate_comparison_recommendations(self, model_results: Dict[str, List[BenchmarkResult]]) -> List[str]:
        """Generate recommendations based on model comparison"""
        try:
            recommendations = []
            
            # Analyze success rates
            success_rates = {}
            for model_name, results in model_results.items():
                if results:
                    success_rate = len([r for r in results if r.status == BenchmarkStatus.COMPLETED]) / len(results)
                    success_rates[model_name] = success_rate
            
            if success_rates:
                best_success_rate = max(success_rates.items(), key=lambda x: x[1])
                worst_success_rate = min(success_rates.items(), key=lambda x: x[1])
                
                if best_success_rate[1] > 0.9:
                    recommendations.append(f"{best_success_rate[0]} has excellent reliability ({best_success_rate[1]:.1%} success rate)")
                
                if worst_success_rate[1] < 0.5:
                    recommendations.append(f"{worst_success_rate[0]} has poor reliability ({worst_success_rate[1]:.1%} success rate) - consider improvements")
            
            # Analyze performance consistency
            for model_name, results in model_results.items():
                if len(results) > 1:
                    completed_results = [r for r in results if r.status == BenchmarkStatus.COMPLETED]
                    if completed_results:
                        # Calculate coefficient of variation for execution time
                        execution_times = [r.execution_time for r in completed_results if r.execution_time > 0]
                        if execution_times and len(execution_times) > 1:
                            cv = np.std(execution_times) / np.mean(execution_times)
                            if cv > 0.5:
                                recommendations.append(f"{model_name} shows high variability in execution time - consider optimization")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating comparison recommendations: {str(e)}")
            return []


# Global benchmarking system instance
_benchmarking_system: Optional[AutomatedBenchmarkingSystem] = None


def get_automated_benchmarking_system(max_concurrent_benchmarks: int = 4) -> AutomatedBenchmarkingSystem:
    """Get or create global automated benchmarking system instance"""
    global _benchmarking_system
    if _benchmarking_system is None:
        _benchmarking_system = AutomatedBenchmarkingSystem(max_concurrent_benchmarks)
    return _benchmarking_system


# Example usage
async def main():
    """Example usage of the automated benchmarking system"""
    system = get_automated_benchmarking_system()
    
    # Create benchmark datasets
    classification_dataset = await system.create_benchmark_dataset(
        name="Classification Dataset",
        benchmark_type=BenchmarkType.CLASSIFICATION,
        data_generator="make_classification",
        parameters={"n_samples": 1000, "n_features": 20, "n_classes": 2, "random_state": 42},
        description="Binary classification dataset"
    )
    
    regression_dataset = await system.create_benchmark_dataset(
        name="Regression Dataset",
        benchmark_type=BenchmarkType.REGRESSION,
        data_generator="make_regression",
        parameters={"n_samples": 1000, "n_features": 20, "noise": 0.1, "random_state": 42},
        description="Regression dataset"
    )
    
    # Create benchmark suite
    suite = await system.create_benchmark_suite(
        name="ML Models Benchmark",
        description="Comprehensive benchmark of machine learning models",
        benchmark_type=BenchmarkType.CLASSIFICATION,
        datasets=[classification_dataset],
        models=[
            {"name": "RandomForest", "class": "RandomForestClassifier", "parameters": {"n_estimators": 100}},
            {"name": "LogisticRegression", "class": "LogisticRegression", "parameters": {"random_state": 42}},
            {"name": "SVC", "class": "SVC", "parameters": {"random_state": 42}}
        ],
        metrics=[BenchmarkMetric.ACCURACY, BenchmarkMetric.F1_SCORE, BenchmarkMetric.PRECISION, BenchmarkMetric.RECALL],
        parallel_execution=True
    )
    
    # Run benchmark suite
    results = await system.run_benchmark_suite(suite.suite_id)
    print(f"Completed {len(results)} benchmarks")
    
    # Get analytics
    analytics = await system.get_benchmark_analytics()
    print(f"Success rate: {analytics.get('success_rate', 0):.1%}")
    
    # Compare models
    comparison = await system.compare_models(["RandomForest", "LogisticRegression", "SVC"])
    print(f"Models compared: {comparison.get('models_compared', [])}")


if __name__ == "__main__":
    asyncio.run(main())

























