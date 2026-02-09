from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from efficient_data_loading import (
        import psutil
        import psutil
from typing import Any, List, Dict, Optional
import logging
"""
Efficient Data Loading Demo

This demo showcases the efficient data loading system with:
- Performance benchmarking and optimization
- Memory usage monitoring and optimization
- Different dataset types and configurations
- Caching and prefetching strategies
- Async data loading capabilities
- Cross-validation and data splitting
"""



    DataLoaderConfig, ThreatDetectionDataset, AnomalyDetectionDataset,
    NetworkTrafficDataset, MalwareDataset, CachedDataset, DataAugmentation,
    DataLoaderFactory, DataLoaderMonitor, MemoryOptimizedDataLoader,
    AsyncDataLoader, DataLoaderBenchmark, create_balanced_sampler,
    split_dataset, get_dataset_info, optimize_dataloader_config
)


class EfficientDataLoadingDemo:
    """Comprehensive demo for efficient data loading."""
    
    def __init__(self) -> Any:
        self.demo_dir = Path("./demo_output")
        self.demo_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.demo_dir / "data").mkdir(exist_ok=True)
        (self.demo_dir / "cache").mkdir(exist_ok=True)
        (self.demo_dir / "logs").mkdir(exist_ok=True)
        
        self.results = {}
        
    async def run_comprehensive_demo(self) -> Any:
        """Run the complete demo showcasing all features."""
        print("🚀 Starting Efficient Data Loading Demo")
        print("=" * 80)
        
        # Generate synthetic datasets
        await self._generate_synthetic_datasets()
        
        # Demo 1: Basic Data Loading
        await self._demo_basic_data_loading()
        
        # Demo 2: Performance Benchmarking
        await self._demo_performance_benchmarking()
        
        # Demo 3: Memory Optimization
        await self._demo_memory_optimization()
        
        # Demo 4: Caching Strategies
        await self._demo_caching_strategies()
        
        # Demo 5: Async Data Loading
        await self._demo_async_data_loading()
        
        # Demo 6: Cross-Validation
        await self._demo_cross_validation()
        
        # Demo 7: Data Augmentation
        await self._demo_data_augmentation()
        
        # Demo 8: System Resource Optimization
        await self._demo_system_optimization()
        
        # Save results
        self._save_demo_results()
        
        print("\n✅ Demo completed successfully!")
        print(f"Results saved to: {self.demo_dir / 'demo_results.json'}")
    
    async def _generate_synthetic_datasets(self) -> Any:
        """Generate synthetic datasets for demonstration."""
        print("\n📊 Generating Synthetic Datasets...")
        
        # Threat Detection Dataset
        X_threat, y_threat = make_classification(
            n_samples=10000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        # Convert to text-like features
        threat_texts = []
        for i, features in enumerate(X_threat):
            text = f"Network packet {i}: {', '.join([f'f{j}={v:.3f}' for j, v in enumerate(features[:10])])}"
            threat_texts.append(text)
        
        threat_df = pd.DataFrame({
            'text': threat_texts,
            'label': y_threat
        })
        threat_df.to_csv(self.demo_dir / "data" / "threat_detection.csv", index=False)
        
        # Anomaly Detection Dataset
        X_anomaly, y_anomaly = make_regression(
            n_samples=8000,
            n_features=50,
            n_informative=30,
            noise=0.1,
            random_state=42
        )
        
        # Add anomalies
        anomaly_indices = np.random.choice(len(X_anomaly), size=800, replace=False)
        for idx in anomaly_indices:
            X_anomaly[idx] += np.random.normal(0, 2, X_anomaly.shape[1])
            y_anomaly[idx] = 1
        y_anomaly[~np.isin(np.arange(len(y_anomaly)), anomaly_indices)] = 0
        
        anomaly_df = pd.DataFrame({
            'features': [json.dumps(features.tolist()) for features in X_anomaly],
            'label': y_anomaly
        })
        anomaly_df.to_csv(self.demo_dir / "data" / "anomaly_detection.csv", index=False)
        
        # Network Traffic Dataset
        X_network, y_network = make_classification(
            n_samples=12000,
            n_features=30,
            n_classes=3,
            random_state=42
        )
        
        # Add timestamp column
        timestamps = pd.date_range('2023-01-01', periods=len(X_network), freq='S')
        
        network_df = pd.DataFrame(X_network, columns=[f'feature_{i}' for i in range(30)])
        network_df['timestamp'] = timestamps
        network_df['label'] = y_network
        network_df.to_csv(self.demo_dir / "data" / "network_traffic.csv", index=False)
        
        # Malware Dataset
        X_malware, y_malware = make_classification(
            n_samples=6000,
            n_features=100,
            n_classes=5,
            random_state=42
        )
        
        # Create binary features and API sequences
        binary_features = []
        api_sequences = []
        
        api_calls = ['CreateFile', 'ReadFile', 'WriteFile', 'DeleteFile', 'RegCreateKey', 
                    'RegSetValue', 'RegDeleteKey', 'CreateProcess', 'CreateThread', 'Connect']
        
        for i, features in enumerate(X_malware):
            # Binary features (first 50 features)
            binary_feat = features[:50].tolist()
            binary_features.append(binary_feat)
            
            # API sequence (random sequence of API calls)
            seq_length = np.random.randint(5, 15)
            api_seq = np.random.choice(api_calls, seq_length).tolist()
            api_sequences.append(api_seq)
        
        malware_df = pd.DataFrame({
            'binary_features': [json.dumps(feat) for feat in binary_features],
            'api_calls': [json.dumps(seq) for seq in api_sequences],
            'label': y_malware
        })
        malware_df.to_csv(self.demo_dir / "data" / "malware.csv", index=False)
        
        print(f"✅ Generated datasets:")
        print(f"   - Threat detection: {len(threat_df)} samples")
        print(f"   - Anomaly detection: {len(anomaly_df)} samples")
        print(f"   - Network traffic: {len(network_df)} samples")
        print(f"   - Malware: {len(malware_df)} samples")
    
    async def _demo_basic_data_loading(self) -> Any:
        """Demo basic data loading with different dataset types."""
        print("\n📥 Demo 1: Basic Data Loading")
        print("-" * 50)
        
        # Test different dataset types
        dataset_types = [
            ("threat_detection", ThreatDetectionDataset),
            ("anomaly_detection", AnomalyDetectionDataset),
            ("network_traffic", NetworkTrafficDataset),
            ("malware", MalwareDataset)
        ]
        
        config = DataLoaderConfig(
            batch_size=32,
            num_workers=2,
            pin_memory=True,
            enable_caching=False
        )
        
        for dataset_name, dataset_class in dataset_types:
            print(f"Loading {dataset_name} dataset...")
            
            try:
                # Create dataset
                dataset_path = self.demo_dir / "data" / f"{dataset_name}.csv"
                dataset = dataset_class(str(dataset_path), config)
                
                # Create DataLoader
                dataloader = DataLoaderFactory.create_dataloader(
                    dataset, config, dataset_name
                )
                
                # Test loading a few batches
                start_time = time.time()
                batch_count = 0
                
                for batch in dataloader:
                    batch_count += 1
                    if batch_count >= 5:  # Load 5 batches
                        break
                
                load_time = time.time() - start_time
                
                # Store results
                self.results[f"{dataset_name}_basic_loading"] = {
                    "dataset_size": len(dataset),
                    "batch_count": batch_count,
                    "load_time_seconds": load_time,
                    "throughput_batches_per_sec": batch_count / load_time,
                    "metadata": dataset.get_metadata()
                }
                
                print(f"   ✅ {dataset_name}: {len(dataset)} samples, {load_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ {dataset_name}: Error - {e}")
                self.results[f"{dataset_name}_basic_loading"] = {"error": str(e)}
    
    async def _demo_performance_benchmarking(self) -> Any:
        """Demo performance benchmarking."""
        print("\n⚡ Demo 2: Performance Benchmarking")
        print("-" * 50)
        
        # Test different configurations
        configs = [
            DataLoaderConfig(batch_size=16, num_workers=1, pin_memory=False),
            DataLoaderConfig(batch_size=32, num_workers=2, pin_memory=True),
            DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
            DataLoaderConfig(batch_size=128, num_workers=8, pin_memory=True)
        ]
        
        dataset_path = self.demo_dir / "data" / "threat_detection.csv"
        dataset = ThreatDetectionDataset(str(dataset_path), DataLoaderConfig())
        
        for i, config in enumerate(configs):
            print(f"Benchmarking configuration {i+1}: batch_size={config.batch_size}, workers={config.num_workers}")
            
            # Create DataLoader
            dataloader = DataLoaderFactory.create_dataloader(dataset, config, "threat_detection")
            
            # Run benchmark
            benchmark_results = DataLoaderBenchmark.benchmark_dataloader(
                dataloader, num_batches=50, warmup_batches=5
            )
            
            # Store results
            self.results[f"benchmark_config_{i+1}"] = {
                "config": config.__dict__,
                "benchmark_results": benchmark_results
            }
            
            print(f"   ✅ Throughput: {benchmark_results['throughput_batches_per_sec']:.2f} batches/sec")
            print(f"   ✅ Avg batch time: {benchmark_results['avg_batch_time_ms']:.2f} ms")
            print(f"   ✅ Memory usage: {benchmark_results['avg_memory_usage_percent']:.1f}%")
    
    async def _demo_memory_optimization(self) -> Any:
        """Demo memory optimization features."""
        print("\n💾 Demo 3: Memory Optimization")
        print("-" * 50)
        
        # Create dataset
        dataset_path = self.demo_dir / "data" / "anomaly_detection.csv"
        config = DataLoaderConfig(
            batch_size=64,
            num_workers=4,
            pin_memory=True
        )
        
        dataset = AnomalyDetectionDataset(str(dataset_path), config)
        dataloader = DataLoaderFactory.create_dataloader(dataset, config, "anomaly_detection")
        
        # Test regular DataLoader
        print("Testing regular DataLoader...")
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            if batch_count >= 20:
                break
        
        memory_after = self._get_memory_usage()
        regular_time = time.time() - start_time
        
        # Test memory-optimized DataLoader
        print("Testing memory-optimized DataLoader...")
        memory_optimized_dataloader = MemoryOptimizedDataLoader(dataloader, max_memory_usage=0.7)
        
        start_time = time.time()
        memory_before_opt = self._get_memory_usage()
        
        batch_count_opt = 0
        for batch in memory_optimized_dataloader:
            batch_count_opt += 1
            if batch_count_opt >= 20:
                break
        
        memory_after_opt = self._get_memory_usage()
        optimized_time = time.time() - start_time
        
        # Get performance report
        performance_report = memory_optimized_dataloader.get_performance_report()
        
        # Store results
        self.results["memory_optimization"] = {
            "regular_dataloader": {
                "time_seconds": regular_time,
                "memory_increase_mb": memory_after - memory_before,
                "batch_count": batch_count
            },
            "memory_optimized_dataloader": {
                "time_seconds": optimized_time,
                "memory_increase_mb": memory_after_opt - memory_before_opt,
                "batch_count": batch_count_opt,
                "performance_report": performance_report
            }
        }
        
        print(f"✅ Regular DataLoader: {regular_time:.2f}s, Memory: +{memory_after - memory_before:.1f}MB")
        print(f"✅ Optimized DataLoader: {optimized_time:.2f}s, Memory: +{memory_after_opt - memory_before_opt:.1f}MB")
    
    async def _demo_caching_strategies(self) -> Any:
        """Demo caching strategies."""
        print("\n🗄️ Demo 4: Caching Strategies")
        print("-" * 50)
        
        # Create dataset
        dataset_path = self.demo_dir / "data" / "threat_detection.csv"
        config = DataLoaderConfig(
            batch_size=32,
            num_workers=2,
            enable_caching=True,
            cache_dir=str(self.demo_dir / "cache")
        )
        
        dataset = ThreatDetectionDataset(str(dataset_path), config)
        
        # Test without caching
        print("Testing without caching...")
        config_no_cache = DataLoaderConfig(
            batch_size=32,
            num_workers=2,
            enable_caching=False
        )
        
        dataloader_no_cache = DataLoaderFactory.create_dataloader(
            dataset, config_no_cache, "threat_detection"
        )
        
        start_time = time.time()
        for batch in dataloader_no_cache:
            pass  # Just iterate through
        no_cache_time = time.time() - start_time
        
        # Test with caching
        print("Testing with caching...")
        dataloader_with_cache = DataLoaderFactory.create_dataloader(
            dataset, config, "threat_detection"
        )
        
        # First pass (cache miss)
        start_time = time.time()
        for batch in dataloader_with_cache:
            pass
        first_pass_time = time.time() - start_time
        
        # Second pass (cache hit)
        start_time = time.time()
        for batch in dataloader_with_cache:
            pass
        second_pass_time = time.time() - start_time
        
        # Get cache stats
        if hasattr(dataloader_with_cache.dataset, 'get_cache_stats'):
            cache_stats = dataloader_with_cache.dataset.get_cache_stats()
        else:
            cache_stats = {"error": "Cache stats not available"}
        
        # Store results
        self.results["caching_strategies"] = {
            "no_cache_time": no_cache_time,
            "first_pass_time": first_pass_time,
            "second_pass_time": second_pass_time,
            "cache_stats": cache_stats,
            "speedup": no_cache_time / second_pass_time if second_pass_time > 0 else 0
        }
        
        print(f"✅ No cache: {no_cache_time:.2f}s")
        print(f"✅ First pass (cache miss): {first_pass_time:.2f}s")
        print(f"✅ Second pass (cache hit): {second_pass_time:.2f}s")
        print(f"✅ Speedup: {no_cache_time / second_pass_time:.2f}x")
    
    async def _demo_async_data_loading(self) -> Any:
        """Demo async data loading."""
        print("\n🔄 Demo 5: Async Data Loading")
        print("-" * 50)
        
        # Create dataset and DataLoader
        dataset_path = self.demo_dir / "data" / "network_traffic.csv"
        config = DataLoaderConfig(batch_size=32, num_workers=2)
        
        dataset = NetworkTrafficDataset(str(dataset_path), config)
        dataloader = DataLoaderFactory.create_dataloader(dataset, config, "network_traffic")
        
        # Create async DataLoader
        async_dataloader = AsyncDataLoader(dataloader, max_queue_size=5)
        
        # Test async loading
        print("Testing async data loading...")
        start_time = time.time()
        
        batch_count = 0
        async for batch in async_dataloader:
            batch_count += 1
            if batch_count >= 10:
                break
        
        async_time = time.time() - start_time
        
        # Test synchronous loading for comparison
        print("Testing synchronous data loading...")
        start_time = time.time()
        
        batch_count_sync = 0
        for batch in dataloader:
            batch_count_sync += 1
            if batch_count_sync >= 10:
                break
        
        sync_time = time.time() - start_time
        
        # Store results
        self.results["async_data_loading"] = {
            "async_time": async_time,
            "sync_time": sync_time,
            "batch_count": batch_count,
            "speedup": sync_time / async_time if async_time > 0 else 0
        }
        
        print(f"✅ Async loading: {async_time:.2f}s")
        print(f"✅ Sync loading: {sync_time:.2f}s")
        print(f"✅ Speedup: {sync_time / async_time:.2f}x")
    
    async def _demo_cross_validation(self) -> Any:
        """Demo cross-validation and data splitting."""
        print("\n✂️ Demo 6: Cross-Validation")
        print("-" * 50)
        
        # Create dataset
        dataset_path = self.demo_dir / "data" / "malware.csv"
        config = DataLoaderConfig(batch_size=32, num_workers=2)
        
        dataset = MalwareDataset(str(dataset_path), config)
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset, train_ratio=0.7, val_ratio=0.15
        )
        
        print(f"Dataset splits:")
        print(f"   - Train: {len(train_dataset)} samples")
        print(f"   - Validation: {len(val_dataset)} samples")
        print(f"   - Test: {len(test_dataset)} samples")
        
        # Create DataLoaders for each split
        train_dataloader = DataLoaderFactory.create_dataloader(
            train_dataset, config, "malware"
        )
        val_dataloader = DataLoaderFactory.create_dataloader(
            val_dataset, config, "malware"
        )
        test_dataloader = DataLoaderFactory.create_dataloader(
            test_dataset, config, "malware"
        )
        
        # Test loading from each split
        splits = [
            ("train", train_dataloader),
            ("validation", val_dataloader),
            ("test", test_dataloader)
        ]
        
        split_results = {}
        for split_name, dataloader in splits:
            print(f"Testing {split_name} split...")
            
            start_time = time.time()
            batch_count = 0
            
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 5:
                    break
            
            load_time = time.time() - start_time
            
            split_results[split_name] = {
                "batch_count": batch_count,
                "load_time": load_time,
                "throughput": batch_count / load_time
            }
        
        # Store results
        self.results["cross_validation"] = {
            "split_sizes": {
                "train": len(train_dataset),
                "validation": len(val_dataset),
                "test": len(test_dataset)
            },
            "split_performance": split_results
        }
        
        print("✅ Cross-validation splits created and tested successfully")
    
    async def _demo_data_augmentation(self) -> Any:
        """Demo data augmentation."""
        print("\n🔄 Demo 7: Data Augmentation")
        print("-" * 50)
        
        # Test text augmentation
        print("Testing text augmentation...")
        original_texts = [
            "Suspicious network activity detected on port 80",
            "Malicious payload identified in email attachment",
            "Unauthorized access attempt from IP 192.168.1.100"
        ]
        
        augmented_texts = []
        for text in original_texts:
            augmented = DataAugmentation.augment_text(text, augmentation_prob=0.5)
            augmented_texts.append(augmented)
        
        # Test feature augmentation
        print("Testing feature augmentation...")
        original_features = np.random.random((10, 20))
        augmented_features = []
        
        for features in original_features:
            augmented = DataAugmentation.augment_features(features, noise_factor=0.01)
            augmented_features.append(augmented)
        
        # Store results
        self.results["data_augmentation"] = {
            "text_augmentation": {
                "original": original_texts,
                "augmented": augmented_texts
            },
            "feature_augmentation": {
                "original_shape": original_features.shape,
                "augmented_shape": np.array(augmented_features).shape,
                "feature_change_ratio": np.mean(np.abs(original_features - np.array(augmented_features)))
            }
        }
        
        print("✅ Text augmentation examples:")
        for i, (orig, aug) in enumerate(zip(original_texts, augmented_texts)):
            print(f"   {i+1}. Original: {orig}")
            print(f"      Augmented: {aug}")
        
        print("✅ Feature augmentation completed")
    
    async def _demo_system_optimization(self) -> Any:
        """Demo system resource optimization."""
        print("\n⚙️ Demo 8: System Resource Optimization")
        print("-" * 50)
        
        # Get system information
        
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        num_cpus = psutil.cpu_count()
        
        print(f"System resources:")
        print(f"   - Total memory: {total_memory_gb:.1f} GB")
        print(f"   - CPU cores: {num_cpus}")
        
        # Optimize configuration
        optimal_config = optimize_dataloader_config(
            dataset_size=10000,
            available_memory_gb=total_memory_gb * 0.8,  # Use 80% of available memory
            num_cpus=num_cpus
        )
        
        print(f"Optimal configuration:")
        print(f"   - Batch size: {optimal_config.batch_size}")
        print(f"   - Workers: {optimal_config.num_workers}")
        print(f"   - Pin memory: {optimal_config.pin_memory}")
        
        # Test with optimal configuration
        dataset_path = self.demo_dir / "data" / "threat_detection.csv"
        dataset = ThreatDetectionDataset(str(dataset_path), optimal_config)
        
        dataloader = DataLoaderFactory.create_dataloader(
            dataset, optimal_config, "threat_detection"
        )
        
        # Benchmark optimal configuration
        benchmark_results = DataLoaderBenchmark.benchmark_dataloader(
            dataloader, num_batches=30, warmup_batches=5
        )
        
        # Store results
        self.results["system_optimization"] = {
            "system_info": {
                "total_memory_gb": total_memory_gb,
                "num_cpus": num_cpus
            },
            "optimal_config": optimal_config.__dict__,
            "benchmark_results": benchmark_results
        }
        
        print(f"✅ Optimal configuration benchmark:")
        print(f"   - Throughput: {benchmark_results['throughput_batches_per_sec']:.2f} batches/sec")
        print(f"   - Avg batch time: {benchmark_results['avg_batch_time_ms']:.2f} ms")
        print(f"   - Memory usage: {benchmark_results['avg_memory_usage_percent']:.1f}%")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _save_demo_results(self) -> Any:
        """Save demo results to file."""
        results_file = self.demo_dir / "demo_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Convert results to JSON-serializable format
        serializable_results = json.loads(
            json.dumps(self.results, default=convert_numpy, indent=2)
        )
        
        with open(results_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📁 Demo results saved to: {results_file}")
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self) -> Any:
        """Generate a summary report of the demo."""
        report_file = self.demo_dir / "demo_summary.md"
        
        with open(report_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("# Efficient Data Loading Demo Summary\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Demo Overview\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("This demo showcases efficient data loading capabilities for cybersecurity applications.\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Key Features Demonstrated\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("1. **Basic Data Loading** - Loading different dataset types efficiently\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("2. **Performance Benchmarking** - Comparing different configurations\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("3. **Memory Optimization** - Reducing memory usage during training\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("4. **Caching Strategies** - Improving data access speed\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("5. **Async Data Loading** - Non-blocking data loading\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("6. **Cross-Validation** - Proper data splitting for ML\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("7. **Data Augmentation** - Increasing training data diversity\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("8. **System Optimization** - Automatic resource optimization\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Results Summary\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "threat_detection_basic_loading" in self.results:
                threat_results = self.results["threat_detection_basic_loading"]
                f.write(f"### Threat Detection Dataset\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Dataset size: {threat_results.get('dataset_size', 'N/A')}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Load time: {threat_results.get('load_time_seconds', 0):.2f} seconds\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Throughput: {threat_results.get('throughput_batches_per_sec', 0):.2f} batches/sec\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "benchmark_config_1" in self.results:
                f.write("### Performance Benchmarking\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                for i in range(1, 5):
                    key = f"benchmark_config_{i}"
                    if key in self.results:
                        results = self.results[key]["benchmark_results"]
                        config = self.results[key]["config"]
                        f.write(f"- Config {i} (batch_size={config['batch_size']}, workers={config['num_workers']}):\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        f.write(f"  - Throughput: {results['throughput_batches_per_sec']:.2f} batches/sec\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        f.write(f"  - Avg batch time: {results['avg_batch_time_ms']:.2f} ms\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "caching_strategies" in self.results:
                cache_results = self.results["caching_strategies"]
                f.write("### Caching Performance\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Speedup with caching: {cache_results.get('speedup', 0):.2f}x\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Cache hit rate: {cache_results.get('cache_stats', {}).get('hit_rate', 0):.2%}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "memory_optimization" in self.results:
                mem_results = self.results["memory_optimization"]
                f.write("### Memory Optimization\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Regular DataLoader memory increase: {mem_results['regular_dataloader']['memory_increase_mb']:.1f} MB\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Optimized DataLoader memory increase: {mem_results['memory_optimized_dataloader']['memory_increase_mb']:.1f} MB\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "system_optimization" in self.results:
                sys_results = self.results["system_optimization"]
                f.write("### System Optimization\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Optimal batch size: {sys_results['optimal_config']['batch_size']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Optimal workers: {sys_results['optimal_config']['num_workers']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Optimal throughput: {sys_results['benchmark_results']['throughput_batches_per_sec']:.2f} batches/sec\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Best Practices\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("1. **Use appropriate batch sizes** based on available memory\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("2. **Enable caching** for frequently accessed data\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("3. **Use multiple workers** for I/O-bound operations\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("4. **Monitor memory usage** and implement garbage collection\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("5. **Use async loading** for non-blocking operations\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("6. **Implement data augmentation** for better model generalization\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("7. **Optimize system resources** automatically\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("8. **Use proper data splitting** for cross-validation\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Files Generated\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `demo_results.json` - Complete demo results\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `data/` - Synthetic datasets\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `cache/` - Cached data\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `logs/` - Performance logs\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Next Steps\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("1. Integrate with your existing training pipeline\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("2. Customize configurations for your specific use case\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("3. Monitor performance in production\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("4. Implement additional optimizations as needed\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("5. Add more dataset types and augmentation strategies\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        print(f"📋 Summary report generated: {report_file}")


async def main():
    """Main demo function."""
    demo = EfficientDataLoadingDemo()
    await demo.run_comprehensive_demo()


match __name__:
    case "__main__":
    asyncio.run(main()) 