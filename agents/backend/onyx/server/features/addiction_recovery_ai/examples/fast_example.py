"""
Fast Inference Example - Addiction Recovery AI
Demonstrates speed optimizations
"""

import torch
import time
from addiction_recovery_ai import (
    create_fast_analyzer,
    create_ultra_fast_engine
)


def benchmark_progress_prediction(analyzer, num_runs=100):
    """Benchmark progress prediction"""
    features = {
        "days_sober": 30,
        "cravings_level": 3,
        "stress_level": 4,
        "support_level": 8,
        "mood_score": 7,
        "sleep_quality": 6,
        "exercise_frequency": 3,
        "therapy_sessions": 2,
        "medication_compliance": 1.0,
        "social_activity": 4
    }
    
    # Warmup
    for _ in range(10):
        _ = analyzer.predict_progress(features)
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        _ = analyzer.predict_progress(features)
    elapsed = time.time() - start
    
    avg_time = (elapsed / num_runs) * 1000  # ms
    throughput = num_runs / elapsed
    
    print(f"Progress Prediction:")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Throughput: {throughput:.2f} predictions/s")
    return avg_time


def benchmark_relapse_risk(analyzer, num_runs=50):
    """Benchmark relapse risk prediction"""
    sequence = [
        {
            "cravings_level": 3,
            "stress_level": 4,
            "mood_score": 7,
            "triggers_count": 1,
            "consumed": 0.0
        }
    ] * 30  # 30 days
    
    # Warmup
    for _ in range(5):
        _ = analyzer.predict_relapse_risk(sequence)
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        _ = analyzer.predict_relapse_risk(sequence)
    elapsed = time.time() - start
    
    avg_time = (elapsed / num_runs) * 1000  # ms
    throughput = num_runs / elapsed
    
    print(f"Relapse Risk Prediction:")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Throughput: {throughput:.2f} predictions/s")
    return avg_time


def benchmark_sentiment(analyzer, num_runs=50):
    """Benchmark sentiment analysis"""
    text = "I'm feeling great today and making good progress!"
    
    # Warmup
    for _ in range(5):
        _ = analyzer.analyze_sentiment(text)
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        _ = analyzer.analyze_sentiment(text)
    elapsed = time.time() - start
    
    avg_time = (elapsed / num_runs) * 1000  # ms
    throughput = num_runs / elapsed
    
    print(f"Sentiment Analysis:")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Throughput: {throughput:.2f} predictions/s")
    return avg_time


def benchmark_batch_processing(engine, num_sequences=32):
    """Benchmark batch processing"""
    sequences = []
    for i in range(num_sequences):
        seq = [
            {
                "cravings_level": 3 + (i % 3),
                "stress_level": 4 + (i % 2),
                "mood_score": 7 - (i % 2),
                "triggers_count": i % 3,
                "consumed": 0.0
            }
        ] * 30
        sequences.append(seq)
    
    # Warmup
    _ = engine.predict_relapse_batch(sequences[:5])
    
    # Benchmark
    start = time.time()
    results = engine.predict_relapse_batch(sequences)
    elapsed = time.time() - start
    
    avg_time = (elapsed / len(sequences)) * 1000  # ms per item
    total_time = elapsed * 1000  # ms
    throughput = len(sequences) / elapsed
    
    print(f"Batch Processing ({num_sequences} sequences):")
    print(f"  Total time: {total_time:.2f}ms")
    print(f"  Average per item: {avg_time:.2f}ms")
    print(f"  Throughput: {throughput:.2f} predictions/s")
    return avg_time


def main():
    """Main benchmark function"""
    print("=" * 60)
    print("Fast Analyzer Benchmark")
    print("=" * 60)
    
    # Create fast analyzer
    print("\n1. Fast Analyzer:")
    fast_analyzer = create_fast_analyzer(use_gpu=torch.cuda.is_available())
    
    benchmark_progress_prediction(fast_analyzer)
    benchmark_relapse_risk(fast_analyzer)
    benchmark_sentiment(fast_analyzer)
    
    print("\n" + "=" * 60)
    print("Ultra-Fast Engine Benchmark")
    print("=" * 60)
    
    # Create ultra-fast engine
    print("\n2. Ultra-Fast Engine:")
    ultra_engine = create_ultra_fast_engine(use_gpu=torch.cuda.is_available())
    
    benchmark_progress_prediction(ultra_engine)
    benchmark_relapse_risk(ultra_engine)
    benchmark_sentiment(ultra_engine)
    
    # Batch processing
    print("\n3. Batch Processing:")
    benchmark_batch_processing(ultra_engine, num_sequences=32)
    
    # Built-in benchmark
    print("\n4. Built-in Benchmark:")
    results = ultra_engine.benchmark(num_runs=100)
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Average time: {metrics['avg_time_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput']:.2f} predictions/s")
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

