"""
Frequency Analyzer Usage Examples
==================================

This file demonstrates how to use the FrequencyAnalyzer class to analyze
frequency components in acceleration and encoder data.
"""

import numpy as np
from frequency_analyzer import FrequencyAnalyzer, FrequencyAnalysisMethod
import matplotlib.pyplot as plt
from typing import Tuple


def generate_test_acceleration_data(
    duration: float = 10.0,
    sampling_rate: float = 1000.0,
    frequencies: Tuple[float, ...] = (5.0, 15.0, 30.0),
    amplitudes: Tuple[float, ...] = (1.0, 0.5, 0.3),
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate synthetic acceleration data for testing.
    
    Args:
        duration: Duration of signal in seconds
        sampling_rate: Sampling rate in Hz
        frequencies: Frequencies to include in signal (Hz)
        amplitudes: Amplitudes for each frequency
        noise_level: Standard deviation of noise
    
    Returns:
        Array of acceleration values
    """
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal = np.zeros_like(t)
    
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    # Add noise
    noise = np.random.normal(0, noise_level, len(signal))
    signal += noise
    
    return signal


def generate_test_encoder_data(
    duration: float = 10.0,
    sampling_rate: float = 1000.0,
    base_frequency: float = 2.0,
    amplitude: float = 1.0,
    drift_rate: float = 0.01
) -> np.ndarray:
    """
    Generate synthetic encoder data for testing.
    
    Args:
        duration: Duration of signal in seconds
        sampling_rate: Sampling rate in Hz
        base_frequency: Base frequency of motion (Hz)
        amplitude: Amplitude of motion
        drift_rate: Linear drift rate
    
    Returns:
        Array of encoder position values
    """
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Periodic motion with harmonics
    signal = amplitude * (
        np.sin(2 * np.pi * base_frequency * t) +
        0.3 * np.sin(2 * np.pi * base_frequency * 2 * t) +
        0.1 * np.sin(2 * np.pi * base_frequency * 3 * t)
    )
    
    # Add linear drift
    signal += drift_rate * t
    
    # Add small noise
    noise = np.random.normal(0, 0.05, len(signal))
    signal += noise
    
    return signal


def example_basic_acceleration_analysis():
    """Example: Basic acceleration frequency analysis."""
    print("\n=== Basic Acceleration Analysis ===")
    
    # Generate test data
    sampling_rate = 1000.0
    accel_data = generate_test_acceleration_data(
        duration=10.0,
        sampling_rate=sampling_rate,
        frequencies=(5.0, 15.0, 30.0),
        amplitudes=(1.0, 0.5, 0.3)
    )
    
    # Create analyzer
    analyzer = FrequencyAnalyzer(
        sampling_rate=sampling_rate,
        method=FrequencyAnalysisMethod.WELCH
    )
    
    # Analyze acceleration
    result = analyzer.analyze_acceleration(
        accel_data,
        remove_dc=True,
        apply_filter=True
    )
    
    # Print results
    print(f"Fundamental Frequency: {result.fundamental_frequency:.2f} Hz")
    print(f"Total Power: {result.total_power:.4f}")
    print(f"SNR: {result.signal_to_noise_ratio:.2f} dB")
    print(f"Bandwidth: {result.bandwidth:.2f} Hz")
    print(f"\nDominant Frequencies:")
    for i, comp in enumerate(result.dominant_frequencies[:5], 1):
        print(f"  {i}. {comp.frequency:.2f} Hz - "
              f"Amplitude: {comp.amplitude:.4f}, Power: {comp.power:.4e}")
    
    if result.harmonics:
        print(f"\nHarmonics Found: {len(result.harmonics)}")
        for harm in result.harmonics[:3]:
            print(f"  Harmonic {harm.harmonic_number}: "
                  f"{harm.frequency:.2f} Hz")


def example_encoder_analysis():
    """Example: Encoder frequency analysis."""
    print("\n=== Encoder Analysis ===")
    
    sampling_rate = 1000.0
    encoder_data = generate_test_encoder_data(
        duration=10.0,
        sampling_rate=sampling_rate,
        base_frequency=2.0
    )
    
    analyzer = FrequencyAnalyzer(
        sampling_rate=sampling_rate,
        method=FrequencyAnalysisMethod.WELCH
    )
    
    result = analyzer.analyze_encoder(
        encoder_data,
        data_type='position',
        remove_trend=True,
        apply_filter=True
    )
    
    print(f"Fundamental Frequency: {result.fundamental_frequency:.2f} Hz")
    print(f"Total Power: {result.total_power:.4f}")
    print(f"\nDominant Frequencies:")
    for i, comp in enumerate(result.dominant_frequencies[:5], 1):
        print(f"  {i}. {comp.frequency:.2f} Hz - "
              f"Amplitude: {comp.amplitude:.4f}")


def example_combined_analysis():
    """Example: Combined acceleration and encoder analysis."""
    print("\n=== Combined Analysis ===")
    
    sampling_rate = 1000.0
    
    # Generate correlated data
    t = np.linspace(0, 10.0, int(sampling_rate * 10.0))
    common_freq = 5.0
    
    accel_data = np.sin(2 * np.pi * common_freq * t) + \
                 0.5 * np.sin(2 * np.pi * common_freq * 2 * t) + \
                 np.random.normal(0, 0.1, len(t))
    
    encoder_data = np.sin(2 * np.pi * common_freq * t + np.pi/4) + \
                   0.3 * np.sin(2 * np.pi * common_freq * 2 * t) + \
                   0.01 * t + np.random.normal(0, 0.05, len(t))
    
    analyzer = FrequencyAnalyzer(sampling_rate=sampling_rate)
    
    combined_result = analyzer.analyze_combined(
        accel_data,
        encoder_data,
        correlation_threshold=0.7
    )
    
    print("Acceleration Analysis:")
    accel_result = combined_result['acceleration']
    print(f"  Fundamental: {accel_result.fundamental_frequency:.2f} Hz")
    
    print("\nEncoder Analysis:")
    encoder_result = combined_result['encoder']
    print(f"  Fundamental: {encoder_result.fundamental_frequency:.2f} Hz")
    
    print("\nCommon Frequencies:")
    for common in combined_result['common_frequencies'][:5]:
        print(f"  {common['frequency']:.2f} Hz - "
              f"Diff: {common['frequency_difference']:.4f} Hz")
    
    print("\nCross-Correlation:")
    corr = combined_result['cross_correlation']
    print(f"  Max Correlation: {corr['max_correlation']:.4f}")
    print(f"  Lag: {corr['lag_time_seconds']:.4f} seconds")


def example_multi_axis_acceleration():
    """Example: Multi-axis acceleration analysis."""
    print("\n=== Multi-Axis Acceleration Analysis ===")
    
    sampling_rate = 1000.0
    duration = 10.0
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Generate 3-axis acceleration data
    accel_3d = np.zeros((len(t), 3))
    accel_3d[:, 0] = np.sin(2 * np.pi * 5.0 * t)  # X-axis: 5 Hz
    accel_3d[:, 1] = np.sin(2 * np.pi * 10.0 * t)  # Y-axis: 10 Hz
    accel_3d[:, 2] = np.sin(2 * np.pi * 15.0 * t)  # Z-axis: 15 Hz
    
    analyzer = FrequencyAnalyzer(sampling_rate=sampling_rate)
    
    # Analyze magnitude (all axes combined)
    result_magnitude = analyzer.analyze_acceleration(
        accel_3d,
        axis=None,  # Analyze magnitude
        remove_dc=True,
        apply_filter=True
    )
    
    print("Magnitude Analysis (All Axes):")
    print(f"  Fundamental: {result_magnitude.fundamental_frequency:.2f} Hz")
    
    # Analyze individual axes
    for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
        result = analyzer.analyze_acceleration(
            accel_3d,
            axis=axis_idx,
            remove_dc=True,
            apply_filter=True
        )
        print(f"\n{axis_name}-Axis Analysis:")
        print(f"  Fundamental: {result.fundamental_frequency:.2f} Hz")
        print(f"  Dominant Frequencies:")
        for comp in result.dominant_frequencies[:3]:
            print(f"    {comp.frequency:.2f} Hz")


def example_performance_comparison():
    """Example: Compare different analysis methods."""
    print("\n=== Performance Comparison ===")
    
    sampling_rate = 1000.0
    accel_data = generate_test_acceleration_data(
        duration=10.0,
        sampling_rate=sampling_rate
    )
    
    methods = [
        FrequencyAnalysisMethod.FFT,
        FrequencyAnalysisMethod.WELCH
    ]
    
    for method in methods:
        analyzer = FrequencyAnalyzer(
            sampling_rate=sampling_rate,
            method=method
        )
        
        result = analyzer.analyze_acceleration(accel_data)
        
        print(f"\n{method.value.upper()} Method:")
        print(f"  Fundamental: {result.fundamental_frequency:.2f} Hz")
        print(f"  SNR: {result.signal_to_noise_ratio:.2f} dB")
        print(f"  Bandwidth: {result.bandwidth:.2f} Hz")


if __name__ == "__main__":
    print("Frequency Analyzer - Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_acceleration_analysis()
    example_encoder_analysis()
    example_combined_analysis()
    example_multi_axis_acceleration()
    example_performance_comparison()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


