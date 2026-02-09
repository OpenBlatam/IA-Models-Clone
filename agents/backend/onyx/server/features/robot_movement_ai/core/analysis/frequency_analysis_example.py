"""
Ejemplo de Uso - Frequency Analyzer
====================================

Demuestra cómo usar el FrequencyAnalyzer para analizar
datos de aceleración de sensores y lecturas de encoders.
"""

import numpy as np
from core.analysis.frequency_analyzer import FrequencyAnalyzer, FrequencyAnalysisResult


def example_acceleration_analysis():
    """Ejemplo de análisis de datos de aceleración."""
    
    analyzer = FrequencyAnalyzer(
        sampling_rate=1000.0,
        window_type="hann",
        min_frequency=1.0,
        max_frequency=500.0
    )
    
    n_samples = 10000
    time = np.linspace(0, 10, n_samples)
    
    signal_1 = 2.0 * np.sin(2 * np.pi * 10 * time)
    signal_2 = 1.5 * np.sin(2 * np.pi * 25 * time)
    signal_3 = 1.0 * np.sin(2 * np.pi * 50 * time)
    noise = 0.1 * np.random.randn(n_samples)
    
    acceleration_3d = np.column_stack([
        signal_1 + noise,
        signal_2 + noise,
        signal_3 + noise
    ])
    
    result = analyzer.analyze_acceleration_data(
        acceleration_3d,
        axis=None
    )
    
    print("Análisis de Aceleración:")
    print(f"Frecuencia pico: {result.peak_frequency:.2f} Hz")
    print(f"Ancho de banda: {result.bandwidth:.2f} Hz")
    print(f"Potencia total: {result.total_power:.2f}")
    print(f"\nFrecuencias dominantes:")
    for i, freq in enumerate(result.dominant_frequencies[:5], 1):
        print(f"  {i}. {freq:.2f} Hz")
    
    print(f"\nComponentes de frecuencia significativos: {len(result.frequency_components)}")
    for comp in result.frequency_components[:5]:
        print(f"  Frecuencia: {comp.frequency:.2f} Hz, "
              f"Potencia: {comp.power:.2f}, "
              f"Relativa: {comp.relative_power:.2%}")
    
    return result


def example_encoder_analysis():
    """Ejemplo de análisis de lecturas de encoder."""
    
    analyzer = FrequencyAnalyzer(
        sampling_rate=1000.0,
        window_type="hann"
    )
    
    n_samples = 5000
    time = np.linspace(0, 5, n_samples)
    
    encoder_signal = (
        5.0 * np.sin(2 * np.pi * 15 * time) +
        3.0 * np.sin(2 * np.pi * 30 * time) +
        0.5 * np.random.randn(n_samples)
    )
    
    result = analyzer.analyze_encoder_readings(
        encoder_signal,
        detrend=True,
        normalize=False
    )
    
    print("\nAnálisis de Encoder:")
    print(f"Frecuencia pico: {result.peak_frequency:.2f} Hz")
    print(f"Ancho de banda: {result.bandwidth:.2f} Hz")
    print(f"Potencia total: {result.total_power:.2f}")
    print(f"\nFrecuencias dominantes:")
    for i, freq in enumerate(result.dominant_frequencies[:5], 1):
        print(f"  {i}. {freq:.2f} Hz")
    
    return result


def example_comparison():
    """Ejemplo de comparación entre aceleración y encoder."""
    
    analyzer = FrequencyAnalyzer(sampling_rate=1000.0)
    
    accel_result = example_acceleration_analysis()
    encoder_result = example_encoder_analysis()
    
    comparison = analyzer.compare_frequency_analysis(
        accel_result,
        encoder_result
    )
    
    print("\nComparación:")
    print(f"Frecuencias comunes: {comparison['common_frequencies']}")
    print(f"Correlación: {comparison['frequency_correlation']:.3f}")
    print(f"Ratio de potencia: {comparison['power_ratio']:.3f}")
    print(f"Pico aceleración: {comparison['acceleration_peak']:.2f} Hz")
    print(f"Pico encoder: {comparison['encoder_peak']:.2f} Hz")


def example_export():
    """Ejemplo de exportación de resultados."""
    
    analyzer = FrequencyAnalyzer(sampling_rate=1000.0)
    
    n_samples = 2000
    time = np.linspace(0, 2, n_samples)
    signal = np.sin(2 * np.pi * 50 * time) + 0.3 * np.random.randn(n_samples)
    
    result = analyzer.analyze_encoder_readings(signal)
    
    analyzer.export_analysis(result, "frequency_analysis.json", format="json")
    analyzer.export_analysis(result, "frequency_analysis.csv", format="csv")
    analyzer.export_analysis(result, "frequency_analysis.npz", format="npy")
    
    print("\nResultados exportados a:")
    print("  - frequency_analysis.json")
    print("  - frequency_analysis.csv")
    print("  - frequency_analysis.npz")


if __name__ == "__main__":
    print("=" * 60)
    print("Ejemplos de Análisis de Frecuencia")
    print("=" * 60)
    
    example_acceleration_analysis()
    example_encoder_analysis()
    example_comparison()
    example_export()
    
    print("\n" + "=" * 60)
    print("Ejemplos completados")
    print("=" * 60)


