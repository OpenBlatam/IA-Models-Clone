# Frequency Analysis Improvements and Recommendations

## Overview

This document provides specific suggestions for enhancing the frequency analysis code to better serve its purpose of analyzing all frequency components in acceleration sensor data and encoder readings.

## Current Implementation Strengths

✅ **Comprehensive Type Annotations**: Full type hints throughout
✅ **Detailed Docstrings**: Complete documentation for all methods
✅ **Multiple Analysis Methods**: FFT and Welch's method support
✅ **Robust Error Handling**: Input validation and exception handling
✅ **Flexible Configuration**: Configurable window types, overlap ratios
✅ **Combined Analysis**: Cross-correlation between sensor and encoder data

## Performance Improvements

### 1. **Optimize FFT Computation**

**Current**: Uses standard FFT for all signals
**Recommendation**: Use Real FFT (rfft) for real-valued signals

```python
# Instead of:
fft_result = fft(signal_data, n=n)

# Use:
fft_result = rfft(signal_data, n=n)
frequencies = rfftfreq(n, 1.0 / self.sampling_rate)
```

**Benefits**:
- ~2x faster for real signals
- Uses half the memory
- Same frequency resolution

### 2. **Caching Window Functions**

**Current**: Window recomputed when parameters change
**Recommendation**: Cache windows for common sizes

```python
from functools import lru_cache

@lru_cache(maxsize=10)
def _get_window(self, size: int, window_type: str) -> np.ndarray:
    """Cached window function generation."""
    # ... window generation code
```

**Benefits**:
- Faster repeated analyses
- Reduced memory allocation

### 3. **Vectorized Operations**

**Current**: Some loops in peak finding
**Recommendation**: Use NumPy vectorized operations

```python
# Instead of loops, use:
peak_mask = (psd > min_height) & (np.diff(np.sign(np.diff(psd))) < 0)
peaks = np.where(peak_mask)[0]
```

**Benefits**:
- Faster execution
- More readable code

### 4. **Parallel Processing for Multi-Axis Data**

**Recommendation**: Process multiple axes in parallel

```python
from concurrent.futures import ThreadPoolExecutor

def analyze_multi_axis_parallel(self, accel_3d: np.ndarray) -> List[FrequencyAnalysisResult]:
    """Analyze all axes in parallel."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(
            lambda axis: self.analyze_acceleration(accel_3d, axis=axis),
            range(3)
        ))
    return results
```

**Benefits**:
- 3x speedup for 3-axis data
- Better CPU utilization

## Functionality Enhancements

### 1. **Short-Time Fourier Transform (STFT)**

**Recommendation**: Add STFT for time-frequency analysis

```python
def analyze_stft(self, signal_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Perform Short-Time Fourier Transform for time-frequency analysis.
    
    Useful for:
    - Non-stationary signals
    - Frequency changes over time
    - Transient events
    """
    f, t, Zxx = signal.stft(
        signal_data,
        fs=self.sampling_rate,
        window=self.window_type,
        nperseg=self.nperseg,
        noverlap=int(self.nperseg * self.overlap_ratio)
    )
    return {
        'frequencies': f,
        'time': t,
        'spectrogram': np.abs(Zxx)
    }
```

**Use Cases**:
- Analyzing changing frequencies during motion
- Detecting transient events
- Vibration analysis during acceleration/deceleration

### 2. **Continuous Wavelet Transform (CWT)**

**Recommendation**: Add CWT for multi-resolution analysis

```python
from scipy import signal

def analyze_cwt(self, signal_data: np.ndarray, 
                scales: np.ndarray = None) -> Dict[str, np.ndarray]:
    """
    Perform Continuous Wavelet Transform.
    
    Advantages:
    - Better time-frequency resolution
    - Detects both high and low frequency components
    - Good for non-stationary signals
    """
    if scales is None:
        scales = np.arange(1, 128)
    
    widths = scales * self.sampling_rate / (2 * np.pi * 5.0)  # Morlet wavelet
    cwt_matrix = signal.cwt(signal_data, signal.morlet2, widths)
    
    return {
        'scales': scales,
        'cwt': cwt_matrix,
        'frequencies': self.sampling_rate / (2 * np.pi * scales)
    }
```

**Use Cases**:
- Multi-scale frequency analysis
- Detecting frequency modulations
- Analyzing complex vibration patterns

### 3. **Advanced Harmonic Analysis**

**Recommendation**: Enhanced harmonic detection with phase relationships

```python
def _identify_harmonics_advanced(
    self, components: List[FrequencyComponent]
) -> Dict[str, Any]:
    """
    Advanced harmonic analysis with phase relationships.
    
    Returns:
        - Fundamental frequency
        - Harmonic series with phase relationships
        - Harmonic distortion metrics
        - Total Harmonic Distortion (THD)
    """
    # ... implementation with phase analysis
    thd = np.sqrt(sum(h.power for h in harmonics)) / fundamental.power
    return {
        'fundamental': fundamental,
        'harmonics': harmonics,
        'thd': thd,
        'phase_relationships': phase_diffs
    }
```

### 4. **Adaptive Filtering**

**Recommendation**: Adaptive filters that adjust to signal characteristics

```python
def _apply_adaptive_filter(self, data: np.ndarray) -> np.ndarray:
    """
    Apply adaptive filter based on signal characteristics.
    
    - Detects noise level automatically
    - Adjusts cutoff frequencies
    - Removes artifacts while preserving signal
    """
    # Estimate noise level
    noise_level = np.median(np.abs(np.diff(data)))
    
    # Adaptive cutoff based on signal power
    signal_power = np.var(data)
    cutoff_ratio = min(0.5, max(0.1, signal_power / (signal_power + noise_level)))
    
    cutoff = self.sampling_rate * cutoff_ratio / 2
    return self._apply_lowpass_filter(data, cutoff)
```

### 5. **Peak Detection Improvements**

**Recommendation**: More sophisticated peak detection

```python
def _find_dominant_frequencies_advanced(
    self, frequencies: np.ndarray, psd: np.ndarray
) -> List[FrequencyComponent]:
    """
    Advanced peak detection with:
    - Prominence-based filtering
    - Width-based filtering
    - Peak quality scoring
    """
    peaks, properties = find_peaks(
        psd,
        height=np.max(psd) * 0.1,
        prominence=np.max(psd) * 0.05,
        width=len(psd) / 1000,  # Minimum width
        distance=len(psd) / 50   # Minimum distance between peaks
    )
    
    # Score peaks by quality
    peak_scores = (
        properties['prominences'] * 
        properties['widths'] * 
        psd[peaks]
    )
    
    # Sort by quality score
    sorted_indices = np.argsort(peak_scores)[::-1]
    # ... return top quality peaks
```

## Readability Improvements

### 1. **Configuration Class**

**Recommendation**: Use dataclass for configuration

```python
@dataclass
class FrequencyAnalysisConfig:
    """Configuration for frequency analysis."""
    sampling_rate: float
    method: FrequencyAnalysisMethod = FrequencyAnalysisMethod.WELCH
    window_type: str = 'hann'
    nperseg: Optional[int] = None
    overlap_ratio: float = 0.5
    filter_enabled: bool = True
    remove_dc: bool = True
    min_frequency: float = 0.0
    max_frequency: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        # ... other validations
```

### 2. **Result Visualization**

**Recommendation**: Add plotting methods

```python
def plot_spectrum(self, result: FrequencyAnalysisResult) -> None:
    """Plot frequency spectrum."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot PSD
    axes[0].plot(
        result.power_spectral_density[:, 0],
        result.power_spectral_density[:, 1]
    )
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power Spectral Density')
    axes[0].set_title('Power Spectral Density')
    axes[0].grid(True)
    
    # Plot amplitude spectrum
    axes[1].plot(
        result.frequency_spectrum[:, 0],
        result.frequency_spectrum[:, 1]
    )
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Frequency Spectrum')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
```

### 3. **Batch Processing**

**Recommendation**: Process multiple signals efficiently

```python
def analyze_batch(
    self,
    signals: List[np.ndarray],
    signal_types: List[str]
) -> List[FrequencyAnalysisResult]:
    """
    Analyze multiple signals in batch.
    
    Args:
        signals: List of signal arrays
        signal_types: List of signal types ('acceleration' or 'encoder')
    
    Returns:
        List of analysis results
    """
    results = []
    for signal_data, sig_type in zip(signals, signal_types):
        if sig_type == 'acceleration':
            result = self.analyze_acceleration(signal_data)
        elif sig_type == 'encoder':
            result = self.analyze_encoder(signal_data)
        else:
            raise ValueError(f"Unknown signal type: {sig_type}")
        results.append(result)
    return results
```

## Specific Recommendations for Robot Movement Analysis

### 1. **Motion-Specific Frequency Bands**

**Recommendation**: Define frequency bands for different motion types

```python
class MotionFrequencyBands:
    """Frequency bands for different robot motion types."""
    WALKING = (0.5, 3.0)      # Hz - Leg motion
    RUNNING = (2.0, 5.0)      # Hz - Faster leg motion
    ARM_MOTION = (0.1, 2.0)   # Hz - Arm movements
    VIBRATION = (10.0, 100.0) # Hz - Mechanical vibrations
    NOISE = (100.0, 500.0)    # Hz - High-frequency noise
```

### 2. **Real-Time Analysis**

**Recommendation**: Streaming analysis for real-time applications

```python
class StreamingFrequencyAnalyzer:
    """Analyze frequency components in streaming data."""
    
    def __init__(self, window_size: int, overlap: int):
        self.window_size = window_size
        self.overlap = overlap
        self.buffer = np.zeros(window_size)
    
    def update(self, new_samples: np.ndarray) -> FrequencyAnalysisResult:
        """Update with new samples and return current analysis."""
        # Shift buffer and add new samples
        self.buffer[:-len(new_samples)] = self.buffer[len(new_samples):]
        self.buffer[-len(new_samples):] = new_samples
        
        # Analyze current window
        return self.analyzer.analyze_acceleration(self.buffer)
```

### 3. **Anomaly Detection**

**Recommendation**: Detect anomalous frequency patterns

```python
def detect_anomalies(
    self,
    result: FrequencyAnalysisResult,
    baseline: FrequencyAnalysisResult
) -> Dict[str, Any]:
    """
    Detect anomalies by comparing to baseline.
    
    Returns:
        - Anomaly score
        - Unusual frequencies
        - Frequency shifts
    """
    # Compare dominant frequencies
    baseline_freqs = {c.frequency for c in baseline.dominant_frequencies}
    current_freqs = {c.frequency for c in result.dominant_frequencies}
    
    new_freqs = current_freqs - baseline_freqs
    missing_freqs = baseline_freqs - current_freqs
    
    # Calculate anomaly score
    freq_diff = abs(result.fundamental_frequency - baseline.fundamental_frequency)
    power_diff = abs(result.total_power - baseline.total_power) / baseline.total_power
    
    anomaly_score = (len(new_freqs) + len(missing_freqs)) * 0.3 + \
                    freq_diff * 0.4 + power_diff * 0.3
    
    return {
        'anomaly_score': anomaly_score,
        'new_frequencies': list(new_freqs),
        'missing_frequencies': list(missing_freqs),
        'frequency_shift': freq_diff,
        'power_change': power_diff
    }
```

## Summary of Priority Improvements

### High Priority (Immediate Impact)
1. ✅ Use Real FFT (rfft) for performance
2. ✅ Add STFT for time-frequency analysis
3. ✅ Improve peak detection with prominence
4. ✅ Add visualization methods

### Medium Priority (Enhanced Functionality)
1. ⚠️ Add CWT for multi-resolution analysis
2. ⚠️ Implement adaptive filtering
3. ⚠️ Add batch processing
4. ⚠️ Enhance harmonic analysis

### Low Priority (Nice to Have)
1. 📋 Add streaming analysis
2. 📋 Implement anomaly detection
3. 📋 Add motion-specific frequency bands
4. 📋 Parallel processing for multi-axis

## Testing Recommendations

1. **Unit Tests**: Test each method with known signals
2. **Integration Tests**: Test with real sensor data
3. **Performance Tests**: Benchmark against reference implementations
4. **Accuracy Tests**: Compare results with MATLAB/Octave implementations

## Conclusion

The current implementation provides a solid foundation for frequency analysis. The suggested improvements will enhance:
- **Performance**: 2-3x speedup with optimizations
- **Functionality**: More analysis methods and features
- **Usability**: Better visualization and batch processing
- **Robustness**: Better peak detection and filtering

All improvements maintain backward compatibility while adding new capabilities.
