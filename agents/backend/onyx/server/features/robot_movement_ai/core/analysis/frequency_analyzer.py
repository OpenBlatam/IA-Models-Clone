"""
Frequency Analysis Module
=========================

This module provides comprehensive frequency analysis for acceleration data
from sensors and encoder readings. It analyzes all frequency components present
in the data using advanced signal processing techniques.

The main goal is to identify and characterize frequency components in:
1. Acceleration data from IMU/accelerometer sensors
2. Encoder scope readings (position/velocity measurements)

Core Features:
- Fast Fourier Transform (FFT) analysis with Real FFT optimization
- Power Spectral Density (PSD) calculation using Welch's method
- Short-Time Fourier Transform (STFT) for time-frequency analysis
- Continuous Wavelet Transform (CWT) for multi-scale analysis
- Dominant frequency identification with phase information
- Harmonic analysis and fundamental frequency detection
- Total Harmonic Distortion (THD) calculation
- Noise filtering and signal enhancement (bandpass, lowpass)
- Cross-correlation between sensor and encoder data
- Coherence analysis between signals
- Anomaly detection in frequency components
- Parallel processing for multi-axis data
- Export to multiple formats (JSON, CSV, NumPy, MATLAB)

Performance Optimizations:
- Real FFT (rfft) for real-valued signals (~2x faster)
- Window function caching for repeated analyses
- Parallel processing for multi-axis acceleration data
- Efficient memory usage and computation

Usage Example:
    >>> from core.analysis.frequency_analyzer import FrequencyAnalyzer
    >>> analyzer = FrequencyAnalyzer(sampling_rate=1000.0)
    >>> result = analyzer.analyze_acceleration(accel_data)
    >>> anomalies = analyzer.detect_anomalies(result)
    >>> analyzer.export_results(result, 'analysis.json')
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import welch, find_peaks, butter, filtfilt, stft, cwt, morlet2
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class FrequencyAnalysisMethod(Enum):
    """Enumeration of available frequency analysis methods."""
    FFT = "fft"
    WELCH = "welch"
    STFT = "stft"
    CWT = "cwt"


class MotionFrequencyBands:
    """
    Frequency bands for different robot motion types and mechanical phenomena.
    
    These predefined frequency bands are useful for categorizing and analyzing
    frequency content in acceleration and encoder data according to the type of
    motion or mechanical phenomena they represent. This classification helps in:
    - Identifying motion patterns (walking, running, arm movements)
    - Detecting mechanical issues (vibrations, noise)
    - Filtering frequency ranges of interest
    - Comparing frequency content across different motion types
    
    Attributes:
        WALKING: Frequency range for walking motion (0.5-3.0 Hz)
                 Typical leg motion frequency during normal walking
        RUNNING: Frequency range for running motion (2.0-5.0 Hz)
                 Faster leg motion frequency during running
        ARM_MOTION: Frequency range for arm movements (0.1-2.0 Hz)
                   Typical frequency of robotic arm movements
        VIBRATION: Frequency range for mechanical vibrations (10.0-100.0 Hz)
                  Mechanical vibrations from motors, gears, or structural resonance
        NOISE: Frequency range for high-frequency noise (100.0-500.0 Hz)
               Electrical noise, sensor noise, or high-frequency artifacts
        LOW_FREQUENCY: General low-frequency content (0.1-1.0 Hz)
                      Slow movements, drift, or low-frequency disturbances
        MEDIUM_FREQUENCY: General medium-frequency content (1.0-10.0 Hz)
                        Moderate speed movements and transitions
        HIGH_FREQUENCY: General high-frequency content (10.0-100.0 Hz)
                       Fast movements, vibrations, or rapid oscillations
    
    Example:
        >>> bands = MotionFrequencyBands()
        >>> walking_band = bands.WALKING  # (0.5, 3.0)
        >>> all_bands = bands.get_all_bands()
        >>> # Filter frequencies in walking band
        >>> walking_freqs = [f for f in frequencies 
        ...                  if walking_band[0] <= f <= walking_band[1]]
    """
    WALKING: Tuple[float, float] = (0.5, 3.0)      # Hz - Leg motion during walking
    RUNNING: Tuple[float, float] = (2.0, 5.0)      # Hz - Faster leg motion during running
    ARM_MOTION: Tuple[float, float] = (0.1, 2.0)   # Hz - Arm movements
    VIBRATION: Tuple[float, float] = (10.0, 100.0) # Hz - Mechanical vibrations
    NOISE: Tuple[float, float] = (100.0, 500.0)    # Hz - High-frequency noise
    LOW_FREQUENCY: Tuple[float, float] = (0.1, 1.0)    # Hz - General low-frequency content
    MEDIUM_FREQUENCY: Tuple[float, float] = (1.0, 10.0)  # Hz - General medium-frequency content
    HIGH_FREQUENCY: Tuple[float, float] = (10.0, 100.0)  # Hz - General high-frequency content
    
    @classmethod
    def get_all_bands(cls) -> Dict[str, Tuple[float, float]]:
        """
        Get all frequency bands as a dictionary.
        
        Returns a dictionary mapping human-readable band names to their
        corresponding frequency ranges (low, high) in Hz. This is useful
        for programmatic access to all available frequency bands.
        
        Returns:
            Dictionary mapping band names to (low, high) frequency tuples in Hz.
            Keys: 'walking', 'running', 'arm_motion', 'vibration', 'noise',
                  'low_frequency', 'medium_frequency', 'high_frequency'
        
        Example:
            >>> bands = MotionFrequencyBands.get_all_bands()
            >>> print(bands['walking'])  # (0.5, 3.0)
            >>> for name, (low, high) in bands.items():
            ...     print(f"{name}: {low}-{high} Hz")
        """
        return {
            'walking': cls.WALKING,
            'running': cls.RUNNING,
            'arm_motion': cls.ARM_MOTION,
            'vibration': cls.VIBRATION,
            'noise': cls.NOISE,
            'low_frequency': cls.LOW_FREQUENCY,
            'medium_frequency': cls.MEDIUM_FREQUENCY,
            'high_frequency': cls.HIGH_FREQUENCY
        }
    
    @classmethod
    def classify_frequency(
        cls,
        frequency: float
    ) -> List[str]:
        """
        Classify a frequency into one or more motion bands.
        
        A frequency may belong to multiple bands if bands overlap.
        This method identifies all bands that contain the given frequency.
        
        Args:
            frequency: Frequency value in Hz to classify
        
        Returns:
            List of band names that contain this frequency.
            Empty list if frequency doesn't match any band.
        
        Example:
            >>> bands = MotionFrequencyBands()
            >>> bands.classify_frequency(2.5)  # ['walking', 'running', 'medium_frequency']
            >>> bands.classify_frequency(50.0)  # ['vibration', 'high_frequency']
        """
        matching_bands: List[str] = []
        all_bands = cls.get_all_bands()
        
        for band_name, (low, high) in all_bands.items():
            if low <= frequency <= high:
                matching_bands.append(band_name)
        
        return matching_bands
    
    @classmethod
    def get_band_power(
        cls,
        result: 'FrequencyAnalysisResult',
        band_name: str
    ) -> float:
        """
        Calculate total power in a specific frequency band.
        
        Integrates the power spectral density over the specified frequency band
        to determine the total power content in that band. Useful for comparing
        power distribution across different motion types.
        
        Args:
            result: FrequencyAnalysisResult containing PSD data
            band_name: Name of the frequency band (e.g., 'walking', 'vibration')
        
        Returns:
            Total power in the specified band (integrated PSD)
        
        Raises:
            ValueError: If band_name is not recognized
        
        Example:
            >>> result = analyzer.analyze_acceleration(accel_data)
            >>> walking_power = MotionFrequencyBands.get_band_power(result, 'walking')
            >>> vibration_power = MotionFrequencyBands.get_band_power(result, 'vibration')
        """
        all_bands = cls.get_all_bands()
        
        if band_name not in all_bands:
            raise ValueError(
                f"Unknown band name '{band_name}'. "
                f"Available bands: {list(all_bands.keys())}"
            )
        
        low, high = all_bands[band_name]
        
        # Extract frequencies and PSD
        freqs = result.power_spectral_density[:, 0]
        psd = result.power_spectral_density[:, 1]
        
        # Find indices within band
        band_mask = (freqs >= low) & (freqs <= high)
        
        if not np.any(band_mask):
            return 0.0
        
        # Integrate PSD over band
        band_freqs = freqs[band_mask]
        band_psd = psd[band_mask]
        
        if len(band_freqs) > 1:
            return float(np.trapz(band_psd, band_freqs))
        else:
            return float(band_psd[0]) if len(band_psd) > 0 else 0.0
    
    @classmethod
    def get_power_distribution(
        cls,
        result: 'FrequencyAnalysisResult'
    ) -> Dict[str, float]:
        """
        Calculate power distribution across all frequency bands.
        
        This method provides a comprehensive view of how power is distributed
        across different motion frequency bands, useful for understanding
        the dominant motion types in the signal.
        
        Args:
            result: FrequencyAnalysisResult containing PSD data
        
        Returns:
            Dictionary mapping band names to their power percentages
            (0.0 to 100.0) relative to total power
        
        Example:
            >>> result = analyzer.analyze_acceleration(accel_data)
            >>> distribution = MotionFrequencyBands.get_power_distribution(result)
            >>> print(f"Walking: {distribution['walking']:.1f}%")
            >>> print(f"Vibration: {distribution['vibration']:.1f}%")
        """
        all_bands = cls.get_all_bands()
        distribution: Dict[str, float] = {}
        
        total_power = result.total_power
        
        if total_power <= 0:
            # Return zero distribution if no power
            return {band: 0.0 for band in all_bands.keys()}
        
        for band_name in all_bands.keys():
            band_power = cls.get_band_power(result, band_name)
            percentage = (band_power / total_power) * 100.0
            distribution[band_name] = float(percentage)
        
        return distribution


@dataclass
class FrequencyComponent:
    """
    Represents a single frequency component identified in the signal.
    
    Attributes:
        frequency: Frequency value in Hz
        amplitude: Amplitude/magnitude of the component
        phase: Phase angle in radians
        power: Power spectral density at this frequency
        harmonic_number: Harmonic number if this is a harmonic (1 = fundamental)
    """
    frequency: float
    amplitude: float
    phase: float
    power: float
    harmonic_number: Optional[int] = None


@dataclass
class FrequencyAnalysisResult:
    """
    Complete frequency analysis results for a signal.
    
    This dataclass contains all the results from frequency analysis including
    dominant frequencies, power spectral density, harmonics, and quality metrics.
    It provides a comprehensive representation of the frequency content in the
    analyzed signal.
    
    Attributes:
        dominant_frequencies: List of dominant frequency components identified
                            in the signal, sorted by power (descending)
        frequency_spectrum: 2D array of shape (n_frequencies, 2) containing
                           [frequencies, amplitudes] for the full spectrum
        power_spectral_density: 2D array of shape (n_frequencies, 2) containing
                               [frequencies, psd_values] for power spectral density
        total_power: Total power in the signal (integrated PSD)
        signal_to_noise_ratio: Estimated signal-to-noise ratio in dB
        fundamental_frequency: Fundamental frequency in Hz if detected, None otherwise
        harmonics: List of harmonic frequency components identified
        bandwidth: Effective bandwidth containing 90% of signal power
        sampling_rate: Sampling rate in Hz used for the analysis
    
    Methods:
        get_frequency_range(): Get the frequency range [min, max] in Hz
    """
    dominant_frequencies: List[FrequencyComponent]
    frequency_spectrum: np.ndarray
    power_spectral_density: np.ndarray
    total_power: float
    signal_to_noise_ratio: float
    fundamental_frequency: Optional[float]
    harmonics: List[FrequencyComponent]
    bandwidth: float
    sampling_rate: float
    
    def get_frequency_range(self) -> Tuple[float, float]:
        """
        Get the frequency range covered by the analysis.
        
        Returns the minimum and maximum frequencies present in the
        frequency spectrum. This is useful for understanding the
        frequency span of the analyzed signal.
        
        Returns:
            Tuple of (min_frequency, max_frequency) in Hz
        
        Example:
            >>> result = analyzer.analyze_acceleration(accel_data)
            >>> freq_min, freq_max = result.get_frequency_range()
            >>> print(f"Frequency range: {freq_min:.2f} - {freq_max:.2f} Hz")
        """
        if self.frequency_spectrum.size == 0:
            return (0.0, 0.0)
        
        frequencies = self.frequency_spectrum[:, 0]
        return (float(np.min(frequencies)), float(np.max(frequencies)))


class FrequencyAnalyzer:
    """
    Comprehensive frequency analyzer for sensor and encoder data.
    
    This class provides methods to analyze frequency components in:
    - Acceleration data from IMU/accelerometer sensors
    - Encoder readings (position, velocity, or angle measurements)
    
    The analyzer uses various signal processing techniques including FFT,
    Welch's method, and advanced filtering to extract meaningful frequency
    information from the input signals.
    """
    
    def __init__(
        self,
        sampling_rate: float,
        method: FrequencyAnalysisMethod = FrequencyAnalysisMethod.WELCH,
        window_type: str = 'hann',
        nperseg: Optional[int] = None,
        overlap_ratio: float = 0.5
    ) -> None:
        """
        Initialize the frequency analyzer.
        
        Args:
            sampling_rate: Sampling rate of the input data in Hz
            method: Frequency analysis method to use
            window_type: Type of window function for spectral analysis
                        ('hann', 'hamming', 'blackman', 'rectangular')
            nperseg: Length of each segment for Welch's method.
                    If None, uses sampling_rate / 10
            overlap_ratio: Overlap ratio between segments (0.0 to 1.0)
        
        Raises:
            ValueError: If sampling_rate <= 0 or invalid parameters
        """
        if sampling_rate <= 0:
            raise ValueError(f"Sampling rate must be positive, got {sampling_rate}")
        
        if not (0.0 <= overlap_ratio <= 1.0):
            raise ValueError(f"Overlap ratio must be between 0 and 1, got {overlap_ratio}")
        
        self.sampling_rate: float = sampling_rate
        self.method: FrequencyAnalysisMethod = method
        self.window_type: str = window_type
        self.nperseg: int = nperseg or int(sampling_rate / 10)
        self.overlap_ratio: float = overlap_ratio
        
        # Pre-compute window function for performance
        self._window: Optional[np.ndarray] = None
        self._update_window()
        
        logger.info(
            f"FrequencyAnalyzer initialized: "
            f"sampling_rate={sampling_rate}Hz, method={method.value}, "
            f"nperseg={self.nperseg}"
        )
    
    def _update_window(self) -> None:
        """
        Update the window function based on current configuration.
        
        This method creates or updates the window function used for spectral
        analysis. The window is pre-computed and cached to improve performance
        for repeated analyses with the same parameters.
        
        The window function is applied to the signal before FFT/Welch analysis
        to reduce spectral leakage and improve frequency resolution.
        
        Note:
            The window is only updated when window parameters change.
            For repeated analyses with the same window size, the cached
            window is reused for optimal performance.
        """
        """
        Update the window function based on current settings.
        
        Pre-computes the window function to improve performance during
        repeated analysis operations. The window is used to reduce spectral
        leakage in frequency domain analysis.
        
        Raises:
            ValueError: If nperseg is invalid
        """
        if self.nperseg <= 0:
            raise ValueError(f"nperseg must be positive, got {self.nperseg}")
        
        window_map: Dict[str, Callable[[int], np.ndarray]] = {
            'hann': signal.windows.hann,
            'hamming': signal.windows.hamming,
            'blackman': signal.windows.blackman,
            'bartlett': signal.windows.bartlett,
            'rectangular': lambda n: np.ones(n)
        }
        
        window_func = window_map.get(self.window_type.lower())
        if window_func is None:
            logger.warning(
                f"Unknown window type '{self.window_type}', using Hann window"
            )
            window_func = signal.windows.hann
        
        self._window = window_func(self.nperseg)
    
    def analyze_acceleration(
        self,
        acceleration_data: np.ndarray,
        axis: Optional[int] = None,
        remove_dc: bool = True,
        apply_filter: bool = True,
        filter_cutoff: Optional[Tuple[float, float]] = None
    ) -> FrequencyAnalysisResult:
        """
        Analyze frequency components in acceleration data.
        
        This method processes acceleration data from sensors (typically IMU)
        and extracts all significant frequency components. It can analyze
        individual axes or the magnitude of all axes.
        
        Args:
            acceleration_data: Array of acceleration values.
                              Shape: (n_samples,) for single axis or
                                     (n_samples, 3) for x, y, z axes
            axis: Specific axis to analyze (0=x, 1=y, 2=z).
                  If None, analyzes magnitude of all axes
            remove_dc: If True, removes DC component (mean) before analysis
            apply_filter: If True, applies bandpass filter to reduce noise
            filter_cutoff: Tuple of (low_cutoff, high_cutoff) in Hz for filter.
                          If None, uses (0.1, sampling_rate/2.5)
        
        Returns:
            FrequencyAnalysisResult containing all frequency analysis data
        
        Raises:
            ValueError: If acceleration_data is invalid
            RuntimeError: If analysis fails
        """
        if acceleration_data.size == 0:
            raise ValueError("Acceleration data cannot be empty")
        
        # Handle multi-axis data
        if acceleration_data.ndim == 2:
            if axis is not None:
                if axis < 0 or axis >= acceleration_data.shape[1]:
                    raise ValueError(f"Axis index {axis} out of range")
                signal_data = acceleration_data[:, axis]
            else:
                # Calculate magnitude
                signal_data = np.linalg.norm(acceleration_data, axis=1)
        elif acceleration_data.ndim == 1:
            signal_data = acceleration_data.copy()
        else:
            raise ValueError(
                f"Acceleration data must be 1D or 2D, got shape {acceleration_data.shape}"
            )
        
        # Remove DC component
        if remove_dc:
            signal_data = signal_data - np.mean(signal_data)
        
        # Apply filtering if requested
        if apply_filter:
            if filter_cutoff is None:
                low_cutoff = 0.1
                high_cutoff = self.sampling_rate / 2.5
            else:
                low_cutoff, high_cutoff = filter_cutoff
            
            signal_data = self._apply_bandpass_filter(
                signal_data, low_cutoff, high_cutoff
            )
        
        # Perform frequency analysis
        return self._analyze_signal(signal_data)
    
    def analyze_encoder(
        self,
        encoder_data: np.ndarray,
        data_type: str = 'position',
        remove_trend: bool = True,
        apply_filter: bool = True
    ) -> FrequencyAnalysisResult:
        """
        Analyze frequency components in encoder readings.
        
        This method processes encoder data (position, velocity, or angle)
        and extracts frequency components. Encoder data typically contains
        periodic patterns related to mechanical motion.
        
        Args:
            encoder_data: Array of encoder readings.
                         Can be position, velocity, or angle measurements
            data_type: Type of encoder data ('position', 'velocity', 'angle')
            remove_trend: If True, removes linear trend (drift) from data
            apply_filter: If True, applies low-pass filter to reduce noise
        
        Returns:
            FrequencyAnalysisResult containing all frequency analysis data
        
        Raises:
            ValueError: If encoder_data is invalid
            RuntimeError: If analysis fails
        """
        if encoder_data.size == 0:
            raise ValueError("Encoder data cannot be empty")
        
        if encoder_data.ndim != 1:
            raise ValueError(f"Encoder data must be 1D, got shape {encoder_data.shape}")
        
        signal_data = encoder_data.copy()
        
        # Remove linear trend (drift) if requested
        if remove_trend:
            x = np.arange(len(signal_data))
            coeffs = np.polyfit(x, signal_data, 1)
            trend = np.polyval(coeffs, x)
            signal_data = signal_data - trend
        
        # Apply low-pass filter for encoder data
        if apply_filter:
            # Encoder data typically has lower frequency content
            cutoff = self.sampling_rate / 4.0
            signal_data = self._apply_lowpass_filter(signal_data, cutoff)
        
        # Perform frequency analysis
        return self._analyze_signal(signal_data)
    
    def analyze_combined(
        self,
        acceleration_data: np.ndarray,
        encoder_data: np.ndarray,
        correlation_threshold: float = 0.7
    ) -> Dict[str, Union[FrequencyAnalysisResult, Dict[str, float]]]:
        """
        Analyze frequency components in both acceleration and encoder data.
        
        This method performs joint analysis of acceleration and encoder data,
        identifying common frequency components and cross-correlations.
        This is useful for understanding the relationship between sensor
        measurements and mechanical motion.
        
        Args:
            acceleration_data: Acceleration data array (1D or 2D)
            encoder_data: Encoder readings array (1D)
            correlation_threshold: Minimum correlation to consider frequencies
                                  as related (0.0 to 1.0)
        
        Returns:
            Dictionary containing:
            - 'acceleration': FrequencyAnalysisResult for acceleration
            - 'encoder': FrequencyAnalysisResult for encoder
            - 'common_frequencies': List of frequencies present in both
            - 'cross_correlation': Cross-correlation metrics
            - 'phase_relationship': Phase differences at common frequencies
        
        Raises:
            ValueError: If data arrays are invalid or mismatched
        """
        if acceleration_data.size == 0 or encoder_data.size == 0:
            raise ValueError("Both acceleration and encoder data must be non-empty")
        
        # Ensure same length
        min_length = min(len(acceleration_data), len(encoder_data))
        if len(acceleration_data) != len(encoder_data):
            logger.warning(
                f"Data length mismatch: acceleration={len(acceleration_data)}, "
                f"encoder={len(encoder_data)}. Truncating to {min_length}"
            )
        
        # Analyze each signal
        accel_result = self.analyze_acceleration(
            acceleration_data[:min_length],
            remove_dc=True,
            apply_filter=True
        )
        encoder_result = self.analyze_encoder(
            encoder_data[:min_length],
            remove_trend=True,
            apply_filter=True
        )
        
        # Find common frequencies
        common_frequencies = self._find_common_frequencies(
            accel_result, encoder_result, correlation_threshold
        )
        
        # Calculate cross-correlation
        cross_corr = self._calculate_cross_correlation(
            acceleration_data[:min_length],
            encoder_data[:min_length]
        )
        
        # Calculate phase relationships
        phase_relationship = self._calculate_phase_relationship(
            accel_result, encoder_result, common_frequencies
        )
        
        return {
            'acceleration': accel_result,
            'encoder': encoder_result,
            'common_frequencies': common_frequencies,
            'cross_correlation': cross_corr,
            'phase_relationship': phase_relationship
        }
    
    def _analyze_signal(self, signal_data: np.ndarray) -> FrequencyAnalysisResult:
        """
        Core signal analysis method.
        
        Routes the signal to the appropriate analysis method based on the
        configured analysis method. Validates input and handles edge cases.
        
        Args:
            signal_data: 1D array of signal values
        
        Returns:
            FrequencyAnalysisResult with complete frequency analysis
        
        Raises:
            ValueError: If signal_data is invalid
            RuntimeError: If analysis fails
        """
        if signal_data.size == 0:
            raise ValueError("Signal data cannot be empty")
        
        if signal_data.ndim != 1:
            raise ValueError(
                f"Signal data must be 1D, got shape {signal_data.shape}"
            )
        
        # Check for invalid values
        if np.any(np.isnan(signal_data)):
            logger.warning("Signal contains NaN values, replacing with 0")
            signal_data = np.nan_to_num(signal_data, nan=0.0)
        
        if np.any(np.isinf(signal_data)):
            logger.warning("Signal contains Inf values, replacing with 0")
            signal_data = np.nan_to_num(signal_data, posinf=0.0, neginf=0.0)
        
        # Route to appropriate analysis method
        if self.method == FrequencyAnalysisMethod.WELCH:
            return self._analyze_welch(signal_data)
        elif self.method == FrequencyAnalysisMethod.FFT:
            return self._analyze_fft(signal_data)
        elif self.method == FrequencyAnalysisMethod.STFT:
            # STFT returns time-frequency representation, extract frequency summary
            return self._analyze_stft_summary(signal_data)
        elif self.method == FrequencyAnalysisMethod.CWT:
            # CWT returns scale-frequency representation, extract frequency summary
            return self._analyze_cwt_summary(signal_data)
        else:
            # Default to Welch for best frequency resolution and noise reduction
            logger.warning(
                f"Unknown method {self.method}, defaulting to Welch's method"
            )
            return self._analyze_welch(signal_data)
    
    def _analyze_welch(
        self, signal_data: np.ndarray
    ) -> FrequencyAnalysisResult:
        """
        Analyze signal using Welch's method for better frequency resolution.
        
        Welch's method provides better frequency resolution and noise reduction
        compared to standard FFT, especially for noisy signals.
        
        Args:
            signal_data: 1D signal array
        
        Returns:
            FrequencyAnalysisResult
        """
        # Calculate PSD using Welch's method
        frequencies, psd = welch(
            signal_data,
            fs=self.sampling_rate,
            window=self.window_type,
            nperseg=self.nperseg,
            noverlap=int(self.nperseg * self.overlap_ratio),
            scaling='density'
        )
        
        # Calculate FFT for phase information using Real FFT for better performance
        n = len(signal_data)
        # Use Real FFT (rfft) for real-valued signals - ~2x faster
        fft_result = rfft(signal_data, n=n)
        fft_frequencies_pos = rfftfreq(n, 1.0 / self.sampling_rate)
        fft_result_pos = fft_result
        
        # Calculate amplitudes and phases
        amplitudes = np.abs(fft_result_pos)
        phases = np.angle(fft_result_pos)
        
        # Find dominant frequencies with phase information
        dominant_freqs = self._find_dominant_frequencies(
            frequencies, psd, amplitudes, phases
        )
        
        # Calculate fundamental frequency and harmonics
        fundamental, harmonics = self._identify_fundamental_and_harmonics(
            dominant_freqs
        )
        
        # Calculate total power
        total_power = np.trapz(psd, frequencies)
        
        # Estimate SNR
        snr = self._estimate_snr(psd, frequencies)
        
        # Calculate bandwidth (frequency range containing 90% of power)
        bandwidth = self._calculate_bandwidth(frequencies, psd)
        
        return FrequencyAnalysisResult(
            dominant_frequencies=dominant_freqs,
            frequency_spectrum=np.column_stack([fft_frequencies_pos, amplitudes]),
            power_spectral_density=np.column_stack([frequencies, psd]),
            total_power=total_power,
            signal_to_noise_ratio=snr,
            fundamental_frequency=fundamental,
            harmonics=harmonics,
            bandwidth=bandwidth,
            sampling_rate=self.sampling_rate
        )
    
    def _analyze_fft(self, signal_data: np.ndarray) -> FrequencyAnalysisResult:
        """
        Analyze signal using Fast Fourier Transform (FFT).
        
        This method uses Real FFT (rfft) for real-valued signals, which is
        approximately 2x faster and uses half the memory compared to standard FFT,
        while providing the same frequency resolution for positive frequencies.
        
        Args:
            signal_data: 1D signal array (real-valued)
        
        Returns:
            FrequencyAnalysisResult with complete frequency analysis
        
        Note:
            For real-valued signals, only positive frequencies are computed,
            which is sufficient for most frequency analysis applications.
            The method automatically detects real signals and uses the optimal FFT variant.
        """
        n = len(signal_data)
        
        if n < 2:
            raise ValueError(f"Signal must have at least 2 samples, got {n}")
        
        # Apply window if available and size matches
        if self._window is not None and len(self._window) == n:
            windowed_signal = signal_data * self._window
        elif self._window is not None:
            # Window size doesn't match, create new window using cached function
            window_func = self._get_window_function(self.window_type)
            if window_func is not None:
                window = window_func(n)
                windowed_signal = signal_data * window
            else:
                windowed_signal = signal_data
        else:
            windowed_signal = signal_data
        
        # Detect if signal is real-valued
        is_real_signal = np.isrealobj(windowed_signal)
        if is_real_signal and hasattr(windowed_signal, 'dtype'):
            is_real_signal = np.issubdtype(windowed_signal.dtype, np.floating)
        
        # Use Real FFT for real-valued signals (more efficient)
        if is_real_signal:
            # Real FFT: faster and uses less memory
            fft_result = rfft(windowed_signal, n=n)
            frequencies_pos = rfftfreq(n, 1.0 / self.sampling_rate)
        else:
            # Complex signal: use standard FFT
            fft_result = fft(windowed_signal, n=n)
            frequencies = fftfreq(n, 1.0 / self.sampling_rate)
            positive_mask = frequencies >= 0
            frequencies_pos = frequencies[positive_mask]
            fft_result = fft_result[positive_mask]
        
        # Calculate amplitudes, phases, and PSD
        amplitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        # Power Spectral Density: |X(f)|² / (N * fs)
        # For rfft, we need to account for the fact that negative frequencies
        # are not computed, so we multiply by 2 (except DC and Nyquist)
        if is_real_signal and n > 1:
            psd = np.zeros_like(amplitudes, dtype=np.float64)
            psd[0] = amplitudes[0] ** 2 / (n * self.sampling_rate)  # DC component
            if n % 2 == 0:
                # Even length: Nyquist frequency is at index n//2
                psd[1:-1] = 2 * amplitudes[1:-1] ** 2 / (n * self.sampling_rate)
                psd[-1] = amplitudes[-1] ** 2 / (n * self.sampling_rate)  # Nyquist
            else:
                # Odd length: no Nyquist frequency
                psd[1:] = 2 * amplitudes[1:] ** 2 / (n * self.sampling_rate)
        else:
            psd = amplitudes ** 2 / (n * self.sampling_rate)
        
        # Find dominant frequencies with phase information
        dominant_freqs = self._find_dominant_frequencies(
            frequencies_pos, psd, amplitudes, phases
        )
        
        # Calculate fundamental and harmonics
        fundamental, harmonics = self._identify_fundamental_and_harmonics(
            dominant_freqs
        )
        
        # Calculate metrics using trapezoidal integration for accuracy
        if len(frequencies_pos) > 1:
            total_power = np.trapz(psd, frequencies_pos)
        else:
            total_power = float(psd[0]) if len(psd) > 0 else 0.0
        
        snr = self._estimate_snr(psd, frequencies_pos)
        bandwidth = self._calculate_bandwidth(frequencies_pos, psd)
        
        return FrequencyAnalysisResult(
            dominant_frequencies=dominant_freqs,
            frequency_spectrum=np.column_stack([frequencies_pos, amplitudes]),
            power_spectral_density=np.column_stack([frequencies_pos, psd]),
            total_power=total_power,
            signal_to_noise_ratio=snr,
            fundamental_frequency=fundamental,
            harmonics=harmonics,
            bandwidth=bandwidth,
            sampling_rate=self.sampling_rate
        )
    
    def _find_dominant_frequencies(
        self,
        frequencies: np.ndarray,
        psd: np.ndarray,
        amplitudes: np.ndarray,
        phases: Optional[np.ndarray] = None,
        n_peaks: int = 10,
        min_height_ratio: float = 0.1
    ) -> List[FrequencyComponent]:
        """
        Identify dominant frequency components in the signal.
        
        This method finds all significant frequency peaks in the power spectral
        density and extracts their characteristics including frequency, amplitude,
        phase, and power. It uses peak detection with configurable thresholds
        to identify meaningful frequency components.
        
        Args:
            frequencies: Frequency array in Hz
            psd: Power spectral density array
            amplitudes: Amplitude array from FFT
            phases: Phase array from FFT. If None, phases are set to 0.0
            n_peaks: Maximum number of dominant peaks to identify
            min_height_ratio: Minimum peak height as ratio of maximum PSD value
                           (0.0 to 1.0). Peaks below this threshold are ignored.
        
        Returns:
            List of FrequencyComponent objects sorted by power (descending)
        
        Note:
            The method ensures that peaks are sufficiently separated to avoid
            identifying multiple peaks from the same frequency component.
        """
        if len(frequencies) == 0 or len(psd) == 0:
            return []
        
        if len(frequencies) != len(psd) or len(frequencies) != len(amplitudes):
            raise ValueError(
                f"Array length mismatch: frequencies={len(frequencies)}, "
                f"psd={len(psd)}, amplitudes={len(amplitudes)}"
            )
        
        # Find peaks in PSD using scipy's peak detection
        max_psd = np.max(psd)
        if max_psd <= 0:
            logger.warning("All PSD values are non-positive, no peaks found")
            return []
        
        min_height = max_psd * min_height_ratio
        min_distance = max(1, int(len(psd) / (n_peaks * 2)))
        
        try:
            peaks, properties = find_peaks(
                psd,
                height=min_height,
                distance=min_distance,
                prominence=max_psd * 0.05  # Minimum prominence for peak
            )
        except Exception as e:
            logger.error(f"Error in peak detection: {e}")
            # Fallback: return peak with maximum power
            max_idx = np.argmax(psd)
            peaks = np.array([max_idx])
        
        if len(peaks) == 0:
            # No peaks found, return the maximum
            max_idx = np.argmax(psd)
            peaks = np.array([max_idx])
        
        # Sort by power (descending)
        peak_powers = psd[peaks]
        sorted_indices = np.argsort(peak_powers)[::-1]
        top_peaks = peaks[sorted_indices][:n_peaks]
        
        # Extract phase information if available
        if phases is None or len(phases) != len(frequencies):
            phases_array = np.zeros_like(frequencies)
        else:
            phases_array = phases
        
        # Create frequency components with complete information
        components: List[FrequencyComponent] = []
        for peak_idx in top_peaks:
            # Find exact frequency index
            freq_idx = int(peak_idx)
            if freq_idx >= len(frequencies):
                continue
            
            # Extract phase at this frequency
            phase_value = float(phases_array[freq_idx]) if freq_idx < len(phases_array) else 0.0
            
            component = FrequencyComponent(
                frequency=float(frequencies[freq_idx]),
                amplitude=float(amplitudes[freq_idx]),
                phase=phase_value,
                power=float(psd[freq_idx]),
                harmonic_number=None  # Will be set later if identified as harmonic
            )
            components.append(component)
        
        return components
    
    def _identify_fundamental_and_harmonics(
        self, components: List[FrequencyComponent]
    ) -> Tuple[Optional[float], List[FrequencyComponent]]:
        """
        Identify fundamental frequency and its harmonics.
        
        This method analyzes frequency components to identify the fundamental
        frequency (typically the lowest frequency with significant power) and
        its integer multiples (harmonics). This is useful for understanding
        periodic signals and mechanical vibrations.
        
        Args:
            components: List of frequency components sorted by power
        
        Returns:
            Tuple containing:
            - fundamental_frequency: Fundamental frequency in Hz, or None if not found
            - harmonics: List of FrequencyComponent objects identified as harmonics,
                       with harmonic_number attribute set
        
        Note:
            The method uses a tolerance-based approach to identify harmonics,
            allowing for small frequency deviations due to noise or measurement errors.
        """
        if not components:
            return None, []
        
        if len(components) == 1:
            # Only one component, it's the fundamental
            return components[0].frequency, []
        
        # Sort by power (already sorted, but ensure)
        sorted_components = sorted(components, key=lambda x: x.power, reverse=True)
        fundamental = sorted_components[0]
        fundamental_freq = fundamental.frequency
        
        if fundamental_freq <= 0:
            logger.warning(f"Invalid fundamental frequency: {fundamental_freq}")
            return None, []
        
        # Find harmonics (integer multiples of fundamental)
        harmonics: List[FrequencyComponent] = []
        harmonic_tolerance = 0.1  # 10% tolerance for harmonic detection
        
        for comp in sorted_components[1:]:
            if comp.frequency <= 0:
                continue
            
            # Calculate ratio to fundamental
            ratio = comp.frequency / fundamental_freq
            
            # Check if it's approximately an integer multiple
            rounded_ratio = round(ratio)
            if rounded_ratio < 2:  # Skip if ratio < 2 (would be subharmonic)
                continue
            
            deviation = abs(ratio - rounded_ratio)
            if deviation < harmonic_tolerance:
                harmonic_num = int(rounded_ratio)
                # Create a copy to avoid modifying original
                harmonic_comp = FrequencyComponent(
                    frequency=comp.frequency,
                    amplitude=comp.amplitude,
                    phase=comp.phase,
                    power=comp.power,
                    harmonic_number=harmonic_num
                )
                harmonics.append(harmonic_comp)
                logger.debug(
                    f"Identified harmonic: {comp.frequency:.2f} Hz "
                    f"as {harmonic_num}x of fundamental {fundamental_freq:.2f} Hz"
                )
        
        return fundamental.frequency, harmonics
    
    def _estimate_snr(
        self, psd: np.ndarray, frequencies: np.ndarray
    ) -> float:
        """
        Estimate signal-to-noise ratio from PSD.
        
        Args:
            psd: Power spectral density
            frequencies: Frequency array
        
        Returns:
            SNR in dB
        """
        # Identify signal and noise regions
        # Signal: peaks in PSD
        # Noise: baseline/floor of PSD
        
        # Use median as noise floor estimate
        noise_floor = np.median(psd)
        
        # Signal power is total power above noise floor
        signal_power = np.sum(psd[psd > noise_floor * 2])
        noise_power = noise_floor * len(psd)
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
        else:
            snr_db = float('inf')
        
        return float(snr_db)
    
    def _calculate_bandwidth(
        self, frequencies: np.ndarray, psd: np.ndarray, power_fraction: float = 0.9
    ) -> float:
        """
        Calculate effective bandwidth containing specified power fraction.
        
        Args:
            frequencies: Frequency array
            psd: Power spectral density
            power_fraction: Fraction of total power (default 0.9 = 90%)
        
        Returns:
            Bandwidth in Hz
        """
        total_power = np.trapz(psd, frequencies)
        target_power = total_power * power_fraction
        
        # Find frequency range containing target power
        cumulative_power = np.cumsum(psd) * (frequencies[1] - frequencies[0])
        
        # Find indices where cumulative power reaches target
        mask = cumulative_power >= target_power
        if np.any(mask):
            bandwidth = frequencies[mask][0] - frequencies[0]
        else:
            bandwidth = frequencies[-1] - frequencies[0]
        
        return float(bandwidth)
    
    def _apply_bandpass_filter(
        self, data: np.ndarray, low_cutoff: float, high_cutoff: float
    ) -> np.ndarray:
        """
        Apply bandpass filter to signal.
        
        Filters the signal to retain only frequencies between low_cutoff and
        high_cutoff, removing low-frequency drift and high-frequency noise.
        Uses a 4th-order Butterworth filter with zero-phase filtering (filtfilt).
        
        Args:
            data: Input signal array
            low_cutoff: Lower cutoff frequency in Hz (must be > 0)
            high_cutoff: Upper cutoff frequency in Hz (must be < Nyquist)
        
        Returns:
            Filtered signal array with same shape as input
        
        Raises:
            ValueError: If cutoff frequencies are invalid
        """
        if low_cutoff <= 0:
            raise ValueError(f"Low cutoff must be positive, got {low_cutoff}")
        
        nyquist = self.sampling_rate / 2.0
        if high_cutoff >= nyquist:
            raise ValueError(
                f"High cutoff ({high_cutoff} Hz) must be less than "
                f"Nyquist frequency ({nyquist} Hz)"
            )
        
        if low_cutoff >= high_cutoff:
            raise ValueError(
                f"Low cutoff ({low_cutoff} Hz) must be less than "
                f"high cutoff ({high_cutoff} Hz)"
            )
        
        # Normalize frequencies to [0, 1] range (Nyquist = 1.0)
        low_normalized = low_cutoff / nyquist
        high_normalized = min(high_cutoff / nyquist, 0.99)
        
        try:
            # Design 4th-order Butterworth bandpass filter
            b, a = butter(4, [low_normalized, high_normalized], btype='band')
            # Apply zero-phase filtering (forward and backward)
            filtered = filtfilt(b, a, data)
            return filtered
        except Exception as e:
            logger.error(f"Error applying bandpass filter: {e}")
            return data
    
    def _apply_lowpass_filter(
        self, data: np.ndarray, cutoff: float
    ) -> np.ndarray:
        """
        Apply low-pass filter to signal.
        
        Filters the signal to remove high-frequency noise while preserving
        low-frequency content. Uses a 4th-order Butterworth filter with
        zero-phase filtering (filtfilt).
        
        Args:
            data: Input signal array
            cutoff: Cutoff frequency in Hz (must be < Nyquist frequency)
        
        Returns:
            Filtered signal array with same shape as input
        
        Raises:
            ValueError: If cutoff frequency is invalid
        """
        if cutoff <= 0:
            raise ValueError(f"Cutoff frequency must be positive, got {cutoff}")
        
        nyquist = self.sampling_rate / 2.0
        if cutoff >= nyquist:
            raise ValueError(
                f"Cutoff frequency ({cutoff} Hz) must be less than "
                f"Nyquist frequency ({nyquist} Hz)"
            )
        
        normalized_cutoff = min(cutoff / nyquist, 0.99)
        
        try:
            # Design 4th-order Butterworth low-pass filter
            b, a = butter(4, normalized_cutoff, btype='low')
            # Apply zero-phase filtering
            filtered = filtfilt(b, a, data)
            return filtered
        except Exception as e:
            logger.error(f"Error applying low-pass filter: {e}")
            return data
    
    def _find_common_frequencies(
        self,
        result1: FrequencyAnalysisResult,
        result2: FrequencyAnalysisResult,
        threshold: float
    ) -> List[Dict[str, float]]:
        """Find frequencies common to both analysis results."""
        common = []
        tolerance = 0.05  # 5% frequency tolerance
        
        for comp1 in result1.dominant_frequencies:
            for comp2 in result2.dominant_frequencies:
                freq_diff = abs(comp1.frequency - comp2.frequency)
                if freq_diff / max(comp1.frequency, comp2.frequency) < tolerance:
                    common.append({
                        'frequency': comp1.frequency,
                        'acceleration_power': comp1.power,
                        'encoder_power': comp2.power,
                        'frequency_difference': freq_diff
                    })
        
        return common
    
    def _calculate_cross_correlation(
        self, data1: np.ndarray, data2: np.ndarray
    ) -> Dict[str, float]:
        """Calculate cross-correlation between two signals."""
        # Normalize signals
        data1_norm = (data1 - np.mean(data1)) / (np.std(data1) + 1e-10)
        data2_norm = (data2 - np.mean(data2)) / (np.std(data2) + 1e-10)
        
        # Cross-correlation
        correlation = np.correlate(data1_norm, data2_norm, mode='full')
        max_corr = np.max(correlation)
        max_corr_idx = np.argmax(correlation)
        
        # Lag in samples
        lag = max_corr_idx - (len(data1) - 1)
        lag_time = lag / self.sampling_rate
        
        return {
            'max_correlation': float(max_corr),
            'lag_samples': int(lag),
            'lag_time_seconds': float(lag_time)
        }
    
    def _calculate_phase_relationship(
        self,
        result1: FrequencyAnalysisResult,
        result2: FrequencyAnalysisResult,
        common_frequencies: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate phase relationships at common frequencies.
        
        This method extracts phase information from both frequency analysis
        results at frequencies that are present in both signals. The phase
        difference provides insight into the temporal relationship between
        acceleration and encoder signals.
        
        Args:
            result1: First frequency analysis result (typically acceleration)
            result2: Second frequency analysis result (typically encoder)
            common_frequencies: List of dictionaries containing common frequency info
        
        Returns:
            Dictionary mapping frequency strings to phase relationship data:
            - 'phase_difference': Phase difference in radians
            - 'phase_difference_degrees': Phase difference in degrees
            - 'acceleration_phase': Phase in result1 at this frequency
            - 'encoder_phase': Phase in result2 at this frequency
            - 'time_delay': Estimated time delay in seconds
        """
        phase_relationships: Dict[str, Dict[str, float]] = {}
        
        if not common_frequencies:
            return phase_relationships
        
        # Extract phase information from frequency components
        # Build frequency-to-phase maps for both results
        freq_to_phase1 = {
            comp.frequency: comp.phase
            for comp in result1.dominant_frequencies
        }
        freq_to_phase2 = {
            comp.frequency: comp.phase
            for comp in result2.dominant_frequencies
        }
        
        for common in common_frequencies:
            freq = common['frequency']
            freq_key = f"{freq:.2f}Hz"
            
            # Get phases from both results
            phase1 = freq_to_phase1.get(freq, 0.0)
            phase2 = freq_to_phase2.get(freq, 0.0)
            
            # Calculate phase difference (normalized to [-π, π])
            phase_diff = phase2 - phase1
            phase_diff = ((phase_diff + np.pi) % (2 * np.pi)) - np.pi
            
            # Convert to degrees
            phase_diff_deg = np.degrees(phase_diff)
            
            # Estimate time delay: Δt = Δφ / (2π * f)
            if freq > 0:
                time_delay = phase_diff / (2 * np.pi * freq)
            else:
                time_delay = 0.0
            
            phase_relationships[freq_key] = {
                'phase_difference': float(phase_diff),
                'phase_difference_degrees': float(phase_diff_deg),
                'acceleration_phase': float(phase1),
                'encoder_phase': float(phase2),
                'time_delay': float(time_delay)
            }
        
        return phase_relationships
    
    def _analyze_stft_summary(
        self, signal_data: np.ndarray
    ) -> FrequencyAnalysisResult:
        """
        Analyze signal using STFT and extract frequency summary.
        
        This method computes STFT and then extracts frequency information
        by averaging the spectrogram across time windows.
        
        Args:
            signal_data: 1D signal array
        
        Returns:
            FrequencyAnalysisResult with frequency analysis
        """
        stft_result = self.analyze_stft(signal_data, return_spectrogram=True)
        
        # Average spectrogram across time to get frequency spectrum
        avg_spectrum = np.mean(stft_result['spectrogram'], axis=1)
        frequencies = stft_result['frequencies']
        psd = avg_spectrum ** 2
        
        # Create result from spectrum
        return self._create_result_from_spectrum(frequencies, psd, avg_spectrum)
    
    def _analyze_cwt_summary(
        self, signal_data: np.ndarray
    ) -> FrequencyAnalysisResult:
        """
        Analyze signal using CWT and extract frequency summary.
        
        This method computes CWT and then extracts frequency information
        by averaging the magnitude across time.
        
        Args:
            signal_data: 1D signal array
        
        Returns:
            FrequencyAnalysisResult with frequency analysis
        """
        cwt_result = self.analyze_cwt(signal_data)
        
        # Average magnitude across time
        avg_magnitude = np.mean(cwt_result['magnitude'], axis=1)
        frequencies = cwt_result['frequencies']
        psd = avg_magnitude ** 2
        
        # Create result from spectrum
        return self._create_result_from_spectrum(frequencies, psd, avg_magnitude)
    
    def _create_result_from_spectrum(
        self,
        frequencies: np.ndarray,
        psd: np.ndarray,
        amplitudes: np.ndarray
    ) -> FrequencyAnalysisResult:
        """
        Create FrequencyAnalysisResult from frequency spectrum data.
        
        Helper method to create a result object from spectrum data,
        used by STFT and CWT analysis methods.
        
        Args:
            frequencies: Frequency array in Hz
            psd: Power spectral density array
            amplitudes: Amplitude array
        
        Returns:
            FrequencyAnalysisResult with complete analysis
        """
        # Find dominant frequencies
        dominant_freqs = self._find_dominant_frequencies(
            frequencies, psd, amplitudes
        )
        
        # Calculate fundamental and harmonics
        fundamental, harmonics = self._identify_fundamental_and_harmonics(
            dominant_freqs
        )
        
        # Calculate metrics
        if len(psd) > 1:
            total_power = float(np.trapz(psd, frequencies))
        else:
            total_power = float(psd[0]) if len(psd) > 0 else 0.0
        
        snr = self._estimate_snr(psd, frequencies)
        bandwidth = self._calculate_bandwidth(frequencies, psd)
        
        return FrequencyAnalysisResult(
            dominant_frequencies=dominant_freqs,
            frequency_spectrum=np.column_stack([frequencies, amplitudes]),
            power_spectral_density=np.column_stack([frequencies, psd]),
            total_power=total_power,
            signal_to_noise_ratio=snr,
            fundamental_frequency=fundamental,
            harmonics=harmonics,
            bandwidth=bandwidth,
            sampling_rate=self.sampling_rate
        )
    
    def analyze_stft(
        self,
        signal_data: np.ndarray,
        return_spectrogram: bool = True
    ) -> Dict[str, Union[np.ndarray, FrequencyAnalysisResult]]:
        """
        Perform Short-Time Fourier Transform (STFT) for time-frequency analysis.
        
        STFT provides a time-frequency representation of the signal, showing how
        frequency content changes over time. This is particularly useful for:
        - Non-stationary signals (frequencies change over time)
        - Transient events (sudden frequency changes)
        - Analyzing acceleration/deceleration phases
        - Vibration analysis during motion transitions
        
        Args:
            signal_data: 1D signal array to analyze
            return_spectrogram: If True, returns spectrogram data. If False, returns
                             frequency analysis result for each time window
        
        Returns:
            Dictionary containing:
            - 'frequencies': Frequency array in Hz
            - 'time': Time array in seconds
            - 'spectrogram': 2D array of magnitude spectrogram (frequencies x time)
            - 'phase_spectrogram': 2D array of phase spectrogram
            - 'power_spectrogram': 2D array of power spectrogram
        
        Raises:
            ValueError: If signal_data is invalid
        """
        if signal_data.size == 0:
            raise ValueError("Signal data cannot be empty")
        
        if signal_data.ndim != 1:
            raise ValueError(f"Signal data must be 1D, got shape {signal_data.shape}")
        
        # Compute STFT
        f, t, Zxx = stft(
            signal_data,
            fs=self.sampling_rate,
            window=self.window_type,
            nperseg=self.nperseg,
            noverlap=int(self.nperseg * self.overlap_ratio),
            boundary='zeros',
            padded=True
        )
        
        # Calculate magnitude, phase, and power spectrograms
        magnitude_spectrogram = np.abs(Zxx)
        phase_spectrogram = np.angle(Zxx)
        power_spectrogram = magnitude_spectrogram ** 2
        
        return {
            'frequencies': f,
            'time': t,
            'spectrogram': magnitude_spectrogram,
            'phase_spectrogram': phase_spectrogram,
            'power_spectrogram': power_spectrogram,
            'complex_spectrogram': Zxx
        }
    
    def analyze_cwt(
        self,
        signal_data: np.ndarray,
        scales: Optional[np.ndarray] = None,
        wavelet: str = 'morlet'
    ) -> Dict[str, np.ndarray]:
        """
        Perform Continuous Wavelet Transform (CWT) for multi-resolution analysis.
        
        CWT provides excellent time-frequency resolution and is particularly useful for:
        - Multi-scale frequency analysis
        - Detecting frequency modulations
        - Analyzing complex vibration patterns
        - Identifying both high and low frequency components simultaneously
        
        Args:
            signal_data: 1D signal array to analyze
            scales: Array of scales for wavelet transform. If None, automatically
                   generates scales from 1 to sampling_rate/4
            wavelet: Wavelet function to use ('morlet' or 'ricker')
        
        Returns:
            Dictionary containing:
            - 'scales': Scale array used for transform
            - 'coefficients': 2D CWT coefficients array (scales x time)
            - 'frequencies': Corresponding frequencies in Hz for each scale
            - 'magnitude': Magnitude of CWT coefficients
            - 'phase': Phase of CWT coefficients
            - 'energy': Energy distribution across scales
        
        Raises:
            ValueError: If signal_data is invalid or wavelet is unknown
            RuntimeError: If CWT computation fails
        """
        if signal_data.size == 0:
            raise ValueError("Signal data cannot be empty")
        
        if signal_data.ndim != 1:
            raise ValueError(f"Signal data must be 1D, got shape {signal_data.shape}")
        
        # Generate scales if not provided
        if scales is None:
            max_scale = int(self.sampling_rate / 4)
            scales = np.arange(1, min(max_scale + 1, 128))  # Limit to reasonable range
        
        # Select wavelet function
        if wavelet == 'morlet':
            wavelet_func = morlet2
        elif wavelet == 'ricker':
            wavelet_func = signal.ricker
        else:
            raise ValueError(f"Unknown wavelet: {wavelet}. Use 'morlet' or 'ricker'")
        
        # Compute CWT
        try:
            cwt_matrix = cwt(signal_data, wavelet_func, scales)
        except Exception as e:
            logger.error(f"Error computing CWT with {wavelet}: {e}")
            raise RuntimeError(f"Failed to compute CWT: {e}") from e
        
        # Calculate corresponding frequencies
        # For Morlet wavelet: frequency ≈ sampling_rate / (2 * pi * scale)
        if wavelet == 'morlet':
            frequencies = self.sampling_rate / (2 * np.pi * scales)
        else:
            # Approximate for Ricker wavelet
            frequencies = self.sampling_rate / (2 * np.pi * scales)
        
        # Calculate magnitude, phase, and energy
        magnitude = np.abs(cwt_matrix)
        phase = np.angle(cwt_matrix)
        energy = magnitude ** 2
        
        return {
            'scales': scales,
            'coefficients': cwt_matrix,
            'frequencies': frequencies,
            'magnitude': magnitude,
            'phase': phase,
            'energy': energy
        }
    
    def analyze_multi_axis_parallel(
        self,
        acceleration_data: np.ndarray,
        max_workers: Optional[int] = None
    ) -> List[FrequencyAnalysisResult]:
        """
        Analyze multiple acceleration axes in parallel for improved performance.
        
        This method processes each axis of multi-axis acceleration data in parallel,
        significantly reducing processing time for 3-axis accelerometer data.
        Typical speedup: 2-3x for 3-axis data on multi-core systems.
        
        Args:
            acceleration_data: 3D acceleration data array of shape (n_samples, 3)
            max_workers: Maximum number of worker threads. If None, uses min(3, CPU count)
        
        Returns:
            List of FrequencyAnalysisResult objects, one for each axis [X, Y, Z]
        
        Raises:
            ValueError: If acceleration_data is not 2D with 3 columns
        
        Example:
            >>> analyzer = FrequencyAnalyzer(sampling_rate=1000.0)
            >>> accel_3d = np.random.randn(10000, 3)  # 3-axis data
            >>> results = analyzer.analyze_multi_axis_parallel(accel_3d)
            >>> x_result, y_result, z_result = results
        """
        if acceleration_data.ndim != 2:
            raise ValueError(
                f"Acceleration data must be 2D (n_samples, 3), got shape {acceleration_data.shape}"
            )
        
        if acceleration_data.shape[1] != 3:
            raise ValueError(
                f"Expected 3 axes, got {acceleration_data.shape[1]} columns"
            )
        
        if max_workers is None:
            import os
            max_workers = min(3, os.cpu_count() or 1)
        
        def analyze_axis(axis_idx: int) -> FrequencyAnalysisResult:
            """Analyze a single axis."""
            return self.analyze_acceleration(
                acceleration_data,
                axis=axis_idx,
                remove_dc=True,
                apply_filter=True
            )
        
        # Process axes in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(analyze_axis, range(3)))
        
        logger.info(f"Analyzed {len(results)} axes in parallel using {max_workers} workers")
        return results
    
    @lru_cache(maxsize=10)
    def _get_cached_window(
        self, size: int, window_type: str
    ) -> np.ndarray:
        """
        Get cached window function for improved performance.
        
        Window functions are cached to avoid recomputation for repeated analyses
        with the same parameters. This provides significant speedup when analyzing
        multiple signals with the same window settings.
        
        Args:
            size: Window size in samples
            window_type: Type of window function ('hann', 'hamming', 'blackman', 'rectangular')
        
        Returns:
            Window array of specified size and type
        
        Note:
            Cache size is limited to 10 entries to prevent excessive memory usage.
        """
        if window_type == 'hann':
            return signal.windows.hann(size)
        elif window_type == 'hamming':
            return signal.windows.hamming(size)
        elif window_type == 'blackman':
            return signal.windows.blackman(size)
        elif window_type == 'rectangular':
            return np.ones(size)
        else:
            logger.warning(f"Unknown window type {window_type}, using Hann")
            return signal.windows.hann(size)
    
    def get_total_harmonic_distortion(
        self, result: FrequencyAnalysisResult
    ) -> Dict[str, float]:
        """
        Calculate Total Harmonic Distortion (THD) and related metrics.
        
        THD measures the harmonic content of a signal relative to the fundamental
        frequency. This is useful for:
        - Assessing signal quality
        - Detecting mechanical issues (high THD may indicate problems)
        - Characterizing vibration patterns
        
        Args:
            result: FrequencyAnalysisResult from analysis
        
        Returns:
            Dictionary containing:
            - 'thd': Total Harmonic Distortion as percentage
            - 'thd_db': THD in decibels
            - 'fundamental_power': Power at fundamental frequency
            - 'harmonic_power': Total power in harmonics
            - 'harmonic_count': Number of harmonics detected
        """
        if result.fundamental_frequency is None:
            return {
                'thd': 0.0,
                'thd_db': float('-inf'),
                'fundamental_power': 0.0,
                'harmonic_power': 0.0,
                'harmonic_count': 0
            }
        
        # Find fundamental component
        fundamental_power = 0.0
        for comp in result.dominant_frequencies:
            if abs(comp.frequency - result.fundamental_frequency) < 0.1:
                fundamental_power = comp.power
                break
        
        # Sum harmonic power
        harmonic_power = sum(h.power for h in result.harmonics)
        harmonic_count = len(result.harmonics)
        
        # Calculate THD
        if fundamental_power > 0:
            thd_ratio = np.sqrt(harmonic_power / fundamental_power)
            thd_percent = thd_ratio * 100.0
            thd_db = 20 * np.log10(thd_ratio) if thd_ratio > 0 else float('-inf')
        else:
            thd_percent = 0.0
            thd_db = float('-inf')
        
        return {
            'thd': float(thd_percent),
            'thd_db': float(thd_db),
            'fundamental_power': float(fundamental_power),
            'harmonic_power': float(harmonic_power),
            'harmonic_count': int(harmonic_count)
        }
    
    def analyze_frequency_bands(
        self,
        result: FrequencyAnalysisResult,
        frequency_bands: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze frequency content within specific frequency bands.
        
        This method is particularly useful for robot movement analysis, where
        different frequency bands correspond to different types of motion:
        - Low frequencies: Slow movements, walking
        - Medium frequencies: Running, arm motion
        - High frequencies: Vibrations, mechanical noise
        
        Args:
            result: FrequencyAnalysisResult from previous analysis
            frequency_bands: Dictionary mapping band names to (low, high) frequency tuples in Hz.
                           Example: {'walking': (0.5, 3.0), 'vibration': (10.0, 100.0)}
        
        Returns:
            Dictionary mapping band names to analysis metrics:
            - 'power': Total power in this band
            - 'power_percentage': Percentage of total power
            - 'dominant_frequency': Dominant frequency in this band
            - 'component_count': Number of frequency components in this band
            - 'peak_amplitude': Peak amplitude in this band
        
        Example:
            >>> bands = {
            ...     'walking': (0.5, 3.0),
            ...     'running': (2.0, 5.0),
            ...     'vibration': (10.0, 100.0)
            ... }
            >>> band_analysis = analyzer.analyze_frequency_bands(result, bands)
        """
        band_analysis: Dict[str, Dict[str, float]] = {}
        
        # Get frequency and power data
        frequencies = result.power_spectral_density[:, 0]
        psd = result.power_spectral_density[:, 1]
        amplitudes = result.frequency_spectrum[:, 1]
        
        total_power = result.total_power
        
        for band_name, (low_freq, high_freq) in frequency_bands.items():
            if low_freq >= high_freq:
                logger.warning(
                    f"Invalid frequency band {band_name}: low >= high ({low_freq} >= {high_freq})"
                )
                continue
            
            # Find frequencies within this band
            band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            
            if not np.any(band_mask):
                band_analysis[band_name] = {
                    'power': 0.0,
                    'power_percentage': 0.0,
                    'dominant_frequency': 0.0,
                    'component_count': 0,
                    'peak_amplitude': 0.0
                }
                continue
            
            # Calculate power in this band
            band_psd = psd[band_mask]
            band_frequencies = frequencies[band_mask]
            band_amplitudes = amplitudes[band_mask]
            
            # Integrate power
            if len(band_psd) > 1:
                band_power = float(np.trapz(band_psd, band_frequencies))
            else:
                band_power = float(band_psd[0]) if len(band_psd) > 0 else 0.0
            
            # Find dominant frequency in band
            max_idx = np.argmax(band_psd)
            dominant_freq = float(band_frequencies[max_idx])
            peak_amplitude = float(band_amplitudes[max_idx])
            
            # Count components in this band
            component_count = sum(
                1 for comp in result.dominant_frequencies
                if low_freq <= comp.frequency <= high_freq
            )
            
            # Calculate percentage
            power_percentage = (band_power / total_power * 100.0) if total_power > 0 else 0.0
            
            band_analysis[band_name] = {
                'power': band_power,
                'power_percentage': power_percentage,
                'dominant_frequency': dominant_freq,
                'component_count': component_count,
                'peak_amplitude': peak_amplitude
            }
        
        return band_analysis
    
    def analyze_batch(
        self,
        signals: List[np.ndarray],
        signal_types: List[str],
        parallel: bool = False
    ) -> List[FrequencyAnalysisResult]:
        """
        Analyze multiple signals in batch for efficient processing.
        
        This method processes multiple signals of the same or different types
        (acceleration or encoder) in a single call. Useful for analyzing
        multiple time windows or multiple sensor channels.
        
        Args:
            signals: List of signal arrays to analyze
            signal_types: List of signal types ('acceleration' or 'encoder'),
                        must match length of signals
            parallel: If True, processes signals in parallel using ThreadPoolExecutor
        
        Returns:
            List of FrequencyAnalysisResult objects, one for each input signal
        
        Raises:
            ValueError: If signals and signal_types have different lengths
            ValueError: If unknown signal type is provided
        
        Example:
            >>> signals = [accel_data1, accel_data2, encoder_data1]
            >>> types = ['acceleration', 'acceleration', 'encoder']
            >>> results = analyzer.analyze_batch(signals, types, parallel=True)
        """
        if len(signals) != len(signal_types):
            raise ValueError(
                f"Signals and signal_types must have same length: "
                f"{len(signals)} != {len(signal_types)}"
            )
        
        def analyze_single(signal_data: np.ndarray, sig_type: str) -> FrequencyAnalysisResult:
            """Analyze a single signal."""
            if sig_type == 'acceleration':
                return self.analyze_acceleration(
                    signal_data,
                    remove_dc=True,
                    apply_filter=True
                )
            elif sig_type == 'encoder':
                return self.analyze_encoder(
                    signal_data,
                    remove_trend=True,
                    apply_filter=True
                )
            else:
                raise ValueError(f"Unknown signal type: {sig_type}. Use 'acceleration' or 'encoder'")
        
        if parallel and len(signals) > 1:
            # Process in parallel
            import os
            max_workers = min(len(signals), os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(
                    lambda args: analyze_single(args[0], args[1]),
                    zip(signals, signal_types)
                ))
            logger.info(f"Processed {len(signals)} signals in parallel using {max_workers} workers")
        else:
            # Process sequentially
            results = [analyze_single(sig, sig_type) for sig, sig_type in zip(signals, signal_types)]
        
        return results
    
    def compare_results(
        self,
        result1: FrequencyAnalysisResult,
        result2: FrequencyAnalysisResult,
        tolerance: float = 0.05
    ) -> Dict[str, Union[float, List[Dict[str, float]]]]:
        """
        Compare two frequency analysis results to identify differences.
        
        This method is useful for:
        - Detecting changes in frequency content over time
        - Comparing baseline vs. current measurements
        - Identifying anomalies or deviations
        - Validating consistency between measurements
        
        Args:
            result1: First frequency analysis result (e.g., baseline)
            result2: Second frequency analysis result (e.g., current measurement)
            tolerance: Frequency matching tolerance as fraction (default 0.05 = 5%)
        
        Returns:
            Dictionary containing comparison metrics:
            - 'fundamental_frequency_diff': Difference in fundamental frequencies (Hz)
            - 'total_power_ratio': Ratio of total powers (result2 / result1)
            - 'snr_diff': Difference in SNR (dB)
            - 'bandwidth_ratio': Ratio of bandwidths
            - 'common_frequencies': List of frequencies present in both results
            - 'new_frequencies': List of frequencies only in result2
            - 'missing_frequencies': List of frequencies only in result1
            - 'similarity_score': Overall similarity score (0.0 to 1.0)
        
        Example:
            >>> baseline = analyzer.analyze_acceleration(baseline_data)
            >>> current = analyzer.analyze_acceleration(current_data)
            >>> comparison = analyzer.compare_results(baseline, current)
            >>> if comparison['similarity_score'] < 0.8:
            ...     print("Significant deviation detected!")
        """
        # Compare fundamental frequencies
        fund1 = result1.fundamental_frequency or 0.0
        fund2 = result2.fundamental_frequency or 0.0
        fundamental_diff = abs(fund2 - fund1)
        
        # Compare total power
        power_ratio = result2.total_power / result1.total_power if result1.total_power > 0 else 0.0
        
        # Compare SNR
        snr_diff = result2.signal_to_noise_ratio - result1.signal_to_noise_ratio
        
        # Compare bandwidth
        bandwidth_ratio = result2.bandwidth / result1.bandwidth if result1.bandwidth > 0 else 0.0
        
        # Find common, new, and missing frequencies
        freq1_set = {comp.frequency for comp in result1.dominant_frequencies}
        freq2_set = {comp.frequency for comp in result2.dominant_frequencies}
        
        common_frequencies = []
        for comp1 in result1.dominant_frequencies:
            for comp2 in result2.dominant_frequencies:
                freq_diff = abs(comp1.frequency - comp2.frequency)
                if freq_diff / max(comp1.frequency, comp2.frequency) < tolerance:
                    common_frequencies.append({
                        'frequency': comp1.frequency,
                        'power1': comp1.power,
                        'power2': comp2.power,
                        'power_ratio': comp2.power / comp1.power if comp1.power > 0 else 0.0
                    })
                    break
        
        new_frequencies = [
            {'frequency': comp.frequency, 'power': comp.power, 'amplitude': comp.amplitude}
            for comp in result2.dominant_frequencies
            if not any(
                abs(comp.frequency - f1) / max(comp.frequency, f1) < tolerance
                for f1 in freq1_set
            )
        ]
        
        missing_frequencies = [
            {'frequency': comp.frequency, 'power': comp.power, 'amplitude': comp.amplitude}
            for comp in result1.dominant_frequencies
            if not any(
                abs(comp.frequency - f2) / max(comp.frequency, f2) < tolerance
                for f2 in freq2_set
            )
        ]
        
        # Calculate similarity score (0.0 to 1.0)
        # Based on fundamental frequency match, power ratio, and common frequencies
        fund_similarity = 1.0 / (1.0 + fundamental_diff / max(fund1, fund2)) if max(fund1, fund2) > 0 else 0.0
        power_similarity = 1.0 / (1.0 + abs(1.0 - power_ratio))
        freq_overlap = len(common_frequencies) / max(len(result1.dominant_frequencies), len(result2.dominant_frequencies), 1)
        
        similarity_score = (fund_similarity * 0.3 + power_similarity * 0.3 + freq_overlap * 0.4)
        
        return {
            'fundamental_frequency_diff': float(fundamental_diff),
            'total_power_ratio': float(power_ratio),
            'snr_diff': float(snr_diff),
            'bandwidth_ratio': float(bandwidth_ratio),
            'common_frequencies': common_frequencies,
            'new_frequencies': new_frequencies,
            'missing_frequencies': missing_frequencies,
            'similarity_score': float(similarity_score)
        }
    
    def detect_anomalies(
        self,
        result: FrequencyAnalysisResult,
        baseline: FrequencyAnalysisResult,
        thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Union[bool, float, List[str]]]:
        """
        Detect anomalies by comparing current result to baseline.
        
        This method identifies deviations from expected frequency patterns,
        which can indicate mechanical issues, sensor problems, or unusual motion.
        
        Args:
            result: Current frequency analysis result
            baseline: Baseline/reference frequency analysis result
            thresholds: Optional dictionary of custom thresholds:
                       - 'frequency_tolerance': Max allowed fundamental frequency change (Hz)
                       - 'power_tolerance': Max allowed power ratio deviation
                       - 'snr_tolerance': Max allowed SNR change (dB)
                       - 'similarity_threshold': Minimum similarity score (0.0 to 1.0)
        
        Returns:
            Dictionary containing:
            - 'is_anomalous': Boolean indicating if anomalies were detected
            - 'anomaly_score': Anomaly score (0.0 = normal, 1.0 = highly anomalous)
            - 'anomaly_reasons': List of detected anomaly reasons
            - 'metrics': Dictionary of comparison metrics
        
        Example:
            >>> baseline = analyzer.analyze_acceleration(normal_data)
            >>> current = analyzer.analyze_acceleration(suspicious_data)
            >>> anomalies = analyzer.detect_anomalies(current, baseline)
            >>> if anomalies['is_anomalous']:
            ...     print(f"Anomalies detected: {anomalies['anomaly_reasons']}")
        """
        if thresholds is None:
            thresholds = {
                'frequency_tolerance': 2.0,  # Hz
                'power_tolerance': 0.5,  # 50% change
                'snr_tolerance': 10.0,  # dB
                'similarity_threshold': 0.7
            }
        
        # Compare results
        comparison = self.compare_results(result, baseline)
        
        anomaly_reasons: List[str] = []
        anomaly_score = 0.0
        
        # Check fundamental frequency deviation
        if comparison['fundamental_frequency_diff'] > thresholds['frequency_tolerance']:
            anomaly_reasons.append(
                f"Fundamental frequency shifted by {comparison['fundamental_frequency_diff']:.2f} Hz"
            )
            anomaly_score += 0.3
        
        # Check power ratio
        power_ratio = comparison['total_power_ratio']
        if abs(1.0 - power_ratio) > thresholds['power_tolerance']:
            anomaly_reasons.append(
                f"Total power changed by {(1.0 - power_ratio) * 100:.1f}%"
            )
            anomaly_score += 0.2
        
        # Check SNR
        if abs(comparison['snr_diff']) > thresholds['snr_tolerance']:
            anomaly_reasons.append(
                f"SNR changed by {comparison['snr_diff']:.2f} dB"
            )
            anomaly_score += 0.2
        
        # Check similarity
        if comparison['similarity_score'] < thresholds['similarity_threshold']:
            anomaly_reasons.append(
                f"Low similarity score: {comparison['similarity_score']:.2f}"
            )
            anomaly_score += 0.3
        
        # Check for new frequencies
        if len(comparison['new_frequencies']) > 2:
            anomaly_reasons.append(
                f"{len(comparison['new_frequencies'])} new frequencies detected"
            )
            anomaly_score += 0.1 * min(len(comparison['new_frequencies']), 5)
        
        # Check for missing frequencies
        if len(comparison['missing_frequencies']) > 2:
            anomaly_reasons.append(
                f"{len(comparison['missing_frequencies'])} frequencies disappeared"
            )
            anomaly_score += 0.1 * min(len(comparison['missing_frequencies']), 5)
        
        # Normalize anomaly score to [0, 1]
        anomaly_score = min(1.0, anomaly_score)
        
        is_anomalous = (
            anomaly_score > 0.5 or
            comparison['similarity_score'] < thresholds['similarity_threshold']
        )
        
        return {
            'is_anomalous': is_anomalous,
            'anomaly_score': float(anomaly_score),
            'anomaly_reasons': anomaly_reasons,
            'metrics': comparison
        }
    
    def export_results(
        self,
        result: FrequencyAnalysisResult,
        filepath: str,
        format: str = 'json'
    ) -> bool:
        """
        Export frequency analysis results to file.
        
        Supports multiple formats for integration with other tools and
        long-term storage of analysis results.
        
        Args:
            result: FrequencyAnalysisResult to export
            filepath: Path to output file
            format: Export format ('json', 'csv', 'numpy')
        
        Returns:
            True if export was successful, False otherwise
        
        Raises:
            ValueError: If format is not supported
            IOError: If file writing fails
        """
        import json
        from pathlib import Path
        
        file_path = Path(filepath)
        
        try:
            if format == 'json':
                # Convert result to dictionary
                data = {
                    'fundamental_frequency': result.fundamental_frequency,
                    'total_power': result.total_power,
                    'signal_to_noise_ratio': result.signal_to_noise_ratio,
                    'bandwidth': result.bandwidth,
                    'sampling_rate': result.sampling_rate,
                    'dominant_frequencies': [
                        {
                            'frequency': comp.frequency,
                            'amplitude': comp.amplitude,
                            'phase': comp.phase,
                            'power': comp.power,
                            'harmonic_number': comp.harmonic_number
                        }
                        for comp in result.dominant_frequencies
                    ],
                    'harmonics': [
                        {
                            'frequency': h.frequency,
                            'amplitude': h.amplitude,
                            'phase': h.phase,
                            'power': h.power,
                            'harmonic_number': h.harmonic_number
                        }
                        for h in result.harmonics
                    ]
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format == 'csv':
                # Export frequency spectrum and PSD to CSV
                import csv
                
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Frequency (Hz)', 'Amplitude', 'PSD'])
                    
                    for i in range(len(result.frequency_spectrum)):
                        freq = result.frequency_spectrum[i, 0]
                        amp = result.frequency_spectrum[i, 1]
                        psd = result.power_spectral_density[i, 1]
                        writer.writerow([freq, amp, psd])
            
            elif format == 'numpy':
                # Export as NumPy archive
                np.savez(
                    file_path,
                    frequency_spectrum=result.frequency_spectrum,
                    power_spectral_density=result.power_spectral_density,
                    fundamental_frequency=result.fundamental_frequency,
                    total_power=result.total_power,
                    signal_to_noise_ratio=result.signal_to_noise_ratio,
                    bandwidth=result.bandwidth,
                    sampling_rate=result.sampling_rate
                )
            
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'json', 'csv', or 'numpy'")
            
            logger.info(f"Exported results to {file_path} in {format} format")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting results to {file_path}: {e}")
            raise IOError(f"Failed to export results: {e}") from e
    
    def calculate_coherence(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        nperseg: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate magnitude-squared coherence between two signals.
        
        Coherence measures the linear relationship between two signals as a
        function of frequency. Values range from 0 (no relationship) to 1
        (perfect linear relationship). This is useful for:
        - Identifying frequency-dependent relationships between sensors
        - Detecting common frequency components
        - Validating sensor synchronization
        
        Args:
            signal1: First signal array (1D)
            signal2: Second signal array (1D), must have same length as signal1
            nperseg: Segment length for coherence calculation.
                   If None, uses self.nperseg
        
        Returns:
            Tuple containing:
            - frequencies: Frequency array in Hz
            - coherence: Coherence values (0 to 1) at each frequency
        
        Raises:
            ValueError: If signals are invalid or have different lengths
            RuntimeError: If coherence calculation fails
        
        Example:
            >>> analyzer = FrequencyAnalyzer(sampling_rate=1000.0)
            >>> accel = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            >>> encoder = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            >>> freqs, coh = analyzer.calculate_coherence(accel, encoder)
            >>> # High coherence at 10 Hz indicates strong relationship
        """
        if signal1.size == 0 or signal2.size == 0:
            raise ValueError("Both signals must be non-empty")
        
        if len(signal1) != len(signal2):
            raise ValueError(
                f"Signals must have same length: {len(signal1)} != {len(signal2)}"
            )
        
        if signal1.ndim != 1 or signal2.ndim != 1:
            raise ValueError("Both signals must be 1D arrays")
        
        nperseg = nperseg or self.nperseg
        nperseg = min(nperseg, len(signal1) // 4)  # Ensure reasonable segment size
        
        try:
            from scipy.signal import coherence
            
            frequencies, coh = coherence(
                signal1,
                signal2,
                fs=self.sampling_rate,
                window=self.window_type,
                nperseg=nperseg,
                noverlap=int(nperseg * self.overlap_ratio)
            )
            
            return frequencies, coh
        except ImportError:
            raise RuntimeError(
                "scipy.signal.coherence not available. "
                "Please ensure scipy is properly installed."
            )
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            raise RuntimeError(f"Failed to calculate coherence: {e}") from e
    
    def detect_anomalies(
        self,
        result: FrequencyAnalysisResult,
        threshold_snr: float = 10.0,
        threshold_power_deviation: float = 3.0
    ) -> Dict[str, Union[List[FrequencyComponent], Dict[str, float]]]:
        """
        Detect anomalous frequency components in the analysis result.
        
        Anomalies are identified based on:
        - Unusually high or low power at specific frequencies
        - Frequencies with poor signal-to-noise ratio
        - Unexpected frequency components
        - Abnormal harmonic relationships
        
        This is useful for:
        - Fault detection in mechanical systems
        - Identifying sensor malfunctions
        - Detecting unexpected vibrations
        - Quality control in manufacturing
        
        Args:
            result: FrequencyAnalysisResult to analyze for anomalies
            threshold_snr: Minimum SNR in dB to consider a component valid.
                          Components below this are flagged as anomalies
            threshold_power_deviation: Number of standard deviations from mean
                                      power to consider anomalous
        
        Returns:
            Dictionary containing:
            - 'anomalous_frequencies': List of FrequencyComponent objects
                                      identified as anomalies
            - 'low_snr_frequencies': Frequencies with poor SNR
            - 'high_power_anomalies': Frequencies with unusually high power
            - 'low_power_anomalies': Frequencies with unusually low power
            - 'statistics': Statistical summary of anomaly detection
        """
        anomalies: List[FrequencyComponent] = []
        low_snr: List[FrequencyComponent] = []
        high_power: List[FrequencyComponent] = []
        low_power: List[FrequencyComponent] = []
        
        if not result.dominant_frequencies:
            return {
                'anomalous_frequencies': [],
                'low_snr_frequencies': [],
                'high_power_anomalies': [],
                'low_power_anomalies': [],
                'statistics': {
                    'total_components': 0,
                    'anomaly_count': 0,
                    'anomaly_percentage': 0.0
                }
            }
        
        # Calculate statistics
        powers = np.array([comp.power for comp in result.dominant_frequencies])
        mean_power = np.mean(powers)
        std_power = np.std(powers)
        
        # Identify anomalies
        for comp in result.dominant_frequencies:
            is_anomaly = False
            
            # Check SNR (if we can estimate it for this component)
            # Simplified: compare component power to median of all powers
            median_power = np.median(powers)
            component_snr_estimate = 10 * np.log10(
                comp.power / (median_power + 1e-10)
            )
            
            if component_snr_estimate < threshold_snr:
                low_snr.append(comp)
                is_anomaly = True
            
            # Check power deviation
            if std_power > 0:
                z_score = (comp.power - mean_power) / std_power
                
                if z_score > threshold_power_deviation:
                    high_power.append(comp)
                    is_anomaly = True
                elif z_score < -threshold_power_deviation:
                    low_power.append(comp)
                    is_anomaly = True
            
            if is_anomaly:
                anomalies.append(comp)
        
        # Remove duplicates (a component can be in multiple categories)
        unique_anomalies = list({
            id(comp): comp for comp in anomalies
        }.values())
        
        return {
            'anomalous_frequencies': unique_anomalies,
            'low_snr_frequencies': low_snr,
            'high_power_anomalies': high_power,
            'low_power_anomalies': low_power,
            'statistics': {
                'total_components': len(result.dominant_frequencies),
                'anomaly_count': len(unique_anomalies),
                'anomaly_percentage': (
                    len(unique_anomalies) / len(result.dominant_frequencies) * 100
                    if result.dominant_frequencies else 0.0
                ),
                'mean_power': float(mean_power),
                'std_power': float(std_power),
                'median_power': float(np.median(powers))
            }
        }
    
    def export_results(
        self,
        result: FrequencyAnalysisResult,
        filepath: str,
        format: str = 'json',
        include_spectrum: bool = True,
        include_psd: bool = True
    ) -> str:
        """
        Export frequency analysis results to file in various formats.
        
        Supports multiple export formats for integration with other tools,
        data archival, and visualization. All formats preserve the essential
        frequency analysis information.
        
        Args:
            result: FrequencyAnalysisResult to export
            filepath: Path to output file (extension may be added if missing)
            format: Export format ('json', 'csv', 'numpy', 'npz', 'matlab')
            include_spectrum: If True, includes full frequency spectrum
            include_psd: If True, includes full power spectral density
        
        Returns:
            Path to the exported file (may differ from input if extension added)
        
        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be written
            RuntimeError: If export fails
        
        Example:
            >>> analyzer = FrequencyAnalyzer(sampling_rate=1000.0)
            >>> result = analyzer.analyze_acceleration(accel_data)
            >>> analyzer.export_results(result, 'analysis.json', format='json')
            'analysis.json'
        """
        import json
        import csv
        from pathlib import Path
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        format_lower = format.lower()
        
        # Add extension if missing
        extensions = {
            'json': '.json',
            'csv': '.csv',
            'numpy': '.npz',
            'npz': '.npz',
            'np': '.npz',
            'matlab': '.mat'
        }
        
        if format_lower in extensions and not filepath.suffix:
            filepath = filepath.with_suffix(extensions[format_lower])
        
        try:
            if format_lower == 'json':
                export_data = {
                    'metadata': {
                        'sampling_rate': result.sampling_rate,
                        'total_power': result.total_power,
                        'signal_to_noise_ratio': result.signal_to_noise_ratio,
                        'fundamental_frequency': result.fundamental_frequency,
                        'bandwidth': result.bandwidth
                    },
                    'dominant_frequencies': [
                        {
                            'frequency': comp.frequency,
                            'amplitude': comp.amplitude,
                            'phase': comp.phase,
                            'power': comp.power,
                            'harmonic_number': comp.harmonic_number
                        }
                        for comp in result.dominant_frequencies
                    ],
                    'harmonics': [
                        {
                            'frequency': h.frequency,
                            'amplitude': h.amplitude,
                            'phase': h.phase,
                            'power': h.power,
                            'harmonic_number': h.harmonic_number
                        }
                        for h in result.harmonics
                    ]
                }
                
                if include_spectrum:
                    export_data['frequency_spectrum'] = {
                        'frequencies': result.frequency_spectrum[:, 0].tolist(),
                        'amplitudes': result.frequency_spectrum[:, 1].tolist()
                    }
                
                if include_psd:
                    export_data['power_spectral_density'] = {
                        'frequencies': result.power_spectral_density[:, 0].tolist(),
                        'psd': result.power_spectral_density[:, 1].tolist()
                    }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format_lower == 'csv':
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Frequency (Hz)', 'Amplitude', 'PSD', 'Phase (rad)'
                    ])
                    
                    # Write spectrum data
                    for i in range(len(result.frequency_spectrum)):
                        freq = result.frequency_spectrum[i, 0]
                        amp = result.frequency_spectrum[i, 1]
                        psd_val = result.power_spectral_density[i, 1]
                        phase = 0.0  # Phase not directly in spectrum
                        
                        # Try to get phase from dominant frequencies
                        for comp in result.dominant_frequencies:
                            if abs(comp.frequency - freq) < 0.01:
                                phase = comp.phase
                                break
                        
                        writer.writerow([freq, amp, psd_val, phase])
            
            elif format_lower in ['numpy', 'npz', 'np']:
                np.savez(
                    filepath,
                    frequencies=result.frequency_spectrum[:, 0],
                    amplitudes=result.frequency_spectrum[:, 1],
                    psd_frequencies=result.power_spectral_density[:, 0],
                    psd=result.power_spectral_density[:, 1],
                    dominant_frequencies=np.array([
                        c.frequency for c in result.dominant_frequencies
                    ]),
                    dominant_amplitudes=np.array([
                        c.amplitude for c in result.dominant_frequencies
                    ]),
                    dominant_powers=np.array([
                        c.power for c in result.dominant_frequencies
                    ]),
                    dominant_phases=np.array([
                        c.phase for c in result.dominant_frequencies
                    ]),
                    sampling_rate=result.sampling_rate,
                    total_power=result.total_power,
                    snr=result.signal_to_noise_ratio,
                    fundamental_frequency=(
                        result.fundamental_frequency
                        if result.fundamental_frequency is not None
                        else np.nan
                    ),
                    bandwidth=result.bandwidth
                )
            
            elif format_lower == 'matlab':
                try:
                    from scipy.io import savemat
                    
                    mat_data = {
                        'frequencies': result.frequency_spectrum[:, 0],
                        'amplitudes': result.frequency_spectrum[:, 1],
                        'psd_frequencies': result.power_spectral_density[:, 0],
                        'psd': result.power_spectral_density[:, 1],
                        'sampling_rate': result.sampling_rate,
                        'total_power': result.total_power,
                        'snr': result.signal_to_noise_ratio,
                        'fundamental_frequency': (
                            result.fundamental_frequency
                            if result.fundamental_frequency is not None
                            else np.nan
                        ),
                        'bandwidth': result.bandwidth
                    }
                    
                    savemat(filepath, mat_data)
                except ImportError:
                    raise ValueError(
                        "scipy.io.savemat not available. "
                        "Install scipy with MATLAB support or use another format."
                    )
            
            else:
                raise ValueError(
                    f"Unsupported format '{format}'. "
                    f"Supported formats: 'json', 'csv', 'numpy', 'matlab'"
                )
            
            logger.info(
                f"Results exported to {filepath} in {format} format "
                f"({filepath.stat().st_size / 1024:.2f} KB)"
            )
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Error exporting results to {filepath}: {e}")
            raise RuntimeError(f"Failed to export results: {e}") from e
    
    def get_statistical_summary(
        self, result: FrequencyAnalysisResult
    ) -> Dict[str, Union[float, int, Dict[str, float]]]:
        """
        Generate comprehensive statistical summary of frequency analysis.
        
        This method provides detailed statistics about the frequency content,
        including distribution metrics, power statistics, and frequency spread.
        Useful for characterizing signal properties and comparing different signals.
        
        Args:
            result: FrequencyAnalysisResult to summarize
        
        Returns:
            Dictionary containing:
            - 'frequency_statistics': Mean, std, min, max of dominant frequencies
            - 'power_statistics': Mean, std, min, max of power values
            - 'amplitude_statistics': Mean, std, min, max of amplitudes
            - 'frequency_spread': Range and variance of frequencies
            - 'power_distribution': Percentiles of power distribution
            - 'dominant_count': Number of dominant frequencies
            - 'harmonic_count': Number of harmonics
        
        Example:
            >>> result = analyzer.analyze_acceleration(accel_data)
            >>> stats = analyzer.get_statistical_summary(result)
            >>> print(f"Mean frequency: {stats['frequency_statistics']['mean']:.2f} Hz")
        """
        if not result.dominant_frequencies:
            return {
                'frequency_statistics': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'power_statistics': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'amplitude_statistics': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'frequency_spread': {'range': 0.0, 'variance': 0.0},
                'power_distribution': {'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p95': 0.0},
                'dominant_count': 0,
                'harmonic_count': len(result.harmonics)
            }
        
        frequencies = np.array([comp.frequency for comp in result.dominant_frequencies])
        powers = np.array([comp.power for comp in result.dominant_frequencies])
        amplitudes = np.array([comp.amplitude for comp in result.dominant_frequencies])
        
        return {
            'frequency_statistics': {
                'mean': float(np.mean(frequencies)),
                'std': float(np.std(frequencies)),
                'min': float(np.min(frequencies)),
                'max': float(np.max(frequencies)),
                'median': float(np.median(frequencies))
            },
            'power_statistics': {
                'mean': float(np.mean(powers)),
                'std': float(np.std(powers)),
                'min': float(np.min(powers)),
                'max': float(np.max(powers)),
                'median': float(np.median(powers)),
                'total': float(result.total_power)
            },
            'amplitude_statistics': {
                'mean': float(np.mean(amplitudes)),
                'std': float(np.std(amplitudes)),
                'min': float(np.min(amplitudes)),
                'max': float(np.max(amplitudes)),
                'median': float(np.median(amplitudes))
            },
            'frequency_spread': {
                'range': float(np.max(frequencies) - np.min(frequencies)),
                'variance': float(np.var(frequencies)),
                'iqr': float(np.percentile(frequencies, 75) - np.percentile(frequencies, 25))
            },
            'power_distribution': {
                'p25': float(np.percentile(powers, 25)),
                'p50': float(np.percentile(powers, 50)),
                'p75': float(np.percentile(powers, 75)),
                'p95': float(np.percentile(powers, 95)),
                'p99': float(np.percentile(powers, 99))
            },
            'dominant_count': len(result.dominant_frequencies),
            'harmonic_count': len(result.harmonics),
            'fundamental_frequency': result.fundamental_frequency,
            'bandwidth': result.bandwidth,
            'snr': result.signal_to_noise_ratio
        }
    
    def validate_signal_quality(
        self, signal_data: np.ndarray, min_snr: float = 10.0
    ) -> Dict[str, Union[bool, float, List[str]]]:
        """
        Validate signal quality before analysis.
        
        This method performs pre-analysis checks to ensure the signal is
        suitable for frequency analysis. Identifies potential issues that
        could affect analysis quality.
        
        Args:
            signal_data: Signal array to validate
            min_snr: Minimum expected SNR in dB (for warning, not rejection)
        
        Returns:
            Dictionary containing:
            - 'is_valid': Boolean indicating if signal is suitable for analysis
            - 'quality_score': Quality score from 0.0 (poor) to 1.0 (excellent)
            - 'warnings': List of warning messages
            - 'issues': List of critical issues
            - 'metrics': Dictionary of quality metrics
        
        Example:
            >>> validation = analyzer.validate_signal_quality(accel_data)
            >>> if not validation['is_valid']:
            ...     print(f"Issues: {validation['issues']}")
        """
        warnings: List[str] = []
        issues: List[str] = []
        metrics: Dict[str, float] = {}
        quality_score = 1.0
        
        # Check for empty signal
        if signal_data.size == 0:
            issues.append("Signal is empty")
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'warnings': warnings,
                'issues': issues,
                'metrics': metrics
            }
        
        # Check for NaN values
        nan_count = np.sum(np.isnan(signal_data))
        if nan_count > 0:
            issues.append(f"Signal contains {nan_count} NaN values")
            quality_score -= 0.3
            signal_data = np.nan_to_num(signal_data, nan=0.0)
        
        # Check for Inf values
        inf_count = np.sum(np.isinf(signal_data))
        if inf_count > 0:
            issues.append(f"Signal contains {inf_count} Inf values")
            quality_score -= 0.3
            signal_data = np.nan_to_num(signal_data, posinf=np.max(np.abs(signal_data[~np.isinf(signal_data)])), neginf=-np.max(np.abs(signal_data[~np.isinf(signal_data)])))
        
        # Check signal length
        if len(signal_data) < self.nperseg:
            warnings.append(
                f"Signal length ({len(signal_data)}) is less than segment length ({self.nperseg})"
            )
            quality_score -= 0.1
        
        # Check for constant signal
        if np.std(signal_data) < 1e-10:
            issues.append("Signal is constant (no variation)")
            quality_score -= 0.5
        
        # Check for clipping (saturation)
        max_val = np.max(np.abs(signal_data))
        if max_val > 0.95 * np.finfo(signal_data.dtype).max:
            warnings.append("Signal may be clipped (near maximum value)")
            quality_score -= 0.1
        
        # Estimate SNR (simplified)
        signal_power = np.var(signal_data)
        # Estimate noise as high-frequency content
        if len(signal_data) > 10:
            high_freq_power = np.var(np.diff(signal_data, n=2))
            estimated_snr = 10 * np.log10(signal_power / (high_freq_power + 1e-10))
            metrics['estimated_snr'] = float(estimated_snr)
            
            if estimated_snr < min_snr:
                warnings.append(f"Low estimated SNR: {estimated_snr:.2f} dB")
                quality_score -= 0.2
        
        # Check dynamic range
        dynamic_range = np.max(signal_data) - np.min(signal_data)
        metrics['dynamic_range'] = float(dynamic_range)
        
        if dynamic_range < 1e-6:
            warnings.append("Very small dynamic range")
            quality_score -= 0.1
        
        # Check for excessive noise
        if len(signal_data) > 100:
            # Check if signal looks like noise (high frequency content)
            diff_signal = np.diff(signal_data)
            if np.std(diff_signal) > 2 * np.std(signal_data):
                warnings.append("Signal appears to be mostly noise")
                quality_score -= 0.2
        
        quality_score = max(0.0, quality_score)
        is_valid = len(issues) == 0 and quality_score > 0.5
        
        return {
            'is_valid': is_valid,
            'quality_score': float(quality_score),
            'warnings': warnings,
            'issues': issues,
            'metrics': metrics
        }
    
    def get_frequency_resolution(self) -> float:
        """
        Get the frequency resolution of the analysis.
        
        Frequency resolution determines the smallest frequency difference
        that can be distinguished. It depends on the sampling rate and
        the number of samples (or segment length for Welch's method).
        
        Returns:
            Frequency resolution in Hz
        
        Formula:
            resolution = sampling_rate / n_samples (for FFT)
            resolution ≈ sampling_rate / nperseg (for Welch)
        """
        if self.method == FrequencyAnalysisMethod.WELCH:
            return self.sampling_rate / self.nperseg
        else:
            # For FFT, resolution depends on signal length
            # Return typical resolution based on nperseg
            return self.sampling_rate / self.nperseg
    
    def get_nyquist_frequency(self) -> float:
        """
        Get the Nyquist frequency (maximum analyzable frequency).
        
        The Nyquist frequency is half the sampling rate and represents
        the highest frequency that can be accurately analyzed without
        aliasing.
        
        Returns:
            Nyquist frequency in Hz
        """
        return self.sampling_rate / 2.0
    
    def plot_analysis(
        self,
        result: FrequencyAnalysisResult,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (15, 10)
    ) -> Optional[object]:
        """
        Create comprehensive visualization of frequency analysis results.
        
        Generates a multi-panel plot showing:
        - Frequency spectrum (amplitude vs frequency)
        - Power spectral density
        - Dominant frequencies highlighted
        - Harmonic relationships
        
        Args:
            result: FrequencyAnalysisResult to visualize
            save_path: Optional path to save the figure. If None, figure is not saved
            show: If True, displays the figure (requires matplotlib)
            figsize: Figure size as (width, height) in inches
        
        Returns:
            Matplotlib figure object if matplotlib is available, None otherwise
        
        Raises:
            ImportError: If matplotlib is not installed
        
        Example:
            >>> result = analyzer.analyze_acceleration(accel_data)
            >>> analyzer.plot_analysis(result, save_path='analysis.png')
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Frequency Spectrum
        ax1 = fig.add_subplot(gs[0, :])
        frequencies = result.frequency_spectrum[:, 0]
        amplitudes = result.frequency_spectrum[:, 1]
        ax1.plot(frequencies, amplitudes, 'b-', linewidth=0.8, alpha=0.7, label='Spectrum')
        
        # Highlight dominant frequencies
        for comp in result.dominant_frequencies[:10]:  # Top 10
            ax1.axvline(comp.frequency, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax1.plot(comp.frequency, comp.amplitude, 'ro', markersize=8, alpha=0.7)
        
        ax1.set_xlabel('Frequency (Hz)', fontsize=11)
        ax1.set_ylabel('Amplitude', fontsize=11)
        ax1.set_title('Frequency Spectrum with Dominant Frequencies', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, min(np.max(frequencies), self.get_nyquist_frequency()))
        
        # 2. Power Spectral Density
        ax2 = fig.add_subplot(gs[1, 0])
        psd_freqs = result.power_spectral_density[:, 0]
        psd = result.power_spectral_density[:, 1]
        ax2.semilogy(psd_freqs, psd, 'g-', linewidth=1.0)
        ax2.set_xlabel('Frequency (Hz)', fontsize=10)
        ax2.set_ylabel('PSD (log scale)', fontsize=10)
        ax2.set_title('Power Spectral Density', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(np.max(psd_freqs), self.get_nyquist_frequency()))
        
        # 3. Dominant Frequencies Bar Chart
        ax3 = fig.add_subplot(gs[1, 1])
        if result.dominant_frequencies:
            top_freqs = result.dominant_frequencies[:10]
            freq_vals = [comp.frequency for comp in top_freqs]
            power_vals = [comp.power for comp in top_freqs]
            ax3.barh(range(len(freq_vals)), power_vals, color='coral', alpha=0.7)
            ax3.set_yticks(range(len(freq_vals)))
            ax3.set_yticklabels([f'{f:.2f} Hz' for f in freq_vals], fontsize=9)
            ax3.set_xlabel('Power', fontsize=10)
            ax3.set_title('Top 10 Dominant Frequencies', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Harmonic Relationships
        ax4 = fig.add_subplot(gs[2, 0])
        if result.harmonics:
            harmonic_nums = [h.harmonic_number for h in result.harmonics if h.harmonic_number is not None]
            harmonic_powers = [h.power for h in result.harmonics if h.harmonic_number is not None]
            if harmonic_nums:
                ax4.bar(harmonic_nums, harmonic_powers, color='purple', alpha=0.7)
                ax4.set_xlabel('Harmonic Number', fontsize=10)
                ax4.set_ylabel('Power', fontsize=10)
                ax4.set_title('Harmonic Power Distribution', fontsize=11, fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Summary Statistics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        summary_text = f"""
        Analysis Summary
        {'='*30}
        Fundamental Frequency: {result.fundamental_frequency:.2f} Hz
        Total Power: {result.total_power:.4e}
        SNR: {result.signal_to_noise_ratio:.2f} dB
        Bandwidth: {result.bandwidth:.2f} Hz
        Dominant Frequencies: {len(result.dominant_frequencies)}
        Harmonics: {len(result.harmonics)}
        Sampling Rate: {result.sampling_rate:.1f} Hz
        Frequency Resolution: {self.get_frequency_resolution():.4f} Hz
        """
        
        ax5.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Frequency Analysis Results', fontsize=14, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def get_spectral_features(
        self, result: FrequencyAnalysisResult
    ) -> Dict[str, float]:
        """
        Extract spectral features for machine learning or classification.
        
        This method computes a set of features that characterize the frequency
        content of the signal. These features can be used for:
        - Signal classification
        - Machine learning models
        - Pattern recognition
        - Anomaly detection
        
        Args:
            result: FrequencyAnalysisResult to extract features from
        
        Returns:
            Dictionary of spectral features:
            - 'spectral_centroid': Center of mass of the spectrum
            - 'spectral_spread': Spread of the spectrum
            - 'spectral_skewness': Skewness of the spectrum
            - 'spectral_kurtosis': Kurtosis of the spectrum
            - 'spectral_rolloff': Frequency below which 85% of energy is contained
            - 'spectral_flux': Rate of change of the spectrum
            - 'zero_crossing_rate': Rate of sign changes
            - 'spectral_flatness': Measure of noisiness (0=tonal, 1=noisy)
            - 'spectral_crest': Ratio of peak to mean
            - 'spectral_slope': Linear slope of the spectrum
        
        Example:
            >>> result = analyzer.analyze_acceleration(accel_data)
            >>> features = analyzer.get_spectral_features(result)
            >>> # Use features for classification
        """
        frequencies = result.frequency_spectrum[:, 0]
        amplitudes = result.frequency_spectrum[:, 1]
        psd = result.power_spectral_density[:, 1]
        
        # Avoid division by zero
        amplitudes = amplitudes + 1e-10
        psd = psd + 1e-10
        
        # Spectral centroid (center of mass)
        spectral_centroid = np.sum(frequencies * amplitudes) / np.sum(amplitudes)
        
        # Spectral spread (standard deviation around centroid)
        spectral_spread = np.sqrt(
            np.sum(((frequencies - spectral_centroid) ** 2) * amplitudes) / np.sum(amplitudes)
        )
        
        # Spectral skewness
        spectral_skewness = (
            np.sum(((frequencies - spectral_centroid) ** 3) * amplitudes) /
            (np.sum(amplitudes) * (spectral_spread ** 3))
        )
        
        # Spectral kurtosis
        spectral_kurtosis = (
            np.sum(((frequencies - spectral_centroid) ** 4) * amplitudes) /
            (np.sum(amplitudes) * (spectral_spread ** 4))
        )
        
        # Spectral rolloff (85% energy)
        cumsum_energy = np.cumsum(psd)
        total_energy = cumsum_energy[-1]
        rolloff_idx = np.where(cumsum_energy >= 0.85 * total_energy)[0]
        spectral_rolloff = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else frequencies[-1]
        
        # Spectral flux (rate of change)
        spectral_flux = np.mean(np.abs(np.diff(amplitudes)))
        
        # Spectral flatness (geometric mean / arithmetic mean)
        geometric_mean = np.exp(np.mean(np.log(amplitudes + 1e-10)))
        arithmetic_mean = np.mean(amplitudes)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Spectral crest (peak / mean)
        spectral_crest = np.max(amplitudes) / (np.mean(amplitudes) + 1e-10)
        
        # Spectral slope (linear fit)
        if len(frequencies) > 1:
            slope, _ = np.polyfit(frequencies, amplitudes, 1)
        else:
            slope = 0.0
        
        return {
            'spectral_centroid': float(spectral_centroid),
            'spectral_spread': float(spectral_spread),
            'spectral_skewness': float(spectral_skewness),
            'spectral_kurtosis': float(spectral_kurtosis),
            'spectral_rolloff': float(spectral_rolloff),
            'spectral_flux': float(spectral_flux),
            'spectral_flatness': float(spectral_flatness),
            'spectral_crest': float(spectral_crest),
            'spectral_slope': float(slope),
            'fundamental_frequency': result.fundamental_frequency or 0.0,
            'total_power': float(result.total_power),
            'snr': result.signal_to_noise_ratio,
            'bandwidth': result.bandwidth
        }
    
    def analyze_robot_motion_pattern(
        self,
        acceleration_data: np.ndarray,
        encoder_data: Optional[np.ndarray] = None
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Analyze robot motion pattern from acceleration and encoder data.
        
        This method performs comprehensive analysis to identify the type of
        robot motion (walking, running, arm motion, etc.) based on frequency
        content. It combines acceleration and encoder analysis to provide
        a complete picture of the motion pattern.
        
        Args:
            acceleration_data: Acceleration data array (1D or 2D)
            encoder_data: Optional encoder data array (1D). If provided,
                        enables more accurate motion classification
        
        Returns:
            Dictionary containing:
            - 'motion_type': Identified motion type ('walking', 'running', 'arm_motion', etc.)
            - 'confidence': Confidence score (0.0 to 1.0) for the classification
            - 'dominant_band': Name of the frequency band with highest power
            - 'band_powers': Dictionary of power in each motion band
            - 'frequency_characteristics': Dictionary of frequency metrics
            - 'acceleration_result': FrequencyAnalysisResult for acceleration
            - 'encoder_result': FrequencyAnalysisResult for encoder (if provided)
        
        Example:
            >>> analyzer = FrequencyAnalyzer(sampling_rate=1000.0)
            >>> pattern = analyzer.analyze_robot_motion_pattern(accel_data, encoder_data)
            >>> print(f"Motion type: {pattern['motion_type']}")
            >>> print(f"Confidence: {pattern['confidence']:.2%}")
        """
        # Analyze acceleration
        accel_result = self.analyze_acceleration(
            acceleration_data,
            remove_dc=True,
            apply_filter=True
        )
        
        # Analyze encoder if provided
        encoder_result: Optional[FrequencyAnalysisResult] = None
        if encoder_data is not None:
            encoder_result = self.analyze_encoder(
                encoder_data,
                remove_trend=True,
                apply_filter=True
            )
        
        # Analyze frequency bands
        bands = MotionFrequencyBands.get_all_bands()
        band_analysis = self.analyze_frequency_bands(accel_result, bands)
        
        # Find dominant band (highest power percentage)
        dominant_band = max(
            band_analysis.items(),
            key=lambda x: x[1]['power_percentage']
        )[0] if band_analysis else 'unknown'
        
        # Calculate band powers
        band_powers = {
            name: data['power_percentage']
            for name, data in band_analysis.items()
        }
        
        # Classify motion type based on dominant band and power distribution
        motion_type = 'unknown'
        confidence = 0.0
        
        if dominant_band == 'walking' and band_powers.get('walking', 0) > 30:
            motion_type = 'walking'
            confidence = min(1.0, band_powers.get('walking', 0) / 50.0)
        elif dominant_band == 'running' and band_powers.get('running', 0) > 30:
            motion_type = 'running'
            confidence = min(1.0, band_powers.get('running', 0) / 50.0)
        elif dominant_band == 'arm_motion' and band_powers.get('arm_motion', 0) > 25:
            motion_type = 'arm_motion'
            confidence = min(1.0, band_powers.get('arm_motion', 0) / 40.0)
        elif band_powers.get('vibration', 0) > 40:
            motion_type = 'vibration'
            confidence = min(1.0, band_powers.get('vibration', 0) / 60.0)
        elif band_powers.get('noise', 0) > 50:
            motion_type = 'noise_dominant'
            confidence = min(1.0, band_powers.get('noise', 0) / 70.0)
        else:
            # Mixed or unclear pattern
            motion_type = 'mixed'
            max_power = max(band_powers.values()) if band_powers else 0
            confidence = max_power / 100.0
        
        # Get frequency characteristics
        frequency_characteristics = {
            'fundamental_frequency': accel_result.fundamental_frequency or 0.0,
            'bandwidth': accel_result.bandwidth,
            'snr': accel_result.signal_to_noise_ratio,
            'total_power': accel_result.total_power,
            'dominant_frequency_count': len(accel_result.dominant_frequencies)
        }
        
        # Add encoder characteristics if available
        if encoder_result is not None:
            frequency_characteristics['encoder_fundamental'] = (
                encoder_result.fundamental_frequency or 0.0
            )
            frequency_characteristics['encoder_bandwidth'] = encoder_result.bandwidth
        
        return {
            'motion_type': motion_type,
            'confidence': float(confidence),
            'dominant_band': dominant_band,
            'band_powers': band_powers,
            'frequency_characteristics': frequency_characteristics,
            'acceleration_result': accel_result,
            'encoder_result': encoder_result
        }
    
    def get_frequency_component_summary(
        self, result: FrequencyAnalysisResult, top_n: int = 10
    ) -> Dict[str, Union[List[Dict[str, float]], Dict[str, float]]]:
        """
        Generate a human-readable summary of frequency components.
        
        This method creates a structured summary of the most important
        frequency components, making it easy to understand the frequency
        content of the signal at a glance.
        
        Args:
            result: FrequencyAnalysisResult to summarize
            top_n: Number of top frequency components to include (default: 10)
        
        Returns:
            Dictionary containing:
            - 'top_frequencies': List of top N frequency components with details
            - 'summary': Dictionary with summary statistics
            - 'harmonics_info': Information about harmonic relationships
        
        Example:
            >>> result = analyzer.analyze_acceleration(accel_data)
            >>> summary = analyzer.get_frequency_component_summary(result, top_n=5)
            >>> for comp in summary['top_frequencies']:
            ...     print(f"{comp['frequency']:.2f} Hz: {comp['power']:.2e}")
        """
        # Get top N dominant frequencies
        top_frequencies = result.dominant_frequencies[:top_n]
        
        # Format frequency components
        formatted_components = [
            {
                'frequency': comp.frequency,
                'amplitude': comp.amplitude,
                'power': comp.power,
                'phase': comp.phase,
                'harmonic_number': comp.harmonic_number,
                'power_percentage': (
                    comp.power / result.total_power * 100.0
                    if result.total_power > 0 else 0.0
                )
            }
            for comp in top_frequencies
        ]
        
        # Create summary statistics
        summary = {
            'total_components': len(result.dominant_frequencies),
            'fundamental_frequency': result.fundamental_frequency,
            'total_power': result.total_power,
            'snr_db': result.signal_to_noise_ratio,
            'bandwidth_hz': result.bandwidth,
            'harmonic_count': len(result.harmonics)
        }
        
        # Harmonic information
        harmonics_info = {
            'has_harmonics': len(result.harmonics) > 0,
            'harmonic_count': len(result.harmonics),
            'fundamental_power': 0.0,
            'harmonic_power_total': sum(h.power for h in result.harmonics)
        }
        
        if result.fundamental_frequency is not None:
            for comp in result.dominant_frequencies:
                if abs(comp.frequency - result.fundamental_frequency) < 0.1:
                    harmonics_info['fundamental_power'] = comp.power
                    break
        
        if harmonics_info['fundamental_power'] > 0:
            harmonics_info['thd_percentage'] = (
                np.sqrt(harmonics_info['harmonic_power_total'] /
                       harmonics_info['fundamental_power']) * 100.0
            )
        else:
            harmonics_info['thd_percentage'] = 0.0
        
        return {
            'top_frequencies': formatted_components,
            'summary': summary,
            'harmonics_info': harmonics_info
        }
    
    def compare_results(
        self,
        result1: FrequencyAnalysisResult,
        result2: FrequencyAnalysisResult,
        tolerance: float = 0.05
    ) -> Dict[str, Union[List[float], float, Dict[str, float]]]:
        """
        Compare two frequency analysis results and identify similarities/differences.
        
        This method is useful for:
        - Comparing acceleration and encoder analysis results
        - Tracking changes in frequency content over time
        - Validating sensor consistency
        - Detecting system changes
        
        Args:
            result1: First frequency analysis result
            result2: Second frequency analysis result
            tolerance: Relative frequency tolerance for matching (0.05 = 5%)
        
        Returns:
            Dictionary containing:
            - 'common_frequencies': List of frequencies present in both results
            - 'unique_to_result1': Frequencies only in result1
            - 'unique_to_result2': Frequencies only in result2
            - 'frequency_differences': Differences in common frequencies
            - 'power_correlation': Correlation of power at common frequencies
            - 'bandwidth_difference': Difference in bandwidth
            - 'snr_difference': Difference in SNR
        """
        # Extract frequencies from both results
        freq1 = [comp.frequency for comp in result1.dominant_frequencies]
        freq2 = [comp.frequency for comp in result2.dominant_frequencies]
        
        # Find common frequencies
        common: List[float] = []
        unique1: List[float] = []
        unique2: List[float] = []
        freq_diffs: Dict[str, float] = {}
        power_corr_data: List[Tuple[float, float]] = []
        
        # Check frequencies in result1
        for comp1 in result1.dominant_frequencies:
            matched = False
            for comp2 in result2.dominant_frequencies:
                rel_diff = abs(comp1.frequency - comp2.frequency) / max(
                    comp1.frequency, comp2.frequency
                )
                if rel_diff < tolerance:
                    common.append((comp1.frequency + comp2.frequency) / 2)
                    freq_diffs[f"{comp1.frequency:.2f}Hz"] = abs(
                        comp1.frequency - comp2.frequency
                    )
                    power_corr_data.append((comp1.power, comp2.power))
                    matched = True
                    break
            
            if not matched:
                unique1.append(comp1.frequency)
        
        # Check frequencies in result2
        for comp2 in result2.dominant_frequencies:
            matched = False
            for comp1 in result1.dominant_frequencies:
                rel_diff = abs(comp1.frequency - comp2.frequency) / max(
                    comp1.frequency, comp2.frequency
                )
                if rel_diff < tolerance:
                    matched = True
                    break
            
            if not matched:
                unique2.append(comp2.frequency)
        
        # Calculate power correlation
        if power_corr_data:
            powers1, powers2 = zip(*power_corr_data)
            power_correlation = float(np.corrcoef(powers1, powers2)[0, 1])
            if np.isnan(power_correlation):
                power_correlation = 0.0
        else:
            power_correlation = 0.0
        
        return {
            'common_frequencies': sorted(list(set(common))),
            'unique_to_result1': sorted(unique1),
            'unique_to_result2': sorted(unique2),
            'frequency_differences': freq_diffs,
            'power_correlation': power_correlation,
            'bandwidth_difference': abs(result1.bandwidth - result2.bandwidth),
            'snr_difference': abs(
                result1.signal_to_noise_ratio - result2.signal_to_noise_ratio
            )
        }
    
    def analyze_batch(
        self,
        signals: List[np.ndarray],
        signal_types: Optional[List[str]] = None,
        max_workers: Optional[int] = None
    ) -> List[FrequencyAnalysisResult]:
        """
        Analyze multiple signals in batch for efficient processing.
        
        This method processes multiple signals efficiently, optionally in parallel,
        making it ideal for processing large datasets or multiple measurements.
        
        Args:
            signals: List of signal arrays to analyze (each must be 1D)
            signal_types: Optional list of signal type identifiers for each signal.
                        If None, all signals are analyzed as generic signals
            max_workers: Maximum number of parallel workers. If None, uses
                        min(len(signals), CPU count)
        
        Returns:
            List of FrequencyAnalysisResult objects, one for each input signal
        
        Raises:
            ValueError: If signals list is empty or contains invalid signals
        
        Example:
            >>> analyzer = FrequencyAnalyzer(sampling_rate=1000.0)
            >>> signals = [accel_x, accel_y, accel_z, encoder_data]
            >>> results = analyzer.analyze_batch(signals, max_workers=4)
        """
        if not signals:
            raise ValueError("Signals list cannot be empty")
        
        if signal_types is not None and len(signal_types) != len(signals):
            raise ValueError(
                f"signal_types length ({len(signal_types)}) must match "
                f"signals length ({len(signals)})"
            )
        
        # Validate all signals
        for i, sig in enumerate(signals):
            if sig.size == 0:
                raise ValueError(f"Signal {i} is empty")
            if sig.ndim != 1:
                raise ValueError(
                    f"Signal {i} must be 1D, got shape {sig.shape}"
                )
        
        if max_workers is None:
            import os
            max_workers = min(len(signals), os.cpu_count() or 1)
        
        def analyze_single(signal_data: np.ndarray) -> FrequencyAnalysisResult:
            """Analyze a single signal."""
            return self._analyze_signal(signal_data)
        
        # Process in parallel if multiple signals and workers > 1
        if len(signals) > 1 and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(analyze_single, signals))
        else:
            # Sequential processing for single signal or single worker
            results = [analyze_single(sig) for sig in signals]
        
        logger.info(
            f"Analyzed {len(signals)} signals in batch "
            f"(workers: {max_workers if len(signals) > 1 else 1})"
        )
        
        return results
    
    def plot_analysis(
        self,
        result: FrequencyAnalysisResult,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Generate comprehensive visualization of frequency analysis results.
        
        Creates publication-quality plots showing:
        - Power Spectral Density (PSD)
        - Frequency spectrum (amplitude)
        - Dominant frequencies highlighted
        - Harmonics marked
        - Statistical information
        
        Args:
            result: FrequencyAnalysisResult to visualize
            save_path: Optional path to save the figure. If None, figure is not saved
            show_plot: If True, displays the plot. If False, only saves (if path provided)
            figsize: Figure size as (width, height) in inches
        
        Raises:
            ImportError: If matplotlib is not installed
            RuntimeError: If plotting fails
        
        Example:
            >>> analyzer = FrequencyAnalyzer(sampling_rate=1000.0)
            >>> result = analyzer.analyze_acceleration(accel_data)
            >>> analyzer.plot_analysis(result, save_path='analysis.png')
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )
        
        try:
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Plot 1: Power Spectral Density
            ax1 = fig.add_subplot(gs[0, :])
            freqs_psd = result.power_spectral_density[:, 0]
            psd = result.power_spectral_density[:, 1]
            ax1.semilogy(freqs_psd, psd, 'b-', linewidth=1.5, label='PSD')
            
            # Highlight dominant frequencies
            for comp in result.dominant_frequencies[:10]:  # Top 10
                ax1.axvline(
                    comp.frequency,
                    color='r',
                    linestyle='--',
                    alpha=0.5,
                    linewidth=1
                )
                ax1.plot(
                    comp.frequency,
                    comp.power,
                    'ro',
                    markersize=8,
                    label='Dominant' if comp == result.dominant_frequencies[0] else ''
                )
            
            # Highlight harmonics
            for h in result.harmonics[:5]:  # Top 5 harmonics
                ax1.axvline(
                    h.frequency,
                    color='g',
                    linestyle=':',
                    alpha=0.5,
                    linewidth=1
                )
                ax1.plot(
                    h.frequency,
                    h.power,
                    'go',
                    markersize=6,
                    label='Harmonic' if h == result.harmonics[0] else ''
                )
            
            ax1.set_xlabel('Frequency (Hz)', fontsize=11)
            ax1.set_ylabel('Power Spectral Density', fontsize=11)
            ax1.set_title('Power Spectral Density with Dominant Frequencies', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            
            # Plot 2: Amplitude Spectrum
            ax2 = fig.add_subplot(gs[1, 0])
            freqs_amp = result.frequency_spectrum[:, 0]
            amps = result.frequency_spectrum[:, 1]
            ax2.plot(freqs_amp, amps, 'b-', linewidth=1.5)
            ax2.set_xlabel('Frequency (Hz)', fontsize=11)
            ax2.set_ylabel('Amplitude', fontsize=11)
            ax2.set_title('Frequency Spectrum', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Dominant Frequencies Bar Chart
            ax3 = fig.add_subplot(gs[1, 1])
            top_freqs = result.dominant_frequencies[:10]
            if top_freqs:
                freq_vals = [comp.frequency for comp in top_freqs]
                power_vals = [comp.power for comp in top_freqs]
                ax3.barh(range(len(freq_vals)), power_vals, color='steelblue')
                ax3.set_yticks(range(len(freq_vals)))
                ax3.set_yticklabels([f'{f:.2f} Hz' for f in freq_vals])
                ax3.set_xlabel('Power', fontsize=11)
                ax3.set_title('Top 10 Dominant Frequencies', fontsize=12)
                ax3.grid(True, alpha=0.3, axis='x')
            
            # Plot 4: Statistics Text
            ax4 = fig.add_subplot(gs[2, :])
            ax4.axis('off')
            
            stats_text = f"""
            Analysis Statistics:
            • Sampling Rate: {result.sampling_rate:.1f} Hz
            • Total Power: {result.total_power:.4e}
            • Signal-to-Noise Ratio: {result.signal_to_noise_ratio:.2f} dB
            • Bandwidth: {result.bandwidth:.2f} Hz
            • Fundamental Frequency: {result.fundamental_frequency:.2f} Hz (if detected)
            • Number of Dominant Frequencies: {len(result.dominant_frequencies)}
            • Number of Harmonics: {len(result.harmonics)}
            """
            
            if result.fundamental_frequency is None:
                stats_text = stats_text.replace(
                    '• Fundamental Frequency: {result.fundamental_frequency:.2f} Hz (if detected)',
                    '• Fundamental Frequency: Not detected'
                )
            
            ax4.text(
                0.1, 0.5,
                stats_text,
                fontsize=10,
                verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            
            plt.suptitle('Frequency Analysis Results', fontsize=14, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
        
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            raise RuntimeError(f"Failed to create plot: {e}") from e
    
    def detect_resonances(
        self,
        result: FrequencyAnalysisResult,
        q_threshold: float = 10.0,
        min_power_ratio: float = 0.1
    ) -> List[Dict[str, float]]:
        """
        Detect resonance frequencies based on Q-factor analysis.
        
        Resonances are characterized by high Q-factor (narrow, high peaks).
        Q-factor = center_frequency / bandwidth. High Q indicates sharp resonance.
        
        This is useful for:
        - Identifying mechanical resonances
        - Detecting structural vibration modes
        - Finding optimal operating frequencies
        - Avoiding resonance frequencies
        
        Args:
            result: FrequencyAnalysisResult to analyze
            q_threshold: Minimum Q-factor to consider a resonance (default 10.0)
            min_power_ratio: Minimum power relative to maximum to consider
        
        Returns:
            List of dictionaries containing resonance information:
            - 'frequency': Resonance frequency in Hz
            - 'q_factor': Estimated Q-factor
            - 'power': Power at resonance
            - 'bandwidth_estimate': Estimated bandwidth in Hz
        """
        resonances: List[Dict[str, float]] = []
        
        if not result.dominant_frequencies:
            return resonances
        
        # Find maximum power for normalization
        max_power = max(comp.power for comp in result.dominant_frequencies)
        power_threshold = max_power * min_power_ratio
        
        # Estimate bandwidth for each dominant frequency
        # Simplified: use power drop-off around peak
        freqs = result.power_spectral_density[:, 0]
        psd = result.power_spectral_density[:, 1]
        
        for comp in result.dominant_frequencies:
            if comp.power < power_threshold:
                continue
            
            # Find frequency index
            freq_idx = np.argmin(np.abs(freqs - comp.frequency))
            
            # Estimate bandwidth: find frequencies where power drops to 1/sqrt(2) of peak
            # (3dB bandwidth)
            half_power = comp.power / np.sqrt(2)
            
            # Find left and right boundaries
            left_idx = freq_idx
            right_idx = freq_idx
            
            # Search left
            while left_idx > 0 and psd[left_idx] > half_power:
                left_idx -= 1
            
            # Search right
            while right_idx < len(psd) - 1 and psd[right_idx] > half_power:
                right_idx += 1
            
            # Calculate bandwidth
            if right_idx > left_idx:
                bandwidth_est = freqs[right_idx] - freqs[left_idx]
            else:
                bandwidth_est = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            
            # Calculate Q-factor
            if bandwidth_est > 0 and comp.frequency > 0:
                q_factor = comp.frequency / bandwidth_est
                
                if q_factor >= q_threshold:
                    resonances.append({
                        'frequency': comp.frequency,
                        'q_factor': q_factor,
                        'power': comp.power,
                        'bandwidth_estimate': bandwidth_est
                    })
        
        # Sort by Q-factor (highest first)
        resonances.sort(key=lambda x: x['q_factor'], reverse=True)
        
        return resonances
    
    def apply_adaptive_filter(
        self,
        signal_data: np.ndarray,
        noise_estimation_method: str = 'median'
    ) -> np.ndarray:
        """
        Apply adaptive filter based on signal characteristics.
        
        Automatically estimates noise level and adjusts filter parameters
        to preserve signal while removing noise. This is particularly useful
        for signals with varying noise characteristics.
        
        Args:
            signal_data: Input signal array (1D)
            noise_estimation_method: Method for noise estimation
                                   ('median', 'std', 'percentile')
        
        Returns:
            Filtered signal array
        
        Raises:
            ValueError: If signal_data is invalid
        """
        if signal_data.size == 0:
            raise ValueError("Signal data cannot be empty")
        
        if signal_data.ndim != 1:
            raise ValueError(f"Signal data must be 1D, got shape {signal_data.shape}")
        
        # Estimate noise level
        if noise_estimation_method == 'median':
            noise_level = np.median(np.abs(np.diff(signal_data)))
        elif noise_estimation_method == 'std':
            noise_level = np.std(signal_data)
        elif noise_estimation_method == 'percentile':
            noise_level = np.percentile(np.abs(signal_data), 25)
        else:
            logger.warning(f"Unknown noise estimation method {noise_estimation_method}, using median")
            noise_level = np.median(np.abs(np.diff(signal_data)))
        
        # Estimate signal power
        signal_power = np.var(signal_data)
        
        # Adaptive cutoff based on signal-to-noise ratio
        if signal_power > 0 and noise_level > 0:
            snr_estimate = signal_power / (noise_level ** 2 + 1e-10)
            # Higher SNR -> can use higher cutoff
            # Lower SNR -> need lower cutoff to remove noise
            cutoff_ratio = min(0.5, max(0.1, 1.0 / (1.0 + 1.0 / snr_estimate)))
        else:
            cutoff_ratio = 0.25  # Default conservative cutoff
        
        cutoff = self.sampling_rate * cutoff_ratio / 2.0
        
        # Apply low-pass filter
        filtered = self._apply_lowpass_filter(signal_data, cutoff)
        
        logger.debug(
            f"Adaptive filter applied: noise_level={noise_level:.4f}, "
            f"cutoff={cutoff:.2f} Hz"
        )
        
        return filtered
    
    def generate_analysis_report(
        self,
        result: FrequencyAnalysisResult,
        include_details: bool = True,
        max_harmonics: int = 10,
        max_dominant: int = 10
    ) -> str:
        """
        Generate a comprehensive text report of frequency analysis results.
        
        Creates a human-readable report summarizing all key findings from
        the frequency analysis, including dominant frequencies, harmonics,
        statistics, and quality metrics. The report is formatted for easy
        reading and can be saved to a file or printed to console.
        
        This method is particularly useful for:
        - Generating analysis reports for documentation
        - Quick overview of frequency content
        - Sharing analysis results with non-technical stakeholders
        - Debugging and validation of analysis results
        
        Args:
            result: FrequencyAnalysisResult to report on
            include_details: If True, includes detailed statistical information
            max_harmonics: Maximum number of harmonics to include in report
            max_dominant: Maximum number of dominant frequencies to include
        
        Returns:
            Multi-line string containing formatted report with sections:
            - Basic Information (sampling rate, frequency range, resolution)
            - Signal Quality Metrics (SNR, total power, bandwidth)
            - Fundamental Frequency & Harmonics
            - Dominant Frequencies
            - Statistical Summary (if include_details=True)
            - Power Distribution by Frequency Bands
        
        Example:
            >>> result = analyzer.analyze_acceleration(accel_data)
            >>> report = analyzer.generate_analysis_report(result)
            >>> print(report)
            >>> # Save to file
            >>> with open('analysis_report.txt', 'w') as f:
            ...     f.write(report)
        """
        report_lines: List[str] = []
        report_lines.append("=" * 70)
        report_lines.append("FREQUENCY ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Basic Information
        report_lines.append("BASIC INFORMATION")
        report_lines.append("-" * 70)
        report_lines.append(f"Sampling Rate: {result.sampling_rate:.1f} Hz")
        report_lines.append(f"Frequency Range: {result.get_frequency_range()[0]:.2f} - {result.get_frequency_range()[1]:.2f} Hz")
        report_lines.append(f"Frequency Resolution: {self.get_frequency_resolution():.4f} Hz")
        report_lines.append(f"Nyquist Frequency: {self.get_nyquist_frequency():.2f} Hz")
        report_lines.append("")
        
        # Signal Quality
        report_lines.append("SIGNAL QUALITY METRICS")
        report_lines.append("-" * 70)
        report_lines.append(f"Signal-to-Noise Ratio: {result.signal_to_noise_ratio:.2f} dB")
        report_lines.append(f"Total Power: {result.total_power:.4e}")
        report_lines.append(f"Effective Bandwidth: {result.bandwidth:.2f} Hz")
        report_lines.append("")
        
        # Fundamental and Harmonics
        report_lines.append("FUNDAMENTAL FREQUENCY & HARMONICS")
        report_lines.append("-" * 70)
        if result.fundamental_frequency is not None:
            report_lines.append(f"Fundamental Frequency: {result.fundamental_frequency:.2f} Hz")
            report_lines.append(f"Number of Harmonics: {len(result.harmonics)}")
            
            if result.harmonics:
                report_lines.append("Harmonic Series:")
                for h in result.harmonics[:max_harmonics]:
                    report_lines.append(
                        f"  Harmonic {h.harmonic_number}: "
                        f"{h.frequency:.2f} Hz, Power: {h.power:.4e}"
                    )
        else:
            report_lines.append("Fundamental Frequency: Not detected")
        report_lines.append("")
        
        # Dominant Frequencies
        report_lines.append("DOMINANT FREQUENCIES")
        report_lines.append("-" * 70)
        report_lines.append(f"Total Dominant Frequencies: {len(result.dominant_frequencies)}")
        
        if result.dominant_frequencies:
            report_lines.append(f"Top {min(max_dominant, len(result.dominant_frequencies))} Dominant Frequencies:")
            for i, comp in enumerate(result.dominant_frequencies[:max_dominant], 1):
                harmonic_info = f" (Harmonic {comp.harmonic_number})" if comp.harmonic_number else ""
                report_lines.append(
                    f"  {i:2d}. {comp.frequency:8.2f} Hz - "
                    f"Power: {comp.power:10.4e}, "
                    f"Amplitude: {comp.amplitude:8.4f}{harmonic_info}"
                )
        report_lines.append("")
        
        # Statistical Summary
        if include_details:
            stats = self.get_statistical_summary(result)
            report_lines.append("STATISTICAL SUMMARY")
            report_lines.append("-" * 70)
            report_lines.append("Frequency Statistics:")
            freq_stats = stats['frequency_statistics']
            report_lines.append(f"  Mean: {freq_stats['mean']:.2f} Hz")
            report_lines.append(f"  Std:  {freq_stats['std']:.2f} Hz")
            report_lines.append(f"  Min:  {freq_stats['min']:.2f} Hz")
            report_lines.append(f"  Max:  {freq_stats['max']:.2f} Hz")
            report_lines.append(f"  Median: {freq_stats['median']:.2f} Hz")
            report_lines.append("")
            report_lines.append("Power Statistics:")
            power_stats = stats['power_statistics']
            report_lines.append(f"  Mean: {power_stats['mean']:.4e}")
            report_lines.append(f"  Std:  {power_stats['std']:.4e}")
            report_lines.append(f"  Min:  {power_stats['min']:.4e}")
            report_lines.append(f"  Max:  {power_stats['max']:.4e}")
            report_lines.append("")
        
        # Frequency Band Analysis
        try:
            distribution = MotionFrequencyBands.get_power_distribution(result)
            report_lines.append("POWER DISTRIBUTION BY FREQUENCY BANDS")
            report_lines.append("-" * 70)
            for band_name, percentage in sorted(
                distribution.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                report_lines.append(f"  {band_name:20s}: {percentage:6.2f}%")
            report_lines.append("")
        except Exception as e:
            logger.warning(f"Could not calculate band distribution: {e}")
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def find_frequency_peaks_in_range(
        self,
        result: FrequencyAnalysisResult,
        freq_min: float,
        freq_max: float,
        min_power_ratio: float = 0.1
    ) -> List[FrequencyComponent]:
        """
        Find all frequency peaks within a specific frequency range.
        
        This method filters dominant frequencies to only include those within
        a specified frequency range. It's particularly useful for:
        - Analyzing specific motion bands (walking, running, vibration)
        - Focusing on frequency ranges of interest
        - Filtering out noise or unwanted frequency components
        - Comparing frequency content across different bands
        
        The method applies both frequency range filtering and power threshold
        filtering to ensure only significant components are returned.
        
        Args:
            result: FrequencyAnalysisResult to search for peaks
            freq_min: Minimum frequency in Hz (inclusive)
            freq_max: Maximum frequency in Hz (inclusive)
            min_power_ratio: Minimum power relative to maximum power (0.0 to 1.0).
                           Components with power below this threshold are excluded.
                           Default: 0.1 (10% of maximum power)
        
        Returns:
            List of FrequencyComponent objects within the specified frequency
            range and above the power threshold, sorted by power (descending).
            Empty list if no peaks match the criteria.
        
        Raises:
            ValueError: If freq_min >= freq_max
        
        Example:
            >>> result = analyzer.analyze_acceleration(accel_data)
            >>> # Find peaks in vibration band (10-100 Hz)
            >>> vibration_peaks = analyzer.find_frequency_peaks_in_range(
            ...     result, 10.0, 100.0, min_power_ratio=0.05
            ... )
            >>> print(f"Found {len(vibration_peaks)} vibration frequencies")
            >>> for peak in vibration_peaks:
            ...     print(f"  {peak.frequency:.2f} Hz: {peak.power:.4e}")
            
            >>> # Find peaks in walking band
            >>> walking_peaks = analyzer.find_frequency_peaks_in_range(
            ...     result, 0.5, 3.0
            ... )
        """
        if freq_min >= freq_max:
            raise ValueError(f"freq_min ({freq_min}) must be less than freq_max ({freq_max})")
        
        if not result.dominant_frequencies:
            return []
        
        max_power = max(comp.power for comp in result.dominant_frequencies)
        min_power = max_power * min_power_ratio
        
        peaks_in_range = [
            comp for comp in result.dominant_frequencies
            if freq_min <= comp.frequency <= freq_max
            and comp.power >= min_power
        ]
        
        # Sort by power (descending)
        peaks_in_range.sort(key=lambda x: x.power, reverse=True)
        
        return peaks_in_range
    
    def calculate_frequency_stability(
        self,
        results: List[FrequencyAnalysisResult],
        reference_frequency: Optional[float] = None,
        tolerance_percent: float = 10.0
    ) -> Dict[str, Union[float, int]]:
        """
        Calculate frequency stability across multiple analysis results.
        
        Measures how consistent frequency components are across multiple
        measurements. This is useful for:
        - Assessing signal stability over time
        - Validating measurement repeatability
        - Detecting frequency drift or instability
        - Quality control in manufacturing or testing
        - Comparing stability across different conditions or sensors
        
        The method tracks a reference frequency across multiple measurements
        and calculates statistical metrics to quantify stability. A high
        stability score indicates consistent frequency measurements.
        
        Args:
            results: List of FrequencyAnalysisResult objects from multiple analyses.
                   Should contain results from the same or similar signals
                   analyzed at different times or conditions
            reference_frequency: Optional reference frequency in Hz to track.
                               If None, uses fundamental frequency from first result.
                               The method finds the closest frequency to this
                               reference in each result
            tolerance_percent: Percentage tolerance for matching frequencies (default: 10%).
                             Frequencies within this tolerance of the reference
                             are considered matches
        
        Returns:
            Dictionary containing stability metrics:
            - 'mean_frequency': Mean frequency across all matched measurements (Hz)
            - 'std_frequency': Standard deviation of frequency (Hz) - lower is more stable
            - 'coefficient_of_variation': CV = std/mean (dimensionless) - normalized stability
            - 'frequency_drift': Maximum frequency change (Hz) - difference between min and max
            - 'stability_score': Overall stability score (0.0 to 1.0) - higher is more stable
            - 'n_measurements': Number of successful frequency matches
        
        Raises:
            ValueError: If results list is empty or no reference frequency can be determined
        
        Example:
            >>> # Analyze same signal multiple times to check stability
            >>> results = [analyzer.analyze_acceleration(data) for _ in range(10)]
            >>> stability = analyzer.calculate_frequency_stability(results)
            >>> print(f"Stability score: {stability['stability_score']:.3f}")
            >>> print(f"Frequency drift: {stability['frequency_drift']:.4f} Hz")
            >>> print(f"Coefficient of variation: {stability['coefficient_of_variation']:.4f}")
            
            >>> # Track specific frequency across measurements
            >>> stability = analyzer.calculate_frequency_stability(
            ...     results, reference_frequency=10.0, tolerance_percent=5.0
            ... )
        """
        if not results:
            raise ValueError("Results list cannot be empty")
        
        if reference_frequency is None:
            reference_frequency = results[0].fundamental_frequency
            if reference_frequency is None:
                raise ValueError(
                    "No reference frequency provided and no fundamental "
                    "frequency found in first result"
                )
        
        # Extract frequencies near reference from all results
        frequencies: List[float] = []
        tolerance = reference_frequency * (tolerance_percent / 100.0)
        
        for result in results:
            # Find closest frequency to reference
            closest_freq = None
            min_diff = float('inf')
            
            for comp in result.dominant_frequencies:
                diff = abs(comp.frequency - reference_frequency)
                if diff < tolerance and diff < min_diff:
                    min_diff = diff
                    closest_freq = comp.frequency
            
            if closest_freq is not None:
                frequencies.append(closest_freq)
        
        if not frequencies:
            return {
                'mean_frequency': 0.0,
                'std_frequency': float('inf'),
                'coefficient_of_variation': float('inf'),
                'frequency_drift': 0.0,
                'stability_score': 0.0
            }
        
        freq_array = np.array(frequencies)
        mean_freq = float(np.mean(freq_array))
        std_freq = float(np.std(freq_array))
        cv = std_freq / mean_freq if mean_freq > 0 else float('inf')
        drift = float(np.max(freq_array) - np.min(freq_array))
        
        # Stability score: 1.0 = perfect stability, 0.0 = no stability
        # Based on coefficient of variation (lower CV = higher stability)
        if cv < 0.01:
            stability_score = 1.0
        elif cv > 1.0:
            stability_score = 0.0
        else:
            stability_score = 1.0 - cv
        
        return {
            'mean_frequency': mean_freq,
            'std_frequency': std_freq,
            'coefficient_of_variation': cv,
            'frequency_drift': drift,
            'stability_score': float(stability_score),
            'n_measurements': len(frequencies)
        }
