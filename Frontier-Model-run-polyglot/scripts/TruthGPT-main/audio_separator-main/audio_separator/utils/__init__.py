# Audio Separator Utilities

from .constants import (
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
    COMMON_SAMPLE_RATES,
    DEFAULT_SAMPLE_RATE as UTILS_DEFAULT_SAMPLE_RATE,
    DEFAULT_SILENCE_THRESHOLD,
    DEFAULT_TARGET_PEAK,
    DEFAULT_TARGET_RMS,
    MAX_NUM_SOURCES
)
from .audio_helpers import (
    pad_audio_to_length,
    calculate_rms,
    calculate_peak,
    amplitude_to_db,
    db_to_amplitude,
    normalize_by_peak,
    normalize_by_rms,
    ensure_same_length
)
from .audio_utils import (
    get_audio_info,
    resample_audio,
    convert_to_mono,
    normalize_audio
)
from .device_utils import (
    get_device,
    move_to_device,
    get_device_info
)
from .validation_utils import (
    validate_audio_file,
    validate_output_format,
    validate_sample_rate,
    validate_num_sources,
    validate_audio_array,
    validate_output_dir
)
from .progress_utils import (
    ProgressTracker,
    track_progress
)
from .cache_utils import CacheManager
from .audio_enhancement import (
    denoise_audio,
    normalize_audio_peak,
    normalize_audio_rms,
    apply_fade,
    apply_compression
)
from .format_converter import (
    convert_format,
    batch_convert
)
from .performance import (
    timeit,
    timer,
    PerformanceMonitor,
    performance_monitor,
    profile_memory
)
from .parallel_processing import (
    process_parallel,
    batch_process_files,
    chunk_audio_processing
)
from .visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_separation_comparison,
    create_separation_report
)
from .audio_analysis import (
    analyze_audio,
    detect_silence,
    calculate_loudness,
    detect_beats,
    extract_features
)
from .quality_metrics import (
    calculate_separation_quality,
    assess_audio_quality,
    compare_separations
)
from .batch_optimizer import BatchOptimizer
from .audio_merger import (
    merge_sources,
    create_mix,
    blend_audio
)
from .export_utils import (
    export_separation_metadata,
    export_separation_report,
    import_separation_metadata,
    create_separation_summary
)
from .backup_utils import (
    backup_separation_results,
    restore_separation_results,
    list_backups
)

__all__ = [
    # Constants
    "SUPPORTED_AUDIO_FORMATS",
    "SUPPORTED_OUTPUT_FORMATS",
    "COMMON_SAMPLE_RATES",
    "UTILS_DEFAULT_SAMPLE_RATE",
    "DEFAULT_SILENCE_THRESHOLD",
    "DEFAULT_TARGET_PEAK",
    "DEFAULT_TARGET_RMS",
    "MAX_NUM_SOURCES",
    # Audio helpers (common operations)
    "pad_audio_to_length",
    "calculate_rms",
    "calculate_peak",
    "amplitude_to_db",
    "db_to_amplitude",
    "normalize_by_peak",
    "normalize_by_rms",
    "ensure_same_length",
    # Audio utilities
    "get_audio_info",
    "resample_audio",
    "convert_to_mono",
    "normalize_audio",
    # Device utilities
    "get_device",
    "move_to_device",
    "get_device_info",
    # Validation utilities
    "validate_audio_file",
    "validate_output_format",
    "validate_sample_rate",
    "validate_num_sources",
    "validate_audio_array",
    "validate_output_dir",
    # Progress utilities
    "ProgressTracker",
    "track_progress",
    # Cache utilities
    "CacheManager",
    # Audio enhancement
    "denoise_audio",
    "normalize_audio_peak",
    "normalize_audio_rms",
    "apply_fade",
    "apply_compression",
    # Format conversion
    "convert_format",
    "batch_convert",
    # Performance
    "timeit",
    "timer",
    "PerformanceMonitor",
    "performance_monitor",
    "profile_memory",
    # Parallel processing
    "process_parallel",
    "batch_process_files",
    "chunk_audio_processing",
    # Visualization
    "plot_waveform",
    "plot_spectrogram",
    "plot_separation_comparison",
    "create_separation_report",
    # Audio analysis
    "analyze_audio",
    "detect_silence",
    "calculate_loudness",
    "detect_beats",
    "extract_features",
    # Quality metrics
    "calculate_separation_quality",
    "assess_audio_quality",
    "compare_separations",
    # Batch optimization
    "BatchOptimizer",
    # Audio merging
    "merge_sources",
    "create_mix",
    "blend_audio",
    # Export utilities
    "export_separation_metadata",
    "export_separation_report",
    "import_separation_metadata",
    "create_separation_summary",
    # Backup utilities
    "backup_separation_results",
    "restore_separation_results",
    "list_backups",
]
