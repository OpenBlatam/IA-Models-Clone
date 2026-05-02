import numpy as np
from scipy.io import wavfile

def normalize_volume(input_path, output_path):
    """
    Load a WAV file, normalize its volume to 0 dBFS (peak normalization),
    and save the result.
    """
    # Read the WAV file (returns sample rate and data array)
    sample_rate, data = wavfile.read(input_path)
    
    # Handle mono or multi-channel
    # Convert to float for processing
    original_dtype = data.dtype
    if np.issubdtype(original_dtype, np.floating):
        # Already floating point; assume range [-1, 1]
        data_float = data.astype(np.float64)
    else:
        # Integer PCM; convert to float in range [-1, 1]
        if original_dtype == np.int16:
            data_float = data.astype(np.float64) / 32768.0
        elif original_dtype == np.int32:
            data_float = data.astype(np.float64) / 2147483648.0
        elif original_dtype == np.uint8:
            # Unsigned 8-bit: bias offset 128
            data_float = (data.astype(np.float64) - 128.0) / 128.0
        else:
            # Fallback: treat as signed integer with half range
            max_val = np.iinfo(original_dtype).max
            min_val = np.iinfo(original_dtype).min
            # For symmetric signed integer, max absolute is max(abs(min), max)
            range_ = max(abs(min_val), abs(max_val))
            data_float = data.astype(np.float64) / range_
    
    # Find the maximum absolute sample value (peak)
    peak = np.max(np.abs(data_float))
    
    # Avoid division by zero
    if peak == 0:
        # No signal, keep as is (or could leave unchanged)
        normalized = data_float
    else:
        # Scale so that the peak becomes 1.0 (0 dBFS)
        scale = 1.0 / peak
        normalized = data_float * scale
    
    # Convert back to original dtype before saving
    if np.issubdtype(original_dtype, np.floating):
        # Keep as float; clip to avoid overshoot
        normalized = np.clip(normalized, -1.0, 1.0).astype(original_dtype)
    else:
        # Convert back to integer PCM
        if original_dtype == np.int16:
            normalized = np.clip(normalized * 32767, -32768, 32767).astype(np.int16)
        elif original_dtype == np.int32:
            normalized = np.clip(normalized * 2147483647, -2147483648, 2147483647).astype(np.int32)
        elif original_dtype == np.uint8:
            normalized = np.clip(normalized * 128 + 128, 0, 255).astype(np.uint8)
        else:
            # Generic signed integer
            max_val = np.iinfo(original_dtype).max
            min_val = np.iinfo(original_dtype).min
            # For symmetric scaling, use half range
            half = max(abs(min_val), abs(max_val))
            normalized = np.clip(normalized * half, min_val, max_val).astype(original_dtype)
    
    # Write the normalized audio to the output file
    wavfile.write(output_path, sample_rate, normalized)

if __name__ == "__main__":
    normalize_volume('exports/neural_melody.wav', 'exports/neural_melody_EDITED.wav')