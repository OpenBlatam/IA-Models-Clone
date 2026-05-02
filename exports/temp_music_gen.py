import numpy as np
import scipy.io.wavfile as wav
import os

# Parameters
sample_rate = 44100
duration = 10.0  # seconds
total_samples = int(sample_rate * duration)

# Ensure export directory exists
os.makedirs('exports', exist_ok=True)

def generate_note(freq, start_time, dur, sr=sample_rate, attack=0.02, release=0.05):
    """Generate a sine wave with a simple ADSR-like envelope (linear attack & release)."""
    start = int(start_time * sr)
    num_samples = int(dur * sr)
    if start + num_samples > total_samples:
        num_samples = total_samples - start
    t = np.linspace(0, dur, num_samples, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)

    # Envelope
    env = np.ones(num_samples)
    attack_s = int(attack * sr)
    release_s = int(release * sr)

    if attack_s > 0:
        env[:attack_s] = np.linspace(0, 1, attack_s)
    if release_s > 0:
        env[-release_s:] = np.linspace(1, 0, release_s)

    return start, wave * env

def generate_chord(frequencies, start_time, dur, sr=sample_rate, attack=0.1, release=0.2):
    """Generate a chord (sum of sine waves) with an envelope."""
    start = int(start_time * sr)
    num_samples = int(dur * sr)
    if start + num_samples > total_samples:
        num_samples = total_samples - start
    t = np.linspace(0, dur, num_samples, endpoint=False)
    wave = sum(np.sin(2 * np.pi * f * t) for f in frequencies)

    env = np.ones(num_samples)
    attack_s = int(attack * sr)
    release_s = int(release * sr)
    if attack_s > 0:
        env[:attack_s] = np.linspace(0, 1, attack_s)
    if release_s > 0:
        env[-release_s:] = np.linspace(1, 0, release_s)

    return start, wave * env

# Initialize output array
output = np.zeros(total_samples)

# ---- Melody (20 notes, each 0.5 sec) ----
# Frequencies (C4 to C5 range)
melody_freqs = [
    261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 392.00, 349.23,
    329.63, 293.66, 261.63, 329.63, 392.00, 523.25, 659.25, 587.33,
    523.25, 440.00, 392.00, 349.23
]
note_dur = 0.5
melody_start = 0.0

for freq in melody_freqs:
    if melody_start + note_dur > duration:
        break
    start_idx, note_wave = generate_note(freq, melody_start, note_dur)
    end_idx = start_idx + len(note_wave)
    output[start_idx:end_idx] += note_wave
    melody_start += note_dur

# ---- Pad chords (background progression) ----
chords = [
    # C major: C4, E4, G4
    ([261.63, 329.63, 392.00], 0.0, 2.0),
    # G major: G3, B3, D4
    ([196.00, 246.94, 293.66], 2.0, 2.0),
    # A minor: A3, C4, E4
    ([220.00, 261.63, 329.63], 4.0, 2.0),
    # F major: F3, A3, C4 (hold for last 4 seconds)
    ([174.61, 220.00, 261.63], 6.0, 4.0)
]

for freqs, start, dur in chords:
    start_idx, chord_wave = generate_chord(freqs, start, dur)
    end_idx = start_idx + len(chord_wave)
    output[start_idx:end_idx] += chord_wave

# Normalize to avoid clipping and convert to int16
max_val = np.max(np.abs(output))
if max_val > 0:
    output = output / max_val * 0.9  # headroom
output_int16 = np.int16(output * 32767)

# Write to file
wav.write('exports/neural_melody.wav', sample_rate, output_int16)