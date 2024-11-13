import soundfile as sf
import numpy as np
import librosa

# Parameters
sampling_rate = 16000  # Sampling rate (Hz)
duration = 1.0         # Duration of each sine wave (seconds)

# Frequencies for each letter
frequencies = {'A': 200, 'B': 300, 'C': 400, 'D': 500}

# Function to generate and save sine wave
def save_sine_wave(letter, frequency, duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # amplitude scaled to 0.5
    filename = f"{letter}_{frequency}Hz.wav"
    sf.write(filename, sine_wave, sampling_rate)
    print(f"Saved {letter} as {filename}")

# Generate and save each sine wave
for letter, freq in frequencies.items():
    save_sine_wave(letter, freq, duration, sampling_rate)
