import numpy as np
import soundfile as sf

# Load the original audio
audio, sample_rate = sf.read('03-01-01-01-02-02-10.wav')

# Generate white noise
duration = len(audio) / sample_rate  # Duration of the audio in seconds
noise = np.random.normal(0, 0.1, len(audio))  # Generate random Gaussian noise

# Adjust the noise level to control the noise-to-signal ratio
noise_level = 0.1  # Example noise level (adjust as desired)
noise *= noise_level

# Add the noise to the original audio
noisy_audio = audio + noise

# Save the noisy audio
sf.write('noisy_audio.wav', noisy_audio, sample_rate)
