# import tensorflow as tf
# import librosa
# import numpy as np
# import wave
# import matplotlib.pyplot as plt
# from scipy.signal import istft
# from scipy.io import wavfile


# n_fft = 512
# hop_length = 256
# sample_rate = 8000

# def convert_to_spectrogram(wav):
#     spectrogram = tf.signal.stft(wav, frame_length=n_fft, frame_step=hop_length)
#     spectrogram = tf.abs(spectrogram)
#     spectrogram = tf.math.log(spectrogram + 1e-8)  # Apply logarithm for dynamic range compression
#     spectrogram = tf.math.divide(spectrogram, tf.reduce_max(spectrogram))  # Normalize between 0 and 1
#     return spectrogram

# def spectrogram_abs_to_wav(spec):
#     spec = tf.expand_dims(spec, axis=0)  # Add a batch dimension
#     spec = tf.expand_dims(spec, axis=-1)  # Add a channel dimension
#     _, wav = istft(spec, window='hann', nperseg=n_fft, noverlap=hop_length, input_onesided=True)
#     return wav


# # Load the audio enhancement model
# model = tf.keras.models.load_model('speech_enhancement_model_by_19.h5')

# # Load the audio file
# audio, sr = librosa.load('noisy_audio.wav', sr=sample_rate)

# noisy_spec = convert_to_spectrogram(audio)
# denoised_spec = model.predict(tf.expand_dims(noisy_spec, axis=0))
# denoised_spec = tf.squeeze(denoised_spec, axis=0)  # Squeeze the batch dimension
# denoised_spec = tf.squeeze(denoised_spec, axis=-1)  # Squeeze the last dimension
# denoised_wav = spectrogram_abs_to_wav(denoised_spec)
# denoised_wav = np.squeeze(denoised_wav)  # Remove any extra dimensions


# # Save the output audio file
# output_file = 'denoised_audio.wav'
# denoised_wav_normalized = denoised_wav / tf.reduce_max(tf.abs(denoised_wav))  # Normalize the waveform
# denoised_wav_int16 = (denoised_wav_normalized * tf.int16.max).numpy().astype(np.int16)  # Convert to int16
# wavfile.write(output_file, sample_rate, denoised_wav_int16)

# # Plot the input and output waveforms
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(audio)
# plt.title('Input Audio')
# plt.xlabel('Samples')
# plt.ylabel('Amplitude')
# plt.subplot(2, 1, 2)
# plt.plot(denoised_wav)
# plt.title('Output Audio')
# plt.xlabel('Samples')
# plt.ylabel('Amplitude')
# plt.tight_layout()
# plt.show()


import tensorflow as tf
import librosa
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.signal import istft
import soundfile as sf


n_fft = 512
hop_length = 256
sample_rate = 8000

# def convert_to_spectrogram(wav):
#     spectrogram = tf.signal.stft(wav, frame_length=n_fft, frame_step=hop_length)
#     spectrogram = tf.abs(spectrogram)
#     spectrogram = tf.math.log(spectrogram + 1e-8)  # Apply logarithm for dynamic range compression
#     spectrogram = tf.math.divide(spectrogram, tf.reduce_max(spectrogram))  # Normalize between 0 and 1
#     return spectrogram

def convert_to_spectrogram(wav):
    spectrogram = tf.signal.stft(wav, frame_length=n_fft, frame_step=hop_length)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.divide(spectrogram, tf.reduce_max(spectrogram))  # Normalize between 0 and 1
    return spectrogram

def spectrogram_abs_to_wav(spec):
    spec = tf.transpose(spec, perm=[1, 0])  # Transpose dimensions
    _, wav = istft(spec, window='hann', nperseg=n_fft, noverlap=hop_length)
    return wav

# Load the audio enhancement model
model = tf.keras.models.load_model('speech_enhancement_model_by_19.h5')

# Load the audio file
audio, sr = librosa.load('noisy_audio.wav', sr=sample_rate)

noisy_spec = convert_to_spectrogram(audio)
denoised_spec = model.predict(tf.expand_dims(noisy_spec, axis=0))
denoised_spec = tf.squeeze(denoised_spec, axis=0)  # Squeeze the batch dimension
denoised_spec = tf.squeeze(denoised_spec, axis=-1)  # Squeeze the last dimension
denoised_wav = spectrogram_abs_to_wav(denoised_spec)

output_file = 'denoised_audio.wav'
denoised_wav_normalized = denoised_wav / np.max(np.abs(denoised_wav))  # Normalize the waveform
sf.write(output_file, denoised_wav_normalized, sample_rate)

# Plot the input and output waveforms
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(audio)
plt.title('Input Audio')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.plot(denoised_wav)
plt.title('Output Audio')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()