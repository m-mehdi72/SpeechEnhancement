import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft
from pystoi import stoi
import glob

n_fft = 512
hop_length = 256

def load_wav(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(waveform, axis = -1)

def convert_to_spectrogram(wav):
    spectrogram = tf.signal.stft(wav, frame_length=n_fft, frame_step=hop_length)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.log(spectrogram + 1e-8)  # Apply logarithm for dynamic range compression
    spectrogram = tf.math.divide(spectrogram, tf.reduce_max(spectrogram))  # Normalize between 0 and 1
    return spectrogram

def spectrogram_abs_to_wav(spec):
    spec = tf.transpose(spec, perm=[1, 0])  # Transpose dimensions
    _, wav = istft(spec, window='hann', nperseg=n_fft, noverlap=hop_length)
    return wav


def plot_wave(waveform):
    plt.figure(figsize=(10, 4))
    plt.plot(waveform)
    plt.show()

def calculate_stoi(clean_wav, denoised_wav, sample_rate=8000):
    stoi_score = stoi(clean_wav.squeeze(), denoised_wav.squeeze(), sample_rate)
    return stoi_score

def evaluate_model(model, clean_files, noisy_files, sample_rate=8000):
    stoi_scores = []
    for clean_file, noisy_file in zip(clean_files, noisy_files):
        clean_wav = load_wav(clean_file)
        noisy_wav = load_wav(noisy_file)
        clean_spec = convert_to_spectrogram(clean_wav)
        noisy_spec = convert_to_spectrogram(noisy_wav)
        denoised_spec = model.predict(tf.expand_dims(noisy_spec, axis=0))
        denoised_spec = tf.squeeze(denoised_spec, axis=0)  # Squeeze the batch dimension
        denoised_spec = tf.squeeze(denoised_spec, axis=-1)  # Squeeze the last dimension
        denoised_wav = spectrogram_abs_to_wav(denoised_spec)
        clean_wav = spectrogram_abs_to_wav(clean_spec)

        if noisy_spec.shape == denoised_spec.shape:
            stoi_score = calculate_stoi(clean_wav, denoised_wav, sample_rate)
            stoi_scores.append(stoi_score)
            
            # Plot input and output waveforms
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(noisy_wav)
            plt.title('Noisy Audio')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.subplot(2, 1, 2)
            plt.plot(denoised_wav)
            plt.title('Denoised Audio')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.show()
            
        else:
            print(f"Skipping file pair: {clean_file} and {noisy_file} due to different spectrogram lengths.")
    
    return stoi_scores


def build_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(None, n_fft // 2 + 1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(1, (3, 3), activation='tanh', padding='same'))
    model.add(layers.Lambda(lambda x: x[:, :, :n_fft // 2 + 1, :]))  # Crop to desired shape
    return model



# def train_model(model, clean_files, noisy_files, epochs=1, batch_size=32):
#     optimizer = keras.optimizers.Adam(learning_rate=0.001)
def train_model(model, clean_files, noisy_files, epochs=20, batch_size=16):
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    if len(clean_files) == 0 or len(noisy_files) == 0:
        raise ValueError("No audio files found.")

    clean_specs = []
    noisy_specs = []

    for clean_file, noisy_file in zip(clean_files, noisy_files):
        clean_wav = load_wav(clean_file)
        noisy_wav = load_wav(noisy_file)
        clean_spec = convert_to_spectrogram(clean_wav)
        noisy_spec = convert_to_spectrogram(noisy_wav)
        clean_specs.append(clean_spec)
        noisy_specs.append(noisy_spec)

    clean_specs = np.array(clean_specs)
    noisy_specs = np.array(noisy_specs)

    history = model.fit(noisy_specs, clean_specs, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[checkpoint])
    model.save('speech_enhancement_model_by_19.h5')
    print("Model downloaded!!!")
    return history


def main():
    clean_files = glob.glob('ravdess_rewritten_8k/*.wav')
    noisy_files = glob.glob('urbansound_8k/*.wav')

    # model = build_model()
    # train_model(model, clean_files, noisy_files, epochs=20, batch_size=16)

    model = tf.keras.models.load_model('speech_enhancement_model_by_19.h5')
    stoi_scores = evaluate_model(model, clean_files, noisy_files)
    average_stoi_score = np.mean(stoi_scores)
    print("Average STOI Score:", float(average_stoi_score)*100)

if __name__ == '__main__':
    main()
