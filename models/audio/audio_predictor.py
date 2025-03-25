from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np
import os
import librosa
import base64
from io import BytesIO
import time  # For unique filenames

audio_anomaly_model_bp = Blueprint('audio_anomaly_model', __name__)

# Model configuration
TARGET_LENGTH = 5  # 5 seconds
SR = 22050        # Sample rate
N_MELS = 128      # Number of Mel bands
HOP_LENGTH = 512  # Hop length for spectrogram

# Directory to save audio files
# SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_audio')
# os.makedirs(SAVE_DIR, exist_ok=True)  # Create directory if it doesn’t exist

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'audio_anomaly_autoencoder.h5')
model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

def load_and_preprocess_audio(audio_bytes, target_length=TARGET_LENGTH, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    """
    Load an audio file from bytes, pad it to 5 seconds, compute its Mel spectrogram, and normalize it.
    """
    # Load audio from bytes
    y, _ = librosa.load(BytesIO(audio_bytes), sr=sr)
    # Pad or truncate to 5 seconds
    if len(y) < target_length * sr:
        y = np.pad(y, (0, target_length * sr - len(y)), mode='constant')
    else:
        y = y[:target_length * sr]
    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    # Convert to dB scale and normalize to [0,1]
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = np.maximum(S_dB, -80)  # Clip values below -80 dB
    S_dB = (S_dB + 80) / 80      # Scale to [0,1]
    S_dB = np.expand_dims(S_dB, axis=(0, -1))  # Shape: (1, 128, 216, 1)
    return S_dB



@audio_anomaly_model_bp.route('/predict', methods=['POST'])
def predict():
    try:
        print("Audio Anomaly Detection")
        # Get data from request
        data = request.get_json()
        if not data or 'audio' not in data:
            return jsonify({'error': 'No audio provided'}), 400
        if 'threshold' not in data:
            return jsonify({'error': 'No threshold provided'}), 400
        
        # Extract base64 audio and threshold
        base64_audio = data['audio']
        threshold = float(data['threshold'])  # Convert to float
        
        # Decode base64 audio
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',')[1]  # Remove data URI prefix
        audio_bytes = base64.b64decode(base64_audio)
        print("Audio decoded")

        # Save the audio file
        # timestamp = int(time.time())
        # save_path = os.path.join(SAVE_DIR, f'audio_{timestamp}.wav')
        # with open(save_path, 'wb') as f:
        #     f.write(audio_bytes)
        # print(f"Audio saved to: {save_path}")

        # Preprocess audio
        spectrogram = load_and_preprocess_audio(audio_bytes)
        print(f"Spectrogram shape: {spectrogram.shape}, min: {np.min(spectrogram)}, max: {np.max(spectrogram)}")

        # Predict reconstruction
        reconstruction = model.predict(spectrogram, verbose=0)
        mse = np.mean(np.square(spectrogram - reconstruction))
        is_anomaly = bool(mse > threshold)

        # Format response
        result = {
            'reconstruction_error': float(mse),  # Convert to float for JSON
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            # 'saved_audio_path': save_path,
            'status': 'success'
        }
        
        return jsonify(result), 200
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500