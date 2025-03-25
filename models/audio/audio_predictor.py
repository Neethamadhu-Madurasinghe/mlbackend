from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np
import os
import librosa
import base64
from io import BytesIO
import time

audio_anomaly_model_bp = Blueprint('audio_anomaly_model', __name__)

# Model configuration (same for all models)
TARGET_LENGTH = 5  # 5 seconds
SR = 22050        # Sample rate
N_MELS = 128      # Number of Mel bands
HOP_LENGTH = 512  # Hop length for spectrogram

# Load all models
model_dir = os.path.dirname(__file__)
models = {
    'a': tf.keras.models.load_model(
        os.path.join(model_dir, 'a.h5'), 
        custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
    ),
    'ada': tf.keras.models.load_model(
        os.path.join(model_dir, 'ada.h5'), 
        custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
    ),
    'dha': tf.keras.models.load_model(
        os.path.join(model_dir, 'dha.h5'), 
        custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
    )
}

def load_and_preprocess_audio(audio_bytes, target_length=TARGET_LENGTH, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    """
    Load an audio file from bytes, pad it to 5 seconds, compute its Mel spectrogram, and normalize it.
    """
    y, _ = librosa.load(BytesIO(audio_bytes), sr=sr)
    if len(y) < target_length * sr:
        y = np.pad(y, (0, target_length * sr - len(y)), mode='constant')
    else:
        y = y[:target_length * sr]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = np.maximum(S_dB, -80)
    S_dB = (S_dB + 80) / 80
    S_dB = np.expand_dims(S_dB, axis=(0, -1))
    return S_dB

def predict_audio(model, audio_bytes, threshold):
    """Helper function to process audio and make prediction"""
    spectrogram = load_and_preprocess_audio(audio_bytes)
    reconstruction = model.predict(spectrogram, verbose=0)
    mse = np.mean(np.square(spectrogram - reconstruction))
    is_anomaly = bool(mse > threshold)
    return mse, is_anomaly

# Route for model 'a'
@audio_anomaly_model_bp.route('/predict/a', methods=['POST'])
def predict_a():
    try:
        data = request.get_json()
        if not data or 'audio' not in data:
            return jsonify({'error': 'No audio provided'}), 400
        if 'threshold' not in data:
            return jsonify({'error': 'No threshold provided'}), 400

        base64_audio = data['audio']
        threshold = float(data['threshold'])
        
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',')[1]
        audio_bytes = base64.b64decode(base64_audio)

        mse, is_anomaly = predict_audio(models['a'], audio_bytes, threshold)

        result = {
            'model': 'a',
            'reconstruction_error': float(mse),
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'status': 'success'
        }
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# Route for model 'ada'
@audio_anomaly_model_bp.route('/predict/ada', methods=['POST'])
def predict_ada():
    try:
        data = request.get_json()
        if not data or 'audio' not in data:
            return jsonify({'error': 'No audio provided'}), 400
        if 'threshold' not in data:
            return jsonify({'error': 'No threshold provided'}), 400

        base64_audio = data['audio']
        threshold = float(data['threshold'])
        
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',')[1]
        audio_bytes = base64.b64decode(base64_audio)

        mse, is_anomaly = predict_audio(models['ada'], audio_bytes, threshold)

        result = {
            'model': 'ada',
            'reconstruction_error': float(mse),
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'status': 'success'
        }
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# Route for model 'dha'
@audio_anomaly_model_bp.route('/predict/dha', methods=['POST'])
def predict_dha():
    try:
        data = request.get_json()
        if not data or 'audio' not in data:
            return jsonify({'error': 'No audio provided'}), 400
        if 'threshold' not in data:
            return jsonify({'error': 'No threshold provided'}), 400

        base64_audio = data['audio']
        threshold = float(data['threshold'])
        
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',')[1]
        audio_bytes = base64.b64decode(base64_audio)

        mse, is_anomaly = predict_audio(models['dha'], audio_bytes, threshold)

        result = {
            'model': 'dha',
            'reconstruction_error': float(mse),
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'status': 'success'
        }
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500