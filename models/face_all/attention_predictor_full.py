from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
import base64
from io import BytesIO
import time
import pickle

engagement_model_full_bp = Blueprint('engagement_model_full', __name__)

# Model configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Directory to save images
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_images')
os.makedirs(SAVE_DIR, exist_ok=True)

# Load the first model (image-based engagement)
model_path_image = os.path.join(os.path.dirname(__file__), 'finalengagement+student.h5')
model_image = tf.keras.models.load_model(model_path_image)

# Load the second model (activity difficulty)
model_path_difficulty = os.path.join(os.path.dirname(__file__), 'finalmodel3.dat')
with open(model_path_difficulty, 'rb') as f:
    model_difficulty = pickle.load(f)

# Define class names for engagement (first model)
engagement_class_names = ['disengage', 'engage', 'midengage']
# Mapping for engagement to numeric values (second model expects 0, 1, 2)
engagement_mapping = {'disengage': 0, 'midengage': 1, 'engage': 2}

# Activity type mapping (from training: Scramble Sentence -> 0, Picture Sequencing -> 1)
activity_type_mapping = {'Scramble Sentence': 0, 'Picture Sequencing': 1}

# Activity difficulty class names (from training data)
difficulty_class_names = ['Easy', 'Hard']  # Adjust based on your training data

def preprocess_image(image):
    """
    Crops 10% from top and bottom, resizes to target size, and normalizes to [0, 1].
    """
    height, width = image.shape[:2]
    crop_height = int(height * 0.2)
    cropped_image = image[crop_height:height - crop_height, :]
    resized_image = cv2.resize(cropped_image, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
    return resized_image

@engagement_model_full_bp.route('/predict', methods=['POST'])
def predict():
    try:
        print("Engagement and Difficulty Prediction")
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Required additional fields
        required_fields = ['time_taken', 're_attempts', 'errors', 'activity_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Extract additional inputs
        time_taken = float(data['time_taken'])  # Time in seconds
        re_attempts = int(data['re_attempts'])  # Number of re-attempts
        errors = int(data['errors'])            # Number of errors
        activity_type = data['activity_type']   # String: 'Scramble Sentence' or 'Picture Sequencing'
        
        # Map activity_type to numeric
        if activity_type not in activity_type_mapping:
            return jsonify({'error': f'Invalid activity_type. Must be one of: {list(activity_type_mapping.keys())}'}), 400
        activity_type_numeric = activity_type_mapping[activity_type]

        # Process images
        image_fields = ['image1', 'image2', 'image3', 'image4', 'image5','image6', 'image7', 'image8', 'image9', 'image10']
        images = {}
        predictions = {}
        saved_paths = {}

        for field in image_fields:
            if field in data and data[field]:
                base64_image = data[field]
                if ',' in base64_image:
                    base64_image = base64_image.split(',')[1]
                img_bytes = base64.b64decode(base64_image)
                print(f"{field} decoded")

                img = Image.open(BytesIO(img_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                timestamp = int(time.time())
                save_path = os.path.join(SAVE_DIR, f'{field}_{timestamp}.png')
                img.save(save_path)
                print(f"{field} saved to: {save_path}")
                saved_paths[field] = save_path

                img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                img_array = preprocess_image(img_array)
                images[field] = img_array

        if not images:
            return jsonify({'error': 'No valid images provided'}), 400

        # Batch prediction for engagement
        img_batch = np.stack([images[field] for field in images], axis=0)
        print(f"Batch preprocessed - shape: {img_batch.shape}, min: {np.min(img_batch)}, max: {np.max(img_batch)}")
        batch_predictions = model_image.predict(img_batch)

        # Store individual predictions
        for i, field in enumerate(images.keys()):
            pred = batch_predictions[i]
            predicted_class_idx = np.argmax(pred)
            predicted_class = engagement_class_names[predicted_class_idx]
            predictions[field] = {
                'prediction': pred.tolist(),
                'predicted_class': predicted_class,
                'saved_image_path': saved_paths[field]
            }

        # Compute average engagement prediction
        avg_prediction = np.mean(batch_predictions, axis=0)
        avg_class_idx = np.argmax(avg_prediction)
        avg_class = engagement_class_names[avg_class_idx]
        engagement_numeric = engagement_mapping[avg_class]  # Map to numeric for second model

        # Prepare input for difficulty model
        difficulty_input = np.array([[engagement_numeric, time_taken, re_attempts, errors, activity_type_numeric]])
        print(f"Difficulty input: {difficulty_input}")

        # Predict activity difficulty
        difficulty_pred = model_difficulty.predict(difficulty_input)[0]
        print(f"Predicted difficulty: {difficulty_pred}")

        # Format response
        result = {
            'individual_predictions': predictions,
            'average_prediction': {
                'prediction': avg_prediction.tolist(),
                'predicted_class': avg_class,
                'engagement_numeric': engagement_numeric
            },
            'activity_difficulty': difficulty_pred,
            'status': 'success'
        }
        
        return jsonify(result), 200
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500