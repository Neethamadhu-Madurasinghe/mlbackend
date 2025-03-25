from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
import base64
from io import BytesIO
import time

engagement_model_bp = Blueprint('engagement_model', __name__)

# Model configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Directory to save images
# SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_images')
# os.makedirs(SAVE_DIR, exist_ok=True)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'finalengagement+student.h5')
model = tf.keras.models.load_model(model_path)

# Define class names based on training class indices
class_names = ['disengage', 'engage', 'midengage']

def preprocess_image(image):
    """
    Crops 10% from top and bottom, resizes to target size, and normalizes to [0, 1].
    """
    height, width = image.shape[:2]
    crop_height = int(height * 0.2)  # 20% total (10% top, 10% bottom)
    cropped_image = image[crop_height:height - crop_height, :]
    resized_image = cv2.resize(cropped_image, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
    return resized_image

@engagement_model_bp.route('/predict', methods=['POST'])
def predict():
    try:
        print("Engagement Prediction")
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # List of possible image fields
        image_fields = ['image1', 'image2', 'image3', 'image4', 'image5','image6', 'image7', 'image8', 'image9', 'image10']
        images = {}
        predictions = {}
        saved_paths = {}

        # Process each image field if present
        for field in image_fields:
            if field in data and data[field]:
                # Decode base64 image
                base64_image = data[field]
                if ',' in base64_image:
                    base64_image = base64_image.split(',')[1]
                img_bytes = base64.b64decode(base64_image)
                print(f"{field} decoded")

                # Save the image
                img = Image.open(BytesIO(img_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                timestamp = int(time.time())
                # save_path = os.path.join(SAVE_DIR, f'{field}_{timestamp}.png')
                # img.save(save_path)
                # print(f"{field} saved to: {save_path}")
                # saved_paths[field] = save_path

                # Preprocess image
                img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                img_array = preprocess_image(img_array)
                images[field] = img_array

        if not images:
            return jsonify({'error': 'No valid images provided'}), 400

        # Stack images for batch prediction
        img_batch = np.stack([images[field] for field in images], axis=0)
        print(f"Batch preprocessed - shape: {img_batch.shape}, min: {np.min(img_batch)}, max: {np.max(img_batch)}")

        # Predict for all images at once
        batch_predictions = model.predict(img_batch)

        # Store individual predictions
        for i, field in enumerate(images.keys()):
            pred = batch_predictions[i]
            predicted_class_idx = np.argmax(pred)
            predicted_class = class_names[predicted_class_idx]
            predictions[field] = {
                'prediction': pred.tolist(),
                'predicted_class': predicted_class,
                # 'saved_image_path': saved_paths[field]
            }

        # Compute average prediction
        avg_prediction = np.mean(batch_predictions, axis=0)
        avg_class_idx = np.argmax(avg_prediction)
        avg_class = class_names[avg_class_idx]

        # Format response
        result = {
            'individual_predictions': predictions,
            'average_prediction': {
                'prediction': avg_prediction.tolist(),
                'predicted_class': avg_class
            },
            'status': 'success'
        }
        
        return jsonify(result), 200
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500