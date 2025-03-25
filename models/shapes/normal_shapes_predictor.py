from flask import Blueprint, request, jsonify
from utils.image_processor import ImageProcessor
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import time  # For unique filenames

normal_shape_model_bp = Blueprint('normal_shape_model', __name__)

# Model configuration
IMG_HEIGHT = 180
IMG_WIDTH = 180

# Directory to save images
# SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_images')
# os.makedirs(SAVE_DIR, exist_ok=True)  # Create directory if it doesn’t exist

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'normal_shapes.h5')
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = ['S1-correct', 'S1-incorrect', 'S2-correct', 'S2-incorrect', 'S3-correct', 'S3-incorrect', 'S4-correct', 'S4-incorrect', 'S5-correct', 'S5-incorrect']
# Define normalization layer
normalization_layer = tf.keras.layers.Rescaling(1./255)

@normal_shape_model_bp.route('/predict', methods=['POST'])
def predict():
    try:
        print("Normal Shapes")
        # Get base64 image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Process image using existing ImageProcessor
        img_array = ImageProcessor.decode_base64_image(data['image'])
        print("Image decoded")

        # Save the image before preprocessing
        # Convert tensor back to PIL Image
        # img = Image.fromarray(np.uint8(img_array[0]))  # Remove batch dim and convert to uint8
        # timestamp = int(time.time())  # Unique filename with timestamp
        # save_path = os.path.join(SAVE_DIR, f'image_{timestamp}.png')
        # img.save(save_path)
        # print(f"Image saved to: {save_path}")

        # Ensure image matches training preprocessing
        img_array = tf.image.resize(img_array, [IMG_HEIGHT, IMG_WIDTH])
        print(f"Image resized to: {img_array.shape}")

        # Normalize to [0, 1]
        #  TODO: Add this later
        # img_array = normalization_layer(img_array)
        print(f"Image normalized - min: {tf.reduce_min(img_array)}, max: {tf.reduce_max(img_array)}")

        # Make prediction
        prediction = model.predict(img_array)
        
        # Get predicted class index and name
        predicted_class_idx = tf.argmax(prediction[0]).numpy()
        predicted_class = class_names[predicted_class_idx]

        # Format response
        result = {
            'prediction': prediction.tolist(),
            'predicted_class': predicted_class,
            'status': 'success'
        }
        
        return jsonify(result), 200
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500