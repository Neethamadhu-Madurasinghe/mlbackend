import base64
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

class ImageProcessor:
    @staticmethod
    def decode_base64_image(base64_string):
        try:
            # Remove data URI prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            img_bytes = base64.b64decode(base64_string)
            
            # Convert bytes to PIL Image
            img = Image.open(BytesIO(img_bytes))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array (no resizing or normalization here)
            img_array = np.array(img)
            
            # Add batch dimension
            img_array = tf.expand_dims(img_array, 0)  # Shape: (1, H, W, 3)
            
            return img_array
        except Exception as e:
            raise ValueError(f"Error decoding image: {str(e)}")