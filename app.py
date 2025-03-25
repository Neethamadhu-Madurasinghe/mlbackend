from flask import Flask
import os
from models.letters.normal_letter_predictor import normal_letter_model_bp
from models.letters.dotted_letter_predictor import dotted_letter_model_bp
from models.shapes.dotted_shapes_predictor import dotted_shape_model_bp
from models.shapes.normal_shapes_predictor import normal_shape_model_bp
from models.words.word_predictor import word_model_bp
from models.audio.audio_predictor import audio_anomaly_model_bp
from models.face.attention_predictor import engagement_model_bp
from models.face_all.attention_predictor_full import engagement_model_full_bp

# Disable GPU to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_app():
    app = Flask(__name__)
    
    # Register blueprints for each model's routes
    app.register_blueprint(normal_letter_model_bp, url_prefix='/normal_letters')
    app.register_blueprint(dotted_letter_model_bp, url_prefix='/dotted_letters')
    app.register_blueprint(dotted_shape_model_bp, url_prefix='/dotted_shapes')
    app.register_blueprint(normal_shape_model_bp, url_prefix='/normal_shapes')
    app.register_blueprint(word_model_bp, url_prefix='/words')
    app.register_blueprint(audio_anomaly_model_bp, url_prefix='/audio')
    app.register_blueprint(engagement_model_bp, url_prefix='/engagement')
    app.register_blueprint(engagement_model_full_bp, url_prefix='/engagement-full')
    

    @app.route('/')
    def health_check():
        return {'status': 'API is running'}, 200
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)