## This code does not include ML models include those in correct locations

#### Audio related model (a.h5, ada.h5....)  
/models/audio/


#### Attention related models
/models/face/
/models/face/all


#### Letters related models
/models/letters/


#### Shapes related models
/models/shapes/


#### Words related models
/models/words/

After placing models, check if the the code contains correct model name
eg: model_path = os.path.join(os.path.dirname(__file__), 'word_model.keras')


## How to run 

### Step 1:
pip install flask tensorflow numpy pillow opencv-python librosa
pip install dlib

(if dlibe does not install: automatic face cropping feauture has to be removed: contact Inusha)


### Step 2: 
copy models into correct locations

### Step 3:
python3 app.py 



## How to add more audio models (new words ...)

audio_predictor.py contains all the routes for audio related models

1. Add the new model to this code at the top
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

2. There are routes already for a few sounds (a, ada, dha) copy one for those, past at the bottom and edit that to to use new model


@audio_anomaly_model_bp.route('/predict/a', methods=['POST'])                   ===== Add the new route
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

        mse, is_anomaly = predict_audio(models['a'], audio_bytes, threshold)   ========== Which model to use

        result = {                                                      
            'model': 'a',                                                       ================== Change this too
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
