from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'ai_model/model.h5'
model = load_model(MODEL_PATH)

# Define class indices in the same order used during training
class_indices = {'Garbage': 0, 'Pothole': 1, 'StreetlightDamage': 2, 'WaterLeakage': 3}
class_labels = list(class_indices.keys())

@app.route('/')
def index():
    return "CivicLens Prediction API (MobileNetV2 Model)"

@app.route('/report', methods=['POST'])
def report():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            img_path = os.path.join('ai_model', 'images', file.filename)
            file.save(img_path)

            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions[0])]
            timestamp = datetime.now().isoformat()

            result = {
                'filename': file.filename,
                'prediction': predicted_class,
                'timestamp': timestamp
            }
            return jsonify(result)

        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({'error': 'Prediction failed or image not recognized'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
