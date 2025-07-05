from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Create folders if not exist
UPLOAD_FOLDER = 'images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model('ai_model/model.h5')
labels = ['Garbage', 'Pothole', 'Water Leakage', 'Streetlight Damage']

# SQLite setup
DB_PATH = 'reports.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Function to classify image
def classify_issue(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    return labels[predicted_index]

# POST endpoint to report an issue
@app.route('/report', methods=['POST'])
def report_issue():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    prediction = classify_issue(filepath)
    timestamp = datetime.now().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO reports (filename, prediction, timestamp)
        VALUES (?, ?, ?)
    ''', (filename, prediction, timestamp))
    conn.commit()
    conn.close()

    return jsonify({
        'filename': filename,
        'prediction': prediction,
        'timestamp': timestamp
    })

# GET endpoint to view all reports
@app.route('/reports', methods=['GET'])
def get_reports():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, prediction, timestamp FROM reports ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()

    reports = []
    for row in rows:
        reports.append({
            'id': row[0],
            'filename': row[1],
            'prediction': row[2],
            'timestamp': row[3]
        })

    return jsonify(reports)

# Root
@app.route('/')
def home():
    return "👋 CivicLens Backend is Running"

# Run server
if __name__ == '__main__':
    app.run(debug=True)

