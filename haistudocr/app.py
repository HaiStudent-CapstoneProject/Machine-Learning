from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import tensorflow as tf 
import cv2
from tensorflow.keras.models import load_model
import CharacterSegmentation as cs

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = './static/'
MODEL_PATH = './model/haistudocr.h5'
SEGMENTED_DIR = './segmented/'
MAPPING_PATH = './data/emnist/processed-mapping.csv'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained OCR model
model = load_model(MODEL_PATH)

# Load mapping file for converting predictions to characters
mapping_df = pd.read_csv(MAPPING_PATH)
code2char = {row['id']: row['char'] for _, row in mapping_df.iterrows()}

@app.route('/')
def home():
    return "Welcome to the OCR Flask API!"

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if file is provided
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process the image
    try:
        ocr_result = process_image(filepath)
        return jsonify({'text': ocr_result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/output/<filename>')
def get_output_image(filename):
    file_path = os.path.join(SEGMENTED_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return jsonify({'error': 'File not found'}), 404

def process_image(filepath):
    """
    Complete OCR pipeline:
    1. Segment characters using CharacterSegmentation module.
    2. Preprocess segmented characters for EMNIST format.
    3. Use the trained model for character recognition.
    """
    # Step 1: Character Segmentation
    cs.image_segmentation(filepath)

    # Load segmented images
    segmented_images = []
    for file in os.listdir(SEGMENTED_DIR):
        if file.endswith('.jpg'):
            segmented_images.append(os.path.join(SEGMENTED_DIR, file))

    # Step 2: Preprocess segmented images to EMNIST format
    X_data = []
    for img_path in segmented_images:
        img = Image.open(img_path).resize((28, 28))
        inv_img = ImageOps.invert(img)
        flatten = np.array(inv_img).flatten() / 255
        flatten = np.where(flatten > 0.5, 1, 0)
        X_data.append(flatten)

    X_data = np.array(X_data).reshape(-1, 28, 28, 1)

    # Step 3: Predict using the trained model
    predictions = model.predict(X_data)
    predicted_indices = np.argmax(predictions, axis=1)

    # Convert indices to characters
    recognized_text = ''.join([code2char[idx] for idx in predicted_indices])
    return recognized_text

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
