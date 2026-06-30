import os
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify

# Add src to path if running from root
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from grad_cam import GradCAM

app = Flask(__name__)

# Constants
MODEL_PATH = 'models/dummy_model.h5'
CLASSES = ['Actinic keratoses (akiec)', 'Basal cell carcinoma (bcc)', 
           'Benign keratosis-like lesions (bkl)', 'Dermatofibroma (df)', 
           'Melanoma (mel)', 'Melanocytic nevi (nv)', 'Vascular lesions (vasc)']
IMG_SIZE = (224, 224)

# Load model globally
model = None
def load_cached_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            print("Model not found. Please run src/generate_dummy_model.py or train a model.")
    return model

def prepare_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_orig = np.array(img)
    
    img = img.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Preprocessing expected by EfficientNetV2 is handled internally, 
    # but we need batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Original image resized for GradCAM overlay
    img_orig_resized = cv2.resize(img_orig, IMG_SIZE) if 'cv2' in sys.modules else np.array(img.resize(IMG_SIZE))
    
    return img_array, img_orig_resized

def encode_image_to_base64(img_array):
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
            
        img_bytes = file.read()
        
        import cv2 # ensure cv2 is available here
        model = load_cached_model()
        if model is None:
            return jsonify({'error': 'Model not loaded. Server error.'})
            
        # Prepare image
        processed_img, orig_img = prepare_image(img_bytes)
        
        # Predict
        preds = model.predict(processed_img)[0]
        class_idx = np.argmax(preds)
        confidence = float(preds[class_idx])
        pred_class = CLASSES[class_idx]
        
        # Generate Grad-CAM
        try:
            cam = GradCAM(model, class_idx)
            heatmap = cam.compute_heatmap(processed_img)
            overlay = cam.overlay_heatmap(heatmap, orig_img)
            
            # Encode overlay to base64
            heatmap_b64 = encode_image_to_base64(overlay)
        except Exception as e:
            print(f"GradCAM error: {e}")
            heatmap_b64 = None
            
        return jsonify({
            'success': True,
            'prediction': pred_class,
            'confidence': f"{confidence:.2%}",
            'heatmap': heatmap_b64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Initialize model on startup
    load_cached_model()
    app.run(debug=True, port=5000)
