import numpy as np
import io
import os
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf # We use TF only for the Lite Interpreter

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------
# LOAD TFLITE MODEL
# -----------------------------------------------------------
MODEL_PATH = 'model.tflite'
interpreter = None
input_details = None
output_details = None

try:
    if os.path.exists(MODEL_PATH):
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        # Get input and output details.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ TFLite Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"❌ Error: {MODEL_PATH} not found in the current directory.")
        print("Current directory files:", os.listdir('.'))

except Exception as e:
    print(f"❌ Critical Error loading TFLite model: {e}")

CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224) 

def preprocess_image(image_bytes):
    """
    Preprocess image for TFLite (Float32 input)
    """
    img = Image.open(io.BytesIO(image_bytes))
    
    # Ensure RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(IMG_SIZE)
    
    # Convert to array and float32 (Required for TFLite)
    img_array = np.array(img, dtype=np.float32)
    
    # Expand dims to (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    return img_array

@app.route('/', methods=['GET'])
def home():
    status = "Running" if interpreter else "Failed to Load Model"
    return f"Brain Tumor Detection API (TFLite) is {status}!"

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({'error': 'Model failed to load on server start. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Preprocess
        processed_img = preprocess_image(file.read())
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Decode results
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_label = CLASS_LABELS[class_idx]
        
        return jsonify({
            'class': predicted_label,
            'confidence': f"{confidence:.2%}",
            'all_probabilities': {label: float(prob) for label, prob in zip(CLASS_LABELS, predictions[0])}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use the port provided by Render, or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
