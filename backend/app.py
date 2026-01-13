import tensorflow as tf
import numpy as np
import io
import os
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import specific layers needed for the custom classes
from tensorflow.keras.layers import (
    Layer, Dense, GlobalAveragePooling2D, Conv2D, Reshape, 
    Multiply, GlobalMaxPooling2D, Add, Activation, Input
)
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# 1. Define Custom Layers (Must match training code exactly)
# ---------------------------------------------------------

class ChannelAttention(Layer):
    def __init__(self, channels, reduction=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction

    def build(self, input_shape):
        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()
        self.fc1 = Dense(self.channels // self.reduction, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.fc2 = Dense(self.channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        super(ChannelAttention, self).build(input_shape)

    def call(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        return Multiply()([x, Reshape((1, 1, self.channels))(out)])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'channels': self.channels, 'reduction': self.reduction})
        return config

class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = Conv2D(1, self.kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        super(SpatialAttention, self).build(input_shape)

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=3, keepdims=True)
        max_out = tf.reduce_max(x, axis=3, keepdims=True)
        out = tf.concat([avg_out, max_out], axis=3)
        return Multiply()([x, self.conv(out)])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

class CBAM(Layer):
    def __init__(self, channels, reduction=16, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def call(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({'channels': self.channels, 'reduction': self.reduction, 'kernel_size': self.kernel_size})
        return config

# -----------------------------------------------------------
# FLASK SETUP
# -----------------------------------------------------------

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------
# LOAD MODEL (FIXED)
# -----------------------------------------------------------
try:
    # We must explicitly tell Keras what 'CBAM', 'ChannelAttention', etc. mean
    custom_objects = {
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention,
        'CBAM': CBAM
    }
    
    model = load_model('model.h5', custom_objects=custom_objects)
    print("✅ Model loaded successfully with Custom Layers.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Fallback for debugging locally
    model = None

CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224) 

# -----------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# -----------------------------------------------------------
# API ROUTES
# -----------------------------------------------------------
@app.route('/', methods=['GET'])
def home():
    return "Brain Tumor Detection API (with CBAM) is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load on server start.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        processed_img = preprocess_image(file.read())
        predictions = model.predict(processed_img)
        
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
    app.run(debug=True, port=5000)