#!/usr/bin/env python3
"""
Predict waste type from an image file
Usage: python predict_image.py <image_path>
"""
import sys
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

def load_model():
    """Load the trained model"""
    print("🏗️  Loading model...")
    
    # Recreate architecture
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
    ])
    
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    model.load_weights('waste_classifier_best.h5')
    
    return model

def preprocess_image(image_path):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_path):
    """Predict waste type for an image"""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    # Class names
    class_names = ['O', 'R']
    
    # Load model
    model = load_model()
    
    # Predict
    print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    
    # Results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    
    for i, class_name in enumerate(class_names):
        confidence = predictions[0][i] * 100
        bar_length = int(confidence / 2)
        bar = "█" * bar_length
        
        waste_type = {
            'O': 'Organic/Wet Waste',
            'R': 'Recyclable/Dry Waste'
        }.get(class_name, class_name)
        
        print(f"\n{waste_type}:")
        print(f"  {bar} {confidence:.2f}%")
    
    # Final prediction
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    
    waste_type = {
        'O': '🍂 Organic/Wet Waste',
        'R': '♻️  Recyclable/Dry Waste'
    }.get(predicted_class, predicted_class)
    
    print("\n" + "=" * 50)
    print(f"✅ CLASSIFICATION: {waste_type}")
    print(f"   Confidence: {confidence:.1f}%")
    print("=" * 50)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
        print("\nExample:")
        print("  python predict_image.py DATASET/TEST/O/O_12345.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict(image_path)
