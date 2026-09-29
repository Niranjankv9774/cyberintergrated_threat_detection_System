import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "dataset", "best_model.keras")

# Global variable to hold the model
_model = None

def get_model():
    global _model
    if _model is None:
        if os.path.exists(model_path):
            _model = load_model(model_path, compile=False)
            print("Model loaded successfully!")
        else:
            print(f"Model not found at {model_path}")
    return _model

import tempfile

def preprocess_ela(img_path, quality=90):
    """
    Replicates ELA preprocessing for the model.
    """
    image = Image.open(img_path).convert('RGB')
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        temp_resaved = tmp.name
    
    try:
        # Save at lower quality
        image.save(temp_resaved, 'JPEG', quality=quality)
        resaved_image = Image.open(temp_resaved)
        
        # Calculate difference
        ela_image = ImageChops.difference(image, resaved_image)
        
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale * 15)
        return ela_image
    finally:
        # Clean up
        if os.path.exists(temp_resaved):
            os.remove(temp_resaved)

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(img_path):
    model = get_model()
    if model is None:
        return "Error", 0

    try:
        # Step 1: Apply ELA preprocessing
        ela_img = preprocess_ela(img_path)
        
        # Step 2: Convert to grayscale and resize as required by the model (64, 64, 1)
        img = ela_img.convert('L')
        img = img.resize((64, 64))
        
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, 64, 64, 1)

        pred = model.predict(img_array)[0][0]
        
        # FIXED LOGIC:
        # In ELA mode, higher values (more noise) = Fake.
        # Based on diagnostics: Untitled_design.png is ~0.47.
        # We use a threshold of 0.33 to be more sensitive to subtle edits.
        threshold = 0.33
        
        # Label Mapping: Low values = Real, High values = Fake
        label = "Fake" if pred >= threshold else "Real"
        
        # Calculate confidence
        if label == "Fake":
            # 0.4 to 1.0 range
            confidence = round(((pred - threshold) / (1 - threshold)) * 100, 2)
            confidence = min(max(confidence, 50), 100)
        else:
            # 0.0 to 0.4 range
            confidence = round(((threshold - pred) / threshold) * 100, 2)
            confidence = min(max(confidence, 50), 100)

        return label, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error", 0
