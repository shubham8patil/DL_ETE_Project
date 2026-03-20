import os
import numpy as np
import tensorflow as tf
from PIL import Image
from Backend.config import IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES

def load_trained_model(model_path):
    """
    Loads and returns the trained Keras model.
    Includes a fallback for loading weights if the full architecture
    fails to deserialize (e.g., when moving models from Colab/Keras 3 to local/Keras 2).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Standard load failed, attempting weight-only fallback... ({e})")
        from Backend.model import build_model
        model = build_model()
        # Expecting some warnings/errors here if the format is completely incompatible, 
        # but load_weights often succeeds between Keras versions when load_model fails.
        model.load_weights(model_path)
        return model

def preprocess_image(image_path_or_pil):
    """
    Prepares an image for model prediction (resize, normalize, expand dims).
    Accepts either a file path or a PIL Image object.
    """
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil)
    else:
        img = image_path_or_pil
        
    # Ensure image is RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_tumor(model, image_input):
    """
    Predicts the tumor class for a given image.
    Returns the predicted class name and confidence dictionary.
    """
    processed_image = preprocess_image(image_input)
    predictions = model.predict(processed_image, verbose=0)[0]
    
    predicted_class_idx = np.argmax(predictions)
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    
    confidence_dict = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    
    return predicted_class_name, confidence_dict

if __name__ == '__main__':
    # Simple test (assuming a model exists)
    from Backend.config import OUTPUT_DIR, TEST_DIR
    model_path = os.path.join(OUTPUT_DIR, 'brain_tumor_model.keras')
    
    try:
        model = load_trained_model(model_path)
        # Pick a random test image
        test_img_path = os.path.join(TEST_DIR, 'glioma', os.listdir(os.path.join(TEST_DIR, 'glioma'))[0])
        print(f"Testing prediction on: {test_img_path}")
        
        pred_class, conf_dict = predict_tumor(model, test_img_path)
        print(f"\\nPredicted: {pred_class}")
        print("Confidences:")
        for cls_name, conf in conf_dict.items():
            print(f"  {cls_name}: {conf*100:.2f}%")
            
    except FileNotFoundError as e:
        print(e)
