import os
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

# Get absolute path to model files (dynamically relative to the current file)
current_dir = os.path.dirname(os.path.abspath(__file__))  # Current file directory

image_validator_model_path = os.path.join(current_dir, 'waters_view_validator.h5')
model_path = os.path.join(current_dir, 'inceptionV3_updated.h5')

# Labels
class_names = ["mild", "moderate", "severe"]

# Preprocess image for validation
def preprocess_image_for_validate(file_bytes):
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    img = Image.open(file_bytes).convert("RGB")
    img = np.array(img)
    if img is None:
        raise ValueError("Could not read image")
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Validate image
def validate_image(file_bytes):
    try:
         img = preprocess_image_for_validate(file_bytes)
         image_validator_model = load_model(image_validator_model_path, compile=False)
         prediction = image_validator_model.predict(img)
         probability = prediction[0][0]
         return (True, probability) if probability > 0.5 else (False, probability)
    except Exception as e:
         raise ValueError(f"Error: {e}")

# Preprocess image for prediction
def preprocess_image(file):
    try:
         image = Image.open(file).convert("RGB")
        
         # Resize the image to (224, 224) as required by InceptionV3
         # Resize the image to (224, 224) as required by ResNet50
         # Resize the image to (224, 224) as required by VGG16
         size = (224, 224)
         
         image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
         image_array = np.asarray(image)
         
         # Normalize the image (same as the original InceptionV3 model preprocessing)
         normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
         
         # Create the array of the right shape for the model
         data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
         data[0] = normalized_image_array
         return data
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {e}")

# Predict image
def predict_image(file_bytes):
   try:
      # Load the trained models
      model = load_model(model_path, compile=False)
      
      # Preprocess image
      data = preprocess_image(file_bytes)
      
      prediction = model.predict(data)
      index = np.argmax(prediction) # get the highest probability for classes
      class_name = class_names[index]
      confidence_score = prediction[0][index]
      return class_name, confidence_score
   except Exception as e:
      raise ValueError(f"Error during prediction: {e}")

# Main file processing function
async def process_xray_image(file):
   try:
      file_bytes = io.BytesIO(await file.read())
      valid, probability = validate_image(file_bytes)

      if not valid:
            return {
               "success": True,
               "message": "Invalid Image",
               "data": {
                  "status": "failed",
                  "confidenceScore": float(probability),
                  "prediction": "invalid",
                  "isSinusitis": False,
                  "label": "Invalid Water's View X Ray",
                  "suggestions": "Please upload valid Water's View X Ray"
               }
            }

      # Predict severity
      class_name, confidence_score = predict_image(file_bytes)

      return {
            "success": True,
            "message": "success",
            "data": {
               "status": "diagnosed",
               "isSinusitis": class_name.strip() != "mild",
               "severity": get_descriptive_title(class_name.strip()),
               "suggestions": get_suggestions(class_name.strip()),
               "confidenceScore": float(confidence_score),
               "prediction": "valid",
               "predictionClass": class_name.strip()
            }
      }
   except Exception as e:
       return {
           "success": False,
           "message": "Failed",
           "data": {
               "status": "failed",
               "isSinusitis": False,
               "stage": "N/A",
               "suggestions": "",
               "confidenceScore": "0.00",
               "prediction": "N/A"
           },
           "error": str(e)
       }

def get_descriptive_title(class_name):
    class_info = {
        "mild": "Mild (Healthy or Minimal Sinus Congestion)",
        "moderate": "Moderate (Significant Inflammation in Sinus Cavity)",
        "severe": "Severe (Advanced Blockage or Potential Complications)"
    }
    return class_info.get(class_name, "Invalid class name. Please provide a valid class name.")

def get_suggestions(class_name):
    suggestions = {
        "mild": "No treatment necessary. Maintain hydration, use saline sprays, and monitor symptoms if present.",
        "moderate": "Consider consulting a healthcare provider. Use nasal decongestants or antihistamines as recommended.",
        "severe": "Urgent medical evaluation required. Imaging studies (e.g., CT scan) may be necessary for treatment planning."
    }
    return suggestions.get(class_name, "Invalid class name. Please provide a valid class name.")
