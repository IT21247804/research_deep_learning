import os
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
import logging

# Get absolute path to model files (dynamically relative to the current file)
current_dir = os.path.dirname(os.path.abspath(__file__))

image_validator_model_path = os.path.join(current_dir, 'ResNet50_endoscopy_image_validator.h5')
model_path = os.path.join(current_dir, 'InceptionV3_cholesteatoma_identifier_pp1.h5')
# model_path = os.path.join(current_dir, 'InceptionV3_cholesteatoma_identifier_pp2.h5')

# Class labels
class_names = ["Normal", "Stage 1", "Stage 2", "Stage 3"]

# Preprocess image for cholesteatoma validation
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

# Preprocess image for cholesteatoma prediction
def preprocess_image(file, target_size=(224, 224)):
    try:
         # Open the image and ensure it's in RGB mode
         image = Image.open(file).convert("RGB")

         # Resize the image to the target size
         image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)

         # Convert image to numpy array
         image_array = np.asarray(image)

         # Normalize the image (same as the original InceptionV3 model preprocessing)
         normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

         # Expand dimensions to match model input
         return np.expand_dims(normalized_image_array, axis=0), image

    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {e}")

# Predict image
def predict_image(file_bytes):
   try:
      # Load the trained models
      model = load_model(model_path, compile=False)

      # Preprocess image
      data, _ = preprocess_image(file_bytes)  # Extract only the first item (processed image)

      # Perform the prediction
      prediction = model.predict(data)
      # get the highest probability for classes
      index = np.argmax(prediction) 
      class_name = class_names[index]
      confidence_score = prediction[0][index]

      return class_name, confidence_score

   except Exception as e:
      raise ValueError(f"Error during prediction: {e}")

# Image file processing main function
async def process_endoscopy_image(file):
   try:
      file_bytes = io.BytesIO(await file.read())
      valid, probability = validate_image(file_bytes)

      logging.info("> ========== Endoscopy image validation ==========")
      logging.info("> Validity: " + "Valid Image" if valid else "Invalid Image")
      logging.info("> Probability: " + str(probability))
      logging.info("> =====================================================")

      if not valid:
            return {
               "success": True,
               "message": "Invalid image",
               "data": {
                  "status": "failed",
                  "confidenceScore": float(probability),
                  "prediction": "invalid"
               }
            }

      # Predict stages of cholesteatoma
      class_name, confidence_score = predict_image(file_bytes)

      # Check cholesteatoma by class name
      isCholesteatoma = False if str(class_name).strip().lower() == "normal" else True

      logging.info("> ========== Predict stages of cholesteatoma ==========")
      logging.info("> Class: " + str(class_name))
      logging.info("> Confidence Score: " + str(confidence_score))
      logging.info("> =====================================================")

      return {
            "success": True,
            "message": "Prediction successful",
            "data": {
               "status": "diagnosed",
               "isCholesteatoma": isCholesteatoma,
               "stage": get_descriptive_stage(class_name.strip()),
               "suggestions": get_suggestions(class_name.strip()),
               "confidenceScore": float(confidence_score),
               "prediction": "valid"
            }
      }

   except Exception as e:
      return {
         "success": False,
         "message": "Failed",
         "data": {
            "status": "failed",
            "isCholesteatoma": False,
            "stage": "N/A",
            "suggestions": "",
            "confidenceScore": "0.00",
            "prediction": "N/A"
         },
         "error": str(e)
      }


# Descriptive stages based on class names
def get_descriptive_stage(class_name):
    class_info = {
        "Normal": "Normal (Healthy Eardrum)",
        "Stage 1": "Stage 1 (Single Quadrant Involved)",
        "Stage 2": "Stage 2 (Multiple Quadrants Involved)",
        "Stage 3": "Stage 3 (Ossicular Involvement)",
        "Stage 4": "Stage 4 (Mastoid Disease/Extension)"
    }

    return class_info.get(class_name, "Invalid class name. Please provide a valid class name.")

# Suggestions based on class names
def get_suggestions(class_name):
    suggestions = {
        "Normal": "No action required. Maintain good ear hygiene and periodic checkups.",
        "Stage 1": "Mild involvement. Consider monitoring the condition and consulting an ENT specialist.",
        "Stage 2": "Involves multiple quadrants. Seek advice from an ENT specialist for further evaluation.",
        "Stage 3": "Ossicular involvement detected. Mastoid disease might be present: A CT scan is highly recommended.",
        "Stage 4": "Critical stage with mastoid disease extension. Immediate CT scan and ENT consultation required."
    }

    return suggestions.get(class_name, "Invalid class name. Please provide a valid class name.")
