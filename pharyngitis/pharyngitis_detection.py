import os
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import numpy as np
import io
import cv2


# Get absolute path to model files (dynamically relative to the current file)
current_dir = os.path.dirname(os.path.abspath(__file__))  # Current file directory
 
# File paths
model_path = os.path.join(current_dir, 'pharyngitis_model_mobilenetv2.h5')
oral_img_validator_model_path = os.path.join(current_dir, 'oral_img_validator_model.h5')

# Function to preprocess an image
def preprocess_image_for_validate(file_bytes):
      IMG_HEIGHT, IMG_WIDTH = 224, 224
      img = Image.open(file_bytes).convert("RGB")
      img = np.array(img)
      #  img = cv2.imread(image_path)  # Load image
      if img is None:
         raise ValueError(f"Could not read image")
      img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))  # Resize
      img = img / 255.0  # Normalize
      img = np.expand_dims(img, axis=0)  # Add batch dimension

      return img

# Function to classify the image
def validate_image(file_bytes):
    try:
        img = preprocess_image_for_validate(file_bytes)
        
        oral_img_validator_model = load_model(oral_img_validator_model_path, compile=False)
        prediction = oral_img_validator_model.predict(img)

        print(f"Prediction: {prediction}")

        probability = prediction[0][0]  # Get the prediction score

        if probability > 0.5:
            return True,probability
        else:
            return False,probability
    except Exception as e:
        raise ValueError(f"Validate Error: {e}")
 
async def process_oral_image(file):
   try:
      file = await file.read()
      file_bytes = io.BytesIO(file)
      
      valid,probability = validate_image(file_bytes)

      if not valid:
         return {
            "success": True,
            "message": "Invalid Image",
            "data":{
               "status": "failed",
               "confidenceScore": float(probability),
               "prediction": "invalid",
               "isPharyngitis": False,
               "label": "Invalid Oral Image",
               "suggestions": "Please upload valid Oral Image"
            }
         }
         
      # Load the model
      model = load_model(model_path, compile=False)

      # Define the class names
      class_names = ["normal", "moderate", "tonsillitis"]

      # Prepare input image
      data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
      image = Image.open(file_bytes).convert("RGB")
      image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
      image_array = np.asarray(image)
      normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
      data[0] = normalized_image_array

      # Predict
      prediction = model.predict(data)
      index = np.argmax(prediction)
      class_name = class_names[index]
      confidence_score = prediction[0][index]
 
      return {
         "success": True,
         "message": "success",
         "data":{
            "status": "diagnosed",
            "isPharyngitis": class_name.strip() != "normal",
            "stage": get_descriptive_title(class_name.strip()),
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
               "isPharyngitis": False,
               "stage": "N/A",
               "suggestions": "",
               "confidenceScore": "0.00",
               "prediction": "N/A"
           },
           "error": str(e)
       }


def get_descriptive_title(class_name):
      class_info = {
         "normal": "Healthy (No throat inflammation or symptoms)",
         "moderate": "Moderate (Inflamed or red throat causing mild discomfort)",
         "tonsillitis": "Tonsillitis (A severe subtype of pharyngitis with swollen, pus-filled tonsils with white patches and high fever)"
      } 
      return class_info.get(class_name, "Invalid class name. Please provide a valid class name.")

def get_suggestions(class_name):
      suggestions = {
         "normal": "No treatment needed. Stay hydrated and maintain oral hygiene.",
         "moderate": "Try warm gargles or lozenges. See a doctor if it worsens.",
         "tonsillitis": "Immediate medical care needed. Antibiotics or advanced treatment may be required."
      }

      return suggestions.get(class_name, "Invalid class name. Please provide a valid class name.")