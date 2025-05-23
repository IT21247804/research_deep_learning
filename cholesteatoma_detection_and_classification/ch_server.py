import cv2
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
# from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add the URL of your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
 
# Load the trained model
# image_validator_model_path = 'cholesteatoma_validator.h5'
model_path = 'InceptionV3_cholesteatoma_identifier_pp1.h5'

# Load the model
model = load_model(model_path, compile=False)
# image_validator_model = load_model(image_validator_model_path, compile=False)

# Class labels
class_names = ["Normal", "Stage 1", "Stage 2", "Stage 3"]
 
# Function to preprocess an image
# def preprocess_image_for_validate(file_bytes):
#       IMG_HEIGHT, IMG_WIDTH = 224, 224
#       img = Image.open(file_bytes).convert("RGB")
#       img = np.array(img)
#       #  img = cv2.imread(image_path)  # Load image
#       if img is None:
#          raise ValueError(f"Could not read image")
#       img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))  # Resize
#       img = img / 255.0  # Normalize
#       img = np.expand_dims(img, axis=0)  # Add batch dimension

#       return img

# Function to classify the image
# def validate_image(file_bytes):
#     try:
#         img = preprocess_image_for_validate(file_bytes)
#         prediction = image_validator_model.predict(img)

#         print(f"Prediction: {prediction}")

#         probability = prediction[0][0]  # Get the prediction score

#         if probability > 0.5:
#             return True, probability
#         else:
#             return False, probability
#     except Exception as e:
#         raise ValueError(f"Error: {e}")

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
      data = preprocess_image(file_bytes)

      # Perform the prediction
      prediction = model.predict(data)
      # get the highest probability for classes
      index = np.argmax(prediction) 
      class_name = class_names[index]
      confidence_score = prediction[0][index]

      return class_name, confidence_score

   except Exception as e:
      raise ValueError(f"Error during prediction: {e}")


# Define the API endpoint
@app.post("/analyze")
async def predictAPI(file: UploadFile = File(...)):
   try:
      # Read the uploaded file as bytes
      file = await file.read()
      file_bytes = io.BytesIO(file)

      # result,probability = validate_image(file_bytes)

      # if result == False:
         # return {
         #    "success": True,
         #    "message": "Invalid image",
         #    "data": {
         #       "confidence_score": float(probability),
         #       "prediction": "invalid"
         #    }
         # }
         
      # Preprocess the image
      data = preprocess_image(file_bytes)

      # Predict the severity level
      class_name, confidence_score = predict_image(data)

      # Return the result as JSON
      return {
            "success": True,
            "message": "Prediction successful",
            "data": {
               "status": "diagnosed",
               "isCholesteatoma": True,
               "stage": class_name.strip(),
               "suggestions": "",
               "confidence_score": float(confidence_score),
               "prediction": "valid"
            }
      }
   except Exception as e:
      return {"error": str(e)}
