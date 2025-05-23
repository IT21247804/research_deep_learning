from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import numpy as np
import io
import cv2

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add the URL of your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# File paths
model_path = 'pharyngitis_model_mobilenetv2.h5'
oral_img_validator_model_path = 'oral_img_validator_model.h5'

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
     
# Define the API endpoint
@app.post("/pharyngitis/predict")
async def predictAPI(file: UploadFile = File(...)):
   try:
      file = await file.read()
      file_bytes = io.BytesIO(file)
      
      result,probability = validate_image(file_bytes)

      if result == False:
         return {
            "success": True,
            "message": "Invalid Image",
            "data":{
               "prediction": "invalid",
               "confidence_score": float(probability)  
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
            "prediction": class_name.strip(),
            "confidence_score": float(confidence_score)  
         }
      }
   except Exception as e:
      return {"error": str(e)}