from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient
from PIL import Image
import requests
import io
 
# Initialize the inference client for Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dZvN0FZAZF3VuaVNLn2R"
)

# Default values for Ultralytics API
DEFAULT_MODEL = "https://hub.ultralytics.com/models/5GKhh9bzNR2oE9ortZBq"
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
API_URL = "https://predict.ultralytics.com"
API_KEY = "15006b56cd8ef5e110058e45f8a92a9ef9664cc5ce"
 
async def detect_objects(image):
    """Perform object detection using Roboflow."""
    print(f"Content type: {image.content_type}")
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        # Read and process the image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Perform inference using Roboflow
        result = CLIENT.infer(pil_image, model_id="lateral-neck/9")
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

 
async def run_inference(file):
    """Perform inference using Ultralytics API."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        # Read uploaded image
        image_content = await file.read()

        # Prepare request data
        data = {
            "model": DEFAULT_MODEL,
            "imgsz": DEFAULT_IMGSZ,
            "conf": DEFAULT_CONF,
            "iou": DEFAULT_IOU,
        }
        headers = {"x-api-key": API_KEY}

        # Send request to the inference API
        response = requests.post(
            API_URL, headers=headers, data=data, files={"file": ("image.jpg", image_content)}
        )

        # Handle errors
        response.raise_for_status()

        # Return the API's response
        return JSONResponse(content=response.json(), status_code=response.status_code)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

 



