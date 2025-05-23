from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from routes import router

# For specific route
# from routes import predict

# Initialize FastAPI app
app = FastAPI()

# Set up logging for server startup and runtime information
logging.basicConfig(level=logging.INFO)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust based on your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a root endpoint to provide information about the API
@app.get("/")
async def root():
    return {"message": "Welcome to the Medical Image Analysis API."}

# Include the router with all routes
app.include_router(router, prefix="/api")

# Include the specific route from the routes.py file
# app.include_router(predict.router, prefix="/predict", tags=["predict"])

# Log server startup message with a base URL reference
@app.on_event("startup")
async def startup_event():
    logging.info("> ==============================================================")
    logging.info("> Medical Image Analysis API is running...")
    logging.info("> ==============================================================")


# Running the FastAPI server (set to run on port 4000)
# This only work when run: python server.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
