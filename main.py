# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn

app = FastAPI()

# Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model once at start
MODEL = tf.keras.models.load_model("model.keras")

CLASS_NAMES = [
   "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"
]

IMAGE_SIZE = 256


@app.get("/")
def read_root():
    return {"message": "Potato Disease Detection API Running"}


def read_file_as_image(data):
    image = Image.open(BytesIO(data)).convert("RGB")
    return np.array(image)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = read_file_as_image(await file.read())

        # Resize
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # Normalize
        image = image / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(image, axis=0)

        # Predict
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        print("ERROR:", str(e))  # Print to terminal for debugging
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)