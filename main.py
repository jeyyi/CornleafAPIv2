from typing import Union
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from PIL import Image
import io
import numpy as np

app = FastAPI()


# Configure CORS
origins = [
    "http://localhost:3000",  # Replace with the URL of your React app
    # Add more allowed origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#load the pretrained model
model = load_model('Inception.h5')
labels = ['Gray Leaf Spot', 'Common Rust', 'Healthy', 'Blight']

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    with tf.device("/cpu:0"):
        # Read the image file
        image = await file.read()
        img = Image.open(io.BytesIO(image))

        # Resize the image to the input size expected by the Inception model
        img = img.resize((180, 180))

        # Preprocess the image
        x = img_to_array(img)
        x = x / 255.0  # Normalize the pixel values between 0 and 1
        x = np.expand_dims(x, axis=0)

        # Make predictions using the Inception model
        prediction = model.predict(x)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_label = labels[predicted_class_index]

        return {predicted_class_label}