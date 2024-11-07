import os
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from typing import List
from PIL import Image
import shutil
import torch
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Adding CORS Middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Path to store uploaded images temporarily
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class SearchResponse(BaseModel):
    image_path: str
    similarity_score: float


def encode_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    return model.get_image_features(**inputs)


def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    return model.get_text_features(**inputs)


def similarity(feature1, feature2):
    return torch.cosine_similarity(feature1, feature2).item()


@app.post("/search", response_model=List[SearchResponse])
async def local_image_search(image_file: UploadFile = File(None), folder_path: str = Form(...)):
    # Check if query is provided
    if image_file:
        # Save uploaded image temporarily
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        query_feature = encode_image(image_path)
    else:
        return {"error": "No valid query provided."}

    # Search for similar images in the specified folder
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(folder_path, filename)
            image_feature = encode_image(image_path)
            sim = similarity(query_feature, image_feature)
            results.append({"image_path": image_path, "similarity_score": sim})

    # Sort results by similarity score
    results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
    return results[:4]  # return top 5 results
