import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

app = FastAPI()

# Cargar el modelo YOLO desde un archivo local
model = YOLO('yolov8n.pt')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    results = model(image)
    return {"detections": results.pandas().xyxy[0].to_dict(orient="records")}
