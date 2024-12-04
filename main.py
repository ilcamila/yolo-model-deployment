from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from io import BytesIO

app = FastAPI()

# Cargar el modelo entrenado
model = torch.hub.load('ultralytics/yolov8', 'custom', path='yolov8n.pt')  # Cambia según sea YOLOv5 o YOLOv8

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    results = model(image)
    return {"detections": results.pandas().xyxy[0].to_dict(orient="records")}
