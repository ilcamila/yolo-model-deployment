import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO

app = FastAPI()
from ultralytics import YOLO

# Carga el modelo YOLOv8
model = YOLO('yolov8n.pt')

# Cargar el modelo entrenado desde un archivo local
model = torch.hub.load('.', 'custom', path='yolov8n.pt', source='local')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    results = model(image)
    return {"detections": results.pandas().xyxy[0].to_dict(orient="records")}
