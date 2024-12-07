import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import pandas as pd
from ultralytics import YOLO

app = FastAPI()

# Cargar el modelo YOLO desde un archivo local
model = YOLO("yolov8n.pt")  # Asegúrate de que el archivo "yolov8n.pt" esté en el mismo directorio.

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Leer la imagen cargada por el usuario
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    
    # Pasar la imagen al modelo
    results = model(image)
    
    # Extraer las detecciones
    detections = results[0].boxes.data.cpu().numpy()  # Convertir a NumPy
    
    # Crear un DataFrame con las detecciones
    detections_df = pd.DataFrame(detections, columns=["x_min", "y_min", "x_max", "y_max", "confidence", "class"])
    
    # Convertir el DataFrame a formato JSON y devolverlo
    return {"detections": detections_df.to_dict(orient="records")}
