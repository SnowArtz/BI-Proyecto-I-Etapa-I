from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load
import numpy as np
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import json


origins = [
    "http://localhost.tiangolo.com/",
    "https://localhost.tiangolo.com/",
    "http://localhost/",
    "http://localhost:3000/",
    "http://localhost:8080/",
]




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class TextsData(BaseModel):  # Cambiado el nombre a TextsData
    texts: List[str]  # Ahora espera una lista de textos

loaded_pipeline = load('D:/Universidad/Sem 8/BI/Proyecto 1/BI-Proyecto-I-Etapa-I/models/BOWRandomForest/BOWRandomForest.pkl')
X_train = pd.read_excel("data/raw/cat_6716.xlsx").Textos_espanol
y_train = pd.read_excel("data/raw/cat_6716.xlsx").sdg
loaded_pipeline.fit(X_train, y_train)


@app.get("/")
async def main():
    return {"message": "Hello World"}

@app.get("/metrics")
def metrics():
    # Asumiendo que tu archivo se llama "data.json" y estÃ¡ en la misma carpeta que tu script
    with open("D:/Universidad/Sem 8/BI/Proyecto 1/BI-Proyecto-I-Etapa-I/data/metrics/metrics.json", "r") as file:
        data = json.load(file)
    return data

@app.post("/predict")
def predict(data: TextsData):  # Actualizado el tipo de dato a TextsData
    X_test = pd.Series(data.texts)  # Procesa la lista directamente
    predictions = loaded_pipeline.predict(X_test)
    return {"predictions": predictions.tolist()}  # Devuelve la lista de predicciones
