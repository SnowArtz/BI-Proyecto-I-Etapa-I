from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
from joblib import load, dump
import numpy as np
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import io

from fastapi.responses import FileResponse
from pathlib import Path




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

class TextsData(BaseModel):
    texts: List[str]

loaded_pipeline = load('models/BOWRandomForest/BOWRandomForest.pkl')
X_train = pd.read_excel("data/raw/cat_6716.xlsx").Textos_espanol
y_train = pd.read_excel("data/raw/cat_6716.xlsx").sdg
loaded_pipeline.fit(X_train, y_train)

@app.get("/")
async def main():
    return {"message": "Hello World"}

@app.get("/metrics")
def metrics():
    with open("data/metrics/metrics.json", "r") as file:
        data = json.load(file)
    return data

@app.post("/predict")
def predict(data: TextsData):
    X_test = pd.Series(data.texts)
    predictions = loaded_pipeline.predict(X_test)
    return {"predictions": predictions.tolist()}

@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    contents = await file.read()
    data = pd.read_excel(io.BytesIO(contents))
    

    X_train_original = pd.read_csv("data/interim/X_train_raw.csv").Textos_espanol
    X_test_original = pd.read_csv("data/interim/X_test_raw.csv").Textos_espanol
    y_train_original = pd.read_csv("data/interim/y_train_raw.csv").sdg
    y_test_original = pd.read_csv("data/interim/y_test_raw.csv").sdg

    X_train_new = pd.Series(data.Textos_espanol)
    y_train_new = pd.Series(data.sdg)

    X_train_final = pd.concat([X_train_new, X_train_original])
    y_train_final = pd.concat([y_train_new, y_train_original])

    loaded_pipeline.fit(X_train_final, y_train_final)
    predictions = loaded_pipeline.predict(X_test_original)

    with open("data/metrics/metrics.json", 'w') as json_file:
        json.dump(classification_report(y_test_original, predictions, digits=4, output_dict=True)["weighted avg"], json_file)


    # Matriz de confusión
    cm = confusion_matrix(y_test_original , predictions) 
    display_labels = np.unique(y_test_original) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("data/metrics/confusion_matrix.png")
    plt.close()


    return {"message": "Model retrained!"}


@app.get("/confusion_matrix")
def get_confusion_matrix():
    file_path = Path("data/metrics/confusion_matrix.png")
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Image not found")
