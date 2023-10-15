import pandas as pd
import numpy as np
from joblib import load

X_train = pd.read_excel("data/raw/cat_6716.xlsx").Textos_espanol
y_train = pd.read_excel("data/raw/cat_6716.xlsx").sdg
X_test = pd.read_excel("data/raw/SinEtiquetatest_cat_6716.xlsx").Textos_espanol

loaded_pipeline = load('src/models/pipelines/TFIDFLogisticRegression/TFIDFLogisticRegression.pkl')
loaded_pipeline.fit(X_train, y_train)
np.savetxt("data/predicted/SinEtiquetatest_cat_6716_clasificado.csv", loaded_pipeline.predict(X_test), delimiter=',', fmt='%s')