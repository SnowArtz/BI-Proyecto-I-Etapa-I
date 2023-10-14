import pandas as pd
import time
from train_model import Model
from src.features.BOW import BOW
from src.features.TFIDF import TFIDF

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


X_train = pd.read_csv("data/processed/X_train_processed.csv").text
X_test = pd.read_csv("data/processed/X_test_processed.csv").text
y_train = pd.read_csv("data/interim/y_train_raw.csv").sdg
y_test = pd.read_csv("data/interim/y_test_raw.csv").sdg

BOW = BOW()
BOW.fit_transform(X_train, X_test)

TFIDF = TFIDF()
TFIDF.fit_transform(X_train, X_test)


print("\nRandom Forest TFIDF: ")
RandomForest_TFIDF = Model(RandomForestClassifier(random_state=42), TFIDF.X_tfidf_train, y_train)
start_time = time.time()
RandomForest_TFIDF.model_grid_search(param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4],
    #'bootstrap': [True, False],
    #'max_features': ['auto', 'sqrt', 'log2'],
    #'criterion': ['gini', 'entropy'],
    #'class_weight': [None, 'balanced', 'balanced_subsample']
})

# Registrar el tiempo de finalización y calcular la duración total
end_time = time.time()
duration = end_time - start_time
print(f"Tiempo de ejecución del programa: {duration:.2f} segundos.")