import pandas as pd
import numpy as np

from train_model import Model
from src.features.BOW import BOW
from src.features.TFIDF import TFIDF

#from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


X_train = pd.read_csv("data/processed/X_train_processed.csv").text
X_test = pd.read_csv("data/processed/X_test_processed.csv").text
y_train = pd.read_csv("data/interim/y_train_raw.csv").sdg
y_test = pd.read_csv("data/interim/y_test_raw.csv").sdg

BOW = BOW()
BOW.fit_transform(X_train, X_test)

TFIDF = TFIDF()
TFIDF.fit_transform(X_train, X_test)

print("\nLogistic Regression BOW: ")
LogisticRegression_BOW = Model(LogisticRegression(max_iter=1000, random_state=42), BOW.X_bow_train, y_train)
LogisticRegression_BOW.model_grid_search(param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'class_weight': [None, 'balanced'],
    'fit_intercept': [True, False],
    'multi_class': ['auto', 'ovr', 'multinomial'],
    'warm_start': [True, False],
    'l1_ratio': [None, 0, 0.25,0.5,0.75,1]
})


print("\nLogistic Regression TFIDF: ")
LogisticRegression_TFIDF = Model(LogisticRegression(max_iter=1000, random_state=42), TFIDF.X_tfidf_train, y_train)
LogisticRegression_TFIDF.model_grid_search(param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'fit_intercept': [True, False],
    'multi_class': ['auto', 'ovr', 'multinomial'],
    'warm_start': [True, False],
    'l1_ratio': [None, 0, 0.25,0.5,0.75,1]
})

print("\nRandom Forest BOW: ")
RandomForest_BOW = Model(RandomForestClassifier(random_state=42), BOW.X_bow_train, y_train)
RandomForest_BOW.model_grid_search(param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
})

print("\nRandom Forest TFIDF: ")
RandomForest_TFIDF = Model(RandomForestClassifier(random_state=42), TFIDF.X_tfidf_train, y_train)
RandomForest_TFIDF.model_cross_validate(param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
})

print("\nSVM BOW: ")
SVM_BOW = Model(SVC(random_state=42), BOW.X_bow_train, y_train)
SVM_BOW.model_grid_search({
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10, 100],
    'coef0': [0, 1, 2, 3, 4],
    'shrinking': [True, False],
    'tol': [1e-3, 1e-4, 1e-5],
})

print("\nSVM TFIDF: ")
SVM_TFIDF = Model(SVC(random_state=42), TFIDF.X_tfidf_train, y_train)
SVM_TFIDF.model_grid_search({
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10, 100],
    'coef0': [0, 1, 2, 3, 4],
    'shrinking': [True, False],
    'tol': [1e-3, 1e-4, 1e-5],
})