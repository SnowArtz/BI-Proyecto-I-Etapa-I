import pandas as pd

from model_representation import Model
from src.features.BOW import BOW
from src.features.TFIDF import TFIDF

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier

X_train = pd.read_csv("data/processed/X_train_processed.csv").text
X_test = pd.read_csv("data/processed/X_test_processed.csv").text
y_train = pd.read_csv("data/interim/y_train_raw.csv").sdg
y_test = pd.read_csv("data/interim/y_test_raw.csv").sdg

BOW = BOW()
BOW.fit_transform(X_train, X_test)

TFIDF = TFIDF()
TFIDF.fit_transform(X_train, X_test)

print("\nLogistic Regression:")
print("Logistic Regression BOW: ")
LogisticRegression_BOW = Model(LogisticRegression(max_iter=1000, random_state=42, C=100, class_weight=None, fit_intercept=True, l1_ratio=None, multi_class="auto", penalty="l2", solver="liblinear", warm_start=True), BOW.X_bow_train, y_train)
LogisticRegression_BOW.fit_predict(BOW.X_bow_train, y_train, BOW.X_bow_test, y_test)

print("Logistic Regression TFIDF: ")
LogisticRegression_TFIDF = Model(LogisticRegression(max_iter=1000, random_state=42, C=100, fit_intercept=True, l1_ratio=None, multi_class="auto", penalty="l2", solver="newton-cg", warm_start=True), TFIDF.X_tfidf_train, y_train)
LogisticRegression_TFIDF.fit_predict(TFIDF.X_tfidf_train, y_train, TFIDF.X_tfidf_test, y_test)

print("\nRandom Forest Classifier:")
print("Random Forest Classifier BOW: ")
RandomForest_BOW = Model(RandomForestClassifier(random_state=42, bootstrap=False, criterion="gini", max_depth=None, max_features="log2", min_samples_leaf=1, min_samples_split=5, n_estimators=200), BOW.X_bow_train, y_train)
RandomForest_BOW.fit_predict(BOW.X_bow_train, y_train, BOW.X_bow_test, y_test)

print("Random Forest Classifier TFIDF: ")
RandomForest_TFIDF = Model(RandomForestClassifier(random_state=42, bootstrap=False, criterion="gini", max_depth=None, max_features="sqrt", min_samples_leaf=1, min_samples_split=2, n_estimators=200), TFIDF.X_tfidf_train, y_train)
RandomForest_TFIDF.fit_predict(TFIDF.X_tfidf_train, y_train, TFIDF.X_tfidf_test, y_test)

print("\nSupport Vector Machine:")
print("Support Vector Machine BOW: ")
SupportVectorMachine_BOW = Model(SVC(random_state=42, C=0.1, coef0=2, gamma=0.01, kernel="poly", shrinking=True, tol=0.001), BOW.X_bow_train, y_train)
SupportVectorMachine_BOW.fit_predict(BOW.X_bow_train, y_train, BOW.X_bow_test, y_test)

print("Support Vector Machine TFIDF: ")
SupportVectorMachine_TFIDF = Model(SVC(random_state=42, C=1, coef0=2, gamma="scale", kernel="poly", shrinking=True, tol=0.001), TFIDF.X_tfidf_train, y_train)
SupportVectorMachine_TFIDF.fit_predict(TFIDF.X_tfidf_train, y_train, TFIDF.X_tfidf_test, y_test)

print("\nMultinomial Naive Bayes:")
print("Multinomial Naive Bayes BOW: ")
MultinomialNaiveBayes_BOW = Model(MultinomialNB(alpha=1, class_prior=None, fit_prior=True, force_alpha=True), BOW.X_bow_train, y_train)
MultinomialNaiveBayes_BOW.fit_predict(BOW.X_bow_train, y_train, BOW.X_bow_test, y_test)

print("Multinomial Naive Bayes TFIDF: ")
MultinomialNaiveBayes_TFIDF = Model(MultinomialNB(alpha=1, class_prior=None, fit_prior=True, force_alpha=True), TFIDF.X_tfidf_train, y_train)
MultinomialNaiveBayes_TFIDF.fit_predict(TFIDF.X_tfidf_train, y_train, TFIDF.X_tfidf_test, y_test)



print("Ridge Classifier TFIDF: ")
RidgeClassifier_TFIDF = Model(RidgeClassifier(random_state=42, alpha= 1, class_weight= 'balanced', fit_intercept= False, positive= True, solver= 'auto', tol=  1e-06), TFIDF.X_tfidf_train, y_train)
RidgeClassifier_TFIDF.fit_predict(TFIDF.X_tfidf_train, y_train, TFIDF.X_tfidf_test, y_test)



