from joblib import dump

from src.data.preprocessing import DataProcessor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

TFIDFLogisticRegressionPipeline = Pipeline([
    ('preprocessor', FunctionTransformer(DataProcessor("").process_data, validate=False)),
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, C=100, class_weight=None, fit_intercept=True, l1_ratio=None, multi_class="auto", penalty="l2", solver="liblinear", warm_start=True))
])

TFIDFCSupportVectorPipeline = Pipeline([
    ('preprocessor', FunctionTransformer(DataProcessor("").process_data, validate=False)),
    ('vectorizer', TfidfVectorizer()),
    ('classifier', SVC(random_state=42, C=1, coef0=2, gamma="scale", kernel="poly", shrinking=True, tol=0.001))
])

BOWRandomForestPipeline = Pipeline([
    ('preprocessor', FunctionTransformer(DataProcessor("").process_data, validate=False)),
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier(random_state=42, bootstrap=False, criterion="gini", max_depth=None, max_features="log2", min_samples_leaf=1, min_samples_split=5, n_estimators=200))
])

dump(TFIDFLogisticRegressionPipeline, 'src/models/pipelines/TFIDFLogisticRegression/TFIDFLogisticRegression.pkl')
dump(TFIDFCSupportVectorPipeline, 'src/models/pipelines/TFIDFCSupportVector/TFIDFCSupportVector.pkl')
dump(BOWRandomForestPipeline, 'src/models/pipelines/BOWRandomForest/BOWRandomForest.pkl')
