import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF():
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.X_tfidf_train = None
        self.X_tfidf_test = None

    def fit_transform(self, X_train, X_test):
        self.X_tfidf_train = self.vectorizer.fit_transform(X_train)
        self.X_tfidf_test = self.vectorizer.transform(X_test)

    def write_data(self):
        pd.DataFrame(self.X_tfidf_train.toarray(), columns=self.vectorizer.get_feature_names_out()).to_csv("data/features/TFIDF/X_tfidf_train.csv", index=False)
        pd.DataFrame(self.X_tfidf_test.toarray(), columns=self.vectorizer.get_feature_names_out()).to_csv("data/features/TFIDF/X_tfidf_test.csv", index=False)

    def transform(self, X):
        return self.vectorizer.transform(X)
    

