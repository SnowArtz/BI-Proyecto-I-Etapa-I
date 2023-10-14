import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class BOW():
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.X_bow_train = None
        self.X_bow_test = None

    def fit_transform(self, X_train, X_test):
        self.X_bow_train = self.vectorizer.fit_transform(X_train)
        self.X_bow_test = self.vectorizer.transform(X_test)

    def write_data(self):
        pd.DataFrame(self.X_bow_train.toarray(), columns=self.vectorizer.get_feature_names_out()).to_csv("data/features/BOW/X_bow_train.csv", index=False)
        pd.DataFrame(self.X_bow_test.toarray(), columns=self.vectorizer.get_feature_names_out()).to_csv("data/features/BOW/X_bow_test.csv", index=False)

    def transform(self, X):
        return self.vectorizer.transform(X)
