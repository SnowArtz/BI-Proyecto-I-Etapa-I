import pandas as pd

from sklearn.model_selection import train_test_split

class DataSplitter():
    def __init__(self, raw_data_path):
        if raw_data_path.split(".")[-1].strip() == "xlsx":
            self.df = pd.read_excel(raw_data_path)
        elif raw_data_path.split(".")[-1].strip() == "csv":
            self.df = pd.read_csv(raw_data_path)
        else:
            raise Exception("File format not supported")
    
    def split_data_train_test(self, test_size=0.2):
        X = self.df["Textos_espanol"]
        y = self.df["sdg"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    def write_data(self, path="data/interim/"):
        self.X_train.to_csv(path+"X_train_raw.csv", index=False)
        self.X_test.to_csv(path+"X_test_raw.csv", index=False)
        self.y_train.to_csv(path+"y_train_raw.csv", index=False)
        self.y_test.to_csv(path+"y_test_raw.csv", index=False)