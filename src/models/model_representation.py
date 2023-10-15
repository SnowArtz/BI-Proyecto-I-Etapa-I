
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

class Model():

    def __init__(self, model, X_train, y_train, output_file="output.txt"):
        self.scorer = make_scorer(f1_score, average='weighted')
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.output_file = output_file
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test, output_file):
        predictions = self.model.predict(X_test)
        np.savetxt(output_file, predictions, delimiter=',', fmt='%s')

    def model_cross_validate(self, cv=5):
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv, scoring=self.scorer)
        mean_f1 = cv_scores.mean()
        print("Resultados de Validación Cruzada (K=5):")
        print("F1 Promedio: " + str(mean_f1))
        print("F1 por Pliegue: " + str(cv_scores))
        self.write_to_file("Resultados de Validación Cruzada (K=5):")
        self.write_to_file("F1 Promedio: " + str(mean_f1))
        self.write_to_file("F1 por Pliegue: " + str(cv_scores))

    def model_grid_search(self, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring=self.scorer, verbose=0, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        print("Mejores hiperparámetros: " + str(grid_search.best_params_))
        print("Mejor F1: " + str(grid_search.best_score_))
        self.write_to_file("Mejores hiperparámetros: " + str(grid_search.best_params_))
        self.write_to_file("Mejor F1: " + str(grid_search.best_score_))

    def fit_predict(self, X_train, y_train, X_test, y_test, print_differences=False, plot_confusion_matrix=False):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        if print_differences:
            mismatch_indices = np.where(y_test != y_pred)[0]
            print("Indices and records that are mispredicted:")
            for idx in mismatch_indices:
                print(f"Index: {idx}, Actual: {y_test[idx]}, Predicted: {y_pred[idx]}")
        if plot_confusion_matrix:
            cm = confusion_matrix(y_test, y_pred)
            display_labels = np.unique(y_test)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            disp.plot(cmap=plt.cm.Blues)
            plt.show()
        return classification_report(y_test, y_pred, digits=4)
    
    def write_to_file(self, content):
        with open(self.output_file, "a") as f:
            f.write(content + "\n")