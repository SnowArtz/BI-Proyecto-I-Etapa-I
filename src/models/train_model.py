
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, f1_score

class Model():

    def __init__(self, model, X_train, y_train):
        self.scorer = make_scorer(f1_score, average='weighted')
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def model_cross_validate(self, cv=5):
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv, scoring=self.scorer)
        mean_accuracy = cv_scores.mean()
        print("Resultados de Validación Cruzada (K=5):")
        print("F1 Promedio:", mean_accuracy)
        print("F1 por Pliegue:", cv_scores)

    def model_grid_search(self, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring=self.scorer, verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        print("Mejores hiperparámetros:", grid_search.best_params_)
        print("Mejor F1:", grid_search.best_score_)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)