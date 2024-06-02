from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from kedro.io import DataCatalog


def get_best_fitting(X, y, params, scoring="roc_auc"):
    rf_model = RandomForestClassifier()
    grid = GridSearchCV(estimator=rf_model, param_grid=params, n_jobs=-1, cv=3, verbose=1, scoring=scoring)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def find_best_model(X, y, param_grids):
    best_models = {}
    for key, params in param_grids.items():
        best_model, best_params, best_score = get_best_fitting(X, y, params)
        best_models[key] = {
            "best_params": best_params,
            "best_score": best_score
        }
    return best_models


def find_best_model(X, y, param_grids):
    best_model = None
    best_params = None
    best_score = float('-inf')
    
    for key, params in param_grids.items():
        _, params, score = get_best_fitting(X, y, params)
        if score > best_score:
            best_model = key
            best_params = params
            best_score = score
    
    return best_model, best_params, best_score

