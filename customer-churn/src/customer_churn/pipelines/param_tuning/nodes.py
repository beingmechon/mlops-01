from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from kedro.io import DataCatalog

from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data: pd.DataFrame):
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.to_frame().reset_index()
    y_test = y_test.to_frame().reset_index()

    return X_train, X_test, y_train, y_test


def get_best_fitting(X, y, params, scoring="f1"): # roc_auc
    # y.squeeze(axis=0)
    print(type(X), type(y))
    rf_model = RandomForestClassifier()
    grid = GridSearchCV(estimator=rf_model, param_grid=params, n_jobs=-1, cv=3, verbose=1, scoring=scoring)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# def find_best_model(X, y, param_grids):
#     best_models = {}
#     for key, params in param_grids.items():
#         best_model, best_params, best_score = get_best_fitting(X, y, params)
#         best_models[key] = {
#             "best_params": best_params,
#             "best_score": best_score
#         }
#     return best_models


def find_best_model(X, y, param_grids):
    print(param_grids)
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

