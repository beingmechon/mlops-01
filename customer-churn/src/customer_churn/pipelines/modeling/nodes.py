import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from kedro.io import DataCatalog


def split_data(data: pd.DataFrame):
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.to_frame()
    y_test = y_test.to_frame()

    return X_train, X_test, y_train, y_test


def train_model(training_x, training_y, testing_x, testing_y, param_grid=None, model=None, cols=None, cf="features", catalog: DataCatalog = None):
    training_y = training_y.values.ravel() if isinstance(training_y, pd.DataFrame) else training_y.ravel()
    testing_y = testing_y.values.ravel() if isinstance(testing_y, pd.DataFrame) else testing_y.ravel()

    if param_grid:
        rf_model = RandomForestClassifier()
        grid = GridSearchCV(estimator=rf_model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=1, scoring="roc_auc")
        grid.fit(training_x, training_y)
        best_model = grid.best_estimator_
    else:
        best_model = model

    best_model.fit(training_x, training_y)
    
    predictions = best_model.predict(testing_x)
    probabilities = best_model.predict_proba(testing_x)[:, 1]

    if cf == "coefficients":
        try:
            coefficients = pd.DataFrame(best_model.coef_.ravel())
        except AttributeError:
            raise ValueError("The model does not have coefficients. Set `cf` to 'features'.")
    elif cf == "features":
        coefficients = pd.DataFrame(best_model.feature_importances_, columns=["coefficients"])
    
    cols = training_x.columns
    column_df = pd.DataFrame(cols, columns=["features"])
    coef_sumry = pd.concat([coefficients, column_df], axis=1)
    coef_sumry = coef_sumry.sort_values(by="coefficients", ascending=False)

    classification_rep = classification_report(testing_y, predictions)
    accuracy = accuracy_score(testing_y, predictions)
    conf_matrix = confusion_matrix(testing_y, predictions)
    model_roc_auc = roc_auc_score(testing_y, probabilities)
    
    # Print metrics
    print("\nClassification report:\n", classification_rep)
    print("Accuracy Score:", accuracy)
    print("Area under curve:", model_roc_auc, "\n")
    
    # Plot confusion matrix
    fig = plt.figure(figsize=(12, 12))
    plt.subplot(221)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(testing_y, probabilities)
    plt.subplot(222)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label="AUC: %.3f" % model_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Plot feature importances
    plt.subplot(212)
    sns.barplot(x="features", y="coefficients", data=coef_sumry)
    plt.title('Feature Importances')
    plt.xticks(rotation=90)

    # Return metrics and model
    return best_model, \
    {
        "model": best_model,
        "predictions": predictions,
        "probabilities": probabilities,
        "coef_sumry": coef_sumry if cf == "features" else None,
        "conf_matrix": conf_matrix,
        "roc_auc": model_roc_auc,
        "classification_report": classification_rep,
        "accuracy": accuracy
    }, fig


# def evaluate_model(metrics: dict):
#     for key, value in metrics.items():
#         if isinstance(value, (float, str)):
#             print(f"{key}: {value}")
#         elif isinstance(value, pd.DataFrame):
#             print(f"\n{key}:\n", value)
#         else:
#             print(f"\n{key}:\n", value)
    
#     return metrics
