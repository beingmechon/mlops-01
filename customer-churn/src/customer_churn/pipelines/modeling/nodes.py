import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from kedro.io import DataCatalog

def split_data(data: pd.DataFrame):
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.to_frame().reset_index()
    y_test = y_test.to_frame().reset_index()

    return X_train, X_test, y_train, y_test

def train_model(model, training_x, training_y, testing_x, testing_y, cols, cf='coefficients', catalog: DataCatalog = None):
    # training_y.squeeze(axis=0)
    model.fit(training_x, training_y)
    predictions = model.predict(testing_x)
    probabilities = model.predict_proba(testing_x)[:, 1]
    
    if cf == "coefficients":
        coefficients = pd.DataFrame(model.coef_.ravel())
    elif cf == "features":
        coefficients = pd.DataFrame(model.feature_importances_)
        
    column_df = pd.DataFrame(cols)
    coef_sumry = pd.merge(coefficients, column_df, left_index=True, right_index=True, how="left")
    coef_sumry.columns = ["coefficients", "features"]
    coef_sumry = coef_sumry.sort_values(by="coefficients", ascending=False)

    # Print and store metrics
    classification_rep = classification_report(testing_y, predictions)
    accuracy = accuracy_score(testing_y, predictions)
    conf_matrix = confusion_matrix(testing_y, predictions)
    model_roc_auc = roc_auc_score(testing_y, probabilities)
    
    print("\n Classification report : \n", classification_rep)
    print("Accuracy   Score : ", accuracy)
    print("Area under curve : ", model_roc_auc, "\n")
    
    # Confusion matrix plot
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    sns.heatmap(conf_matrix, fmt="d", annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    
    # ROC AUC plot
    fpr, tpr, thresholds = roc_curve(testing_y, probabilities)
    plt.subplot(222)
    plt.plot(fpr, tpr, color='darkorange', lw=1, label="AUC : %.3f" % model_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Feature importances plot
    plt.subplot(212)
    sns.barplot(x=coef_sumry["features"], y=coef_sumry["coefficients"])
    plt.title('Feature Importances')
    plt.xticks(rotation="vertical")
    
    # Save plots using Kedro's DataCatalog
    if catalog:
        fig_path = catalog.get_save_path("churn_prediction_plots")
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()

    return {
        "model": model,
        "predictions": predictions,
        "probabilities": probabilities,
        "coef_sumry": coef_sumry,
        "conf_matrix": conf_matrix,
        "roc_auc": model_roc_auc,
        "classification_report": classification_rep,
        "accuracy": accuracy
    }

def evaluate_model(metrics: dict):
    for key, value in metrics.items():
        if isinstance(value, (float, str)):
            print(f"{key}: {value}")
        elif isinstance(value, pd.DataFrame):
            print(f"\n{key}:\n", value)
        else:
            print(f"\n{key}:\n", value)
    
    return metrics
