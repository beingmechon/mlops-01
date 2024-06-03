import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve)

def evaluate_model(model, X_test, y_test, catalog=None):
    """
    Evaluates a trained model on a test set.

    Parameters
    ----------
    """
    y_test = y_test.values.ravel() if isinstance(y_test, pd.DataFrame) else y_test.ravel()

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    classification_rep = classification_report(y_test, predictions, output_dict=True)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    model_roc_auc = roc_auc_score(y_test, probabilities)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)

    # Print metrics
    print("\nClassification report:\n", classification_rep)
    # print("Accuracy Score:", accuracy)
    # print("Area under curve:", model_roc_auc)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1-score:", f1)
    # print("Matthews correlation coefficient:", mcc, "\n")

    # Create figure for plotting
    fig = plt.figure(figsize=(12, 12))

    # Plot confusion matrix
    plt.subplot(221)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    plt.subplot(222)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label="AUC: %.3f" % model_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save or show plots
    # catalog.save("reporting_image", fig)
    plt.close()

    # Prepare results dictionary
    results = {
        # "predictions": predictions,
        # "probabilities": probabilities,
        "conf_matrix": conf_matrix.tolist(),
        "roc_auc": model_roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "classification_report": classification_rep,
        "accuracy": accuracy
    }

    # Save results to catalog
    # catalog.save("evaluation_metrics", results)

    return results, fig
