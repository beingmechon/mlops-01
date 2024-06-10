# from ..model import CRNN, weights_init
from ..utils import compute_loss, decode_predictions, correct_predictions
from json import loads, dumps

import torch
import mlflow
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(test_loader, trained_model, idx_to_char):
    crnn = trained_model
    results = []

    with torch.no_grad():
        for image_batch, text_batch in tqdm(test_loader, desc="Evaluation"):
            text_batch_logits = crnn(image_batch.to(device))
            text_batch_pred = decode_predictions(text_batch_logits.cpu(), idx_to_char)
            results.append(pd.DataFrame({'actual': text_batch, 'prediction': text_batch_pred}))

    results = pd.concat(results, ignore_index=True)
    results['prediction_corrected'] = results['prediction'].apply(correct_predictions)

    test_accuracy = accuracy_score(results['actual'], results['prediction_corrected'])
    test_precision = precision_score(results['actual'], results['prediction_corrected'], average='weighted', zero_division=1)
    test_recall = recall_score(results['actual'], results['prediction_corrected'], average='weighted', zero_division=1)
    test_f1 = f1_score(results['actual'], results['prediction_corrected'], average='weighted')
    print(f"test_accuracy: {test_accuracy:.5f}")
    print(f"test_precision: {test_precision:.5f}")
    print(f"test_recall: {test_recall:.5f}")
    print(f"test_f1: {test_f1:.5f}")

    test_metrics = {"test_accuracy": test_accuracy,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1": test_f1}

    mlflow.log_metric("Test Accuracy", test_accuracy)
    mlflow.log_metric("Test Precision", test_precision)
    mlflow.log_metric("Test Recall", test_recall)
    mlflow.log_metric("Test F1-Score", test_f1)

    result_json = results.to_json(orient="records")
    parsed = loads(result_json)
    result = dumps(parsed, indent=4)
    with open("results.json", "w") as f:
        f.write(result)

    mlflow.log_artifact("results.json")

    return result, test_metrics