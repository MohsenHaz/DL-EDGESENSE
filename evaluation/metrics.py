"""All evaluation metrics — F1 (weighted), precision, recall, mAP, MSE."""
import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, mean_squared_error)
from scipy import stats

def evaluate(y_true, y_pred, y_prob=None):
    results = {
        "f1_weighted": f1_score(y_true, y_pred, average="weighted") * 100,
        "precision":   precision_score(y_true, y_pred, average="weighted") * 100,
        "recall":      recall_score(y_true, y_pred, average="weighted") * 100,
    }
    if y_prob is not None:
        results["mAP"] = average_precision_score(
            np.eye(y_prob.shape[1])[y_true], y_prob, average="weighted") * 100
    return results
