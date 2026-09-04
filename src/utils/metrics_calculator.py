"""Metric calculations — F1 (weighted), precision, recall, mAP, MSE."""
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
import numpy as np

def compute_metrics(y_true, y_pred):
    return {
        "f1_weighted": f1_score(y_true, y_pred, average="weighted") * 100,
        "precision": precision_score(y_true, y_pred, average="weighted") * 100,
        "recall": recall_score(y_true, y_pred, average="weighted") * 100,
        "mse": mean_squared_error(y_true, y_pred),
    }
