from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np

def evaluate(y_true, y_pred, y_prob=None):
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    res = {'report': report, 'confusion_matrix': cm.tolist(), 'roc_auc': auc}
    return res
