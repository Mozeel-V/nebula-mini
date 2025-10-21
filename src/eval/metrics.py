import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

def tpr_at_fpr(y_true, y_score, fpr_target=1e-3):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    mask = fpr <= fpr_target
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))

def compute_metrics(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    y_pred = (y_score >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    return {
        "auc": float(auc),
        "f1": float(f1),
        "tpr_at_1e-3": tpr_at_fpr(y_true, y_score, 1e-3),
        "tpr_at_1e-4": tpr_at_fpr(y_true, y_score, 1e-4)
    }
