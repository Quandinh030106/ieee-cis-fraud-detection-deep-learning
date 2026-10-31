import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    save_dir: Optional[Path] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Comprehensive evaluation for fraud detection:
    - ROC-AUC
    - PR-AUC (Average Precision)
    - Confusion Matrix (TP, FP, FN, TN)
    - Precision, Recall, F1 at default and optimal threshold
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # 1. Ranking metrics (threshold-independent)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    # 2. Optimal threshold finding based on F1-score
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid zero division
    f1_scores = np.where((precisions + recalls) > 0, 2 * (precisions * recalls) / (precisions + recalls), 0)
    best_idx = np.argmax(f1_scores)
    optimal_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])

    # 3. Binary metrics at default threshold
    y_pred_default = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_default).ravel()

    precision_default = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall_default = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_default = float(2 * precision_default * recall_default / (precision_default + recall_default)) if (precision_default + recall_default) > 0 else 0.0

    # 4. Binary metrics at optimal threshold
    y_pred_opt = (y_prob >= optimal_thresh).astype(int)
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_true, y_pred_opt).ravel()
    prec_o = float(tp_o / (tp_o + fp_o)) if (tp_o + fp_o) > 0 else 0.0
    rec_o = float(tp_o / (tp_o + fn_o)) if (tp_o + fn_o) > 0 else 0.0

    metrics = {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'default_threshold': threshold,
        'default_metrics': {
            'precision': precision_default,
            'recall': recall_default,
            'f1_score': f1_default,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        },
        'optimal_threshold': optimal_thresh,
        'optimal_metrics': {
            'precision': prec_o,
            'recall': rec_o,
            'f1_score': best_f1,
            'tp': int(tp_o),
            'fp': int(fp_o),
            'fn': int(fn_o),
            'tn': int(tn_o)
        }
    }

    # Save metrics and curves if save_dir provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        json_path = save_dir / f"{model_name.lower().replace(' ', '_')}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Plot ROC and PR curves
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC Curve
        axes[0].plot(fpr, tpr, color='#1f77b4', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
        axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Guess')
        axes[0].set_xlabel('False Positive Rate', fontweight='bold')
        axes[0].set_ylabel('True Positive Rate (Recall)', fontweight='bold')
        axes[0].set_title(f'Receiver Operating Characteristic (ROC) - {model_name}', fontweight='bold')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)

        # Precision-Recall Curve
        baseline_pr = float(y_true.mean())
        axes[1].plot(recalls, precisions, color='#ff7f0e', lw=2, label=f'{model_name} (PR-AUC = {pr_auc:.4f})')
        axes[1].axhline(y=baseline_pr, color='gray', linestyle='--', lw=1.5, label=f'Base Fraud Rate ({baseline_pr*100:.2f}%)')
        axes[1].set_xlabel('Recall', fontweight='bold')
        axes[1].set_ylabel('Precision', fontweight='bold')
        axes[1].set_title(f'Precision-Recall Curve - {model_name}', fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        curve_path = save_dir / f"{model_name.lower().replace(' ', '_')}_roc_pr_curve.png"
        plt.savefig(curve_path, dpi=300)
        plt.close()

    return metrics
