"""
evaluation.py
-------------
Metric computation, model comparison table, and confusion matrix plotting.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless plotting, safe for scripts/servers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import accuracy_score


def compute_metrics(y_true, y_pred) -> dict:
    """Compute accuracy, macro precision/recall/F1. Macro averaging
    weights every class equally regardless of how many examples it has,
    which matters if the dataset is imbalanced (a model that only ever
    predicts the majority class would score high on plain accuracy but
    poorly on macro F1).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """`results` maps model_name -> metrics dict (as returned by
    compute_metrics). Returns a tidy DataFrame ready to print to the
    terminal, sorted by macro F1 (best first).
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision (macro)": round(metrics["precision_macro"], 4),
                "Recall (macro)": round(metrics["recall_macro"], 4),
                "F1 (macro)": round(metrics["f1_macro"], 4),
            }
        )
    df = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    return df


def select_best_model(validation_results: dict) -> str:
    """Pick the model name with the highest macro F1 on the VALIDATION
    set. This is the only place model selection happens -- the test set
    is never used to choose between models, only to report final numbers.
    """
    return max(validation_results, key=lambda name: validation_results[name]["f1_macro"])


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Save a confusion matrix plot to output_path (a .png file path)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return cm
