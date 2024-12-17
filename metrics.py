import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset
from typing import List, Tuple
from matplotlib import pyplot as plt


def calculate_roc(y_true, y_pred, pos_label: int):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def calculate_roc_over_classes(
    dataset: Dataset, shapley_values_per_classes: List[torch.Tensor]
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    num_classes = len(shapley_values_per_classes)
    y_true = np.zeros(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        y_true[i] = int(dataset[i][1])
    result = []
    for cls in range(num_classes):
        y_pred = shapley_values_per_classes[cls].detach().cpu().numpy()
        fpr, tpr, roc_auc = calculate_roc(y_true, y_pred, pos_label=cls)
        result.append((fpr, tpr, roc_auc))
    return result


def plot_roc_curve_over_classes(roc_results: List[Tuple[np.ndarray, np.ndarray, float]]):
    num_classes = len(roc_results)
    max_fig_per_row = 3
    num_row = (num_classes + max_fig_per_row - 1) // max_fig_per_row
    num_col = (num_classes + num_row - 1) // num_row
    fig, axes = plt.subplots(num_row, num_col, figsize=(5 * num_col, 5 * num_row))
    for i, (fpr, tpr, roc_auc) in enumerate(roc_results):
        row = i // num_col
        col = i % num_col
        ax = axes[row][col] if row > 0 else axes[col]
        ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC curve for class {i}")
        ax.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()
