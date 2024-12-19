import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Optional
from os import PathLike
from matplotlib import pyplot as plt


# this function maybe only run on cpu, so it's can be slow
# but I don't have any idea to work on gpu
# so I just turn input to numpy and calculate it
def calculate_roc_for_single_class(y_true: np.ndarray, y_pred: np.ndarray, *, pos_label: int):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


# if shapley_values is list of tensor, it's biased shapley_values for validation set that has the same label
# otherwise, it's unbiased shapley_values for whole validation set
def calculate_roc_over_classes(
    dataset_for_train: Dataset,
    shapley_values: Union[torch.Tensor, List[torch.Tensor]],
    *,
    num_classes: int,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    # get all label of dataset for train
    all_labels = np.array(
        [dataset_for_train[i][1] for i in range(len(dataset_for_train))], dtype=np.int64
    )
    result = []
    for cls in range(num_classes):
        current_shapley_values = (
            (shapley_values[cls] if isinstance(shapley_values, list) else shapley_values)
            .detach()
            .cpu()
            .numpy()
        )
        fpr, tpr, roc_auc = calculate_roc_for_single_class(
            all_labels, current_shapley_values, pos_label=cls
        )
        result.append((fpr, tpr, roc_auc))
    return result


def plot_roc_curve_over_classes(
    roc_results: List[Tuple[np.ndarray, np.ndarray, float]],
    *,
    save_path: Optional[Union[str, PathLike]] = None,
):
    num_classes = len(roc_results)
    max_fig_per_row = 3
    num_row = (num_classes + max_fig_per_row - 1) // max_fig_per_row
    num_col = (num_classes + num_row - 1) // num_row
    fig, axes = plt.subplots(num_row, num_col, figsize=(5 * num_col, 5 * num_row))
    for i, (fpr, tpr, roc_auc) in enumerate(roc_results):
        row = i // num_col
        col = i % num_col
        ax = axes[row, col] if num_row > 1 else axes[col]
        ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC curve for class {i}")
        ax.legend(loc="lower right")
    if save_path is not None:
        fig.savefig(save_path)
        print("save figure to", save_path)
    plt.close(fig)
