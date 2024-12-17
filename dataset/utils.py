from typing import Tuple
from torch.utils.data import Dataset, Subset
import numpy as np


def split_dataset(dataset: Dataset, val_ratio=0.2) -> Tuple[Dataset, Dataset]:
    # split dataset with balance class
    targets = np.array([target for _, target in dataset])
    train_indices = []
    val_indices = []

    for class_idx in np.unique(targets):
        class_indices = np.where(targets == class_idx)[0]
        np.random.shuffle(class_indices)
        val_count = int(len(class_indices) * val_ratio)
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset
