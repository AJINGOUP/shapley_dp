import numpy as np
from typing import Tuple, List, Mapping, Callable, Optional
from pathlib import Path
from torch.utils.data import Dataset, Subset
from dataclasses import dataclass

DATASET_DIR = Path(__file__).parent.parent / Path("data")


@dataclass
class DatasetInfo:
    name: str
    num_classes: int
    image_size: Optional[Tuple[int, int, int]] = None
    vector_size: Optional[int] = None


DatasetPack = Tuple[Dataset, Dataset, Dataset, DatasetInfo]


class DatasetFactory:
    dataset_factory: Mapping[str, Callable[..., DatasetPack]] = {}

    @classmethod
    def get_dataset_by_name(cls, name: str, *args, **kwargs) -> DatasetPack:
        if name not in cls.dataset_factory:
            raise ValueError(f"not support for dataset: {name}")
        return cls.dataset_factory[name](*args, **kwargs)

    @classmethod
    def get_client_datasets(cls, dataset: Dataset, sample_per_client: int = 1) -> List[Dataset]:
        n = len(dataset)
        permuted_indices = np.random.permutation(n)
        client_datasets = []
        for i in range(0, n, sample_per_client):
            indices = permuted_indices[i : i + sample_per_client]
            client_datasets.append(Subset(dataset, indices))
        return client_datasets

    @classmethod
    def group_dataset_by_label(cls, dataset: Dataset, num_classes: int) -> List[Dataset]:
        indices = [[] for _ in range(num_classes)]
        for i, (data, target) in enumerate(dataset):
            indices[target].append(i)
        return [Subset(dataset, i) for i in indices]

    @classmethod
    def get_warmup_dataset(cls, dataset: Dataset, target_label: int, *, exclusive: bool) -> Dataset:
        indices = []
        for i, (data, target) in enumerate(dataset):
            if exclusive:
                if target != target_label:
                    indices.append(i)
            else:
                if target == target_label:
                    indices.append(i)
        return Subset(dataset, indices)


def register_dataset_factory(
    name: str,
) -> Callable[[Callable[..., DatasetPack]], Callable[..., DatasetPack]]:
    """
    A decorator to register a dataset factory function under a given name.

    Args:
        name (str): The name to register the dataset factory function under.

    Returns:
        Callable[[Callable[..., DatasetPack]], Callable[..., DatasetPack]]:
        A decorator that registers the given dataset factory function.
    """

    def decorator(func: Callable[..., DatasetPack]) -> Callable[..., DatasetPack]:
        DatasetFactory.dataset_factory[name] = func
        return func

    return decorator
