from .core import DatasetFactory, DatasetInfo, register_dataset_factory
from ._mnist import dataset_factory as _mnist_dataset_factory  # noqa: F401
from ._adult import dataset_factory as _adult_dataset_factory  # noqa: F401
from ._medmnist import dataset_factory as _medmnist_dataset_factory  # noqa: F401

__all__ = ["DatasetFactory", "DatasetInfo", "register_dataset_factory"]
