from .core import DatasetInfo, DatasetPack, DATASET_DIR, register_dataset_factory
from .utils import split_dataset
import torchvision


@register_dataset_factory(name="mnist")
def dataset_factory(val_ratio: float = 0.2) -> DatasetPack:
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root=DATASET_DIR, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=DATASET_DIR, train=False, download=True, transform=transform
    )
    train_dataset, val_dataset = split_dataset(train_dataset, val_ratio)
    infos = DatasetInfo(name="mnist", num_classes=10, image_size=(1, 28, 28))
    return train_dataset, val_dataset, test_dataset, infos
