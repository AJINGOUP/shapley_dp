import medmnist
from medmnist import INFO
from torchvision import transforms

from .core import DatasetInfo, DatasetPack, register_dataset_factory


@register_dataset_factory(name="medmnist")
def dataset_factory(name: str = "pathmnist") -> DatasetPack:
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )
    train_dataset = DataClass(split="train", transform=transform, download=True)
    val_dataset = DataClass(split="val", transform=transform, download=True)
    test_dataset = DataClass(split="test", transform=transform, download=True)
    infos = DatasetInfo(
        name=name,
        num_classes=info,
        image_size=(info["n_channels"], 28, 28),
    )
    return train_dataset, val_dataset, test_dataset, infos
